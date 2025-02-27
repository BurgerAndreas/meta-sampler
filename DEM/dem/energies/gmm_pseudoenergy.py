import torch
import numpy as np
import copy
import os
import matplotlib

matplotlib.use("Agg")  # Force non-interactive backend
import matplotlib.pyplot as plt  # Import pyplot after setting backend

plt.ioff()  # Turn off interactive model

import scipy.optimize
from tqdm import tqdm
import wandb
from lightning.pytorch.loggers import WandbLogger

import fab.target_distributions.gmm
from dem.energies.base_energy_function import BaseEnergyFunction
from dem.energies.gmm_energy import GMMEnergy
from dem.utils.logging_utils import fig_to_image
from dem.utils.plotting import plot_fn, plot_marginal_pair


class GMMPseudoEnergy(GMMEnergy):
    """GMM pseudo-energy function to find transition points (index-1 saddle points).
    This function should be minimal at the transition points of some potential energy surface.

    Pseudo-energy that combines potential energy and force terms F=dU/dx,
    and possibly (approximations of) the second order derivatives (Hessian).

    Args:
        dimensionality (int): Dimension of input space
        energy_weight (float): Weight for energy term
        force_weight (float): Weight for force term
        force_exponent_eps (float): If force exponent is negative, add this value to the force magnitude to avoid division by zero. Higher value tends to smear out singularity around |force|=0.
        **gmm_kwargs: Additional arguments passed to GMM class
    """

    def __init__(
        self,
        energy_weight=1.0,
        force_weight=1.0,
        forces_norm=None,
        force_exponent=1,
        force_exponent_eps=1e-6,
        force_activation=None,
        hessian_weight=1.0,
        hessian_eigenvalue_penalty="softplus",
        term_aggr="sum",
        use_vmap=True,
        kbT=1.0,
        **gmm_kwargs,
    ):
        # Initialize GMMEnergy base class
        print(f"Initializing GMMPseudoEnergy with kwargs: {gmm_kwargs}")
        super().__init__(**copy.deepcopy(gmm_kwargs))

        self._is_molecule = False

        self.energy_weight = energy_weight
        self.force_weight = force_weight
        self.forces_norm = forces_norm
        self.force_exponent = force_exponent
        self.force_exponent_eps = force_exponent_eps
        self.force_activation = force_activation
        self.hessian_weight = hessian_weight
        self.hessian_eigenvalue_penalty = hessian_eigenvalue_penalty
        self.use_vmap = use_vmap
        self.term_aggr = term_aggr
        self.kbT = kbT
        # transition states of the GMM potential
        self.boundary_points = None
        self.transition_points = None
        self.validation_results = None

        # Accumulated training losses
        self.energy_loss_sum = 0.0
        self.force_loss_sum = 0.0
        self.hessian_loss_sum = 0.0
        self.n_loss_samples = 0

    def log_prob_energy(
        self, samples: torch.Tensor, return_aux_output: bool = False
    ) -> torch.Tensor:
        """Compute unnormalized log-probability of energy.
        Same as GMMEnergy.__call__.

        Args:
            x: Input positions tensor of shape (dimensionality,)
                When used with vmap, this will be automatically vectorized

        Returns:
            Negative of pseudo-energy value (scalar)
        """
        if self.should_unnormalize:
            samples = self.unnormalize(samples)

        # Compute log-probability of potential energy
        if return_aux_output:
            return self.gmm.log_prob(samples), {}
        return self.gmm.log_prob(samples)

    def gmm_potential(
        self, samples: torch.Tensor, return_aux_output: bool = False
    ) -> torch.Tensor:
        """Alias for log_prob_energy. Same as GMMEnergy.__call__."""
        return self.log_prob_energy(samples, return_aux_output=return_aux_output)

    def __call__(
        self, samples: torch.Tensor, return_aux_output: bool = False
    ) -> torch.Tensor:
        """Compute pseudo-energy combining energy, force, and Hessian terms.
        Returns unnormalized log-probability = -pseudo-energy.
        Similar to GMMEnergy.__call__.

        Args:
            x: Input positions tensor of shape (dimensionality,)
                When used with vmap, this will be automatically vectorized

        Returns:
            Negative of pseudo-energy value (scalar)
        """
        if self.should_unnormalize:
            samples = self.unnormalize(samples)

        return self.log_prob(samples, return_aux_output=return_aux_output)

    def log_prob(
        self, samples: torch.Tensor, return_aux_output: bool = False
    ) -> torch.Tensor:
        """Compute unnormalized log-probability of pseudo-energy.
        Corresponds to GMMEnergy.log_prob.

        Args:
            samples: Input positions tensor of shape (dimensionality,)
                When used with vmap, this will be automatically vectorized
            return_aux_output: Whether to return auxiliary outputs

        Returns:
            Negative of pseudo-energy value (scalar)
        """

        # Compute energy (all positive after -log_prob)
        if self.energy_weight > 0:
            energy = -self.gmm.log_prob(samples)
        else:
            energy = torch.zeros_like(samples[:, 0])

        # Compute force penalty 
        if self.force_weight > 0:
            # For computing forces, we need to sum the energy to get a scalar
            def energy_sum(x):
                return self.gmm.log_prob(x).sum()  # TODO: is sum right here?

            # Use functorch.grad to compute forces
            forces = -torch.func.grad(energy_sum)(samples)
            # Compute force magnitude
            force_magnitude = torch.linalg.norm(forces, ord=self.forces_norm, dim=-1)
            if self.force_exponent > 0:
                force_magnitude = force_magnitude**self.force_exponent
            else:
                force_magnitude = -(force_magnitude+self.force_exponent_eps)**self.force_exponent
            # force_magnitude += 1. # [0, inf] -> [1, inf]
            if self.force_activation == "tanh":
                force_magnitude = torch.tanh(force_magnitude)
            elif self.force_activation == "sigmoid":
                force_magnitude = torch.sigmoid(force_magnitude)
            elif self.force_activation == "softplus":
                force_magnitude = torch.nn.functional.softplus(force_magnitude)
            elif self.force_activation in [None, False]:
                pass
            else:
                raise ValueError(f"Invalid force_activation: {self.force_activation}")
        else:
            force_magnitude = torch.zeros_like(energy)

        # Compute two smallest eigenvalues of Hessian
        if self.hessian_weight > 0:
            if len(samples.shape) == 1:
                # Handle single sample
                hessian = torch.func.hessian(self.gmm.log_prob)(samples)
                eigenvalues = torch.linalg.eigvalsh(hessian)
                smallest_eigenvalues = torch.sort(eigenvalues, dim=-1)[0][:2]
            else:
                # Handle batched inputs using vmap # [B, D, D]
                batched_hessian = torch.vmap(torch.func.hessian(self.gmm.log_prob))(samples)  
                # Get eigenvalues for each sample in batch # [B, D]
                batched_eigenvalues = torch.linalg.eigvalsh(batched_hessian)  
                # Sort eigenvalues in ascending order for each sample # [B, D]
                batched_eigenvalues = torch.sort(batched_eigenvalues, dim=-1)[0]  
                # Get 2 smallest eigenvalues for each sample
                smallest_eigenvalues = batched_eigenvalues[..., :2]  # [B, 2]

            # def loss_fn(x):
            #     return self.gmm.log_prob(x)

            # # Get Hessian-vector product using functorch transforms
            # grad_fn = torch.func.grad(loss_fn)
            # def hvp(v):
            #     # Ensure v has the same shape as samples
            #     v = v.reshape(samples.shape)
            #     return torch.func.jvp(grad_fn, (samples,), (v,))[1]

            # # Run Lanczos on the HVP function
            # v0 = torch.randn_like(samples)
            # T, Q_mat = lanczos(hvp, v0, m=100)

            # # Compute eigenvalues of the tridiagonal matrix
            # eigenvalues = torch.linalg.eigvalsh(T)
            # smallest_eigenvalues = eigenvalues[:2]

            # Bias toward index-1 saddle points:
            # - First eigenvalue should be negative (minimize positive values)
            # - Second eigenvalue should be positive (minimize negative values)
            if self.hessian_eigenvalue_penalty == "softplus":
                # Using softplus which is differentiable everywhere but still creates one-sided penalties
                # if first eigenvalue > 0, increase energy
                ev1_bias = torch.nn.functional.softplus(smallest_eigenvalues[:, 0])
                # if second eigenvalue > 0, increase energy
                ev2_bias = torch.nn.functional.softplus(-smallest_eigenvalues[:, 1])
                saddle_bias = ev1_bias + ev2_bias
            elif self.hessian_eigenvalue_penalty == "relu":
                ev1_bias = torch.relu(smallest_eigenvalues[:, 0])
                ev2_bias = torch.relu(-smallest_eigenvalues[:, 1])
                saddle_bias = ev1_bias + ev2_bias
            # elif self.hessian_eigenvalue_penalty == 'heaviside':
            #     # 1 if smallest_eigenvalues[0] > 0 else 0
            #     ev1_bias = torch.heaviside(smallest_eigenvalues[:, 0], torch.tensor(0.))
            #     # 1 if smallest_eigenvalues[1] < 0 else 0
            #     ev2_bias = torch.heaviside(-smallest_eigenvalues[:, 1], torch.tensor(0.))
            #     saddle_bias = ev1_bias + ev2_bias
            elif self.hessian_eigenvalue_penalty == "sigmoid":
                ev1_bias = torch.sigmoid(smallest_eigenvalues[:, 0])
                ev2_bias = torch.sigmoid(-smallest_eigenvalues[:, 1])
                saddle_bias = ev1_bias + ev2_bias
            elif self.hessian_eigenvalue_penalty == "mult":
                # Penalize if both eigenvalues are positive or negative
                ev1_bias = smallest_eigenvalues[:, 0]
                ev2_bias = smallest_eigenvalues[:, 1]
                saddle_bias = ev1_bias * ev2_bias
            elif self.hessian_eigenvalue_penalty == "tanh":
                saddle_bias = torch.tanh(
                    smallest_eigenvalues[:, 0] * smallest_eigenvalues[:, 1]
                )
                saddle_bias = -torch.nn.functional.softplus(-saddle_bias-2.) # [0, 1]
                saddle_bias += 1. # [1, 2]
            elif self.hessian_eigenvalue_penalty == "and":
                saddle_bias = torch.nn.functional.softplus(
                    smallest_eigenvalues[:, 0] * smallest_eigenvalues[:, 1] * 10.
                ) # [0, inf], 0 if good
                saddle_bias = torch.tanh(saddle_bias) # [0, 1]
                # saddle_bias = 1 - saddle_bias # [0, 1] -> [1, 0] # 1 if good
            elif self.hessian_eigenvalue_penalty == "tanh_mult":
                # Penalize if both eigenvalues are positive or negative
                # tanh: ~ -1 for negative, 1 for positive
                # both neg -> 1, both pos -> 1, one neg one pos -> 0
                ev1_bias = torch.tanh(smallest_eigenvalues[:, 0])
                ev2_bias = torch.tanh(smallest_eigenvalues[:, 1])
                saddle_bias = ev1_bias * ev2_bias # [-1, 1]
                # saddle_bias += 1. # [0, 2]
                # saddle_bias = -torch.nn.functional.softplus(-saddle_bias-2.) # [0, 1]
                # saddle_bias += 1. # [1, 2]
            else:
                raise ValueError(
                    f"Invalid penalty function: {self.hessian_eigenvalue_penalty}"
                )
        else:
            # No penalty
            saddle_bias = torch.zeros_like(energy)

        # idx = torch.randperm(samples.shape[0])[:15]
        # for i in idx:
        #     print(f"evs={smallest_eigenvalues[i].tolist()} -> {saddle_bias[i]:.2f}")

        # ensure we have one value per batch
        assert energy.shape == force_magnitude.shape == saddle_bias.shape, \
            f"samples={samples.shape}, energy={energy.shape}, forces={force_magnitude.shape}, hessian={saddle_bias.shape}"
        assert energy.shape[0] == samples.shape[0], \
            f"samples={samples.shape}, energy={energy.shape}, forces={force_magnitude.shape}, hessian={saddle_bias.shape}"

        # Combine loss terms
        energy_loss = self.energy_weight * energy
        force_loss = self.force_weight * force_magnitude
        hessian_loss = self.hessian_weight * saddle_bias

        if self.term_aggr == "sum":
            total_loss = energy_loss + force_loss + hessian_loss
        elif "norm" in self.term_aggr: # "2_norm"
            # p‑Norms provide a way to interpolate between simple addition (p=1) and the maximum function (as p→∞)
            p = float(self.term_aggr.split("_")[0])
            total_loss = (energy_loss**p + force_loss**p + hessian_loss**p)**(1/p)
        elif self.term_aggr == "logsumexp":
            # Smooth Maximum via Log‑Sum‑Exp
            total_loss = torch.logsumexp(torch.stack([energy_loss, force_loss, hessian_loss], dim=-1), dim=-1)
        elif self.term_aggr == "mult":
            total_loss = energy_loss * force_loss * hessian_loss
        elif self.term_aggr == "multfh":
            # multiply acts like an `and` operation
            total_loss = energy_loss + (force_loss * hessian_loss)
        elif self.term_aggr == "1mmultfh":
            # multiply acts like an `and` operation
            # total_loss = 1 - energy_loss + (force_loss * hessian_loss)
            total_loss = energy_loss + 1 - ((1-force_loss) * (1-hessian_loss))
            # total_loss += 2. # [0, inf] -> [1, inf]
        else:
            raise ValueError(f"Invalid term_aggr: {self.term_aggr}")

        
        # Boltzmann distribution
        # increase temperature to wash out the pseudo-potential
        total_loss /= self.kbT

        total_loss *= -1. # log(P) ~ -E
        if return_aux_output:
            aux_output = {
                "energy_loss": energy_loss,
                "force_loss": force_loss,
                "hessian_loss": hessian_loss,
            }
            return total_loss, aux_output
        return total_loss

    def log_on_epoch_end(self, *args, **kwargs):

        # First plot the original GMM energy surface
        super().log_on_epoch_end(*args, **kwargs)
        print(f"Plotted GMM energy surface at epoch {self.curr_epoch}")

        # Now plot the pseudo-energy surface
        if (
            kwargs.get("wandb_logger") is not None
            and kwargs.get("latest_samples") is not None
        ):
            wandb_logger = kwargs["wandb_logger"]
            latest_samples = kwargs["latest_samples"]
            prefix = kwargs.get("prefix", "")

            if len(prefix) > 0 and prefix[-1] != "/":
                prefix += "/"

            if self.curr_epoch % self.plot_samples_epoch_period == 0:
                fig, ax = plt.subplots(1, 1, figsize=(8, 8))

                # Create grid for contour plot
                bounds = (-1.4 * 40, 1.4 * 40)
                x = np.linspace(bounds[0], bounds[1], 200)
                y = np.linspace(bounds[0], bounds[1], 200)
                X, Y = np.meshgrid(x, y)

                # Evaluate pseudo-energy on grid points
                grid_points = torch.tensor(
                    np.stack([X.flatten(), Y.flatten()], axis=1),
                    dtype=torch.float32,
                    device=self.device,
                )
                Z = torch.vmap(self.__call__)(grid_points).reshape(X.shape)

                # Plot contours of pseudo-energy
                ax.contour(X, Y, Z.detach().cpu().numpy(), levels=50)

                # Plot samples if available
                if latest_samples is not None:
                    print(
                        f"latest samples in range: {latest_samples.min().item()} to {latest_samples.max().item()}"
                    )
                    ax.scatter(*latest_samples.detach().cpu().T, alpha=0.5)

                ax.set_title("Pseudo-energy Surface")
                wandb_logger.log_image(
                    f"{prefix}pseudo_energy_surface", [fig_to_image(fig)]
                )
                plt.close()
                # del fig, ax, img
                print(f"Plotted pseudo-energy surface at epoch {self.curr_epoch}")

    def get_single_dataset_fig(
        self,
        samples,
        name,
        n_contour_levels=50,
        plotting_bounds=(-1.4 * 40, 1.4 * 40),
        plot_gaussian_means=False,
        grid_width_n_points=200,
        plot_style="contours",
        with_legend=False,
        plot_prob_kwargs={},
        plot_sample_kwargs={},
        colorbar=False,
        do_exp=False, 
    ):
        """Creates visualization of samples against GMM contours.
        Used in train.py for sample visualization.

        Args:
            samples (torch.Tensor): Samples to plot
            name (str): Title for plot
            plotting_bounds (tuple, optional): Plot bounds. Defaults to (-1.4*40, 1.4*40)
            plot_gaussian_means (bool, optional): Whether to plot the Gaussian centers of the potential. Defaults to False.
            grid_width_n_points (int, optional): Number of points along each dimension for the grid. Defaults to 200.
            plot_style (str, optional): Plot style. Defaults to "contours".
            with_legend (bool, optional): Whether to show the legend. Defaults to False.
            plot_prob_kwargs (dict, optional): Keyword arguments for the plot function. Defaults to {}.
            plot_sample_kwargs (dict, optional): Keyword arguments for the plot function. Defaults to {}.
            colorbar (bool, optional): Whether to show the colorscale of the plot. Defaults to False.
            do_exp (bool, optional): Whether to exponentiate the log probability. Defaults to False.
        
        Returns:
            numpy.ndarray: Plot as image array
        """
        plt.close()
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        self.gmm.to("cpu")
        # self.gmm.to("cpu")
        plot_fn(
            self.log_prob,
            bounds=plotting_bounds,
            ax=ax,
            plot_style=plot_style,
            n_contour_levels=n_contour_levels,
            grid_width_n_points=grid_width_n_points,
            plot_kwargs=plot_prob_kwargs,
            colorbar=colorbar,
            do_exp=do_exp,
        )
        if samples is not None:
            samples = samples.to("cpu")
            plot_marginal_pair(
                samples, ax=ax, bounds=plotting_bounds, 
                plot_kwargs=plot_sample_kwargs
            )
        if name is not None:
            ax.set_title(f"{name}")

        if plot_gaussian_means:
            means = self.gmm.distribution.component_distribution.loc
            ax.scatter(*means.detach().cpu().T, color="red", marker="x")
            # ax.legend()

        self.gmm.to(self.device)
        # self.gmm.to(self.device)

        if with_legend:
            ax.legend()

        return fig_to_image(fig)

    # TODO: plot gmm or pseudo-energy?
    def get_dataset_fig(
        self,
        samples,
        gen_samples=None,
        n_contour_levels=50,
        plotting_bounds=(-1.4 * 40, 1.4 * 40),
        plot_gaussian_means=False,
        plot_style="contours",
        plot_prob_kwargs={},
        plot_sample_kwargs={},
        colorbar=False,
        do_exp=False, 
    ):
        """Creates side-by-side visualization of buffer and generated samples.
        Used in train.py for comparing sample distributions.

        Args:
            samples (torch.Tensor): Buffer samples to plot
            gen_samples (torch.Tensor, optional): Generated samples to plot. Defaults to None
            plotting_bounds (tuple, optional): Plot bounds. Defaults to (-1.4*40, 1.4*40)

        Returns:
            numpy.ndarray: Plot as image array
        """
        plt.close()
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))

        self.gmm.to("cpu")
        plot_fn(
            self.gmm.log_prob,
            bounds=plotting_bounds,
            ax=axs[0],
            plot_style=plot_style,
            n_contour_levels=n_contour_levels,
            grid_width_n_points=200,
            plot_kwargs=plot_prob_kwargs,
            colorbar=colorbar,
            do_exp=do_exp,
        )

        # plot dataset samples
        if samples is not None:
            plot_marginal_pair(samples, ax=axs[0], bounds=plotting_bounds)
            axs[0].set_title("Buffer")

        if gen_samples is not None:
            plot_fn(
                self.gmm.log_prob,
                bounds=plotting_bounds,
                ax=axs[1],
                plot_style=plot_style,
                n_contour_levels=50,
                grid_width_n_points=200,
                plot_kwargs=plot_prob_kwargs,
                colorbar=colorbar,
                do_exp=do_exp,
            )
            # plot generated samples
            plot_marginal_pair(gen_samples, ax=axs[1], bounds=plotting_bounds, plot_kwargs=plot_sample_kwargs)
            axs[1].set_title("Generated samples")

        if plot_gaussian_means:
            means = self.gmm.distribution.component_distribution.loc
            axs[1].scatter(*means.detach().cpu().T, color="red", marker="x")
            # axs[1].legend()

        # delete subplot
        else:
            fig.delaxes(axs[1])

        self.gmm.to(self.device)

        return fig_to_image(fig)

    def log_samples(
        self,
        samples: torch.Tensor,
        wandb_logger: WandbLogger,
        name: str = "",
        should_unnormalize: bool = False,
    ) -> None:
        """Logs sample visualizations against the pseudopotential energy surface.
        Used in train.py for logging sample distributions.

        Args:
            samples (torch.Tensor): Samples to visualize
            wandb_logger (WandbLogger): Logger for visualizations
            name (str, optional): Name for logged images. Defaults to ""
            should_unnormalize (bool, optional): Whether to unnormalize samples. Defaults to False
        """
        if wandb_logger is None:
            return

        if self.should_unnormalize and should_unnormalize:
            samples = self.unnormalize(samples)

        # Create visualization showing samples against pseudopotential surface
        plt.close()
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        # Plot contours of pseudopotential energy surface
        plot_fn(
            self.log_prob,
            bounds=(-1.4 * 40, 1.4 * 40),
            ax=ax,
            plot_style="contours",
            n_contour_levels=50,
            grid_width_n_points=200,
        )

        # Plot samples as scatter points
        if samples is not None:
            plot_marginal_pair(samples, ax=ax, bounds=(-1.4 * 40, 1.4 * 40))

        if name:
            ax.set_title(name)

        # Log the visualization
        wandb_logger.log_image(f"{name}", [fig_to_image(fig)])
        plt.close()

    ###########################################################################
    # Ground truth transition states
    ###########################################################################

    def find_transition_boundaries(self, grid_size=200, bounds=(-56, 56)):
        """Find candidate transition points by detecting boundary cells.

        Decision Boundary Approximation via Log-Likelihood Contours
        Generate a fine grid over the 2D space
        Compute the log-likelihood of each Gaussian component at each grid point.
        Find boundary regions where the highest-likelihood component switches.
        Extract transition points by detecting level-set intersections

        Args:
            grid_size (int): Number of points along each dimension
            bounds (tuple): (min, max) bounds for grid

        Returns:
            torch.Tensor: Coordinates of identified boundary points
        """
        # Create grid
        x = np.linspace(bounds[0], bounds[1], grid_size)
        y = np.linspace(bounds[0], bounds[1], grid_size)
        X, Y = np.meshgrid(x, y)
        grid_points = torch.tensor(
            np.stack([X.flatten(), Y.flatten()], axis=1),
            dtype=torch.float32,
            device=self.device,
        )

        # Get GMM parameters directly from distribution object
        mix = self.gmm.distribution.mixture_distribution
        comp = self.gmm.distribution.component_distribution
        means = comp.loc
        covs = comp.covariance_matrix
        weights = mix.probs

        # Compute log-likelihood for each component
        log_probs = []
        for i in range(len(means)):
            mean = means[i]
            cov = covs[i]
            weight = weights[i]
            diff = grid_points - mean
            log_prob = (
                -0.5 * (torch.matmul(diff, torch.inverse(cov)) * diff).sum(1)
                - 0.5 * torch.log(torch.det(cov))
                + torch.log(weight)
            )
            log_probs.append(log_prob)

        log_probs = torch.stack(log_probs)  # (n_components, n_grid_points)

        # Determine dominant component at each grid point
        dominant_components = torch.argmax(log_probs, dim=0)
        dominant_components = dominant_components.reshape(grid_size, grid_size)

        # Collect cells witch match boundary conditions
        # Collect candidate transition points where the dominant component changes
        candidate_points = []
        for i in range(grid_size - 1):
            for j in range(grid_size - 1):
                if (dominant_components[i, j] != dominant_components[i + 1, j]) or (
                    dominant_components[i, j] != dominant_components[i, j + 1]
                ):
                    x_coord = (x[j] + x[j + 1]) / 2
                    y_coord = (y[i] + y[i + 1]) / 2
                    candidate_points.append([x_coord, y_coord])

        if len(candidate_points) == 0:
            return torch.empty((0, self.dimensionality), device=self.device)

        return torch.tensor(candidate_points, device=self.device, dtype=torch.float32)

    def validate_transition_states(
        self, transition_points, abs_ev_tol=1e-6, grad_tol=1e-1
    ):
        """
        Validate transition states by checking that
        - the gradient is near zero (extremal point or saddle point)
        - exactly one Hessian eigenvalue is negative.
        True transition states (index-1 saddle points) should have exactly one negative eigenvalue.

        Args:
            transition_points (torch.Tensor):
                Tensor of shape (n_points, dimensionality) containing candidate transition state coordinates

        Returns:
            dict: Dictionary containing:
                - 'valid_points': Tensor of validated transition state coordinates
                - 'eigenvals': Corresponding Hessian eigenvalues
                - 'accuracy': Fraction of points that were true transition states
        """

        def neg_log_prob(x):
            # GMM negative log probability
            log_prob = self.gmm.log_prob(x)
            return -log_prob

        def compute_grad_and_hessian(x):
            grad = torch.func.grad(neg_log_prob)(x)
            hessian = torch.func.hessian(neg_log_prob)(x)
            return grad, hessian

        valid_points = []
        valid_point_eigenvals = []
        all_eigenvals = []
        all_grad_norms = []

        for point in transition_points:
            point = point.requires_grad_(True)
            grad, hessian = compute_grad_and_hessian(point)
            eigenvals, _ = torch.linalg.eigh(hessian)

            # 1) Check if gradient is small => near stationary
            if grad.norm() < grad_tol:

                # 2) Check for exactly one negative eigenvalue
                n_negative = torch.sum(eigenvals < -abs_ev_tol)
                if n_negative == 1:
                    valid_points.append(point.detach())
                    valid_point_eigenvals.append(eigenvals.detach())

            all_eigenvals.append(eigenvals.detach())
            all_grad_norms.append(grad.norm().item())
            # print(f"Point: {[round(p, 2) for p in point.tolist()]}")
            # print(f"Gradient: {[round(g, 2) for g in grad.tolist()]}, norm={round(grad.norm().item(), 2)}")
            # print(f"Hessian: {[ [round(h, 2) for h in row] for row in hessian.tolist()]}")
            # print(f"Eigenvalues: {[round(e, 2) for e in eigenvals.tolist()]}")

        # print(f"Smallest gradient norms: {sorted(all_grad_norms)[:10]}")

        if len(valid_points) > 0:
            valid_points = torch.stack(valid_points)
            valid_point_eigenvals = torch.stack(valid_point_eigenvals)
            accuracy = len(valid_points) / len(transition_points)
        else:
            valid_points = torch.tensor([], device=self.device)
            valid_point_eigenvals = torch.tensor([], device=self.device)
            accuracy = 0.0

        return {
            "valid_points": valid_points,
            "eigenvals": valid_point_eigenvals,
            "all_eigenvals": all_eigenvals,
            "saddle_boundary_ratio": accuracy,
        }

    def get_true_transition_states(self, grid_size=200, bounds=(-56, 56)):
        """Find saddle points using scipy.optimize.root and Hessian eigenvalue analysis.

        Args:
            grid_size (int, optional): Number of points along each dimension
            bounds (tuple, optional): (min, max) bounds for grid

        Returns:
            torch.Tensor: Coordinates of identified saddle points
        """
        if self.transition_points is not None:
            return self.transition_points
        
        fname = f"dem_outputs/transition_points_gmm.npy"
        if os.path.exists(fname):
            self.transition_points = torch.tensor(np.load(fname), device=self.device)
            # print(f"Loaded transition points from {fname}")
            return self.transition_points

        # Generate candidate points
        if self.boundary_points is None:
            self.boundary_points = self.find_transition_boundaries()

        # Now validate these candidate points by checking the Hessian eigenvalues.
        # Only keep the true index-1 saddles (one negative eigenvalue).
        validation_results = self.validate_transition_states(
            self.boundary_points, abs_ev_tol=1e-6, grad_tol=1e1
        )
        candidate_points = validation_results["valid_points"]

        # Find stationary points and validate them as saddle points
        saddle_points = []
        for point in tqdm(candidate_points, total=len(candidate_points)):
            # Find stationary point
            result = find_stationary_point(point, self.gmm)

            if not result.success:
                continue

            # Convert to torch tensor for Hessian computation
            x = torch.tensor(
                result.x, dtype=torch.float32, requires_grad=True, device=self.device
            )

            # Compute Hessian and check eigenvalues
            hessian = torch.func.hessian(lambda x: -self.gmm.log_prob(x))(x)
            eigenvals = torch.linalg.eigvalsh(hessian)

            # Check for index-1 saddle point (exactly one negative eigenvalue)
            n_negative = torch.sum(eigenvals < -1e-6)
            if n_negative == 1:
                saddle_points.append(result.x)

        # Remove duplicates by converting to numpy, using unique, and converting back to torch
        unique_saddle_points = np.unique(np.array(saddle_points), axis=0)
        self.transition_points = torch.tensor(
            unique_saddle_points, device=self.device, dtype=torch.float32
        )
        # save them to file
        np.save(fname, self.transition_points.detach().cpu().numpy())
        print(f"Found {len(self.transition_points)} transition states")
        return self.transition_points


def find_stationary_point(initial_guess, gmm, method="hybr"):
    """
    Solve grad_neg_log_prob(x) = 0 starting from initial_guess.

    Args:
        initial_guess: np.ndarray of shape (dim,)
        gmm: instance of GMM class
        method: solver method (e.g., 'hybr', 'lm', etc.)
    Returns:
        result: scipy.optimize.OptimizeResult object
    """
    # Convert initial guess to numpy if it's a torch tensor
    if isinstance(initial_guess, torch.Tensor):
        initial_guess = initial_guess.detach().cpu().numpy()

    result = scipy.optimize.root(
        fun=grad_neg_log_prob, x0=initial_guess, args=(gmm,), method=method
    )
    return result


def grad_neg_log_prob(x_np, gmm):
    """
    Compute gradient of negative log probability.

    Args:
        x_np: numpy array of shape (dim,) -- current guess
        gmm: instance of GMM class with a .log_prob() method
    Returns:
        grad: numpy array of shape (dim,) -- gradient of -log p(x)
    """
    # Convert from numpy to torch tensor
    if not isinstance(x_np, torch.Tensor):
        x_torch = torch.tensor(
            x_np, dtype=torch.float32, requires_grad=True, device=gmm.device
        )
    else:
        x_torch = x_np

    # Negative log probability
    neg_log_p = -gmm.log_prob(x_torch)

    # Compute gradient
    grad_x = torch.autograd.grad(neg_log_p, x_torch, create_graph=False)[0]

    # Move gradient back to CPU and convert to numpy
    return grad_x.detach().cpu().numpy()
