import torch
import numpy as np
import matplotlib.pyplot as plt
from fab.target_distributions import gmm
from fab.utils.plotting import plot_contours, plot_marginal_pair
from dem.energies.base_energy_function import BaseEnergyFunction
from dem.energies.gmm_energy import GMM
from dem.utils.logging_utils import fig_to_image
import scipy.optimize
from tqdm import tqdm


class GMMPseudoEnergy(BaseEnergyFunction):
    """GMM pseudo-energy function to find transition points (index-1 saddle points).
    This function should be minimal at the transition points of some potential energy surface.

    Pseudo-energy that combines potential energy and force terms F=dU/dx,
    and possibly (approximations of) the second order derivatives (Hessian).

    Args:
        dimensionality (int): Dimension of input space
        energy_weight (float): Weight for energy term
        force_weight (float): Weight for force term
        **gmm_kwargs: Additional arguments passed to GMM class
    """

    def __init__(
        self,
        dimensionality=2,
        energy_weight=1.0,
        force_weight=1.0,
        use_vmap=True,
        device="cuda",
        **gmm_kwargs,
    ):
        # Create GMM instance first
        self.gmm_potential = GMM(
            dimensionality=dimensionality, device=device, **gmm_kwargs
        )
        # Then initialize base class
        super().__init__(dimensionality=dimensionality)

        self.energy_weight = energy_weight
        self.force_weight = force_weight
        self.use_vmap = use_vmap
        self.device = device

        self.boundary_points = None
        self.transition_points = None
        self.validation_results = None
        # self.update_transition_states()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Compute pseudo-energy combining energy and force terms.

        Args:
            x: Input positions tensor of shape (dimensionality,)
                When used with vmap, this will be automatically vectorized

        Returns:
            Pseudo-energy value (scalar)
        """
        # Compute energy
        energy = self.gmm_potential(x)

        # For computing forces, we need to sum the energy to get a scalar
        def energy_sum(x):
            return self.gmm_potential(x).sum()

        # Use functorch.grad to compute forces
        forces = -torch.func.grad(energy_sum)(x)

        # Compute force magnitude
        force_magnitude = torch.norm(forces)

        # Combine energy and force terms
        pseudo_energy = (
            self.energy_weight * energy + self.force_weight * force_magnitude
        )

        return pseudo_energy

    # Inherit other methods from GMM
    def setup_test_set(self):
        return self.gmm_potential.setup_test_set()

    def setup_train_set(self):
        return self.gmm_potential.setup_train_set()

    def setup_val_set(self):
        return self.gmm_potential.setup_val_set()

    def log_on_epoch_end(self, *args, **kwargs):
        # First plot the original GMM energy surface
        gmm_result = self.gmm_potential.log_on_epoch_end(*args, **kwargs)

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

            if self.curr_epoch % self.gmm_potential.plot_samples_epoch_period == 0:
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
                    ax.scatter(*latest_samples.detach().cpu().T, alpha=0.5)

                ax.set_title("Pseudo-energy Surface")
                wandb_logger.log_image(
                    f"{prefix}pseudo_energy_surface", [fig_to_image(fig)]
                )
                plt.close()

        return gmm_result

    def log_samples(self, *args, **kwargs):
        return self.gmm_potential.log_samples(*args, **kwargs)

    ###########################################################################
    # Ground truth transition states
    ###########################################################################

    def update_transition_states(self, abs_ev_tol=1e-3, grad_tol=1e-3):
        # Find transition states
        self.boundary_points = self.find_transition_boundaries()

        # Now validate these candidate points by checking the Hessian eigenvalues.
        # Only keep the true index-1 saddles (one negative eigenvalue).
        self.validation_results = self.validate_transition_states(
            self.boundary_points, abs_ev_tol=abs_ev_tol, grad_tol=grad_tol
        )
        self.transition_points = self.validation_results["valid_points"]

        print(f"Found {len(self.transition_points)} transition states")

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
        mix = self.gmm_potential.gmm.distribution.mixture_distribution
        comp = self.gmm_potential.gmm.distribution.component_distribution
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
            log_prob = self.gmm_potential.gmm.log_prob(x)
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

    def find_saddle_points_scipy(self, grid_size=200, bounds=(-56, 56)):
        """Find saddle points using scipy.optimize.root and Hessian eigenvalue analysis.

        Args:
            grid_size (int, optional): Number of points along each dimension
            bounds (tuple, optional): (min, max) bounds for grid

        Returns:
            torch.Tensor: Coordinates of identified saddle points
        """
        # Generate grid of initial points
        # x = np.linspace(bounds[0], bounds[1], grid_size)
        # candidate_points = np.array(np.meshgrid(*[x for _ in range(self.dimensionality)])).T.reshape(-1, self.dimensionality)

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
            result = find_stationary_point(point, self.gmm_potential.gmm)

            if not result.success:
                continue

            # Convert to torch tensor for Hessian computation
            x = torch.tensor(
                result.x, dtype=torch.float32, requires_grad=True, device=self.device
            )

            # Compute Hessian and check eigenvalues
            hessian = torch.func.hessian(lambda x: -self.gmm_potential.gmm.log_prob(x))(
                x
            )
            eigenvals = torch.linalg.eigvalsh(hessian)

            # Check for index-1 saddle point (exactly one negative eigenvalue)
            n_negative = torch.sum(eigenvals < -1e-6)
            if n_negative == 1:
                saddle_points.append(result.x)

        if len(saddle_points) > 0:
            return torch.tensor(saddle_points, device=self.device, dtype=torch.float32)
        else:
            return torch.tensor([], device=self.device)


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
