import torch
from typing import Optional


class SDE(torch.nn.Module):
    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, drift, diffusion):
        super().__init__()
        self.drift = drift
        self.diffusion = diffusion

    def f(self, t, x, *args, **kwargs):
        if t.dim() == 0:
            # repeat the same time for all points if we have a scalar time
            t = t * torch.ones(x.shape[0]).to(x.device)

        return self.drift(t, x, *args, **kwargs)

    def g(self, t, x, *args, **kwargs):
        return self.diffusion(t, x, *args, **kwargs)


class VEReverseSDE(torch.nn.Module):
    """Variance Exploding (VE) Reverse Stochastic Differential Equation.

    This class implements the reverse-time SDE for variance exploding diffusion models.
    It defines the drift and diffusion terms used to generate samples by simulating
    the reverse diffusion process from noise to data.

    Attributes:
        noise_type (str): Type of noise used in the SDE, set to "diagonal".
        sde_type (str): Type of SDE formulation, set to "ito".
        score (callable): Function that computes the score (gradient of log probability).
        noise_schedule (object): Object that defines the noise schedule over time.
    """

    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, score, noise_schedule):
        """Initialize the VE Reverse SDE.

        Args:
            score (callable): Function that computes the score (gradient of log probability).
            noise_schedule: Object that defines the noise schedule over time.
        """
        super().__init__()
        self.score = score
        self.noise_schedule = noise_schedule

    def f(self, t, x, *args, **kwargs):
        """Compute the drift term of the SDE.

        The drift term is calculated as g(t,x)^2 * score(t,x), where g is the
        diffusion coefficient and score is the gradient of log probability.

        Args:
            t (torch.Tensor): Time points at which to evaluate the drift.
            x (torch.Tensor): State at which to evaluate the drift.
            *args: Additional positional arguments to pass to the score function.
            **kwargs: Additional keyword arguments to pass to the score function.

        Returns:
            torch.Tensor: The drift term evaluated at (t,x).
        """
        if t.dim() == 0:
            # repeat the same time for all points if we have a scalar time
            t = t * torch.ones(x.shape[0]).to(x.device)

        score = self.score(t, x, *args, **kwargs)
        return self.g(t, x).pow(2) * score

    def g(self, t, x, *args, **kwargs):
        """Compute the diffusion term of the SDE.

        The diffusion term determines the amount of noise added at each time step.

        Args:
            t (torch.Tensor): Time points at which to evaluate the diffusion.
            x (torch.Tensor): State at which to evaluate the diffusion.
            *args: Additional positional arguments (unused).
            **kwargs: Additional keyword arguments (unused).

        Returns:
            torch.Tensor: The diffusion coefficient evaluated at (t,x).
        """
        g = self.noise_schedule.g(t)
        return g.unsqueeze(1) if g.ndim > 0 else torch.full_like(x, g)


class RegVEReverseSDE(VEReverseSDE):
    def f(self, t, x, *args, **kwargs):
        dx = super().f(t, x[..., :-1], *args, **kwargs)
        quad_reg = 0.5 * dx.pow(2).sum(dim=-1, keepdim=True)
        return torch.cat([dx, quad_reg], dim=-1)

    def g(self, t, x, *args, **kwargs):
        g = self.noise_schedule.g(t)
        if g.ndim > 0:
            return g.unsqueeze(1)
        return torch.cat(
            [torch.full_like(x[..., :-1], g), torch.zeros_like(x[..., -1:])], dim=-1
        )
