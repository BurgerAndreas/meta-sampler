import torch
import numpy as np


def forward(samples):
    # this causes
    # RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.cuda.FloatTensor [8, 1, 2]]
    return torch.linalg.norm(samples)
    # this works fine
    # return torch.sum(samples**2).sqrt()


def pseudoenergy_function(samples):
    def _get_forward(_samples):
        return forward(_samples)

    # works fine
    # energies = torch.vmap(
    #     _get_forward, in_dims=(0)
    # )(samples)

    # works fine
    # forces = -1 * torch.vmap(
    #     torch.func.grad(_get_forward, argnums=0),
    #     in_dims=(0,),
    # )(samples)

    hessian = torch.vmap(
        torch.func.hessian(_get_forward, argnums=0),
        in_dims=(0,),
    )(samples)

    # some reduction to get shape [B]
    pseudoenergy = torch.sum(hessian, dim=(1, 2))

    return pseudoenergy


def log_expectation_reward_vmap(
    t: torch.Tensor,
    x: torch.Tensor,
    energy_function,
    num_mc_samples: int,
):
    repeated_t = t.unsqueeze(0).repeat_interleave(num_mc_samples, dim=0)  # [S]
    repeated_x = x.unsqueeze(0).repeat_interleave(num_mc_samples, dim=0)  # [S, D]

    # Add noise to positions
    h_t = torch.ones_like(repeated_t).unsqueeze(-1)  # [S, 1]
    samples = repeated_x + (torch.randn_like(repeated_x) * h_t.sqrt())  # [S, D]

    # Compute log rewards per MC sample [S]
    log_rewards = energy_function(samples)

    # Average log rewards over MC samples [1]
    reward_val = torch.logsumexp(log_rewards, dim=-1) - np.log(num_mc_samples)
    return reward_val


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # we have two "batch" dimensions: S=num_mc_samples and B=batch_size
    batch_size = 4
    num_mc_samples = 8
    dim = 2

    # Create test inputs
    t = torch.ones(batch_size, device=device) * 0.5  # [B]
    x = torch.randn(batch_size, dim, device=device)  # [B, D]

    # [B], [B,D] -> [], [D]
    grad_out = torch.vmap(
        # argnums=1 -> computes grad w.r.t. to x (zero indexed)
        torch.func.grad(log_expectation_reward_vmap, argnums=1),
        in_dims=(0, 0, None, None),
        randomness="different",
    )(t, x, pseudoenergy_function, num_mc_samples)

    print("grad_out", grad_out.shape)
