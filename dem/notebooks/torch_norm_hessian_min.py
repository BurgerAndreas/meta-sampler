import torch
import numpy as np


def forward(samples):
    # this causes
    # RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation: [torch.cuda.FloatTensor [8, 1, 2]]
    return torch.linalg.norm(samples)

    # this works fine
    # return torch.sum(samples**2).sqrt()


def pseudoenergy_function(samples):

    # works fine with torch.linalg.norm
    # energies = torch.vmap(
    #     forward, in_dims=(0)
    # )(samples)

    # works fine with torch.linalg.norm
    # forces = -1 * torch.vmap(
    #     torch.func.grad(forward, argnums=0),
    #     in_dims=(0,),
    # )(samples)

    hessian = torch.vmap(
        torch.func.hessian(forward, argnums=0),
        in_dims=(0,),
    )(samples)

    # some reduction to get shape [1]
    pseudoenergy = torch.sum(hessian)

    return pseudoenergy


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.autograd.set_detect_anomaly(True)

    batch_size = 8
    dim = 2

    # Create test inputs
    x = torch.randn(batch_size, dim, device=device)  # [B, D]

    grad_out = torch.func.grad(pseudoenergy_function, argnums=0)(x)

    print("grad_out", grad_out.shape)
