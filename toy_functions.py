import numpy as np
import torch
import plotly.graph_objects as go

# ---------------------------
# 1. Define our toy 2D potential_energy_surface and its pseudo_energy.
# ---------------------------
def toy_polynomial(x: torch.Tensor) -> torch.Tensor:
    # x is a tensor of shape (batch, 2) representing (x, y) coordinates.
    # Define f(x,y) = (x^2 - 1)^3 + (y^2 - 1.3)^3.
    x.requires_grad_()
    pes = (x[:, 0] ** 2 - 1) ** 3 + (x[:, 1] ** 2 - 1.3) ** 3

    # pseudo_energy is defined as the
    # norm of the analytical gradient of the potential_energy_surface.
    # The gradients are:
    #   df/dx = 6*x*(x^2-1)^2
    #   df/dy = 6*y*(y^2-2)^2
    grad_x = 6 * x[:, 0] * (x[:, 0] ** 2 - 1) ** 2
    grad_y = 6 * x[:, 1] * (x[:, 1] ** 2 - 1.3) ** 2
    pes_grad_norm = torch.sqrt(grad_x**2 + grad_y**2)
    return pes, pes_grad_norm


def six_hump_camel(point: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Six-Hump Camel function.

    Parameters
    ----------
    point : torch.Tensor
        A 2D point (x, y).

    Returns
    -------
    torch.Tensor
        The value of the Six-Hump Camel function at the given point.
        
    torch.Tensor
        The gradient norm of the Six-Hump Camel function at the given point.

    Notes
    -----
    The function is typically defined by:
        f(x, y) = (4 - 2.1 * x^2 + (x^4) / 3) * x^2 + x * y + (-4 + 4 * y^2) * y^2.

    It has multiple local minima and saddle points.
    The usual recommended domain is x in [-3, 3], y in [-2, 2].
    """
    # Unpack the coordinates
    if isinstance(point, tuple):
        x = torch.tensor(point)
    elif isinstance(point, torch.Tensor):
        x = point
    elif isinstance(point, np.ndarray):
        x = torch.tensor(point)
    else:
        raise ValueError("Invalid input type. Please provide a tuple or a torch.Tensor.")
    assert x.shape[1] == 2, f"x must be a [N, 2], but got shape {x.shape}"
    x.requires_grad_()

    # Compute each term
    term1 = (4 - 2.1 * x[:, 0]**2 + (x[:, 0]**4) / 3.0) * x[:, 0]**2
    term2 = x[:, 0] * x[:, 1]
    term3 = (-4 + 4 * x[:, 1]**2) * x[:, 1]**2
    y = term1 + term2 + term3
    
    # Compute the gradients using PyTorch's autograd.
    grad_f_x = torch.autograd.grad(
        outputs=y, inputs=x, 
        grad_outputs=torch.ones_like(y).to(x.device), create_graph=True
    )[0]
    # Compute the gradient norm, which serves as the pseudo_energy.
    grad_norm = torch.linalg.norm(grad_f_x, dim=1)

    return y, grad_norm




# Example usage:
if __name__ == "__main__":
    pts_to_test = torch.tensor([
        (0.0, 0.0),
        (0.0898, -0.7126),  # Approx. one global minimum
        (-0.0898, 0.7126),  # Approx. another global minimum
    ])
    
    print("Testing toy_polynomial")
    
    print("pts_to_test.shape", pts_to_test.shape)
    
    pes, pes_grad_norm = toy_polynomial(pts_to_test)
    print(f"pes = \n{pes}")
    print(f"pes_grad_norm = \n{pes_grad_norm}")
    
    pes, pes_grad_norm = six_hump_camel(pts_to_test)
    print(f"pes = \n{pes}")
    print(f"pes_grad_norm = \n{pes_grad_norm}")
    
    ##################################################################
    
    pes_fn = six_hump_camel
    plot_pes_and_gradient(pes_fn)
