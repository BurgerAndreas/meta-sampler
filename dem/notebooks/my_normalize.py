import torch


def manual_normalize(x, p=2, dim=1, eps=1e-12):
    # Compute p-norm using torch.sum
    norm = torch.sum(x.abs() ** p, dim=dim, keepdim=True) ** (1.0 / p)
    norm = norm.clamp(min=eps)  # Avoid division by zero
    return x / norm


if __name__ == "__main__":
    x = torch.randn(10, 10)

    xnorm = manual_normalize(x, dim=-1)

    xtrue = torch.nn.functional.normalize(x, dim=-1)

    print(xnorm - xtrue)
