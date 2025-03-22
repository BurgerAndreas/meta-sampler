
change these functions to be able to use torch.vmap


mace/tools/scatter.py
```
def scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
    reduce: str = "sum",
) -> torch.Tensor:
    assert reduce == "sum"  # for now, TODO
    index = _broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        # return out.scatter_add_(dim, index, src)
        return torch.scatter_add(out, dim, index, src)
    else:
        # return out.scatter_add_(dim, index, src)
        return torch.scatter_add(out, dim, index, src)
```


e3nn.o3._spherical_harmonics
```
# @torch.jit.script
def _spherical_harmonics(lmax: int, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
```