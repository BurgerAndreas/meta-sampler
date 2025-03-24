
Change these functions to be able to use torch.vmap


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


mace/modules/utils.py
```
def get_edge_vectors_and_lengths(
    positions: torch.Tensor,  # [n_nodes, 3]
    edge_index: torch.Tensor,  # [2, n_edges]
    shifts: torch.Tensor,  # [n_edges, 3]
    normalize: bool = False,
    eps: float = 1e-9,
) -> Tuple[torch.Tensor, torch.Tensor]:
    sender = edge_index[0]
    receiver = edge_index[1]
    vectors = positions[receiver] - positions[sender] + shifts  # [n_edges, 3]
    # lengths = torch.linalg.norm(vectors, dim=-1, keepdim=True)  # [n_edges, 1]
    lengths = torch.sum(vectors**2, dim=-1, keepdim=True).sqrt()
```


e3nn.o3._spherical_harmonics
```
# @torch.jit.script
def _spherical_harmonics(lmax: int, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
```


e3nn/o3/_spherical_harmonics.py
```
if self.normalize:
    # x = torch.nn.functional.normalize(x, dim=-1)  # forward 0's instead of nan for zero-radius
    
    def manual_normalize(x, p=2, dim=1, eps=1e-12):
        # Compute p-norm using torch.sum
        norm = torch.sum(x.abs()**p, dim=dim, keepdim=True)**(1./p)
        norm = norm.clamp(min=eps)  # Avoid division by zero
        return x / norm
    x = manual_normalize(x, dim=-1)

if self.normalization == 'integral':
    # mul_ -> mul
    sh.mul(torch.cat([
        (math.sqrt(2 * l + 1) / math.sqrt(4 * math.pi)) * torch.ones(2 * l + 1, dtype=sh.dtype, device=sh.device)
        for l in self._ls_list
    ]))
elif self.normalization == 'component':
    # mul_ -> mul
    sh.mul(torch.cat([
        math.sqrt(2 * l + 1) * torch.ones(2 * l + 1, dtype=sh.dtype, device=sh.device)
        for l in self._ls_list
    ]))
```




