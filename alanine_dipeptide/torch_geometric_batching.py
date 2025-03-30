from typing import List, Optional, Any, Union, Tuple

import torch
from torch import Tensor

# excerpts from torch geometric

WITH_PT20 = int(torch.__version__.split('.')[0]) >= 2
WITH_PT21 = WITH_PT20 and int(torch.__version__.split('.')[1]) >= 1
WITH_PT22 = WITH_PT20 and int(torch.__version__.split('.')[1]) >= 2
WITH_PT23 = WITH_PT20 and int(torch.__version__.split('.')[1]) >= 3
WITH_PT24 = WITH_PT20 and int(torch.__version__.split('.')[1]) >= 4
WITH_PT25 = WITH_PT20 and int(torch.__version__.split('.')[1]) >= 5
WITH_PT111 = WITH_PT20 or int(torch.__version__.split('.')[1]) >= 11
WITH_PT112 = WITH_PT20 or int(torch.__version__.split('.')[1]) >= 12
WITH_PT113 = WITH_PT20 or int(torch.__version__.split('.')[1]) >= 13

def is_torch_sparse_tensor(src: Any) -> bool:
    r"""Returns :obj:`True` if the input :obj:`src` is a
    :class:`torch.sparse.Tensor` (in any sparse layout).

    Args:
        src (Any): The input object to be checked.
    """
    if isinstance(src, Tensor):
        if src.layout == torch.sparse_coo:
            return True
        if src.layout == torch.sparse_csr:
            return True
        if (WITH_PT112
                and src.layout == torch.sparse_csc):
            return True
    return False

def maybe_num_nodes(
    edge_index: Union[Tensor, Tuple[Tensor, Tensor]],
    num_nodes: Optional[int] = None,
) -> int:
    if num_nodes is not None:
        return num_nodes
    # elif not torch.jit.is_scripting() and isinstance(edge_index, EdgeIndex):
    #     return max(edge_index.get_sparse_size())
    elif isinstance(edge_index, Tensor):
        if is_torch_sparse_tensor(edge_index):
            return max(edge_index.size(0), edge_index.size(1))

        if torch.jit.is_tracing():
            # Avoid non-traceable if-check for empty `edge_index` tensor:
            tmp = torch.concat([
                edge_index.view(-1),
                edge_index.new_full((1, ), fill_value=-1)
            ])
            return tmp.max() + 1  # type: ignore

        return int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
    elif isinstance(edge_index, tuple):
        return max(
            int(edge_index[0].max()) + 1 if edge_index[0].numel() > 0 else 0,
            int(edge_index[1].max()) + 1 if edge_index[1].numel() > 0 else 0,
        )
    # elif isinstance(edge_index, SparseTensor):
    #     return max(edge_index.size(0), edge_index.size(1))
    raise NotImplementedError

def cumsum(x: Tensor, dim: int = 0) -> Tensor:
    r"""Returns the cumulative sum of elements of :obj:`x`.
    In contrast to :meth:`torch.cumsum`, prepends the output with zero.

    Args:
        x (torch.Tensor): The input tensor.
        dim (int, optional): The dimension to do the operation over.
            (default: :obj:`0`)

    Example:
        >>> x = torch.tensor([2, 4, 1])
        >>> cumsum(x)
        tensor([0, 2, 6, 7])

    """
    size = x.size()[:dim] + (x.size(dim) + 1, ) + x.size()[dim + 1:]
    out = x.new_empty(size)

    out.narrow(dim, 0, 1).zero_()
    torch.cumsum(x, dim=dim, out=out.narrow(dim, 1, x.size(dim)))

    return out

def degree(index: Tensor, num_nodes: Optional[int] = None,
           dtype: Optional[torch.dtype] = None) -> Tensor:
    r"""Computes the (unweighted) degree of a given one-dimensional index
    tensor.

    Args:
        index (LongTensor): Index tensor.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)
        dtype (:obj:`torch.dtype`, optional): The desired data type of the
            returned tensor.

    :rtype: :class:`Tensor`

    Example:
        >>> row = torch.tensor([0, 1, 0, 2, 0])
        >>> degree(row, dtype=torch.long)
        tensor([3, 1, 1])
    """
    N = maybe_num_nodes(index, num_nodes)
    out = torch.zeros((N, ), dtype=dtype, device=index.device)
    one = torch.ones((index.size(0), ), dtype=out.dtype, device=out.device)
    return out.scatter_add_(0, index, one)

def unbatch(
    src: Tensor,
    batch: Tensor,
    dim: int = 0,
    batch_size: Optional[int] = None,
) -> List[Tensor]:
    r"""Splits :obj:`src` according to a :obj:`batch` vector along dimension
    :obj:`dim`.

    Args:
        src (Tensor): The source tensor.
        batch (LongTensor): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            entry in :obj:`src` to a specific example. Must be ordered.
        dim (int, optional): The dimension along which to split the :obj:`src`
            tensor. (default: :obj:`0`)
        batch_size (int, optional): The batch size. (default: :obj:`None`)

    :rtype: :class:`List[Tensor]`

    Example:
        >>> src = torch.arange(7)
        >>> batch = torch.tensor([0, 0, 0, 1, 1, 2, 2])
        >>> unbatch(src, batch)
        (tensor([0, 1, 2]), tensor([3, 4]), tensor([5, 6]))
    """
    sizes = degree(batch, batch_size, dtype=torch.long).tolist()
    return src.split(sizes, dim)


def unbatch_edge_index(
    edge_index: Tensor,
    batch: Tensor,
    batch_size: Optional[int] = None,
) -> List[Tensor]:
    r"""Splits the :obj:`edge_index` according to a :obj:`batch` vector.

    Args:
        edge_index (Tensor): The edge_index tensor. Must be ordered.
        batch (LongTensor): The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example. Must be ordered.
        batch_size (int, optional): The batch size. (default: :obj:`None`)

    :rtype: :class:`List[Tensor]`

    Example:
        >>> edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 4, 5, 5, 6],
        ...                            [1, 0, 2, 1, 3, 2, 5, 4, 6, 5]])
        >>> batch = torch.tensor([0, 0, 0, 0, 1, 1, 1])
        >>> unbatch_edge_index(edge_index, batch)
        (tensor([[0, 1, 1, 2, 2, 3],
                [1, 0, 2, 1, 3, 2]]),
        tensor([[0, 1, 1, 2],
                [1, 0, 2, 1]]))
    """
    deg = degree(batch, batch_size, dtype=torch.long)
    ptr = cumsum(deg)

    edge_batch = batch[edge_index[0]]
    edge_index = edge_index - ptr[edge_batch]
    sizes = degree(edge_batch, batch_size, dtype=torch.long).cpu().tolist()
    return edge_index.split(sizes, dim=1)