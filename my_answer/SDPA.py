import torch, math
import torch.nn as nn
from jaxtyping import Bool, Float, Int
from torch import Tensor

def softmax(in_features: Float[Tensor, " ..."], dim: int) -> Float[Tensor, " ..."]:
    m = in_features.max(dim=dim, keepdim=True).values
    e = torch.exp(in_features - m)
    s = e.sum(dim=dim, keepdim=True)
    return e / s

def SDPA(Q: Float[torch.Tensor, " ... queries d_k"],
    K: Float[torch.Tensor, " ... keys d_k"],
    V: Float[torch.Tensor, " ... keys d_v"],
    mask: Bool[torch.Tensor, " ... queries keys"] | None = None,
):
    d_k = Q.shape[-1]
    temp = Q @ K.transpose(-1, -2) / math.sqrt(d_k)

    if mask is not None:
        temp = temp - ~mask * 1e30

    return softmax(temp, dim=-1) @ V