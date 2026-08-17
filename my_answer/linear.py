import torch, math
import torch.nn as nn

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        std = math.sqrt(2 / (in_features + out_features))
        self.weight = nn.Parameter(nn.init.trunc_normal_(
            torch.empty(out_features, in_features, device=device, dtype=dtype), 
            mean=0,
            std=std,
            a=-3 * std,
            b=3 * std))
        self.device = device
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T