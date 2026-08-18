from my_answer.linear import Linear
import torch
import torch.nn as nn

def SiLU(x: torch.Tensor):
    return x * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff=None, device=None, dtype=None):
        super().__init__()

        if d_ff is None:
            round_to_64 = lambda x: round(x / 64) * 64
            d_ff = round_to_64(d_model * 8 / 3)

        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)
        self.device = device
        self.dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2.forward(SiLU(self.w1.forward(x)) * self.w3.forward(x))