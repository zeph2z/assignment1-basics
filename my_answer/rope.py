import torch
import torch.nn as nn
from einops import rearrange

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        i = torch.arange(0, max_seq_len)
        k = torch.arange(0, d_k // 2)
        freqs = 1 / theta ** (2 * k / d_k)
        angles = i[:, None] * freqs[None, :]

        self.sin_table = torch.sin(angles) # i * k
        self.cos_table = torch.cos(angles)

        self.d_k = d_k
        self.device = device

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        x_view = rearrange(x, "... seq (d h) -> ... seq d h", h=2)
        x_even = x_view[..., 0]
        x_odd = x_view[..., 1]

        sin_for_x = self.sin_table[token_positions]
        cos_for_x = self.cos_table[token_positions]

        out_even = x_even * cos_for_x - x_odd * sin_for_x
        out_odd = x_even * sin_for_x + x_odd * cos_for_x

        out_stack = torch.stack([out_even, out_odd], dim=-1)
        return rearrange(out_stack, "... seq p t -> ... seq (p t)")