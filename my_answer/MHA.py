import torch, einops
import torch.nn as nn
from my_answer.linear import Linear
from my_answer.SDPA import SDPA
from my_answer.rope import RoPE

class MHA(nn.Module):
    def __init__(self, d_in, d_out, num_head, dtype = None, device = None):
        super().__init__()

        self.q_proj = Linear(in_features=d_in, out_features=d_out, dtype=dtype, device=device)
        self.k_proj = Linear(in_features=d_in, out_features=d_out, dtype=dtype, device=device)
        self.v_proj = Linear(in_features=d_in, out_features=d_out, dtype=dtype, device=device)
        self.output_proj = Linear(in_features=d_out, out_features=d_in, dtype=dtype, device=device)

        self.d_in = d_in
        self.d_out = d_out
        self.num_head = num_head
        self.dtype = dtype
        self.device = device

    def forward(self, x, rope: RoPE | None = None, token_positions=None):
        # [batch, seq, d_out]
        Q = self.q_proj.forward(x)
        K = self.k_proj.forward(x)
        V = self.v_proj.forward(x)

        # [batch, head, seq, d_head]
        multi_Q = einops.rearrange(Q, "... s (h d) -> ... h s d", h=self.num_head)
        multi_K = einops.rearrange(K, "... s (h d) -> ... h s d", h=self.num_head)
        multi_V = einops.rearrange(V, "... s (h d) -> ... h s d", h=self.num_head)

        if rope is not None:
            multi_Q = rope.forward(multi_Q, token_positions)
            multi_K = rope.forward(multi_K, token_positions)

        seq = x.shape[-2]
        mask = (torch.triu(torch.ones(seq, seq), diagonal=1) == 0)

        # [batch, head, seq, d_head]
        multi_head = SDPA(multi_Q, multi_K, multi_V, mask)
        # [batch, seq, d_out]
        concat_head = einops.rearrange(multi_head, "... h s d -> ... s (h d)", h=self.num_head)
        return self.output_proj.forward(concat_head)

    # Q = run_linear(d_model, d_model, q_proj_weight, in_features)
    # K = run_linear(d_model, d_model, k_proj_weight, in_features)
    # V = run_linear(d_model, d_model, v_proj_weight, in_features)

    # multi_Q = einops.rearrange(Q, "... s (d h) -> ... d s h", d=num_heads)
    # multi_K = einops.rearrange(K, "... s (d h) -> ... d s h", d=num_heads)
    # multi_V = einops.rearrange(V, "... s (d h) -> ... d s h", d=num_heads)
