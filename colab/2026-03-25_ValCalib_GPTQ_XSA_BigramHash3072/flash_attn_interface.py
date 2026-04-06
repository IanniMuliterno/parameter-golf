from __future__ import annotations

import torch
import torch.nn.functional as F


def _expand_gqa_heads(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if q.size(-2) == k.size(-2):
        return q, k, v
    if q.size(-2) % k.size(-2) != 0:
        raise ValueError(
            f"Incompatible GQA head counts: q_heads={q.size(-2)} kv_heads={k.size(-2)}"
        )
    repeat = q.size(-2) // k.size(-2)
    return q, k.repeat_interleave(repeat, dim=-2), v.repeat_interleave(repeat, dim=-2)


def flash_attn_func(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False) -> torch.Tensor:
    q, k, v = _expand_gqa_heads(q, k, v)
    q = q.permute(0, 2, 1, 3).contiguous()
    k = k.permute(0, 2, 1, 3).contiguous()
    v = v.permute(0, 2, 1, 3).contiguous()
    out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    return out.permute(0, 2, 1, 3).contiguous()
