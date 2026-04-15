# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging

import torch

logger = logging.getLogger(__name__)

# Epsilon for numerical stability — safe for float16 (min ~6e-8) and bfloat16
_EPS = 1e-6


def _merge_on_evict(
    keys: torch.Tensor,
    values: torch.Tensor,
    scores: torch.Tensor,
    n_kept: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Core merge-on-evict: partition by score, fold evicted tokens into nearest survivor."""
    bsz, num_key_value_heads, k_len, head_dim = keys.shape
    n_evict = k_len - n_kept

    # --- Partition into keep / evict ---
    keep_idx = scores.topk(n_kept, dim=-1).indices  # (B, H, n_kept)
    mask = torch.ones(bsz, num_key_value_heads, k_len, device=keys.device, dtype=torch.bool)
    mask.scatter_(2, keep_idx, False)
    evict_idx = mask.nonzero(as_tuple=False)[:, 2].reshape(bsz, num_key_value_heads, n_evict)

    # --- Gather kept and evicted tensors ---
    idx4 = keep_idx.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    kept_keys = keys.gather(2, idx4)
    kept_values = values.gather(2, idx4)
    idx4_e = evict_idx.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    evict_keys = keys.gather(2, idx4_e)
    evict_values = values.gather(2, idx4_e)

    # --- Cosine similarity → nearest survivor ---
    evict_keys_f = evict_keys.float()
    kept_keys_f = kept_keys.float()
    e_norm = evict_keys_f / evict_keys_f.norm(dim=-1, keepdim=True).clamp(min=_EPS)
    s_norm = kept_keys_f / kept_keys_f.norm(dim=-1, keepdim=True).clamp(min=_EPS)
    sim = torch.matmul(e_norm, s_norm.transpose(-2, -1))  # (B, H, n_evict, n_kept)
    max_sim, target_idx = sim.max(dim=-1)  # (B, H, n_evict)

    # --- Uniform-weight scatter-add merge ---
    evict_w = max_sim.clamp(min=0)
    ew = evict_w.unsqueeze(-1)
    tgt = target_idx.unsqueeze(-1).expand_as(evict_keys)

    val_accum = torch.zeros(bsz, num_key_value_heads, n_kept, head_dim, device=keys.device, dtype=torch.float32)
    val_accum.scatter_add_(2, tgt, ew * evict_values.float())
    w_accum = torch.zeros(bsz, num_key_value_heads, n_kept, device=keys.device, dtype=torch.float32)
    w_accum.scatter_add_(2, target_idx, evict_w)

    # --- Normalize ---
    active = w_accum > 0
    total_w = (1.0 + w_accum).unsqueeze(-1)
    new_vals = (kept_values.float() + val_accum) / total_w
    merged_values = torch.where(active.unsqueeze(-1), new_vals.to(kept_values.dtype), kept_values)

    return kept_keys.contiguous(), merged_values.contiguous()
