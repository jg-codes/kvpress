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
    similarity_threshold: float = 0.0,
    merge_keys: bool = False,
    value_norm_weighting: bool = True,
    max_merge_per_token: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Core merge-on-evict kernel for :class:`MergingPress`.

    Given per-token scores, partitions into *keep* and *evict* sets, then folds
    each evicted token into its most cosine-similar survivor via a weighted
    scatter-add instead of discarding it.

    **Perturbation bound.**  For a single query position *t* and evicted token
    *i* routed to survivor *j* with cosine similarity :math:`w = \\cos(k_i, k_j)`:

    .. math::

        \\|\\Delta O_{\\text{merge}}\\| \\leq \\frac{1}{1 + w} \\;\\|\\Delta O_{\\text{evict}}\\|

    where :math:`\\Delta O_{\\text{evict}} = a_{t,i} \\, v_i` is the output
    perturbation from hard eviction and :math:`a_{t,i}` is the attention
    weight.  At :math:`w \\geq 0.7` the merge error is at most 59% of hard-
    eviction error; at :math:`w = 1` it halves exactly.

    Parameters
    ----------
    keys : Tensor, shape ``(B, H, L, D)``
    values : Tensor, shape ``(B, H, L, D)``
    scores : Tensor, shape ``(B, H, L)``
        Higher score → more important (kept).
    n_kept : int
        Number of tokens to survive after compression.
    similarity_threshold : float
        Minimum cosine similarity for a merge to proceed.
    merge_keys : bool
        Whether to merge evicted information into survivor keys.
    value_norm_weighting : bool
        Scale merge weight by relative value-vector L2 norm.
    max_merge_per_token : int, default=0
        Maximum number of evicted tokens that may merge into any single
        survivor.  When a survivor receives more merges than the cap, each
        merge weight is scaled down proportionally so the total deposited
        weight does not exceed ``max_merge_per_token × mean_weight``.
        ``0`` disables the cap (default).
    Returns
    -------
    tuple[Tensor, Tensor]
        ``(merged_keys, merged_values)`` each of shape ``(B, H, n_kept, D)``.

    Notes
    -----
    The perturbation bound above is an original derivation for this implementation.
    The merge routing strategy is inspired by Token Merging (ToMe) but simplified
    from bipartite matching to greedy max-cosine-similarity, and extended with
    multi-stage weighting (similarity × value-norm × score × cap).

    References
    ----------
    .. [1] Bolya et al., "Token Merging: Your ViT But Faster", ICLR 2023.
       https://arxiv.org/abs/2210.09461
    .. [2] Wan et al., "D2O: Dynamic Discriminative Operations for Efficient
       Generative Inference of Large Language Models", 2024.
       https://arxiv.org/abs/2406.13035
    .. [3] Huang et al., "KeepKV: Lossless KV Cache Compression in Large
       Language Models", 2025.
       https://arxiv.org/abs/2504.09936
    """
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

    # --- Threshold gate ---
    merge_mask = max_sim >= similarity_threshold
    if not merge_mask.any():
        return kept_keys.contiguous(), kept_values.contiguous()

    if logger.isEnabledFor(logging.DEBUG):
        n_merged = merge_mask.sum().item()
        merge_count = torch.zeros(bsz, num_key_value_heads, n_kept, device=keys.device)
        merge_count.scatter_add_(2, target_idx, merge_mask.float())
        max_count = merge_count.max().item()
        mean_sim = max_sim[merge_mask].mean().item()
        logger.debug(
            "merge_on_evict: %d/%d evicted tokens merged (%.1f%%), "
            "mean_sim=%.3f, max_merges_per_survivor=%.0f",
            n_merged,
            n_evict * bsz * num_key_value_heads,
            100.0 * n_merged / (n_evict * bsz * num_key_value_heads),
            mean_sim,
            max_count,
        )

    # --- Similarity-weighted scatter-add merge ---
    evict_w = max_sim.clamp(min=0) * merge_mask  # (B, H, n_evict)

    if value_norm_weighting:
        ev_vnorm = evict_values.float().norm(dim=-1)
        target_vnorm = kept_values.float().norm(dim=-1).gather(2, target_idx)
        rel_norm = ev_vnorm / (ev_vnorm + target_vnorm + _EPS)
        evict_w = evict_w * rel_norm

    # --- Merge count cap: prevent survivor dilution ---
    if max_merge_per_token > 0:
        merge_count = torch.zeros(bsz, num_key_value_heads, n_kept, device=keys.device, dtype=torch.float32)
        merge_count.scatter_add_(2, target_idx, merge_mask.float())
        excess = (merge_count / max_merge_per_token).clamp(min=1.0)
        evict_w = evict_w / excess.gather(2, target_idx)

    ew = evict_w.unsqueeze(-1)
    tgt = target_idx.unsqueeze(-1).expand_as(evict_keys)

    val_accum = torch.zeros(bsz, num_key_value_heads, n_kept, head_dim, device=keys.device, dtype=torch.float32)
    val_accum.scatter_add_(2, tgt, ew * evict_values.float())
    w_accum = torch.zeros(bsz, num_key_value_heads, n_kept, device=keys.device, dtype=torch.float32)
    w_accum.scatter_add_(2, target_idx, evict_w)

    # --- Normalize ---
    active = w_accum > 0
    total_w = (1.0 + w_accum).unsqueeze(-1)
    active_mask = active.unsqueeze(-1)
    new_vals = (kept_values.float() + val_accum) / total_w
    merged_values = torch.where(active_mask, new_vals.to(kept_values.dtype), kept_values)

    if merge_keys:
        key_accum = torch.zeros(bsz, num_key_value_heads, n_kept, head_dim, device=keys.device, dtype=torch.float32)
        key_accum.scatter_add_(2, tgt, ew * evict_keys.float())
        new_keys = (kept_keys.float() + key_accum) / total_w
        merged_keys = torch.where(active_mask, new_keys.to(kept_keys.dtype), kept_keys)
    else:
        merged_keys = kept_keys

    return merged_keys.contiguous(), merged_values.contiguous()
