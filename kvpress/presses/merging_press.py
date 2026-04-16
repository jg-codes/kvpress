# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
from dataclasses import dataclass

import torch
from torch import nn

from kvpress.presses.base_press import BasePress
from kvpress.presses.decoding_press import DecodingPress
from kvpress.presses.scorer_press import ScorerPress

logger = logging.getLogger(__name__)

# Epsilon for numerical stability — safe for float16 (min ~6e-8) and bfloat16
_EPS = 1e-6


def _merge_on_evict(
    keys: torch.Tensor,
    values: torch.Tensor,
    scores: torch.Tensor,
    n_kept: int,
    similarity_threshold: float,
    merge_keys: bool,
    value_norm_weighting: bool,
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

    # nonzero() returns indices in row-major (C-contiguous) order, so the
    # reshape below produces a correct (B, H, n_evict) partition.
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

    # --- Batched cosine similarity → nearest survivor ---
    # Guard against zero-norm keys: clamp norms to avoid NaN from F.normalize
    evict_keys_f = evict_keys.float()
    kept_keys_f = kept_keys.float()
    e_norms = evict_keys_f.norm(dim=-1, keepdim=True).clamp(min=_EPS)
    s_norms = kept_keys_f.norm(dim=-1, keepdim=True).clamp(min=_EPS)
    e_norm = evict_keys_f / e_norms
    s_norm = kept_keys_f / s_norms
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
            "merge_on_evict: %d/%d evicted tokens merged (%.1f%%), " "mean_sim=%.3f, max_merges_per_survivor=%.0f",
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

    ew = evict_w.unsqueeze(-1)  # (B, H, n_evict, 1)
    tgt = target_idx.unsqueeze(-1).expand_as(evict_keys)  # (B, H, n_evict, D)

    # Accumulate in float32 for numerical stability
    val_accum = torch.zeros(bsz, num_key_value_heads, n_kept, head_dim, device=keys.device, dtype=torch.float32)
    val_accum.scatter_add_(2, tgt, ew * evict_values.float())

    w_accum = torch.zeros(bsz, num_key_value_heads, n_kept, device=keys.device, dtype=torch.float32)
    w_accum.scatter_add_(2, target_idx, evict_w)

    # --- Normalize: weighted average for active positions ---
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


@dataclass
class MergingPress(BasePress):
    """
    Press-agnostic merge-on-evict wrapper for KV cache compression during prefill.

    Wraps any :class:`BasePress` and replaces hard eviction with merge-on-evict:
    each evicted token is folded into its most similar surviving neighbor rather than
    being discarded.  Values are blended via a similarity-weighted average; keys can
    optionally be merged or left unchanged depending on the ``merge_keys`` flag.

    **Composition modes:**

    * ``MergingPress(ScorerPress)``: calls ``.score()``, applies uniform per-head
      budget, returns truncated tensors with merged survivors.
    * ``MergingPress(AdaKVPress(ScorerPress))``: delegates to AdaKV's adaptive
      per-head budget allocation, then merges evicted tokens into survivors in-place.
      Returns full-length tensors with ``masked_key_indices`` set.
    * ``MergingPress(CriticalAdaKVPress(...))``: same pattern — any mask-based press.

    For any non-ScorerPress inner press, MergingPress delegates ``.compress()`` to
    the inner press, reads back ``module.masked_key_indices``, and merges evicted
    tokens into their nearest cosine-similar survivors.

    Parameters
    ----------
    press : BasePress
        The underlying press.  Can be a :class:`ScorerPress` for uniform merge,
        or a mask-based press like :class:`AdaKVPress` / :class:`CriticalAdaKVPress`
        for adaptive per-head merge.
    similarity_threshold : float, default=0.0
        Minimum cosine similarity between an evicted key and its nearest survivor
        for the merge to proceed.  Evicted tokens below this threshold are dropped
        without merging.
    merge_keys : bool, default=False
        Whether to merge evicted information into kept keys.  When ``False`` (default),
        only values are merged — kept keys are returned unchanged.  This preserves
        RoPE positional encoding in the keys.
    value_norm_weighting : bool, default=True
        When ``True`` (default), the merge weight for each evicted token is additionally
        scaled by the relative L2 norm of its value vector.
    max_merge_per_token : int, default=0
        Maximum number of evicted tokens merged into any single survivor before
        weight scaling kicks in.  ``0`` (default) disables the cap.
    """

    press: BasePress
    similarity_threshold: float = 0.0
    merge_keys: bool = False
    value_norm_weighting: bool = True
    max_merge_per_token: int = 0

    def __post_init__(self):
        assert isinstance(self.press, BasePress), f"MergingPress requires a BasePress, got {type(self.press)}"
        assert 0.0 <= self.similarity_threshold <= 1.0
        assert self.max_merge_per_token >= 0, "max_merge_per_token must be non-negative"

    def post_init_from_model(self, model):
        self.press.post_init_from_model(model)

    @property
    def compression_ratio(self):
        return self.press.compression_ratio

    @compression_ratio.setter
    def compression_ratio(self, value):
        self.press.compression_ratio = value

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.press.compression_ratio == 0:
            return keys, values

        bsz, num_key_value_heads, k_len, head_dim = keys.shape

        # --- ScorerPress path: uniform per-head merge, returns truncated tensors ---
        if isinstance(self.press, ScorerPress):
            scores = self.press.score(module, hidden_states, keys, values, attentions, kwargs)
            n_kept = int(k_len * (1 - self.press.compression_ratio))
            if n_kept >= k_len:
                return keys, values
            if n_kept <= 0:
                return keys[:, :, :0, :].contiguous(), values[:, :, :0, :].contiguous()

            return _merge_on_evict(
                keys,
                values,
                scores,
                n_kept,
                self.similarity_threshold,
                self.merge_keys,
                self.value_norm_weighting,
                self.max_merge_per_token,
            )

        # --- Mask-based press path (AdaKV, CriticalAdaKV, etc.) ---
        # Delegate to the inner press which sets module.masked_key_indices
        # and returns keys/values unchanged.
        keys, values = self.press.compress(module, hidden_states, keys, values, attentions, kwargs)

        mask_indices = getattr(module, "masked_key_indices", None)
        if mask_indices is None:
            return keys, values

        # Build boolean eviction mask from (batch, head, seq) index tuple
        evict_mask = torch.zeros(bsz, num_key_value_heads, k_len, device=keys.device, dtype=torch.bool)
        evict_mask[mask_indices] = True

        # Merge evicted tokens into their nearest cosine-similar survivors
        new_keys, new_values = _merge_on_evict_adaptive(
            keys,
            values,
            evict_mask,
            self.similarity_threshold,
            self.merge_keys,
            self.value_norm_weighting,
            self.max_merge_per_token,
        )
        return new_keys, new_values


@dataclass
class MergingDecodingPress(DecodingPress):
    """
    Merge-on-evict KV cache compression during decoding.

    Extends :class:`DecodingPress` with the same merge-on-evict strategy used by
    :class:`MergingPress`: instead of hard-pruning low-scoring tokens, their key/value
    vectors are folded into the most cosine-similar survivor.  All decoding-phase
    scheduling (buffered hidden states, interval-based triggering) is inherited from
    ``DecodingPress``.

    Compared to :class:`CAMPress`, which merges into *sequential neighbors* using a
    Bernoulli mask derived from cumulative attention, ``MergingDecodingPress`` merges
    into the *most similar survivor* using cosine similarity — making it
    position-agnostic and compatible with any :class:`ScorerPress`.

    Parameters
    ----------
    base_press : ScorerPress
        Scorer used to rank tokens for eviction.
    compression_interval : int, default=512
        Decoding steps between compression passes.
    target_size : int, default=2048
        Number of tokens to keep after each compression.
    hidden_states_buffer_size : int, default=256
        Maximum buffered hidden states for scoring context.
    similarity_threshold : float, default=0.0
        Minimum cosine similarity for a merge to proceed.
    merge_keys : bool, default=False
        Whether to merge evicted keys into survivors.  ``False`` (default)
        preserves RoPE positional encoding.
    value_norm_weighting : bool, default=True
        Scale merge weight by relative value-vector L2 norm.
    max_merge_per_token : int, default=0
        Maximum merges per survivor.  ``0`` disables the cap.
    """

    similarity_threshold: float = 0.0
    merge_keys: bool = False
    value_norm_weighting: bool = True
    max_merge_per_token: int = 0

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Override hard eviction with merge-on-evict during decoding."""
        k_len = keys.shape[2]
        n_kept = self.target_size

        if k_len <= n_kept:
            return keys, values

        target_compression_ratio = self._find_target_compression_ratio(k_len, n_kept)

        original_cr = self.base_press.compression_ratio
        self.base_press.compression_ratio = target_compression_ratio
        scores = self.base_press.score(module, hidden_states, keys, values, attentions, kwargs)
        self.base_press.compression_ratio = original_cr

        return _merge_on_evict(
            keys,
            values,
            scores,
            n_kept,
            self.similarity_threshold,
            self.merge_keys,
            self.value_norm_weighting,
            self.max_merge_per_token,
        )


def _merge_on_evict_adaptive(
    keys: torch.Tensor,
    values: torch.Tensor,
    evict_mask: torch.Tensor,
    similarity_threshold: float,
    merge_keys: bool,
    value_norm_weighting: bool,
    max_merge_per_token: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Merge-on-evict with variable per-head eviction counts for :class:`MergingPress`.

    Unlike :func:`_merge_on_evict` which uses a uniform ``n_kept`` across all KV heads,
    this variant handles variable per-head eviction counts from AdaKV's global budget
    allocation.  Each head may evict a different number of tokens; evicted tokens in each
    head are merged into their nearest cosine-similar survivor independently.

    The merged information is written **in-place** into the survivor positions of the
    full-length tensors.  Evicted positions are left unchanged (they will be masked by
    the attention patch via ``module.masked_key_indices``).

    Parameters
    ----------
    keys : Tensor, shape ``(B, H, L, D)``
    values : Tensor, shape ``(B, H, L, D)``
    evict_mask : Tensor, shape ``(B, H, L)``, dtype bool
        ``True`` at positions to evict, ``False`` at positions to keep.
    similarity_threshold : float
        Minimum cosine similarity for a merge to proceed.
    merge_keys : bool
        Whether to merge evicted information into survivor keys.
    value_norm_weighting : bool
        Scale merge weight by relative value-vector L2 norm.
    max_merge_per_token : int, default=0
        Cap on merges per survivor (0 = unlimited).

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(new_keys, new_values)`` — same shape ``(B, H, L, D)`` as input.
        Survivor positions contain merged information; evicted positions are unchanged.
    """
    bsz, num_kv_heads, k_len, head_dim = keys.shape
    device = keys.device

    # Work on float32 copies for numerical stability
    merged_values = values.float().clone()
    merged_keys = keys.float().clone() if merge_keys else None

    # Iterate over (batch, head) — typically B=1, H=8 for Qwen3-8B = 8 iterations
    for b in range(bsz):
        for h in range(num_kv_heads):
            evict_idx = evict_mask[b, h].nonzero(as_tuple=True)[0]
            keep_idx = (~evict_mask[b, h]).nonzero(as_tuple=True)[0]
            n_evict = evict_idx.shape[0]
            n_kept = keep_idx.shape[0]

            if n_evict == 0 or n_kept == 0:
                continue

            # Cosine similarity between evicted and kept keys
            evict_k = keys[b, h, evict_idx].float()  # (n_e, D)
            kept_k = keys[b, h, keep_idx].float()  # (n_k, D)

            e_norms = evict_k.norm(dim=-1, keepdim=True).clamp(min=_EPS)
            k_norms = kept_k.norm(dim=-1, keepdim=True).clamp(min=_EPS)
            sim = (evict_k / e_norms) @ (kept_k / k_norms).T  # (n_e, n_k)
            max_sim, target = sim.max(dim=-1)  # (n_e,)

            # Threshold gate
            merge_ok = max_sim >= similarity_threshold
            if not merge_ok.any():
                continue

            # Merge weights
            w = max_sim.clamp(min=0) * merge_ok.float()

            if value_norm_weighting:
                evict_v = values[b, h, evict_idx].float()
                target_v = values[b, h, keep_idx[target]].float()
                ev_norm = evict_v.norm(dim=-1)
                tv_norm = target_v.norm(dim=-1)
                w = w * ev_norm / (ev_norm + tv_norm + _EPS)

            if max_merge_per_token > 0:
                count = torch.zeros(n_kept, device=device, dtype=torch.float32)
                count.scatter_add_(0, target, merge_ok.float())
                excess = (count / max_merge_per_token).clamp(min=1.0)
                w = w / excess[target]

            # Scatter-add evicted values into survivors
            w_exp = w.unsqueeze(-1)  # (n_e, 1)
            evict_v = values[b, h, evict_idx].float()

            val_accum = torch.zeros(n_kept, head_dim, device=device, dtype=torch.float32)
            val_accum.scatter_add_(0, target.unsqueeze(-1).expand_as(evict_v), w_exp * evict_v)

            w_accum = torch.zeros(n_kept, device=device, dtype=torch.float32)
            w_accum.scatter_add_(0, target, w)

            # Normalize: weighted average for active survivors
            active = w_accum > 0
            total_w = (1.0 + w_accum).unsqueeze(-1)

            orig_v = merged_values[b, h, keep_idx]
            new_v = (orig_v + val_accum) / total_w
            merged_values[b, h, keep_idx] = torch.where(active.unsqueeze(-1), new_v, orig_v)

            if merge_keys and merged_keys is not None:
                evict_k_orig = keys[b, h, evict_idx].float()
                key_accum = torch.zeros(n_kept, head_dim, device=device, dtype=torch.float32)
                key_accum.scatter_add_(0, target.unsqueeze(-1).expand_as(evict_k_orig), w_exp * evict_k_orig)
                orig_k = merged_keys[b, h, keep_idx]
                new_k = (orig_k + key_accum) / total_w
                merged_keys[b, h, keep_idx] = torch.where(active.unsqueeze(-1), new_k, orig_k)

    result_values = merged_values.to(values.dtype)
    result_keys = merged_keys.to(keys.dtype) if merge_keys else keys
    return result_keys, result_values

