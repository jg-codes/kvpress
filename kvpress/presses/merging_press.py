# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass, field
import logging

import torch
import torch.nn.functional as F
from torch import nn

from kvpress.presses.base_press import BasePress
from kvpress.presses.decoding_press import DecodingPress
from kvpress.presses.scorer_press import ScorerPress

logger = logging.getLogger(__name__)


def _merge_on_evict(
    keys: torch.Tensor,
    values: torch.Tensor,
    scores: torch.Tensor,
    n_kept: int,
    similarity_threshold: float,
    merge_keys: bool,
    value_norm_weighting: bool,
    max_merge_per_token: int = 0,
    collect_diagnostics: bool = False,
    score_weighting: bool = False,
    adaptive_threshold: bool = False,
) -> tuple[torch.Tensor, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, dict]:
    """
    Core merge-on-evict kernel shared by :class:`MergingPress` (prefill) and
    :class:`MergingDecodingPress` (decoding).

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
    score_weighting : bool, default=False
        When ``True``, the merge weight for each evicted token is additionally
        scaled by its normalised importance score (relative to the evict set).
        Tokens that barely missed the keep threshold contribute more to their
        target survivor than truly unimportant tokens.
    adaptive_threshold : bool, default=False
        When ``True``, the similarity threshold is computed dynamically as the
        25th percentile of the per-token maximum cosine similarities instead
        of using the fixed ``similarity_threshold``.  This skips the worst-
        matched evicted tokens (bottom quartile) while still merging the
        majority.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(merged_keys, merged_values)`` each of shape ``(B, H, n_kept, D)``.
    """
    bsz, num_kv_heads, k_len, head_dim = keys.shape
    n_evict = k_len - n_kept

    # --- Partition into keep / evict ---
    keep_idx = scores.topk(n_kept, dim=-1).indices  # (B, H, n_kept)

    # nonzero() returns indices in row-major (C-contiguous) order, so the
    # reshape below produces a correct (B, H, n_evict) partition.
    mask = torch.ones(bsz, num_kv_heads, k_len, device=keys.device, dtype=torch.bool)
    mask.scatter_(2, keep_idx, False)
    evict_idx = mask.nonzero(as_tuple=False)[:, 2].reshape(bsz, num_kv_heads, n_evict)

    # --- Gather kept and evicted tensors ---
    idx4 = keep_idx.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    kept_keys = keys.gather(2, idx4)
    kept_values = values.gather(2, idx4)

    idx4_e = evict_idx.unsqueeze(-1).expand(-1, -1, -1, head_dim)
    evict_keys = keys.gather(2, idx4_e)
    evict_values = values.gather(2, idx4_e)

    # --- Batched cosine similarity → nearest survivor ---
    # Guard against zero-norm keys: clamp norms to avoid NaN from F.normalize
    _EPS = 1e-6  # safe for both float16 (min ~6e-8) and bfloat16 (min ~1e-38 but precision ~1e-3)
    evict_keys_f = evict_keys.float()
    kept_keys_f = kept_keys.float()
    e_norms = evict_keys_f.norm(dim=-1, keepdim=True).clamp(min=_EPS)
    s_norms = kept_keys_f.norm(dim=-1, keepdim=True).clamp(min=_EPS)
    e_norm = evict_keys_f / e_norms
    s_norm = kept_keys_f / s_norms
    sim = torch.matmul(e_norm, s_norm.transpose(-2, -1))  # (B, H, n_evict, n_kept)
    max_sim, target_idx = sim.max(dim=-1)  # (B, H, n_evict)

    # --- Threshold gate (adaptive or fixed) ---
    if adaptive_threshold:
        threshold = torch.quantile(max_sim.float().flatten(-2), 0.25, dim=-1, keepdim=True).unsqueeze(-1)
        threshold = threshold.expand_as(max_sim)
        merge_mask = max_sim >= threshold
    else:
        merge_mask = max_sim >= similarity_threshold
    if not merge_mask.any():
        if collect_diagnostics:
            return kept_keys.contiguous(), kept_values.contiguous(), {"n_merged": 0, "n_evicted": n_evict * bsz * num_kv_heads}
        return kept_keys.contiguous(), kept_values.contiguous()

    if logger.isEnabledFor(logging.DEBUG):
        n_merged = merge_mask.sum().item()
        merge_count = torch.zeros(bsz, num_kv_heads, n_kept, device=keys.device)
        merge_count.scatter_add_(2, target_idx, merge_mask.float())
        max_count = merge_count.max().item()
        mean_sim = max_sim[merge_mask].mean().item()
        logger.debug(
            "merge_on_evict: %d/%d evicted tokens merged (%.1f%%), "
            "mean_sim=%.3f, max_merges_per_survivor=%.0f",
            n_merged, n_evict * bsz * num_kv_heads,
            100.0 * n_merged / (n_evict * bsz * num_kv_heads),
            mean_sim, max_count,
        )

    # --- Similarity-weighted scatter-add merge ---
    evict_w = max_sim.clamp(min=0) * merge_mask  # (B, H, n_evict)

    if score_weighting:
        evict_scores = scores.gather(2, evict_idx).float()  # (B, H, n_evict)
        # Normalise within evict set to [0, 1] so scale is independent of scorer
        s_min = evict_scores.min(dim=-1, keepdim=True).values
        s_max = evict_scores.max(dim=-1, keepdim=True).values
        s_range = (s_max - s_min).clamp(min=_EPS)
        norm_scores = (evict_scores - s_min) / s_range  # 0 = least important, 1 = just missed keep
        evict_w = evict_w * (0.5 + 0.5 * norm_scores)  # floor at 0.5 to avoid zeroing out

    if value_norm_weighting:
        ev_vnorm = evict_values.float().norm(dim=-1)
        target_vnorm = kept_values.float().norm(dim=-1).gather(2, target_idx)
        rel_norm = ev_vnorm / (ev_vnorm + target_vnorm + _EPS)
        evict_w = evict_w * rel_norm

    # --- Merge count cap: prevent survivor dilution ---
    if max_merge_per_token > 0:
        merge_count = torch.zeros(bsz, num_kv_heads, n_kept, device=keys.device, dtype=torch.float32)
        merge_count.scatter_add_(2, target_idx, merge_mask.float())
        excess = (merge_count / max_merge_per_token).clamp(min=1.0)
        evict_w = evict_w / excess.gather(2, target_idx)

    ew = evict_w.unsqueeze(-1)  # (B, H, n_evict, 1)
    tgt = target_idx.unsqueeze(-1).expand_as(evict_keys)  # (B, H, n_evict, D)

    # Accumulate in float32 for numerical stability
    val_accum = torch.zeros(bsz, num_kv_heads, n_kept, head_dim, device=keys.device, dtype=torch.float32)
    val_accum.scatter_add_(2, tgt, ew * evict_values.float())

    w_accum = torch.zeros(bsz, num_kv_heads, n_kept, device=keys.device, dtype=torch.float32)
    w_accum.scatter_add_(2, target_idx, evict_w)

    # --- Normalize: weighted average for active positions ---
    active = w_accum > 0
    total_w = (1.0 + w_accum).unsqueeze(-1)
    active_mask = active.unsqueeze(-1)

    new_vals = (kept_values.float() + val_accum) / total_w
    merged_values = torch.where(active_mask, new_vals.to(kept_values.dtype), kept_values)

    if merge_keys:
        key_accum = torch.zeros(bsz, num_kv_heads, n_kept, head_dim, device=keys.device, dtype=torch.float32)
        key_accum.scatter_add_(2, tgt, ew * evict_keys.float())
        new_keys = (kept_keys.float() + key_accum) / total_w
        merged_keys = torch.where(active_mask, new_keys.to(kept_keys.dtype), kept_keys)
    else:
        merged_keys = kept_keys

    if collect_diagnostics:
        n_merged = merge_mask.sum().item()
        merge_count = torch.zeros(bsz, num_kv_heads, n_kept, device=keys.device)
        merge_count.scatter_add_(2, target_idx, merge_mask.float())
        diag = {
            "n_merged": n_merged,
            "n_evicted": n_evict * bsz * num_kv_heads,
            "merge_ratio": n_merged / (n_evict * bsz * num_kv_heads),
            "mean_sim": max_sim[merge_mask].mean().item(),
            "min_sim": max_sim[merge_mask].min().item(),
            "max_sim": max_sim[merge_mask].max().item(),
            "mean_weight": evict_w[merge_mask].mean().item(),
            "max_merges_per_survivor": merge_count.max().item(),
            "mean_merges_per_active_survivor": merge_count[merge_count > 0].mean().item(),
        }
        return merged_keys.contiguous(), merged_values.contiguous(), diag

    return merged_keys.contiguous(), merged_values.contiguous()


@dataclass
class MergingPress(BasePress):
    """
    Scorer-agnostic merge-on-evict wrapper for KV cache compression during prefill.

    Wraps any :class:`ScorerPress` and replaces its hard eviction with merge-on-evict:
    each evicted token is folded into its most similar surviving neighbor rather than
    being discarded.  Values are blended via a similarity-weighted average; keys can
    optionally be merged or left unchanged depending on the ``merge_keys`` flag.

    The scoring is delegated entirely to the wrapped press; only the eviction step
    changes.  This makes the wrapper composable with all existing scorers.

    See also :class:`MergingDecodingPress` for the decoding-phase counterpart.

    Parameters
    ----------
    press : ScorerPress
        The underlying scoring method whose scores determine which tokens survive.
    similarity_threshold : float, default=0.0
        Minimum cosine similarity between an evicted key and its nearest survivor
        for the merge to proceed.  Evicted tokens below this threshold are dropped
        without merging.  Because cosine similarity can be negative, ``threshold=0.0``
        still drops tokens whose keys point in the opposite direction of all survivors.
    merge_keys : bool, default=False
        Whether to merge evicted information into kept keys.  When ``False`` (default),
        only values are merged — kept keys are returned unchanged.  This preserves
        RoPE positional encoding in the keys.  Empirically, ``merge_keys=True`` hurts
        quality on RoPE models (−2.5 pp on RULER-4096 with Qwen3-8B at CR=0.75).
    value_norm_weighting : bool, default=True
        When ``True`` (default), the merge weight for each evicted token is additionally
        scaled by the relative L2 norm of its value vector.  This allocates more merge
        budget to evicted tokens that carry high-magnitude value content.  Empirically,
        this improves accuracy by ~1.9 pp on RULER-4096.
    max_merge_per_token : int, default=0
        Maximum number of evicted tokens merged into any single survivor before
        weight scaling kicks in.  ``0`` (default) disables the cap.  When set,
        survivors that receive more merges than the cap have each inbound weight
        scaled down proportionally, preventing dilution of high-importance tokens.
    """

    press: ScorerPress
    similarity_threshold: float = 0.0
    merge_keys: bool = False
    value_norm_weighting: bool = True
    max_merge_per_token: int = 0
    score_weighting: bool = False
    adaptive_threshold: bool = False
    collect_diagnostics: bool = False
    diagnostics: list = field(default_factory=list, repr=False)

    def __post_init__(self):
        assert isinstance(self.press, ScorerPress), f"MergingPress requires a ScorerPress, got {type(self.press)}"
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

        bsz, num_kv_heads, k_len, head_dim = keys.shape
        scores = self.press.score(module, hidden_states, keys, values, attentions, kwargs)

        n_kept = int(k_len * (1 - self.press.compression_ratio))
        if n_kept >= k_len:
            return keys, values
        if n_kept <= 0:
            return keys[:, :, :0, :].contiguous(), values[:, :, :0, :].contiguous()

        result = _merge_on_evict(
            keys, values, scores, n_kept, self.similarity_threshold, self.merge_keys, self.value_norm_weighting,
            self.max_merge_per_token, self.collect_diagnostics, self.score_weighting, self.adaptive_threshold,
        )
        if self.collect_diagnostics and len(result) == 3:
            merged_keys, merged_values, diag = result
            self.diagnostics.append(diag)
            return merged_keys, merged_values
        return result[:2]


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
    Bernoulli mask derived from cumulative attention, ``MergingDecodingPress`` merges into
    the *most similar survivor* using cosine similarity — making it position-agnostic.

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
        Maximum merges per survivor before weight scaling.  ``0`` disables.
    """

    base_press: ScorerPress
    compression_interval: int = 512
    target_size: int = 2048
    hidden_states_buffer_size: int = 256
    similarity_threshold: float = 0.0
    merge_keys: bool = False
    value_norm_weighting: bool = True
    max_merge_per_token: int = 0
    score_weighting: bool = False
    adaptive_threshold: bool = False

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
            keys, values, scores, n_kept, self.similarity_threshold, self.merge_keys, self.value_norm_weighting,
            self.max_merge_per_token, score_weighting=self.score_weighting, adaptive_threshold=self.adaptive_threshold,
        )


def _adakv_head_budgets(
    scores: torch.Tensor,
    n_kept: int,
    alpha_safeguard: float,
) -> torch.Tensor:
    """
    Compute per-head token budgets using the AdaKV adaptive allocation algorithm.

    Selects the top ``n_kept * num_heads`` scores globally, then counts how many
    land in each head.  A safeguard ensures every head retains at least
    ``alpha_safeguard * n_kept`` tokens.

    Parameters
    ----------
    scores : Tensor, shape ``(B, H, L)``
    n_kept : int
        Target tokens to keep *per head* (uniform baseline).
    alpha_safeguard : float
        Minimum fraction of ``n_kept`` guaranteed per head.

    Returns
    -------
    Tensor, shape ``(B, H)``
        Per-head token budget (int values stored as long).
    """
    bsz, num_heads, k_len = scores.shape
    n_safe = int(n_kept * alpha_safeguard)
    total_budget = n_kept * num_heads

    # Protect top-n_safe per head from being deprioritised
    protected_scores = scores.clone()
    if n_safe > 0:
        top_safe_idx = torch.topk(protected_scores, min(n_safe, k_len), dim=-1).indices
        protected_scores.scatter_(-1, top_safe_idx, torch.finfo(scores.dtype).max)

    # Global top-B across all heads → count per head
    flat = protected_scores.reshape(bsz, -1)  # (B, H*L)
    global_top_idx = torch.topk(flat, min(total_budget, flat.shape[-1]), dim=-1).indices  # (B, total_budget)
    head_of_idx = global_top_idx // k_len  # (B, total_budget)

    # Count per head
    budgets = torch.zeros(bsz, num_heads, device=scores.device, dtype=torch.long)
    for b in range(bsz):
        budgets[b] = torch.bincount(head_of_idx[b], minlength=num_heads)[:num_heads]

    # Enforce minimum
    budgets = budgets.clamp(min=max(n_safe, 1))
    return budgets


@dataclass
class MergingAdaKVPress(BasePress):
    """
    Adaptive head-wise merge-on-evict KV cache compression.

    Combines the head-wise adaptive budget allocation of :class:`AdaKVPress`
    with the merge-on-evict strategy of :class:`MergingPress`.  Instead of
    uniform per-head compression, each attention head receives a budget
    proportional to its information density — heads with dispersed attention
    patterns get more tokens, sparse heads get fewer.  Evicted tokens are
    merged into their most similar survivor rather than being discarded.

    This is the first method to combine adaptive head-wise allocation with
    merge-on-evict, yielding both optimal budget distribution and information
    preservation.

    Parameters
    ----------
    press : ScorerPress
        The underlying scoring method.
    alpha_safeguard : float, default=0.20
        Minimum fraction of tokens each head must retain (from AdaKV).
    similarity_threshold : float, default=0.0
        Minimum cosine similarity for a merge to proceed.
    merge_keys : bool, default=False
        Whether to merge evicted keys into survivors.
    value_norm_weighting : bool, default=True
        Scale merge weight by relative value-vector L2 norm.
    max_merge_per_token : int, default=0
        Maximum merges per survivor before weight scaling.  ``0`` disables.
    score_weighting : bool, default=False
        Scale merge weight by normalised importance score.
    adaptive_threshold : bool, default=False
        Compute similarity threshold dynamically as the 25th percentile
        of per-token maximum cosine similarities.
    """

    press: ScorerPress
    alpha_safeguard: float = 0.20
    similarity_threshold: float = 0.0
    merge_keys: bool = False
    value_norm_weighting: bool = True
    max_merge_per_token: int = 0
    score_weighting: bool = False
    adaptive_threshold: bool = False
    collect_diagnostics: bool = False
    diagnostics: list = field(default_factory=list, repr=False)

    def __post_init__(self):
        assert isinstance(self.press, ScorerPress), f"MergingAdaKVPress requires a ScorerPress, got {type(self.press)}"
        assert 0.0 <= self.alpha_safeguard <= 1.0
        assert 0.0 <= self.similarity_threshold <= 1.0
        assert self.max_merge_per_token >= 0

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

        bsz, num_kv_heads, k_len, head_dim = keys.shape
        scores = self.press.score(module, hidden_states, keys, values, attentions, kwargs)

        n_kept_uniform = int(k_len * (1 - self.press.compression_ratio))
        if n_kept_uniform >= k_len:
            return keys, values
        if n_kept_uniform <= 0:
            return keys[:, :, :0, :].contiguous(), values[:, :, :0, :].contiguous()

        # --- Adaptive per-head budgets ---
        budgets = _adakv_head_budgets(scores, n_kept_uniform, self.alpha_safeguard)  # (B, H)
        max_budget = budgets.max().item()

        # --- Per-head merge-on-evict ---
        # Process each head with its own budget, pad to max_budget for uniform output
        all_keys = torch.zeros(bsz, num_kv_heads, max_budget, head_dim, device=keys.device, dtype=keys.dtype)
        all_values = torch.zeros(bsz, num_kv_heads, max_budget, head_dim, device=keys.device, dtype=values.dtype)
        all_diags = []

        for h in range(num_kv_heads):
            h_keys = keys[:, h : h + 1, :, :]  # (B, 1, L, D)
            h_values = values[:, h : h + 1, :, :]
            h_scores = scores[:, h : h + 1, :]

            # Use the *minimum* budget across the batch for this head for simplicity
            h_budget = budgets[:, h].min().item()
            h_budget = min(max(h_budget, 1), k_len)

            if h_budget >= k_len:
                # No compression for this head — pad
                all_keys[:, h, :k_len, :] = keys[:, h, :, :]
                all_values[:, h, :k_len, :] = values[:, h, :, :]
                continue

            result = _merge_on_evict(
                h_keys, h_values, h_scores, h_budget,
                self.similarity_threshold, self.merge_keys, self.value_norm_weighting,
                self.max_merge_per_token, self.collect_diagnostics, self.score_weighting,
                self.adaptive_threshold,
            )
            if self.collect_diagnostics and len(result) == 3:
                mk, mv, diag = result
                diag["head"] = h
                all_diags.append(diag)
            else:
                mk, mv = result[:2]

            # mk, mv are (B, 1, h_budget, D) — place in padded output
            all_keys[:, h, :h_budget, :] = mk[:, 0, :, :]
            all_values[:, h, :h_budget, :] = mv[:, 0, :, :]

        if self.collect_diagnostics and all_diags:
            self.diagnostics.append(all_diags)

        return all_keys.contiguous(), all_values.contiguous()
