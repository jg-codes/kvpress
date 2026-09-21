# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn
from transformers import QuantizedCache

from kvpress.presses.base_press import BasePress
from kvpress.presses.decoding_press import DecodingPress
from kvpress.presses.scorer_press import ScorerPress
from kvpress.utils import compute_n_kept, extract_keys_and_values

logger = logging.getLogger(__name__)

# Epsilon for numerical stability — safe for float16 (min ~6e-8) and bfloat16
_EPS = 1e-6

# Largest exponent used when converting a log-mass score difference into a fold weight.
# Evicted tokens score at or below their survivor, so the difference is normally <= 0; the clamp
# only guards rows where a mask-based inner press evicted a token that outscores its target.
_MAX_LOG_WEIGHT = 30.0

FOLDS = ("blend", "mass")
BIASES = ("none", "score")
SCORE_MAPS = ("log", "z")


def _map_scores(scores: torch.Tensor, score_map: str) -> torch.Tensor:
    """Map a press score ``(B, H, L)`` to log-mass units ``s`` (float32).

    ``"log"``: ``s = log(score)`` for attention-type scores (SnapKV, TOVA, ExpectedAttention,
    the KVzip / RestoreKV reconstruction attention), which are non-negative and proportional to
    an attention mass. ``"z"``: per-(batch, head) z-score, for norm-type scores (KnormPress)
    whose scale carries no mass interpretation.
    """
    s = scores.float()
    if score_map == "log":
        if bool((s < 0).any()):
            raise ValueError(
                "score_map='log' needs a non-negative (attention-type) score; this press returned negative "
                "scores (norm-type, e.g. KnormPress) — use score_map='z'"
            )
        return torch.log(s.clamp(min=1e-30))
    if score_map == "z":
        mu = s.mean(dim=-1, keepdim=True)
        sd = s.std(dim=-1, keepdim=True).clamp(min=1e-6)
        return (s - mu) / sd
    raise ValueError(f"score_map must be one of {SCORE_MAPS}, got {score_map!r}")


def _fraction_gate(max_sim: torch.Tensor, merge_ok: torch.Tensor, merge_fraction: float) -> torch.Tensor:
    """Keep only the top ``merge_fraction`` of the *eligible* tokens by similarity, ranked within the
    eligible set of each row (port of NVIDIA/kvpress#287, commit 58bc611; same idiom as
    ``torch.nn.utils.prune.PruningContainer._combine_masks``). ``max_sim`` and ``merge_ok`` are
    ``(..., n_evict)``; the rank is taken along the last dimension.
    """
    if merge_fraction >= 1.0 or not bool(merge_ok.any()):
        return merge_ok
    n_eligible = merge_ok.sum(dim=-1, keepdim=True)
    n_merge = (n_eligible.float() * merge_fraction).round().clamp(min=1).long()
    k_max = int(n_merge.max())
    top_sim = max_sim.masked_fill(~merge_ok, float("-inf")).topk(k_max, dim=-1).values
    threshold = top_sim.gather(-1, (n_merge - 1).clamp(max=k_max - 1))
    return merge_ok & (max_sim >= threshold)


def _merge_on_evict(
    keys: torch.Tensor,
    values: torch.Tensor,
    scores: torch.Tensor,
    n_kept: int,
    similarity_threshold: float,
    merge_keys: bool,
    value_norm_weighting: bool,
    max_merge_per_token: int = 0,
    merge_fraction: float = 1.0,
    perturbation_gate: float = 0.0,
    fold: str = "blend",
    score_map: str = "log",
    target_mask: torch.Tensor | None = None,
    return_stats: bool = False,
):
    """
    Core merge-on-evict kernel for :class:`MergingPress`.

    Given per-token scores, partitions into *keep* and *evict* sets, then folds
    each evicted token into its most cosine-similar survivor via a weighted
    scatter-add instead of discarding it.

    Two value folds are available. ``fold="blend"`` (shipped): weight
    ``w_j = cos(k_j, k_i) * ||v_j|| / (||v_j|| + ||v_i||)``. ``fold="mass"``: weight
    ``w_j = exp(s_j - s_i)`` with ``s`` the press score in log-mass units (see
    :func:`_map_scores`), so that ``v_i' = (e^{s_i} v_i + sum_j e^{s_j} v_j) / (e^{s_i} + sum_j e^{s_j})``;
    similarity is then used for routing only. With ``return_stats=True`` the kernel also returns
    per-survivor merge counts and the score-based logit bias
    ``b_i = log(1 + sum_j exp(s_j - s_i)) = log(1 + sum_j w_j)`` (zero for unmerged survivors),
    which restores the folded attention mass at the survivor's softmax logit (C86).

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
    fold : {"blend", "mass"}, default="blend"
        Value fold (see above). ``"mass"`` ignores ``value_norm_weighting``.
    score_map : {"log", "z"}, default="log"
        Map from press score to log-mass units, used only when ``fold="mass"``.
    target_mask : Tensor, shape ``(B, H, L)``, dtype bool, optional
        ``True`` at positions that may receive merges (e.g. to exclude attention sinks).
    return_stats : bool, default=False
        Also return ``counts`` and ``bias`` of shape ``(B, H, n_kept)`` (float32) in survivor order.

    Returns
    -------
    tuple[Tensor, Tensor] or tuple[Tensor, Tensor, Tensor, Tensor]
        ``(merged_keys, merged_values)`` each of shape ``(B, H, n_kept, D)``;
        with ``return_stats=True`` also ``(counts, bias)``.

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
    if target_mask is not None:
        # Survivors that may not receive merges (sinks, restore slots) are removed from the routing
        kept_target_ok = target_mask.to(keys.device).gather(2, keep_idx)  # (B, H, n_kept)
        sim = sim.masked_fill(~kept_target_ok.unsqueeze(2), float("-inf"))
    max_sim, target_idx = sim.max(dim=-1)  # (B, H, n_evict)

    # --- Threshold gate ---
    merge_mask = max_sim >= similarity_threshold

    # --- Fraction gate (rank within the eligible set of each row, NVIDIA/kvpress#287) ---
    merge_mask = _fraction_gate(max_sim, merge_mask, merge_fraction)

    # --- Perturbation-bound gate: skip merges with high estimated error ---
    # bound_i = ‖v_i‖ * (1 - w) / (1 + w)  where w = cosine similarity
    if perturbation_gate > 0 and merge_mask.any():
        evict_v_norms = evict_values.float().norm(dim=-1)  # (B, H, n_evict)
        error_bound = evict_v_norms * (1 - max_sim) / (1 + max_sim + _EPS)
        merge_mask = merge_mask & (error_bound <= perturbation_gate)

    if not merge_mask.any():
        if return_stats:
            zeros = torch.zeros(bsz, num_key_value_heads, n_kept, device=keys.device, dtype=torch.float32)
            return kept_keys.contiguous(), kept_values.contiguous(), zeros, zeros.clone()
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

    # --- Merge weights ---
    if fold == "mass":
        # w_j = exp(s_j - s_i): the evicted token's attention mass relative to its survivor's
        s = _map_scores(scores, score_map)  # (B, H, L)
        s_evict = s.gather(2, evict_idx)
        s_target = s.gather(2, keep_idx).gather(2, target_idx)
        evict_w = torch.exp((s_evict - s_target).clamp(max=_MAX_LOG_WEIGHT)) * merge_mask
    else:
        # Similarity-weighted (x relative value norm) blend
        evict_w = max_sim.clamp(min=0) * merge_mask  # (B, H, n_evict)
        if value_norm_weighting:
            ev_vnorm = evict_values.float().norm(dim=-1)
            target_vnorm = kept_values.float().norm(dim=-1).gather(2, target_idx)
            rel_norm = ev_vnorm / (ev_vnorm + target_vnorm + _EPS)
            evict_w = evict_w * rel_norm

    # Number of evicted tokens folded into each survivor (the cap rescales weights, not counts)
    merge_count = torch.zeros(bsz, num_key_value_heads, n_kept, device=keys.device, dtype=torch.float32)
    merge_count.scatter_add_(2, target_idx, merge_mask.float())

    # --- Merge count cap: prevent survivor dilution ---
    if max_merge_per_token > 0:
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

    if return_stats:
        # b_i = log(1 + sum_j w_j); equals log(1 + sum_j exp(s_j - s_i)) for fold="mass"
        bias = torch.log1p(w_accum)
        return merged_keys.contiguous(), merged_values.contiguous(), merge_count, bias
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
    * ``MergingPress(DMSPress(ScorerPress))``: delegates to DMSPress's threshold-based
      eviction via its ``forward_hook``, then merges evicted tokens into survivors.
      Combines content-adaptive compression with merge-on-evict.

    For compress-based inner presses, MergingPress delegates ``.compress()`` and reads
    back ``module.masked_key_indices``.  For hook-based inner presses (like DMSPress)
    that override ``forward_hook`` without implementing ``compress()``, MergingPress
    delegates to the inner hook and then merges.

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
    perturbation_gate : float, default=0.0
        Maximum per-merge perturbation bound for a merge to proceed.  For each
        evicted token *i* routed to survivor *j* with cosine similarity *w*, the
        bound is :math:`\\|v_i\\| \\cdot (1 - w) / (1 + w)`.  Merges exceeding
        this threshold are skipped (hard-evicted instead).  ``0.0`` disables the
        gate.  Useful for preventing regressions on exact-match tasks (e.g. QA)
        where high-norm answer tokens should not be blended into context survivors.
    exclude_restore_targets : bool, default=False
        Call-overriding inner presses only (KVzip family).  When ``True``, cache positions
        ``>= press.context_length`` (e.g. the restore slots appended by
        :class:`RestoreKVPress`) are never merge *targets*.  They are never evicted by
        the inner press, so they are never merge sources either.
    exclude_sink_targets : bool, default=False
        Call-overriding inner presses only.  When ``True``, the first ``press.n_sink``
        positions are never merge targets.
    count_logit_bias : bool, default=False
        Call-overriding inner presses only.  When ``True``, every survivor that absorbed
        ``M`` evicted tokens receives an additive attention-logit bias ``log(1 + M)``
        (KeepKV, arXiv:2504.09936) at attention time, applied through the attention
        mask by :func:`kvpress.attention_patch.attention_patch`.  Unmerged positions
        get bias 0.  Requires an attention implementation that accepts an additive
        float mask (eager, sdpa); not compatible with flash attention.
    fold : {"blend", "mass"}, default="blend"
        Value fold.  ``"blend"`` is the shipped similarity x relative-value-norm blend.
        ``"mass"`` folds with weights ``e^{s_j}`` in the inner press's own score mapped to
        log-mass units (``score_map``): ``v_i' = (e^{s_i} v_i + sum_j e^{s_j} v_j) / (e^{s_i} + sum_j e^{s_j})``.
        Cosine similarity is then used for routing only, and ``value_norm_weighting`` is ignored.
        Available for :class:`ScorerPress` inner presses (``score()``) and for call-overriding
        inner presses that expose a per-token ``score_val`` (KVzip family, incl. RestoreKVPress).
    bias : {"none", "score"}, default="none"
        ``"score"`` adds ``b_i = log(1 + sum_j exp(s_j - s_i))`` to the attention logit of every
        survivor ``i`` that absorbed evicted tokens ``j`` (zero for unmerged survivors, never on
        restore slots or sinks when those are excluded as targets), through the same attention-mask
        hook as ``count_logit_bias``.  With ``fold="mass"`` this restores, at the survivor's logit,
        the attention mass the folded tokens carried under the press score.  Requires
        ``fold="mass"``: a logit bias on top of the similarity/value-norm blend is not allowed.
    score_map : {"log", "z"}, default="log"
        Map from press score to log-mass units for ``fold="mass"`` / ``bias="score"``: ``"log"``
        for attention-type scores (SnapKV, TOVA, ExpectedAttention, KVzip/RestoreKV reconstruction
        attention), ``"z"`` for norm-type scores (KnormPress).
    """

    press: BasePress
    similarity_threshold: float = 0.0
    merge_keys: bool = False
    value_norm_weighting: bool = True
    max_merge_per_token: int = 0
    merge_fraction: float = 1.0
    perturbation_gate: float = 0.0
    exclude_restore_targets: bool = False
    exclude_sink_targets: bool = False
    count_logit_bias: bool = False
    fold: str = "blend"
    bias: str = "none"
    score_map: str = "log"

    def __post_init__(self):
        assert isinstance(self.press, BasePress), f"MergingPress requires a BasePress, got {type(self.press)}"
        assert 0.0 <= self.similarity_threshold <= 1.0
        assert self.max_merge_per_token >= 0, "max_merge_per_token must be non-negative"
        assert 0.0 < self.merge_fraction <= 1.0, "merge_fraction must be in (0, 1]"
        assert self.perturbation_gate >= 0.0, "perturbation_gate must be non-negative"
        assert self.fold in FOLDS, f"fold must be one of {FOLDS}, got {self.fold!r}"
        assert self.bias in BIASES, f"bias must be one of {BIASES}, got {self.bias!r}"
        assert self.score_map in SCORE_MAPS, f"score_map must be one of {SCORE_MAPS}, got {self.score_map!r}"
        if self.exclude_restore_targets or self.count_logit_bias:
            assert self._is_call_overriding_press(), (
                "exclude_restore_targets / count_logit_bias are implemented for "
                "call-overriding inner presses (KVzip family, RestoreKVPress) only"
            )
        if self.exclude_sink_targets:
            assert self._is_call_overriding_press() or (
                isinstance(self.press, ScorerPress) and hasattr(self.press, "n_sink")
            ), "exclude_sink_targets needs a call-overriding inner press or a ScorerPress with an n_sink attribute"
        if self.bias == "score":
            assert self.fold == "mass", (
                "bias='score' is the mass-restoring logit bias and must be combined with fold='mass'; "
                "it is not defined on top of the similarity / value-norm blend (fold='blend')"
            )
            assert not self.count_logit_bias, "bias='score' and count_logit_bias are mutually exclusive"
        if self.fold == "mass":
            if not (isinstance(self.press, ScorerPress) or self._exposes_score_val()):
                raise NotImplementedError(
                    "fold='mass' needs a per-token score: a ScorerPress inner (score()) or a call-overriding "
                    f"inner press exposing score_val (KVzip family); got {type(self.press).__name__}"
                )

    def _exposes_score_val(self) -> bool:
        """Call-overriding inner press (KVzip family, incl. RestoreKVPress) whose ``compress_post`` keeps the
        per-token reconstruction score in ``score_val`` of shape ``(n_layers, bsz, num_kv_heads, context_length)``."""
        return self._is_call_overriding_press() and hasattr(self.press, "score_val")

    def post_init_from_model(self, model):
        self.press.post_init_from_model(model)

    @property
    def compression_ratio(self):
        return self.press.compression_ratio

    @compression_ratio.setter
    def compression_ratio(self, value):
        self.press.compression_ratio = value

    @property
    def threshold(self):
        """Passthrough to inner press threshold (e.g. DMSPress)."""
        return getattr(self.press, "threshold", None)

    @threshold.setter
    def threshold(self, value):
        if hasattr(self.press, "threshold"):
            self.press.threshold = value
        else:
            raise AttributeError(f"Inner press {type(self.press).__name__} has no threshold attribute")

    def _is_hook_based_press(self) -> bool:
        """Check if the inner press uses forward_hook instead of compress."""
        return type(self.press).compress is BasePress.compress and type(self.press).forward_hook is not BasePress.forward_hook

    def _is_call_overriding_press(self) -> bool:
        """Check if the inner press drives compression from its own ``__call__`` (KVzip family).

        Such presses score in forward hooks but evict only in ``compress_post`` (called from
        their ``__call__`` after prefill), by setting ``module.masked_key_indices``. Registering
        ``MergingPress.forward_hook`` alone would run neither their scoring passes nor their
        eviction, so the merge must be attached to ``compress_post`` instead.
        """
        return type(self.press).__call__ is not BasePress.__call__ and hasattr(self.press, "compress_post")

    @contextmanager
    def __call__(self, model):
        if not self._is_call_overriding_press():
            with super().__call__(model):
                yield
            return

        inner = self.press
        original_compress_post = inner.compress_post
        language_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
        for layer in language_model.layers:
            layer.self_attn.merge_logit_bias = None
            layer.self_attn.merge_counts = None

        def compress_post_then_merge(model_):
            original_compress_post(model_)
            self.merge_masked_cache(model_, inner._cache)

        inner.compress_post = compress_post_then_merge  # instance attribute shadows the class method
        try:
            with inner(model):
                yield
        finally:
            del inner.compress_post

    def merge_masked_cache(self, model, cache):
        """Merge evicted tokens into survivors for every layer whose ``masked_key_indices`` is set.

        Used for call-overriding inner presses (KVzip family, incl. RestoreKVPress) right after
        their ``compress_post`` has decided the eviction. Evicted positions stay in the cache
        (they are masked at attention time); only survivor values (and keys if ``merge_keys``)
        are rewritten.
        """
        language_model = model.model.language_model if hasattr(model.model, "language_model") else model.model
        context_length = getattr(self.press, "context_length", None)
        n_sink = int(getattr(self.press, "n_sink", 0) or 0)
        if self.exclude_restore_targets and not context_length:
            raise ValueError("exclude_restore_targets requires the inner press to expose a positive context_length")
        for layer in language_model.layers:
            module = layer.self_attn
            module.merge_logit_bias = None
            module.merge_counts = None
            mask_indices = getattr(module, "masked_key_indices", None)
            if mask_indices is None or len(mask_indices[0]) == 0:
                continue
            keys, values = extract_keys_and_values(cache, module.layer_idx)
            bsz, num_kv_heads, k_len, head_dim = keys.shape
            evict_mask = torch.zeros(bsz, num_kv_heads, k_len, device=keys.device, dtype=torch.bool)
            evict_mask[tuple(i.to(keys.device) for i in mask_indices)] = True
            target_mask = None
            if self.exclude_restore_targets or self.exclude_sink_targets:
                target_mask = torch.ones(bsz, num_kv_heads, k_len, device=keys.device, dtype=torch.bool)
                if self.exclude_restore_targets:
                    target_mask[:, :, int(context_length):] = False
                if self.exclude_sink_targets and n_sink > 0:
                    target_mask[:, :, :n_sink] = False
            scores = None
            if self.fold == "mass":
                # Per-token reconstruction score of the KVzip family, (bsz, num_kv_heads, context_length);
                # appended slots (restore tokens) have no score: they are never evicted, and get +inf so that
                # a fold into one of them (only possible without exclude_restore_targets) carries weight 0.
                score_val = self.press.score_val[int(module.layer_idx)]
                s_ctx = _map_scores(score_val.to(keys.device), self.score_map)
                scores = torch.full((bsz, num_kv_heads, k_len), float("inf"), device=keys.device, dtype=torch.float32)
                scores[:, :, : s_ctx.shape[2]] = s_ctx
            new_keys, new_values, counts, bias = _merge_on_evict_adaptive(
                keys,
                values,
                evict_mask,
                self.similarity_threshold,
                self.merge_keys,
                self.value_norm_weighting,
                self.max_merge_per_token,
                self.merge_fraction,
                self.perturbation_gate,
                target_mask=target_mask,
                return_counts=True,
                scores=scores,
                fold=self.fold,
                return_bias=True,
            )
            self._write_back(cache, module.layer_idx, new_keys, new_values)
            module.merge_counts = counts
            if self.count_logit_bias:
                module.merge_logit_bias = torch.log1p(counts)
            elif self.bias == "score":
                module.merge_logit_bias = bias

    @staticmethod
    def _write_back(cache, layer_idx, new_keys, new_values):
        cache_layer = cache.layers[layer_idx]
        if isinstance(cache, QuantizedCache):
            cache_layer._quantized_keys = cache_layer._quantize(new_keys, axis=cache_layer.axis_key)
            cache_layer._quantized_values = cache_layer._quantize(new_values, axis=cache_layer.axis_value)
            cache_layer.keys = torch.zeros(0, dtype=new_keys.dtype, device=new_keys.device)
            cache_layer.values = torch.zeros(0, dtype=new_keys.dtype, device=new_keys.device)
            cache_layer.cumulative_length = new_keys.shape[2]
        else:
            cache_layer.keys = new_keys
            cache_layer.values = new_values

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
            n_kept = compute_n_kept(k_len, self.press.compression_ratio)
            if n_kept >= k_len:
                return keys, values
            if n_kept <= 0:
                return keys[:, :, :0, :].contiguous(), values[:, :, :0, :].contiguous()

            target_mask = None
            n_sink = int(getattr(self.press, "n_sink", 0) or 0)
            if self.exclude_sink_targets and n_sink > 0:
                target_mask = torch.ones(bsz, num_key_value_heads, k_len, device=keys.device, dtype=torch.bool)
                target_mask[:, :, :n_sink] = False

            keys, values, counts, bias = _merge_on_evict(
                keys,
                values,
                scores,
                n_kept,
                self.similarity_threshold,
                self.merge_keys,
                self.value_norm_weighting,
                self.max_merge_per_token,
                self.merge_fraction,
                self.perturbation_gate,
                fold=self.fold,
                score_map=self.score_map,
                target_mask=target_mask,
                return_stats=True,
            )
            # Survivor-ordered stats, (B, H, n_kept): the truncated cache keeps survivors in this order, so the
            # bias aligns with the cache positions the attention patch sees at decode (new tokens get bias 0).
            module.merge_counts = counts
            module.merge_logit_bias = bias if self.bias == "score" else None
            return keys, values

        # --- Mask-based press path (AdaKV, CriticalAdaKV, etc.) ---
        # Delegate to the inner press which sets module.masked_key_indices
        # and returns keys/values unchanged.
        keys, values = self.press.compress(module, hidden_states, keys, values, attentions, kwargs)

        mask_indices = getattr(module, "masked_key_indices", None)
        if mask_indices is None:
            return keys, values

        # Build boolean eviction mask from (batch, head, seq) index tuple
        evict_mask = torch.zeros(bsz, num_key_value_heads, k_len, device=keys.device, dtype=torch.bool)
        evict_mask[tuple(mask_indices)] = True

        # Merge evicted tokens into their nearest cosine-similar survivors
        new_keys, new_values = _merge_on_evict_adaptive(
            keys,
            values,
            evict_mask,
            self.similarity_threshold,
            self.merge_keys,
            self.value_norm_weighting,
            self.max_merge_per_token,
            self.merge_fraction,
            self.perturbation_gate,
        )
        return new_keys, new_values

    def forward_hook(self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list):
        """Override to support hook-based inner presses (e.g. DMSPress).

        For inner presses that implement their logic in ``forward_hook`` rather than
        ``compress()`` (like DMSPress), we delegate to the inner hook first — letting it
        score, accumulate, and set ``module.masked_key_indices`` — then merge evicted
        tokens into their nearest cosine-similar survivors.

        For all other inner presses, falls through to ``BasePress.forward_hook`` which
        calls ``self.compress()``.
        """
        if not self._is_hook_based_press():
            return super().forward_hook(module, input, kwargs, output)

        # --- Hook-based press path (DMSPress, etc.) ---
        # 1. Delegate to inner press hook: scores, accumulates, sets masks
        output = self.press.forward_hook(module, input, kwargs, output)

        # 2. Check if eviction happened this layer
        mask_indices = getattr(module, "masked_key_indices", None)
        if mask_indices is None or len(mask_indices[0]) == 0:
            return output

        # 3. Extract current keys/values from cache
        cache = kwargs["past_key_values"]
        keys, values = extract_keys_and_values(cache, module.layer_idx)
        bsz, num_kv_heads, k_len, head_dim = keys.shape

        # 4. Build boolean eviction mask from index tuple
        evict_mask = torch.zeros(bsz, num_kv_heads, k_len, device=keys.device, dtype=torch.bool)
        evict_mask[tuple(mask_indices)] = True

        # 5. Merge evicted tokens into survivors
        new_keys, new_values = _merge_on_evict_adaptive(
            keys,
            values,
            evict_mask,
            self.similarity_threshold,
            self.merge_keys,
            self.value_norm_weighting,
            self.max_merge_per_token,
            self.merge_fraction,
            self.perturbation_gate,
        )

        # 6. Write merged values back to cache (evicted positions stay masked)
        self._write_back(cache, module.layer_idx, new_keys, new_values)

        return output


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
    perturbation_gate : float, default=0.0
        Maximum per-merge perturbation bound.  See :class:`MergingPress`.
    """

    similarity_threshold: float = 0.0
    merge_keys: bool = False
    value_norm_weighting: bool = True
    max_merge_per_token: int = 0
    merge_fraction: float = 1.0
    perturbation_gate: float = 0.0

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
            self.merge_fraction,
            self.perturbation_gate,
        )


def _merge_on_evict_adaptive(
    keys: torch.Tensor,
    values: torch.Tensor,
    evict_mask: torch.Tensor,
    similarity_threshold: float,
    merge_keys: bool,
    value_norm_weighting: bool,
    max_merge_per_token: int = 0,
    merge_fraction: float = 1.0,
    perturbation_gate: float = 0.0,
    target_mask: torch.Tensor | None = None,
    return_counts: bool = False,
    scores: torch.Tensor | None = None,
    fold: str = "blend",
    return_bias: bool = False,
):
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
    target_mask : Tensor, shape ``(B, H, L)``, dtype bool, optional
        ``True`` at positions that may receive merges.  Survivors with ``False`` are
        left untouched and never selected as nearest neighbour.  ``None`` means every
        survivor is a valid target.
    return_counts : bool, default=False
        Also return ``counts`` of shape ``(B, H, L)`` (float32): the number of evicted
        tokens folded into each position (0 for evicted, non-target and unmerged positions).
    scores : Tensor, shape ``(B, H, L)``, optional
        Per-token press score in log-mass units (the caller applies :func:`_map_scores`, because
        appended slots without a score must be padded after the map). Required for ``fold="mass"``.
    fold : {"blend", "mass"}, default="blend"
        See :func:`_merge_on_evict`.
    return_bias : bool, default=False
        Also return ``bias`` of shape ``(B, H, L)`` (float32): ``log(1 + sum_j w_j)`` at survivors that
        received merges, 0 elsewhere.

    Returns
    -------
    tuple
        ``(new_keys, new_values)`` — same shape ``(B, H, L, D)`` as input, followed by ``counts``
        if ``return_counts`` and ``bias`` if ``return_bias``.
        Survivor positions contain merged information; evicted positions are unchanged.
    """
    bsz, num_kv_heads, k_len, head_dim = keys.shape
    device = keys.device
    if fold == "mass":
        assert scores is not None, "fold='mass' needs per-token scores"
        assert scores.shape == evict_mask.shape, f"scores {tuple(scores.shape)} != evict_mask {tuple(evict_mask.shape)}"
        s_all = scores.to(device=device, dtype=torch.float32)

    # Work on float32 copies for numerical stability
    merged_values = values.float().clone()
    merged_keys = keys.float().clone() if merge_keys else None
    counts = torch.zeros(bsz, num_kv_heads, k_len, device=device, dtype=torch.float32)
    bias_out = torch.zeros(bsz, num_kv_heads, k_len, device=device, dtype=torch.float32)

    keep_mask = ~evict_mask
    if target_mask is not None:
        assert target_mask.shape == evict_mask.shape, f"target_mask {tuple(target_mask.shape)} != {tuple(evict_mask.shape)}"
        keep_mask = keep_mask & target_mask.to(evict_mask.device)

    # Iterate over (batch, head) — typically B=1, H=8 for Qwen3-8B = 8 iterations
    for b in range(bsz):
        for h in range(num_kv_heads):
            evict_idx = evict_mask[b, h].nonzero(as_tuple=True)[0]
            keep_idx = keep_mask[b, h].nonzero(as_tuple=True)[0]
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

            # Fraction gate (rank within the eligible set, NVIDIA/kvpress#287)
            merge_ok = _fraction_gate(max_sim, merge_ok, merge_fraction)

            # Perturbation-bound gate: skip merges with high estimated error
            if perturbation_gate > 0 and merge_ok.any():
                evict_v_norms = values[b, h, evict_idx].float().norm(dim=-1)
                error_bound = evict_v_norms * (1 - max_sim) / (1 + max_sim + _EPS)
                merge_ok = merge_ok & (error_bound <= perturbation_gate)

            if not merge_ok.any():
                continue

            # Merge weights
            if fold == "mass":
                s_row = s_all[b, h]
                w = torch.exp((s_row[evict_idx] - s_row[keep_idx[target]]).clamp(max=_MAX_LOG_WEIGHT)) * merge_ok.float()
            else:
                w = max_sim.clamp(min=0) * merge_ok.float()

            # Number of evicted tokens folded into each survivor (cap rescales weights, not counts)
            cnt = torch.zeros(n_kept, device=device, dtype=torch.float32)
            cnt.scatter_add_(0, target, merge_ok.float())
            counts[b, h, keep_idx] = cnt

            if fold != "mass" and value_norm_weighting:
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
            bias_out[b, h, keep_idx] = torch.log1p(w_accum)  # 0 where w_accum == 0

            if merge_keys and merged_keys is not None:
                evict_k_orig = keys[b, h, evict_idx].float()
                key_accum = torch.zeros(n_kept, head_dim, device=device, dtype=torch.float32)
                key_accum.scatter_add_(0, target.unsqueeze(-1).expand_as(evict_k_orig), w_exp * evict_k_orig)
                orig_k = merged_keys[b, h, keep_idx]
                new_k = (orig_k + key_accum) / total_w
                merged_keys[b, h, keep_idx] = torch.where(active.unsqueeze(-1), new_k, orig_k)

    result_values = merged_values.to(values.dtype)
    result_keys = merged_keys.to(keys.dtype) if merge_keys else keys
    out = [result_keys, result_values]
    if return_counts:
        out.append(counts)
    if return_bias:
        out.append(bias_out)
    return tuple(out)

