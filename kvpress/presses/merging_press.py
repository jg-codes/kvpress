# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from kvpress.presses.base_press import BasePress
from kvpress.presses.scorer_press import ScorerPress


@dataclass
class MergingPress(BasePress):
    """
    Scorer-agnostic merge-on-evict wrapper for KV cache compression.

    Wraps any :class:`ScorerPress` and replaces its hard eviction with merge-on-evict:
    each evicted token is folded into its most similar surviving neighbor rather than
    being discarded.  Values are blended via a similarity-weighted average; keys can
    optionally be merged or left unchanged depending on the ``merge_keys`` flag.

    The scoring is delegated entirely to the wrapped press; only the eviction step
    changes.  This makes the wrapper composable with all existing scorers.

    Parameters
    ----------
    press : ScorerPress
        The underlying scoring method whose scores determine which tokens survive.
    similarity_threshold : float, default=0.0
        Minimum cosine similarity between an evicted key and its nearest survivor
        for the merge to proceed.  Evicted tokens below this threshold are dropped
        without merging.  Because cosine similarity can be negative, ``threshold=0.0``
        still drops tokens whose keys point in the opposite direction of all survivors.
        Use 0.0 to merge same-direction tokens; 0.8 for conservative gating.
    merge_keys : bool, default=True
        Whether to merge evicted information into kept keys.  When ``False``, only
        values are merged — kept keys are returned unchanged.  This preserves RoPE
        positional encoding in the keys and can improve quality on models that use
        rotary embeddings.
    value_norm_weighting : bool, default=False
        When ``True``, the merge weight for each evicted token is additionally scaled
        by the relative L2 norm of its value vector.  This allocates more merge budget
        to evicted tokens that carry high-magnitude value content and less to those
        with near-zero values.
    """

    press: ScorerPress
    similarity_threshold: float = 0.0
    merge_keys: bool = True
    value_norm_weighting: bool = False

    def __post_init__(self):
        assert isinstance(self.press, ScorerPress), f"MergingPress requires a ScorerPress, got {type(self.press)}"
        assert 0.0 <= self.similarity_threshold <= 1.0

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

        # --- 1. Score via wrapped press ---
        scores = self.press.score(module, hidden_states, keys, values, attentions, kwargs)
        # scores: (B, H, k_len)

        n_kept = int(k_len * (1 - self.press.compression_ratio))
        if n_kept >= k_len:
            return keys, values
        if n_kept <= 0:
            return keys[:, :, :0, :].contiguous(), values[:, :, :0, :].contiguous()
        n_evict = k_len - n_kept

        # --- 2. Partition into keep / evict ---
        keep_idx = scores.topk(n_kept, dim=-1).indices  # (B, H, n_kept)

        # Evict mask → evict indices (uniform n_evict per slice).
        # nonzero() returns indices in row-major (C-contiguous) order, so the
        # reshape below produces a correct (B, H, n_evict) partition.
        mask = torch.ones(bsz, num_kv_heads, k_len, device=keys.device, dtype=torch.bool)
        mask.scatter_(2, keep_idx, False)
        evict_idx = mask.nonzero(as_tuple=False)[:, 2].reshape(bsz, num_kv_heads, n_evict)

        # --- 3. Gather kept and evicted tensors ---
        idx4 = keep_idx.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        kept_keys = keys.gather(2, idx4)  # (B, H, n_kept, D)
        kept_values = values.gather(2, idx4)

        idx4_e = evict_idx.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        evict_keys = keys.gather(2, idx4_e)  # (B, H, n_evict, D)
        evict_values = values.gather(2, idx4_e)

        # --- 4. Batched cosine similarity → nearest survivor ---
        e_norm = F.normalize(evict_keys.float(), dim=-1)
        s_norm = F.normalize(kept_keys.float(), dim=-1)
        # (B, H, n_evict, n_kept)
        sim = torch.matmul(e_norm, s_norm.transpose(-2, -1))
        max_sim, target_idx = sim.max(dim=-1)  # (B, H, n_evict)

        # --- 5. Threshold gate ---
        merge_mask = max_sim >= self.similarity_threshold  # (B, H, n_evict)
        if not merge_mask.any():
            return kept_keys.contiguous(), kept_values.contiguous()

        # --- 6. Similarity-weighted scatter-add merge ---
        # Use cosine similarity as the blend factor: the closer an evicted token
        # is to its target survivor, the more it shifts the merged result.  Kept
        # tokens anchor at unit weight.
        evict_w = max_sim.clamp(min=0) * merge_mask  # (B, H, n_evict)

        # Optional: scale merge weight by relative value norm so that
        # high-magnitude evicted values contribute proportionally more.
        if self.value_norm_weighting:
            ev_vnorm = evict_values.float().norm(dim=-1)  # (B, H, n_evict)
            # Gather kept-value norms at the target positions
            kv_vnorm = kept_values.float().norm(dim=-1)  # (B, H, n_kept)
            target_vnorm = kv_vnorm.gather(2, target_idx)  # (B, H, n_evict)
            # Relative norm: evict / (evict + kept), clamped for stability
            rel_norm = ev_vnorm / (ev_vnorm + target_vnorm + 1e-8)
            evict_w = evict_w * rel_norm

        ew = evict_w.unsqueeze(-1)  # (B, H, n_evict, 1)
        tgt = target_idx.unsqueeze(-1).expand_as(evict_keys)  # (B, H, n_evict, D)

        # Accumulate in float32 for numerical stability
        val_accum = torch.zeros(bsz, num_kv_heads, n_kept, head_dim, device=keys.device, dtype=torch.float32)
        val_accum.scatter_add_(2, tgt, ew * evict_values.float())

        w_accum = torch.zeros(bsz, num_kv_heads, n_kept, device=keys.device, dtype=torch.float32)
        w_accum.scatter_add_(2, target_idx, evict_w)

        # --- 7. Normalize: weighted average for active positions ---
        active = w_accum > 0  # (B, H, n_kept)
        total_w = (1.0 + w_accum).unsqueeze(-1)  # (B, H, n_kept, 1), always >= 1
        active_mask = active.unsqueeze(-1)  # (B, H, n_kept, 1)

        # Values are always merged
        new_vals = (kept_values.float() + val_accum) / total_w
        merged_values = torch.where(active_mask, new_vals.to(kept_values.dtype), kept_values)

        # Keys are merged only when merge_keys=True; otherwise kept keys are
        # returned unchanged, preserving their original RoPE encoding.
        if self.merge_keys:
            key_accum = torch.zeros(bsz, num_kv_heads, n_kept, head_dim, device=keys.device, dtype=torch.float32)
            key_accum.scatter_add_(2, tgt, ew * evict_keys.float())
            new_keys = (kept_keys.float() + key_accum) / total_w
            merged_keys = torch.where(active_mask, new_keys.to(kept_keys.dtype), kept_keys)
        else:
            merged_keys = kept_keys

        return merged_keys.contiguous(), merged_values.contiguous()
