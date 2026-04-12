# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    being discarded.  Keys are blended via a score-weighted average; values are added
    with a cosine-similarity weight that limits magnitude inflation while partially
    compensating for attention sag.

    The scoring is delegated entirely to the wrapped press; only the eviction step
    changes.  This makes the wrapper composable with all existing scorers.

    Parameters
    ----------
    press : ScorerPress
        The underlying scoring method whose scores determine which tokens survive.
    similarity_threshold : float, default=0.0
        Minimum cosine similarity between an evicted key and its nearest survivor
        for the merge to proceed.  Evicted tokens below this threshold are dropped
        without merging.  Use 0.0 to merge all; 0.8 for conservative gating.
    """

    press: ScorerPress
    similarity_threshold: float = 0.0

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
        # scores: (bsz, num_kv_heads, k_len)

        n_kept = int(k_len * (1 - self.press.compression_ratio))
        if n_kept >= k_len:
            return keys, values

        # --- 2. Partition into keep / evict ---
        topk = scores.topk(n_kept, dim=-1)
        keep_idx = topk.indices  # (bsz, num_kv_heads, n_kept)

        # Build evict mask and indices
        mask = torch.ones(bsz, num_kv_heads, k_len, device=keys.device, dtype=torch.bool)
        mask.scatter_(2, keep_idx, False)
        evict_idx = mask.nonzero(as_tuple=False)  # (N_evict, 3) — [batch, head, pos]

        if evict_idx.shape[0] == 0:
            # Nothing to evict
            idx4 = keep_idx.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            return keys.gather(2, idx4).contiguous(), values.gather(2, idx4).contiguous()

        # --- 3. Gather kept and evicted tensors ---
        idx4_keep = keep_idx.unsqueeze(-1).expand(-1, -1, -1, head_dim)
        kept_keys = keys.gather(2, idx4_keep)      # (bsz, H, n_kept, D)
        kept_values = values.gather(2, idx4_keep)   # (bsz, H, n_kept, D)

        # --- 4. Vectorised merge-on-evict ---
        # For each (batch, head) slice: compute cosine similarity between evicted and kept keys,
        # find the nearest survivor, and merge if above threshold.
        for b in range(bsz):
            for h in range(num_kv_heads):
                # Indices of evicted positions within this (b, h) slice
                slice_mask = (evict_idx[:, 0] == b) & (evict_idx[:, 1] == h)
                e_pos = evict_idx[slice_mask, 2]  # positions in original seq
                if e_pos.shape[0] == 0:
                    continue

                e_keys = keys[b, h, e_pos]       # (n_evict, D)
                s_keys = kept_keys[b, h]          # (n_kept, D)

                # Cosine similarity: (n_evict, n_kept)
                sim = F.cosine_similarity(
                    e_keys.unsqueeze(1), s_keys.unsqueeze(0), dim=-1
                )
                max_sim, target_idx = sim.max(dim=1)  # (n_evict,)

                # Gate by similarity threshold
                merge_mask = max_sim >= self.similarity_threshold
                if not merge_mask.any():
                    continue

                e_keys_m = e_keys[merge_mask]
                e_vals_m = values[b, h, e_pos[merge_mask]]
                tgt = target_idx[merge_mask]
                cosines = max_sim[merge_mask]

                # Scores for blending weight
                e_scores = scores[b, h, e_pos[merge_mask]].abs()
                s_scores = scores[b, h].gather(0, keep_idx[b, h].gather(0, tgt)).abs()
                alpha = e_scores / (e_scores + s_scores + 1e-8)  # (n_merge,)

                # Key merge: score-weighted interpolation
                # k_survivor += alpha * (k_evicted - k_survivor)
                delta_k = alpha.unsqueeze(-1) * (e_keys_m - kept_keys[b, h, tgt])
                kept_keys[b, h].scatter_add_(0, tgt.unsqueeze(-1).expand_as(delta_k), delta_k.to(kept_keys.dtype))

                # Value merge: score-weighted interpolation (same as keys to preserve magnitude)
                # v_survivor += alpha * (v_evicted - v_survivor)
                delta_v = alpha.unsqueeze(-1) * (e_vals_m - kept_values[b, h, tgt])
                kept_values[b, h].scatter_add_(0, tgt.unsqueeze(-1).expand_as(delta_v), delta_v.to(kept_values.dtype))

        return kept_keys.contiguous(), kept_values.contiguous()
