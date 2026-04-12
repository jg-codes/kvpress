# SPDX-FileCopyrightText: Copyright (c) 2025 Johannes Gast
# SPDX-License-Identifier: Apache-2.0

"""
VmaxAdaKVPress: AdaKV with V_max-weighted per-head safeguard.

Standard AdaKV applies a fixed safeguard alpha to every KV head — each head
retains at least alpha * n_kept tokens before global pooling.  This ignores
that heads contribute differently to the output: a head with large max value
norm (V_max) causes a larger error when its retention drops.

From the Boltzmann Lens error bound:

    ||Δo_h|| ≤ 2 (1 - r_h) V_max_h

it follows that heads with high V_max need stronger protection.  VmaxAdaKV
replaces the fixed safeguard with:

    alpha_h = alpha * (V_max_h / mean(V_max))

so that high-V_max heads get proportionally more protected tokens before
the global pruning step.  The total safeguard budget is unchanged on average.
"""

from dataclasses import dataclass

import torch

from kvpress.presses.base_press import BasePress
from kvpress.presses.scorer_press import ScorerPress


@dataclass
class VmaxAdaKVPress(BasePress):
    """
    AdaKV with value-norm-weighted per-head safeguard.

    Allocates safeguard budget proportionally to each head's max value norm,
    so that heads contributing most to the output are protected from
    over-pruning during global pooled token selection.

    Parameters
    ----------
    press : ScorerPress
        Base scoring method (e.g. SnapKVPress, ExpectedAttentionPress).
    alpha_safeguard : float, default=0.20
        Average fraction of kept tokens protected per head.
        Individual heads receive alpha * V_max_h / mean(V_max).
    """

    press: ScorerPress
    alpha_safeguard: float = 0.20

    def __post_init__(self):
        assert isinstance(self.press, ScorerPress), "VmaxAdaKVPress requires a ScorerPress as input"
        assert 0 <= self.alpha_safeguard <= 1, "alpha_safeguard should be in [0, 1]"

    def post_init_from_model(self, model):
        self.press.post_init_from_model(model)

    @property
    def compression_ratio(self):
        return self.press.compression_ratio

    @compression_ratio.setter
    def compression_ratio(self, value):
        self.press.compression_ratio = value

    def compress(self, module, hidden_states, keys, values, attentions, kwargs):
        if self.compression_ratio == 0:
            return keys, values

        assert module.config._attn_implementation != "eager", "eager mode not supported"

        # Compute scores from the inner scorer
        scores = self.press.score(module, hidden_states, keys, values, attentions, kwargs)
        bsz, num_key_value_heads, k_len = scores.shape

        n_kept = int(k_len * (1 - self.compression_ratio))

        # --- V_max-weighted per-head safeguard ---
        # V_max per head: max L2 norm of value vectors across sequence positions
        v_max = values.float().norm(dim=-1).max(dim=-1).values  # (bsz, num_kv_heads)
        v_mean = v_max.mean(dim=-1, keepdim=True)  # (bsz, 1)

        # Per-head alpha: proportional to V_max, average equals alpha_safeguard
        alpha_per_head = self.alpha_safeguard * (v_max / (v_mean + 1e-8))  # (bsz, num_kv_heads)
        alpha_per_head = alpha_per_head.clamp(0, 1)

        # Protect top n_safe_h tokens per head (vectorized across heads)
        # Compute per-head n_safe as int, then scatter max scores
        n_safe_max = 0
        n_safe_list = []
        for h in range(num_key_value_heads):
            n_safe_h = int(n_kept * alpha_per_head[0, h].item())
            n_safe_h = min(n_safe_h, k_len)
            n_safe_list.append(n_safe_h)
            n_safe_max = max(n_safe_max, n_safe_h)

        if n_safe_max > 0:
            # Get top indices for the maximum safeguard budget
            top_indices = torch.topk(scores, min(n_safe_max, k_len), dim=-1).indices

            # Apply per-head safeguard: only protect up to n_safe_h for each head
            for h in range(num_key_value_heads):
                n_safe_h = n_safe_list[h]
                if n_safe_h > 0:
                    safe_idx = top_indices[:, h, :n_safe_h]
                    scores[:, h, :].scatter_(-1, safe_idx, torch.finfo(scores.dtype).max)

        # Global pooled pruning across heads (same as AdaKV)
        n_pruned = num_key_value_heads * (k_len - n_kept)
        indices = torch.topk(-scores.reshape(bsz, -1), n_pruned, dim=1).indices.flatten()

        batch_indices = torch.arange(bsz).repeat_interleave(n_pruned)
        head_indices = indices // k_len
        seq_indices = indices % k_len
        module.masked_key_indices = (batch_indices, head_indices, seq_indices)
        return keys, values
