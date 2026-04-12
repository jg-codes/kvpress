# SPDX-FileCopyrightText: Copyright (c) 2026 Johannes Gast
# SPDX-License-Identifier: Apache-2.0

"""
RetentionPress: Per-head budget allocation minimizing worst-case retention error.

Standard eviction applies the same budget to every KV head.  AdaKV pools scores
across heads but optimizes a global pruning threshold.  RetentionPress instead
allocates per-head budgets k_h to minimize the worst-case single-layer error
bound from the Boltzmann Lens framework:

    max_h  2 (1 - r_h(k_h)) V_max_h

where r_h(k_h) = sum of the top-k_h attention weights for head h, and
V_max_h = max value-vector norm for head h.

This is solved greedily: at each step, give one extra token to the head whose
(1 - r_h) * V_max_h is currently largest (i.e. the head that would benefit most
from keeping one more token).  This is itself a submodular allocation problem
with a (1 - 1/e) guarantee.
"""

from dataclasses import dataclass

import torch
from torch import nn

from kvpress.presses.base_press import BasePress
from kvpress.presses.scorer_press import ScorerPress


@dataclass
class RetentionPress(BasePress):
    """
    Retention-aware per-head KV cache compression.

    Allocates per-head budgets to minimize worst-case attention retention loss,
    using the error bound ||Δo|| ≤ 2(1-r) V_max as the optimization criterion.

    Parameters
    ----------
    press : ScorerPress
        Inner scorer whose scores drive token ranking within each head.
    min_budget_fraction : float, default=0.05
        Minimum fraction of tokens each head must retain (prevents total starvation).
    use_vnorm_weighting : bool, default=True
        Weight the retention criterion by per-head V_max.  When False, minimize
        max_h (1 - r_h) directly (ignore value-vector norms).
    """

    press: ScorerPress
    min_budget_fraction: float = 0.05
    use_vnorm_weighting: bool = True

    def __post_init__(self):
        assert isinstance(self.press, ScorerPress), "RetentionPress requires a ScorerPress"
        assert 0 <= self.min_budget_fraction <= 1

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

        # --- 1. Get per-head scores from inner scorer ---
        scores = self.press.score(module, hidden_states, keys, values, attentions, kwargs)
        # shape: (bsz, num_kv_heads, seq_len)
        bsz, num_kv_heads, seq_len = scores.shape

        # Total budget across all heads
        n_kept_per_head_uniform = int(seq_len * (1 - self.compression_ratio))
        total_budget = num_kv_heads * n_kept_per_head_uniform
        min_per_head = max(1, int(seq_len * self.min_budget_fraction))

        # --- 2. Compute retention curve per head ---
        # Sort scores descending per head; cumsum of softmax gives r_h(k)
        sorted_scores, sorted_indices = scores.sort(dim=-1, descending=True)
        # Convert scores to attention-like weights via softmax per head
        attn_weights = torch.softmax(sorted_scores, dim=-1)
        # Cumulative retention: r_h(k) = sum of top-k attention weights
        cum_retention = attn_weights.cumsum(dim=-1)  # (bsz, num_kv_heads, seq_len)

        # --- 3. Compute V_max per head (optional weighting) ---
        if self.use_vnorm_weighting:
            # values shape: (bsz, num_kv_heads, seq_len, head_dim)
            v_max = values.float().norm(dim=-1).max(dim=-1).values  # (bsz, num_kv_heads)
        else:
            v_max = torch.ones(bsz, num_kv_heads, device=scores.device)

        # --- 4. Greedy per-head budget allocation ---
        # Start with min_per_head budget for each head
        budgets = torch.full((bsz, num_kv_heads), min_per_head, dtype=torch.long,
                             device=scores.device)
        remaining = total_budget - min_per_head * num_kv_heads

        if remaining > 0:
            # Vectorized greedy allocation:
            # For each head h, compute the marginal error reduction of giving it
            # one more token at each possible budget level.
            # error(h, k) = (1 - cum_retention[h, k-1]) * v_max[h]
            # We pick tokens greedily by largest current error.

            # Marginal error at each budget: (bsz, num_kv_heads, seq_len)
            # error_at_k[b, h, k] = (1 - r_h(k+1)) * V_max_h = error if head h has budget k+1
            error_at_k = (1 - cum_retention) * v_max.unsqueeze(-1)  # (bsz, H, S)

            # For batch processing: iterate greedily
            for _ in range(remaining):
                # Current retention per head
                budget_idx = (budgets - 1).clamp(min=0)
                current_error = error_at_k.gather(
                    2, budget_idx.unsqueeze(-1)
                ).squeeze(-1)

                # Mask heads at max budget
                at_max = budgets >= seq_len
                current_error = current_error.masked_fill(at_max, -1.0)

                # Give one token to the worst head
                worst_head = current_error.argmax(dim=-1)
                budgets.scatter_(
                    1,
                    worst_head.unsqueeze(-1),
                    budgets.gather(1, worst_head.unsqueeze(-1)) + 1,
                )
        else:
            # Budget is too small even for minimums; clip
            budgets = budgets.clamp(max=seq_len)
            # Scale down proportionally
            scale = total_budget / (min_per_head * num_kv_heads)
            budgets = (budgets.float() * scale).long().clamp(min=1, max=seq_len)

        # --- 5. Build per-head masks and apply via masked_key_indices ---
        all_batch_idx = []
        all_head_idx = []
        all_seq_idx = []

        for b in range(bsz):
            for h in range(num_kv_heads):
                k_h = budgets[b, h].item()
                n_pruned_h = seq_len - k_h
                if n_pruned_h > 0:
                    # Prune the lowest-scoring positions for this head
                    pruned_indices = sorted_indices[b, h, k_h:]  # tokens to remove
                    all_batch_idx.append(torch.full((n_pruned_h,), b, dtype=torch.long))
                    all_head_idx.append(torch.full((n_pruned_h,), h, dtype=torch.long))
                    all_seq_idx.append(pruned_indices.cpu())

        if all_batch_idx:
            batch_indices = torch.cat(all_batch_idx)
            head_indices = torch.cat(all_head_idx)
            seq_indices = torch.cat(all_seq_idx)
            module.masked_key_indices = (batch_indices, head_indices, seq_indices)

        return keys, values
