# SPDX-FileCopyrightText: Copyright (c) 2025 Johannes Gast
# SPDX-License-Identifier: Apache-2.0

"""
VmaxCriticalAdaKVPress: CriticalAdaKV with V_max-weighted per-head safeguard.

Combines the two-stage scoring of CriticalKV (base scores × Wo@V L1 norm)
with adaptive head-wise pooling from AdaKV, but replaces the fixed safeguard
with a V_max-proportional safeguard from the Boltzmann Lens error bound.
"""

import logging
from dataclasses import dataclass

import torch
from transformers.models.llama.modeling_llama import repeat_kv

from kvpress.presses.base_press import BasePress
from kvpress.presses.expected_attention_press import ExpectedAttentionPress
from kvpress.presses.scorer_press import ScorerPress

logger = logging.getLogger(__name__)


@dataclass
class VmaxCriticalAdaKVPress(BasePress):
    """
    CriticalAdaKV with V_max-weighted per-head safeguard.

    Parameters
    ----------
    press : ScorerPress
        The underlying scoring method.
    alpha_safeguard : float, default=0.20
        Average fraction of kept tokens protected per head.
    epsilon : float, default=1e-4
        Numerical stability for score rescaling.
    first_stage_ratio : float, default=0.5
        Fraction of budget allocated to first stage.
    """

    press: ScorerPress = None
    alpha_safeguard: float = 0.20
    epsilon: float = 1e-4
    first_stage_ratio: float = 0.5

    def __post_init__(self):
        assert 0 <= self.alpha_safeguard <= 1, "alpha_safeguard should be in [0, 1]"
        assert isinstance(self.press, ScorerPress), "VmaxCriticalAdaKVPress requires a ScorerPress"
        if isinstance(self.press, ExpectedAttentionPress) and self.press.use_vnorm:
            logger.warning("use_vnorm should be disabled for VmaxCriticalAdaKVPress")

    def post_init_from_model(self, model):
        self.press.post_init_from_model(model)

    @staticmethod
    def vwl1norm(values, module):
        """Compute L1 norm of Wo @ V per head, robust to missing config.head_dim."""
        bsz, num_key_value_heads, k_len, _ = values.shape
        num_key_value_groups = module.config.num_attention_heads // num_key_value_heads
        head_dim = getattr(module.config, "head_dim", None) or (module.config.hidden_size // module.config.num_attention_heads)
        Wo = module.o_proj.weight.transpose(0, 1)
        Wo = Wo.view(module.config.num_attention_heads, head_dim, module.config.hidden_size)
        V = repeat_kv(values, num_key_value_groups)

        head_WoV_norm_list = []
        for head in range(V.size(1)):
            head_WoV = V[:, head, :, ...].matmul(Wo[head, ...].unsqueeze(0))
            head_WoV_norm = torch.norm(head_WoV, p=1, dim=-1)
            head_WoV_norm_list.append(head_WoV_norm)

        WoV_norm = torch.stack(head_WoV_norm_list, dim=1)
        WoV_norm = WoV_norm.view(bsz, num_key_value_heads, num_key_value_groups, k_len).mean(dim=2)
        return WoV_norm

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

        scores = self.press.score(module, hidden_states, keys, values, attentions, kwargs)
        bsz, num_key_value_heads, k_len = scores.shape

        n_kept = int(k_len * (1 - self.compression_ratio))

        # --- V_max-weighted per-head safeguard ---
        v_max = values.float().norm(dim=-1).max(dim=-1).values  # (bsz, num_kv_heads)
        v_mean = v_max.mean(dim=-1, keepdim=True)
        alpha_per_head = self.alpha_safeguard * (v_max / (v_mean + 1e-8))
        alpha_per_head = alpha_per_head.clamp(0, 1)

        n_safe_list = []
        n_safe_max = 0
        for h in range(num_key_value_heads):
            n_safe_h = int(n_kept * alpha_per_head[0, h].item())
            n_safe_h = min(n_safe_h, k_len)
            n_safe_list.append(n_safe_h)
            n_safe_max = max(n_safe_max, n_safe_h)

        if n_safe_max > 0:
            top_indices_safe = torch.topk(scores, min(n_safe_max, k_len), dim=-1).indices
            for h in range(num_key_value_heads):
                n_safe_h = n_safe_list[h]
                if n_safe_h > 0:
                    safe_idx = top_indices_safe[:, h, :n_safe_h]
                    scores[:, h, :].scatter_(-1, safe_idx, torch.finfo(scores.dtype).max)

        ############################
        # Start of CriticalKV code #
        ############################

        # Budget allocation via pooled pruning (use safeguard-boosted scores)
        budget_scores = scores.reshape(bsz, -1)
        top_indices = torch.topk(budget_scores, n_kept * num_key_value_heads, dim=-1).indices
        top_indices_head_idx = top_indices // k_len
        head_budgets = torch.zeros(num_key_value_heads, device=keys.device, dtype=torch.int64)
        head_budgets.scatter_add_(0, top_indices_head_idx.flatten(), torch.ones_like(top_indices_head_idx.flatten()))

        # Re-score from scratch for two-stage CriticalKV
        scores = self.press.score(module, hidden_states, keys, values, attentions, kwargs)

        # Apply safeguard again on fresh scores
        if n_safe_max > 0:
            top_indices_safe = torch.topk(scores, min(n_safe_max, k_len), dim=-1).indices
            for h in range(num_key_value_heads):
                n_safe_h = n_safe_list[h]
                if n_safe_h > 0:
                    safe_idx = top_indices_safe[:, h, :n_safe_h]
                    scores[:, h, :].scatter_(-1, safe_idx, torch.finfo(scores.dtype).max)

        # Stage 1: first_stage_ratio of each head's budget
        head_selection_budget_1st = (head_budgets * self.first_stage_ratio).to(torch.int64).tolist()
        top_k_index = torch.topk(scores, max(head_selection_budget_1st), sorted=True, dim=-1).indices
        for head_idx in range(num_key_value_heads):
            phase1_budget = head_selection_budget_1st[head_idx]
            scores[:, head_idx, :].scatter_(
                -1, top_k_index[:, head_idx, :phase1_budget], torch.finfo(scores.dtype).max
            )

        # Stage 2: rescale by Wo@V L1 norm
        projected_norm = self.vwl1norm(values, module)
        scores = (scores + self.epsilon) * projected_norm
        top_k_index = torch.topk(scores, max(head_budgets), sorted=True, dim=-1).indices
        for head_idx in range(num_key_value_heads):
            budget = head_budgets[head_idx]
            scores[:, head_idx, :].scatter_(
                -1, top_k_index[:, head_idx, :budget], torch.finfo(scores.dtype).max
            )

        ##########################
        # End of CriticalKV code #
        ##########################

        # Global pooled pruning
        n_pruned = num_key_value_heads * (k_len - n_kept)
        indices = torch.topk(-scores.reshape(bsz, -1), n_pruned, dim=1).indices.flatten()

        batch_indices = torch.arange(bsz).repeat_interleave(n_pruned)
        head_indices = indices // k_len
        seq_indices = indices % k_len
        module.masked_key_indices = (batch_indices, head_indices, seq_indices)
        return keys, values
