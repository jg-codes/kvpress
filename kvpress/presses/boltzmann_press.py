# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from kvpress.presses.scorer_press import ScorerPress
from kvpress.presses.snapkv_press import SnapKVPress


@dataclass
class BoltzmannPress(ScorerPress):
    """
    Boltzmann (log-space) attention scorer for KV cache eviction.

    Scores tokens by their mean log-attention across a recent window of queries,
    then retains the top-k per KV head. Log-space aggregation is motivated by the
    submodularity of the log-partition f(S) = log Σ_{j∈S} exp(x_j) over attention
    logits, for which greedy marginal-gain selection carries a (1 − 1/e) approximation
    bound. The top-k-by-log-attention heuristic used here is a tractable proxy for
    that greedy procedure (exact submodular greedy is O(n²k), impractical at KV scale);
    the empirical comparison against SnapKV/Knorm decides whether the proxy holds.

    Contrast with SnapKV (mean attention): by Jensen's inequality, mean(log p) ≤
    log(mean p), so log-space is strictly more conservative — it penalizes tokens
    attended inconsistently across queries/heads. Whether that conservatism helps
    or hurts is task-dependent.
    """

    compression_ratio: float = 0.0
    window_size: int = 64
    kernel_size: int = 5

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:

        bsz, num_key_value_heads, k_len, _ = keys.shape
        num_key_value_groups = module.config.num_attention_heads // num_key_value_heads

        assert hidden_states.shape[1] > self.window_size, (
            f"Query length {hidden_states.shape[1]} should be greater than window size {self.window_size}"
        )

        if attentions is not None:
            attn_weights = attentions[..., -self.window_size :, : -self.window_size]
        else:
            attn_weights = SnapKVPress.compute_window_attention(
                module, hidden_states, keys, self.window_size, kwargs["position_embeddings"]
            )

        # Log-space scoring: mean log-attention across window queries.
        # Clamp for numerical stability (exp(-30) ≈ 1e-13, below fp16 precision anyway).
        log_attn = torch.log(attn_weights.clamp_min(1e-30))
        scores = log_attn.mean(dim=-2)

        # SnapKV-style kernel smoothing — soften sharp spikes so nearby tokens co-survive.
        scores = F.avg_pool1d(scores, kernel_size=self.kernel_size, padding=self.kernel_size // 2, stride=1)

        # Aggregate across GQA groups: mean log-attention per KV head.
        scores = scores.view(bsz, num_key_value_heads, num_key_value_groups, k_len - self.window_size)
        scores = scores.mean(2)

        # Protect window tokens — they must be retained for continuity.
        scores = F.pad(scores, (0, self.window_size), value=scores.max().item())

        return scores
