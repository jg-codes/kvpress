# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the C86 options of MergingPress (fold='mass', bias='score', score_map) and for the
port of the merge_fraction fix (NVIDIA/kvpress#287) to both kernels of this branch."""

import math

import pytest
import torch

from kvpress import ExpectedAttentionPress, KnormPress, KVzipPress, MergingPress
from kvpress.presses.merging_press import _map_scores, _merge_on_evict, _merge_on_evict_adaptive


def _synthetic(bsz=1, n_heads=2, L=64, D=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    keys = torch.randn(bsz, n_heads, L, D, generator=g)
    values = torch.randn(bsz, n_heads, L, D, generator=g)
    scores = torch.rand(bsz, n_heads, L, generator=g) + 1e-3  # positive, attention-type
    return keys, values, scores


# ----------------------------------------------------------------------------- construction rules
class TestConstruction:
    def test_defaults_are_off(self):
        p = MergingPress(KnormPress(compression_ratio=0.5))
        assert p.fold == "blend" and p.bias == "none" and p.score_map == "log"

    def test_score_bias_requires_mass_fold(self):
        with pytest.raises(AssertionError, match="fold='mass'"):
            MergingPress(KnormPress(compression_ratio=0.5), bias="score")
        with pytest.raises(AssertionError, match="fold='mass'"):
            MergingPress(KnormPress(compression_ratio=0.5), fold="blend", bias="score", value_norm_weighting=True)

    def test_score_bias_excludes_count_bias(self):
        with pytest.raises(AssertionError):
            MergingPress(
                KVzipPress(compression_ratio=0.5), fold="mass", bias="score", count_logit_bias=True
            )

    def test_mass_fold_needs_a_score(self):
        class NoScore(MergingPress.__mro__[1]):  # BasePress subclass without score()
            compression_ratio: float = 0.5

            def compress(self, *a, **k):  # pragma: no cover
                raise NotImplementedError

        with pytest.raises(NotImplementedError, match="per-token score"):
            MergingPress(NoScore(), fold="mass")

    def test_mass_fold_accepts_scorer_and_kvzip(self):
        MergingPress(ExpectedAttentionPress(compression_ratio=0.5), fold="mass", bias="score")
        MergingPress(KVzipPress(compression_ratio=0.5), fold="mass", bias="score", exclude_sink_targets=True)

    def test_bad_enum_values(self):
        with pytest.raises(AssertionError):
            MergingPress(KnormPress(compression_ratio=0.5), fold="mean")
        with pytest.raises(AssertionError):
            MergingPress(KnormPress(compression_ratio=0.5), score_map="zl")


# ----------------------------------------------------------------------------- score map
class TestScoreMap:
    def test_log_rejects_negative_scores(self):
        with pytest.raises(ValueError, match="score_map='z'"):
            _map_scores(-torch.rand(1, 2, 8), "log")

    def test_z_is_row_standardised(self):
        s = torch.rand(1, 3, 32) * 5
        z = _map_scores(s, "z")
        assert torch.allclose(z.mean(-1), torch.zeros(1, 3), atol=1e-5)
        assert torch.allclose(z.std(-1), torch.ones(1, 3), atol=1e-4)


# ----------------------------------------------------------------------------- kernel: fold + bias
class TestMassFoldKernel:
    def test_bias_and_fold_match_the_closed_form(self):
        keys, values, scores = _synthetic()
        n_kept = 16
        k, v, counts, bias = _merge_on_evict(
            keys, values, scores, n_kept, 0.0, False, True, fold="mass", score_map="log", return_stats=True
        )
        # Independent re-computation from the routing (nearest cosine survivor)
        keep_idx = scores.topk(n_kept, dim=-1).indices
        s = torch.log(scores)
        for h in range(keys.shape[1]):
            kept = keep_idx[0, h]
            evicted = torch.tensor(sorted(set(range(keys.shape[2])) - set(kept.tolist())))
            kk = torch.nn.functional.normalize(keys[0, h, kept], dim=-1)
            ek = torch.nn.functional.normalize(keys[0, h, evicted], dim=-1)
            tgt = (ek @ kk.T).argmax(-1)
            for i in range(n_kept):
                fold = evicted[tgt == i]
                w = torch.exp(s[0, h, fold] - s[0, h, kept[i]])
                b_ref = math.log1p(float(w.sum()))
                assert abs(float(bias[0, h, i]) - b_ref) < 1e-5, (h, i)
                assert int(counts[0, h, i]) == len(fold)
                v_ref = (values[0, h, kept[i]] + (w[:, None] * values[0, h, fold]).sum(0)) / (1 + w.sum())
                assert torch.allclose(v[0, h, i], v_ref, atol=1e-5), (h, i)
        assert torch.equal(k, keys.gather(2, keep_idx.unsqueeze(-1).expand(-1, -1, -1, keys.shape[-1])))

    def test_identity_for_one_query(self):
        """With s = the true logits of a single query, fold='mass' + bias reproduces that query's head output."""
        torch.manual_seed(1)
        D, L, n_kept = 16, 64, 8
        keys = torch.randn(1, 1, L, D).abs()  # non-negative keys: every cosine >= 0, so threshold 0 folds every token
        values = torch.randn(1, 1, L, D)
        q = torch.randn(D)
        logits = (keys[0, 0] @ q) / math.sqrt(D)
        scores = torch.exp(logits)[None, None]  # attention-type score whose log is the logit
        k, v, counts, bias = _merge_on_evict(
            keys, values, scores, n_kept, 0.0, False, True, fold="mass", score_map="log", return_stats=True
        )
        assert int(counts.sum()) == L - n_kept, "identity needs every evicted token folded"
        ref = torch.softmax(logits, -1) @ values[0, 0]
        comp_logits = (k[0, 0] @ q) / math.sqrt(D) + bias[0, 0]
        out = torch.softmax(comp_logits, -1) @ v[0, 0]
        assert torch.allclose(out, ref, atol=1e-4, rtol=1e-4), float((out - ref).norm() / ref.norm())

    def test_empty_fold_is_identity_with_zero_bias(self):
        keys, values, scores = _synthetic()
        n_kept = 16
        # similarity_threshold=1.0: no evicted key reaches cosine 1.0 with a survivor, so nothing is folded
        k, v, counts, bias = _merge_on_evict(
            keys, values, scores, n_kept, 1.0, False, True, fold="mass", return_stats=True
        )
        keep_idx = scores.topk(n_kept, dim=-1).indices.unsqueeze(-1).expand(-1, -1, -1, keys.shape[-1])
        assert torch.equal(v, values.gather(2, keep_idx))
        assert int(counts.sum()) == 0 and float(bias.abs().max()) == 0.0

    def test_target_mask_excludes_positions(self):
        keys, values, scores = _synthetic()
        n_kept = 16
        scores[:, :, :4] = scores.max() + 1  # sinks survive
        tm = torch.ones_like(scores, dtype=torch.bool)
        tm[:, :, :4] = False
        k, v, counts, bias = _merge_on_evict(
            keys, values, scores, n_kept, 0.0, False, True, fold="mass", target_mask=tm, return_stats=True
        )
        keep_idx = scores.topk(n_kept, dim=-1).indices
        sink_slots = (keep_idx < 4)
        assert int(counts[sink_slots].sum()) == 0 and float(bias[sink_slots].abs().max()) == 0.0
        kept_vals = values.gather(2, keep_idx.unsqueeze(-1).expand(-1, -1, -1, keys.shape[-1]))
        assert torch.equal(v[sink_slots], kept_vals[sink_slots])
        assert int(counts.sum()) == keys.shape[1] * (keys.shape[2] - n_kept)  # everything else merged

    def test_adaptive_kernel_matches_uniform_kernel(self):
        keys, values, scores = _synthetic()
        n_kept = 16
        k1, v1, c1, b1 = _merge_on_evict(
            keys, values, scores, n_kept, 0.0, False, True, fold="mass", return_stats=True
        )
        keep_idx = scores.topk(n_kept, dim=-1).indices
        evict_mask = torch.ones_like(scores, dtype=torch.bool).scatter(2, keep_idx, False)
        k2, v2, c2, b2 = _merge_on_evict_adaptive(
            keys, values, evict_mask, 0.0, False, True, return_counts=True,
            scores=_map_scores(scores, "log"), fold="mass", return_bias=True,
        )
        idx = keep_idx.unsqueeze(-1).expand(-1, -1, -1, keys.shape[-1])
        assert torch.allclose(v2.gather(2, idx), v1, atol=1e-6)
        assert torch.allclose(b2.gather(2, keep_idx), b1, atol=1e-6)
        assert torch.equal(c2.gather(2, keep_idx), c1)


# ----------------------------------------------------------------------------- #287 port
def _merged_share(kernel, merge_fraction, rejection_per_head, n=64):
    """Share of threshold-eligible evicted tokens merged, per head (construction of test 58bc611)."""
    num_heads = len(rejection_per_head)
    head_dim = 2 * n
    keys = torch.zeros(1, num_heads, 2 * n, head_dim)
    values = torch.zeros(1, num_heads, 2 * n, head_dim)
    n_eligible = []
    for h, rejection in enumerate(rejection_per_head):
        n_reject = round(rejection * n)
        a = torch.cat([torch.linspace(0.05, 0.45, n_reject), torch.linspace(0.5, 0.99, n - n_reject)])
        idx = torch.arange(n)
        keys[0, h, idx, idx] = 1.0
        keys[0, h, n + idx, idx] = a
        keys[0, h, n + idx, n + idx] = (1 - a**2).sqrt()
        values[0, h, n:] = 1.0
        n_eligible.append(n - n_reject)
    scores = torch.cat([torch.full((n,), 2.0), torch.full((n,), 1.0)]).expand(1, num_heads, 2 * n).clone()
    if kernel == "uniform":
        _, new_values = _merge_on_evict(keys, values, scores, n, 0.5, False, True, merge_fraction=merge_fraction)
        merged = (new_values[0].abs().sum(-1) > 0).sum(-1)
    else:
        evict_mask = torch.zeros(1, num_heads, 2 * n, dtype=torch.bool)
        evict_mask[:, :, n:] = True
        _, new_values = _merge_on_evict_adaptive(keys, values, evict_mask, 0.5, False, True, merge_fraction=merge_fraction)
        merged = (new_values[0, :, :n].abs().sum(-1) > 0).sum(-1)
    return [m / e for m, e in zip(merged.tolist(), n_eligible)]


@pytest.mark.parametrize("kernel", ["uniform", "adaptive"])
@pytest.mark.parametrize("rejection", [0.0, 0.25, 0.5, 0.75])
def test_merge_fraction_is_a_share_of_eligible_tokens(kernel, rejection):
    (share,) = _merged_share(kernel, 0.75, [rejection])
    assert abs(share - 0.75) < 0.02, f"{kernel} rejection={rejection}: merged share {share:.3f} != 0.75"


@pytest.mark.parametrize("kernel", ["uniform", "adaptive"])
def test_merge_fraction_per_row_with_heterogeneous_rejection(kernel):
    shares = _merged_share(kernel, 0.75, [0.0, 0.25, 0.5, 0.75])
    assert all(abs(s - 0.75) < 0.02 for s in shares), shares


@pytest.mark.parametrize("kernel", ["uniform", "adaptive"])
def test_merge_fraction_one_is_unchanged_by_the_gate(kernel):
    assert _merged_share(kernel, 1.0, [0.0, 0.5]) == [1.0, 1.0]
