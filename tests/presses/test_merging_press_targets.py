# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Target exclusion, merge counts and the log(1+count) logit bias of MergingPress (C82)."""

import math
from types import SimpleNamespace

import pytest
import torch

from kvpress.attention_patch import apply_merge_logit_bias
from kvpress.presses.merging_press import _merge_on_evict_adaptive


def _make(bsz=1, heads=2, length=16, dim=8, n_evict=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    keys = torch.randn(bsz, heads, length, dim, generator=g)
    values = torch.randn(bsz, heads, length, dim, generator=g)
    evict = torch.zeros(bsz, heads, length, dtype=torch.bool)
    for h in range(heads):
        idx = torch.randperm(length - 6, generator=g)[:n_evict] + 3  # never evict sinks 0..2 or the last 3 slots
        evict[:, h, idx] = True
    return keys, values, evict


def test_target_mask_excludes_positions():
    keys, values, evict = _make()
    target = torch.ones_like(evict)
    target[:, :, :3] = False  # sinks
    target[:, :, -3:] = False  # restore-like slots
    _, new_values, counts = _merge_on_evict_adaptive(
        keys, values, evict, 0.0, False, True, 0, 1.0, 0.0, target_mask=target, return_counts=True
    )
    changed = (new_values != values).any(-1)
    excluded = ~target & ~evict
    assert not changed[excluded].any(), "excluded survivors must not be rewritten"
    assert counts[excluded].sum() == 0
    assert counts[evict].sum() == 0
    assert changed[~evict & target].any(), "some allowed survivor must have received a merge"


def test_counts_sum_to_merged_evictions_and_survive_the_cap():
    keys, values, evict = _make()
    _, _, c0 = _merge_on_evict_adaptive(keys, values, evict, 0.0, False, True, 0, 1.0, 0.0, return_counts=True)
    _, _, c3 = _merge_on_evict_adaptive(keys, values, evict, 0.0, False, True, 3, 1.0, 0.0, return_counts=True)
    # threshold 0.0 merges every evicted token whose best cosine similarity is >= 0
    assert c0.sum() <= evict.sum()
    assert torch.equal(c0, c3), "the cap rescales merge weights; it does not change which tokens are merged"


def test_default_return_shape_unchanged():
    keys, values, evict = _make()
    out = _merge_on_evict_adaptive(keys, values, evict, 0.0, False, True, 0, 1.0, 0.0)
    assert len(out) == 2 and out[0].shape == keys.shape


@pytest.mark.parametrize("q_len", [1, 5])
def test_logit_bias_multiplies_softmax_odds(q_len):
    torch.manual_seed(0)
    bsz, n_q, n_kv, k_cached, d = 1, 4, 2, 12, 8
    k_len = k_cached + q_len
    module = SimpleNamespace()
    query = torch.randn(bsz, n_q, q_len, d)
    key = torch.randn(bsz, n_kv, k_len, d)
    M, pos, h_kv = 3, 7, 1
    bias = torch.zeros(bsz, n_kv, k_cached)
    bias[0, h_kv, pos] = math.log1p(M)
    mask = apply_merge_logit_bias(module, query, key, None, bias)
    assert mask.shape == (bsz, n_q, q_len, k_len)
    assert module.merge_bias_calls == 1
    kq = key.repeat_interleave(n_q // n_kv, dim=1)
    logits = query @ kq.transpose(-1, -2) / math.sqrt(d)
    # causal structure is preserved for q_len > 1
    if q_len > 1:
        causal = torch.tril(torch.ones(q_len, k_len, dtype=torch.bool), diagonal=k_len - q_len)
        assert torch.isfinite(mask[0, 0][causal]).all() and (mask[0, 0][~causal] < -1e30).all()
    a0 = torch.softmax(logits.masked_fill(mask < -1e30, float("-inf")), -1)
    a1 = torch.softmax(logits + mask, -1)
    for h in range(n_q):
        expect = 1.0 + M if h // (n_q // n_kv) == h_kv else 1.0
        odds0 = a0[0, h, :, pos] / (1 - a0[0, h, :, pos])
        odds1 = a1[0, h, :, pos] / (1 - a1[0, h, :, pos])
        assert torch.allclose(odds1 / odds0, torch.full_like(odds0, expect), rtol=1e-5)


def test_zero_bias_returns_mask_unchanged():
    module = SimpleNamespace()
    query, key = torch.randn(1, 4, 1, 8), torch.randn(1, 2, 9, 8)
    assert apply_merge_logit_bias(module, query, key, None, torch.zeros(1, 2, 8)) is None
    assert not hasattr(module, "merge_bias_calls")
