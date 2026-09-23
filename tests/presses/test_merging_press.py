# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from transformers import DynamicCache

from kvpress import AdaKVPress, KnormPress, KVgradPress, KVzipPress
from kvpress.presses.merging_press import MergingPress
from tests.default_presses import TestRestoreKVPress
from tests.fixtures import kv_press_unit_test_pipeline, unit_test_model  # noqa: F401


def test_merge_differs_from_hard_eviction(unit_test_model):  # noqa: F811
    """Merged values should differ from hard-evicted values."""
    torch.manual_seed(42)
    input_ids = torch.randint(0, 1024, (1, 64), device=unit_test_model.device)

    base = KnormPress(compression_ratio=0.5)
    with base(unit_test_model):
        cache_hard = DynamicCache()
        unit_test_model(input_ids.clone(), past_key_values=cache_hard)

    wrapper = MergingPress(press=KnormPress(compression_ratio=0.5), similarity_threshold=0.0)
    with wrapper(unit_test_model):
        cache_merge = DynamicCache()
        unit_test_model(input_ids.clone(), past_key_values=cache_merge)

    assert cache_hard.get_seq_length() == cache_merge.get_seq_length() == 32
    any_diff = any(
        not torch.equal(cache_hard.layers[i].values, cache_merge.layers[i].values)
        for i in range(len(cache_hard.layers))
    )
    assert any_diff, "Merging produced identical values to hard eviction"


def test_keys_unchanged(unit_test_model):  # noqa: F811
    """Keys must not be modified (RoPE-safe by design)."""
    torch.manual_seed(42)
    input_ids = torch.randint(0, 1024, (1, 64), device=unit_test_model.device)

    base = KnormPress(compression_ratio=0.5)
    with base(unit_test_model):
        cache_hard = DynamicCache()
        unit_test_model(input_ids.clone(), past_key_values=cache_hard)

    wrapper = MergingPress(press=KnormPress(compression_ratio=0.5))
    with wrapper(unit_test_model):
        cache_merge = DynamicCache()
        unit_test_model(input_ids.clone(), past_key_values=cache_merge)

    for i in range(len(cache_hard.layers)):
        assert torch.equal(cache_hard.layers[i].keys, cache_merge.layers[i].keys), (
            f"Layer {i}: keys must not be modified"
        )


def test_merge_preserves_more_info(unit_test_model):  # noqa: F811
    """Merge-on-evict stays closer to uncompressed cache than hard eviction."""
    torch.manual_seed(42)
    input_ids = torch.randint(0, 1024, (1, 64), device=unit_test_model.device)

    cache_ref = DynamicCache()
    unit_test_model(input_ids.clone(), past_key_values=cache_ref)
    ref_values = [layer.values.float() for layer in cache_ref.layers]

    base = KnormPress(compression_ratio=0.7)
    with base(unit_test_model):
        cache_hard = DynamicCache()
        unit_test_model(input_ids.clone(), past_key_values=cache_hard)

    wrapper = MergingPress(press=KnormPress(compression_ratio=0.7), similarity_threshold=0.0)
    with wrapper(unit_test_model):
        cache_merge = DynamicCache()
        unit_test_model(input_ids.clone(), past_key_values=cache_merge)

    def recon_error(cache):
        return sum(
            (layer.values.float() - ref_values[i][:, :, : layer.values.shape[2]]).norm().item()
            for i, layer in enumerate(cache.layers)
        )

    assert recon_error(cache_merge) <= recon_error(cache_hard) + 1e-6


def test_half_precision_no_nan(unit_test_model):  # noqa: F811
    """Float32 accumulation must produce finite results in fp16."""
    model = unit_test_model.to(torch.float16)
    torch.manual_seed(42)
    input_ids = torch.randint(0, 1024, (1, 64), device=model.device)

    wrapper = MergingPress(press=KnormPress(compression_ratio=0.5))
    with wrapper(model):
        cache = DynamicCache()
        model(input_ids, past_key_values=cache)

    for layer in cache.layers:
        assert torch.isfinite(layer.keys).all()
        assert torch.isfinite(layer.values).all()
    model.float()


def test_batch_size_greater_than_one(unit_test_model):  # noqa: F811
    """Kernel must handle batch_size > 1 correctly."""
    torch.manual_seed(42)
    input_ids = torch.randint(0, 1024, (2, 64), device=unit_test_model.device)

    wrapper = MergingPress(press=KnormPress(compression_ratio=0.5))
    with wrapper(unit_test_model):
        cache = DynamicCache()
        unit_test_model(input_ids, past_key_values=cache)

    assert cache.get_seq_length() == 32
    for layer in cache.layers:
        assert layer.keys.shape[0] == 2


def test_merge_method_signature():
    """Lock in the public merge(keys, values, indices) surface from #219.

    `indices` are the kept positions (output of scores.topk). The method
    folds evicted information into the kept slots; other positions are
    unchanged (they get pruned by compress() after this returns).
    """
    torch.manual_seed(42)
    bsz, num_heads, seq_len, head_dim = 1, 2, 8, 4
    keys = torch.randn(bsz, num_heads, seq_len, head_dim)
    values = torch.randn(bsz, num_heads, seq_len, head_dim)
    # Keep first half, evict second half
    kept = torch.arange(seq_len // 2).expand(bsz, num_heads, seq_len // 2)

    press = MergingPress(press=KnormPress(compression_ratio=0.5))
    new_keys, new_values = press.merge(keys, values, kept)

    assert new_keys.shape == keys.shape
    assert new_values.shape == values.shape
    # Keys are returned unchanged (RoPE-safe by design)
    assert torch.equal(new_keys, keys)
    # Kept positions absorb evicted information: values must change there
    assert not torch.equal(new_values[:, :, : seq_len // 2], values[:, :, : seq_len // 2])


def _merged_share(merge_fraction, rejection_per_head, n=64, dtype=torch.float32):
    """Share of threshold-eligible evicted tokens that ``MergingPress.merge`` actually merges.

    Head h has n kept tokens with keys e_0..e_{n-1} and n evicted tokens whose only similar
    survivor is kept token i, with cosine similarity a_i. A share ``rejection_per_head[h]`` of
    the a_i lie below the similarity threshold 0.5, the rest at or above it, so the number of
    eligible tokens is known exactly. Kept values are zero and evicted values are one, so a
    kept value is non-zero after the merge iff its evicted partner was merged.
    """
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
    kept = torch.arange(n).expand(1, num_heads, n)
    press = MergingPress(
        press=KnormPress(compression_ratio=0.5), similarity_threshold=0.5, merge_fraction=merge_fraction
    )
    _, new_values = press.merge(keys.to(dtype), values.to(dtype), kept)
    merged = (new_values[0, :, :n].abs().sum(-1) > 0).sum(-1)
    return [m / e for m, e in zip(merged.tolist(), n_eligible)]


@pytest.mark.parametrize("rejection", [0.0, 0.25, 0.5, 0.75])
def test_merge_fraction_is_a_share_of_eligible_tokens(rejection):
    """merge_fraction=0.75 must merge 75% of the threshold-eligible tokens at any rejection rate."""
    (share,) = _merged_share(0.75, [rejection])
    assert abs(share - 0.75) < 0.02, f"rejection={rejection}: merged share {share:.3f} != 0.75"


def test_merge_fraction_per_row_with_heterogeneous_rejection():
    """The share holds per (batch, head) row when rows reject different fractions."""
    shares = _merged_share(0.75, [0.0, 0.25, 0.5, 0.75])
    assert all(abs(s - 0.75) < 0.02 for s in shares), shares


def test_merge_fraction_one_is_unchanged_by_the_gate():
    """merge_fraction=1.0 merges every eligible token."""
    shares = _merged_share(1.0, [0.0, 0.5])
    assert shares == [1.0, 1.0]


def _prefill(press, model, input_ids):
    cache = DynamicCache()
    with press(model):
        model(input_ids, past_key_values=cache)
    masks = [layer.self_attn.masked_key_indices for layer in model.model.layers]
    return cache, masks


def _evicted_counts(mask, shape):
    evicted = torch.zeros(shape, dtype=torch.bool)
    evicted[mask] = True
    return evicted.sum(-1)


KVZIP_FAMILY = [
    lambda: KVzipPress(compression_ratio=0.5),
    lambda: KVgradPress(compression_ratio=0.5, chunk_size=64),
    lambda: TestRestoreKVPress(compression_ratio=0.5),
]


@pytest.mark.parametrize("make_press", KVZIP_FAMILY, ids=["KVzipPress", "KVgradPress", "RestoreKVPress"])
def test_kvzip_family_inner(unit_test_model, make_press):  # noqa: F811
    """A KVzipPress inner decides the eviction; MergingPress only rewrites survivor values."""
    torch.manual_seed(0)
    input_ids = torch.randint(0, 1024, (1, 128), device=unit_test_model.device)
    cache_bare, masks_bare = _prefill(make_press(), unit_test_model, input_ids)
    cache_merge, masks_merge = _prefill(MergingPress(press=make_press()), unit_test_model, input_ids)

    assert cache_bare.get_seq_length() == cache_merge.get_seq_length()
    any_diff = False
    for layer_bare, layer_merge, mask_bare, mask_merge in zip(
        cache_bare.layers, cache_merge.layers, masks_bare, masks_merge
    ):
        shape = layer_bare.keys.shape[:3]
        # Same evicted pairs, hence the same kept count per (layer, kv-head)
        assert torch.equal(_evicted_counts(mask_bare, shape), _evicted_counts(mask_merge, shape))
        assert all(torch.equal(a, b) for a, b in zip(mask_bare, mask_merge))
        assert torch.equal(layer_bare.keys, layer_merge.keys)
        # Evicted pairs are masked at attention time and left unchanged
        assert torch.equal(layer_bare.values[mask_bare], layer_merge.values[mask_merge])
        any_diff |= not torch.equal(layer_bare.values, layer_merge.values)
    assert any_diff, "MergingPress did not change any survivor value"


def test_merge_mask_matches_merge():
    """With the same number of evicted tokens per head, merge_mask equals merge on the kept positions."""
    torch.manual_seed(0)
    keys, values = torch.randn(2, 3, 16, 8), torch.randn(2, 3, 16, 8)
    kept = torch.rand(2, 3, 16).topk(6, dim=-1).indices.sort(dim=-1).values
    evicted = torch.ones(2, 3, 16, dtype=torch.bool).scatter_(2, kept, False)

    press = MergingPress(press=KnormPress(compression_ratio=0.5))
    _, expected = press.merge(keys, values, kept)
    torch.testing.assert_close(press.merge_mask(keys, values, evicted), expected)


def test_restorekv_inner_pipeline_without_merges(kv_press_unit_test_pipeline, monkeypatch):  # noqa: F811
    """With no merge (similarity_threshold=1.0) the wrapper reproduces the inner press end to end.

    The question must start after the restore slots, as for the bare RestoreKVPress.
    """
    pipe = kv_press_unit_test_pipeline
    context, question = "This is a test article. It was written on 2022-01-01.", "When was the article written?"
    context_lengths, answers = [], []
    generate_answer = pipe.generate_answer

    def spy(question_ids, cache, context_length, max_new_tokens):
        context_lengths.append(context_length)
        return generate_answer(question_ids, cache, context_length, max_new_tokens)

    monkeypatch.setattr(pipe, "generate_answer", spy)
    for wrap in [False, True]:
        press = TestRestoreKVPress(compression_ratio=0.5)
        press = MergingPress(press=press, similarity_threshold=1.0) if wrap else press
        answers.append(pipe(context, question=question, press=press)["answer"])
    assert context_lengths[0] == context_lengths[1]
    assert answers[0] == answers[1]


@pytest.mark.parametrize(
    "inner",
    [
        lambda: MergingPress(press=KnormPress(compression_ratio=0.5)),
        lambda: MergingPress(press=KVzipPress(compression_ratio=0.5)),
        lambda: AdaKVPress(press=KnormPress(compression_ratio=0.5)),
    ],
    ids=["MergingPress(KnormPress)", "MergingPress(KVzipPress)", "AdaKVPress(KnormPress)"],
)
def test_unsupported_inner_raises(inner):
    with pytest.raises(AssertionError, match="requires a ScorerPress or a KVzipPress"):
        MergingPress(press=inner())
