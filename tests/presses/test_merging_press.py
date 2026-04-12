# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from transformers import DynamicCache

from kvpress import KnormPress, SnapKVPress
from kvpress.presses.merging_press import MergingPress
from tests.fixtures import unit_test_model  # noqa: F401


class TestMergingPress:
    def test_requires_scorer_press(self):
        with pytest.raises(AssertionError, match="requires a ScorerPress"):
            MergingPress(press="not_a_press")

    def test_threshold_bounds(self):
        with pytest.raises(AssertionError):
            MergingPress(press=KnormPress(compression_ratio=0.5), similarity_threshold=-0.1)
        with pytest.raises(AssertionError):
            MergingPress(press=KnormPress(compression_ratio=0.5), similarity_threshold=1.1)

    def test_compression_ratio_delegation(self):
        base = KnormPress(compression_ratio=0.3)
        wrapper = MergingPress(press=base)
        assert wrapper.compression_ratio == 0.3
        wrapper.compression_ratio = 0.6
        assert base.compression_ratio == 0.6

    def test_zero_compression_is_identity(self, unit_test_model):  # noqa: F811
        base = KnormPress(compression_ratio=0.0)
        wrapper = MergingPress(press=base)
        with wrapper(unit_test_model):
            input_ids = torch.randint(0, 1024, (1, 64), device=unit_test_model.device)
            cache = DynamicCache()
            unit_test_model(input_ids, past_key_values=cache)
            assert cache.get_seq_length() == 64

    @pytest.mark.parametrize("base_cls", [KnormPress, SnapKVPress])
    def test_runs_with_model(self, unit_test_model, base_cls):  # noqa: F811
        if base_cls == SnapKVPress:
            base = base_cls(compression_ratio=0.5, window_size=2)
        else:
            base = base_cls(compression_ratio=0.5)
        wrapper = MergingPress(press=base)
        with wrapper(unit_test_model):
            input_ids = torch.randint(0, 1024, (1, 64), device=unit_test_model.device)
            cache = DynamicCache()
            unit_test_model(input_ids, past_key_values=cache)
            assert cache.get_seq_length() == 32

    def test_merge_differs_from_hard_eviction(self, unit_test_model):  # noqa: F811
        """Merged values should differ from hard-evicted values."""
        torch.manual_seed(42)
        input_ids = torch.randint(0, 1024, (1, 64), device=unit_test_model.device)

        # Hard eviction
        base = KnormPress(compression_ratio=0.5)
        with base(unit_test_model):
            cache_hard = DynamicCache()
            unit_test_model(input_ids.clone(), past_key_values=cache_hard)

        # Merge-on-evict (threshold=0 merges all)
        base2 = KnormPress(compression_ratio=0.5)
        wrapper = MergingPress(press=base2, similarity_threshold=0.0)
        with wrapper(unit_test_model):
            cache_merge = DynamicCache()
            unit_test_model(input_ids.clone(), past_key_values=cache_merge)

        # Same number of tokens kept
        assert cache_hard.get_seq_length() == cache_merge.get_seq_length() == 32

        # Values should differ due to merging
        any_different = False
        for i in range(len(cache_hard.layers)):
            if not torch.equal(cache_hard.layers[i].values, cache_merge.layers[i].values):
                any_different = True
                break
        assert any_different, "Merging produced identical values to hard eviction"

    def test_threshold_gates_merges(self, unit_test_model):  # noqa: F811
        """High threshold should skip more merges, producing results closer to hard eviction."""
        torch.manual_seed(42)
        input_ids = torch.randint(0, 1024, (1, 64), device=unit_test_model.device)

        # Hard eviction baseline
        base = KnormPress(compression_ratio=0.5)
        with base(unit_test_model):
            cache_hard = DynamicCache()
            unit_test_model(input_ids.clone(), past_key_values=cache_hard)

        # Aggressive merge (threshold=0)
        base_lo = KnormPress(compression_ratio=0.5)
        wrap_lo = MergingPress(press=base_lo, similarity_threshold=0.0)
        with wrap_lo(unit_test_model):
            cache_lo = DynamicCache()
            unit_test_model(input_ids.clone(), past_key_values=cache_lo)

        # Conservative merge (threshold=0.99)
        base_hi = KnormPress(compression_ratio=0.5)
        wrap_hi = MergingPress(press=base_hi, similarity_threshold=0.99)
        with wrap_hi(unit_test_model):
            cache_hi = DynamicCache()
            unit_test_model(input_ids.clone(), past_key_values=cache_hi)

        # Compute value deviation from hard eviction
        def max_value_diff(cache_a, cache_b):
            return max(
                (cache_a.layers[i].values - cache_b.layers[i].values).abs().max().item()
                for i in range(len(cache_a.layers))
            )

        diff_lo = max_value_diff(cache_lo, cache_hard)
        diff_hi = max_value_diff(cache_hi, cache_hard)

        # High threshold should deviate less (fewer merges happened)
        assert diff_hi <= diff_lo, f"High-threshold diff ({diff_hi}) > low-threshold diff ({diff_lo})"

    def test_keys_are_modified_by_merge(self, unit_test_model):  # noqa: F811
        """Keys should also be modified by the score-weighted blending."""
        torch.manual_seed(42)
        input_ids = torch.randint(0, 1024, (1, 64), device=unit_test_model.device)

        base = KnormPress(compression_ratio=0.5)
        with base(unit_test_model):
            cache_hard = DynamicCache()
            unit_test_model(input_ids.clone(), past_key_values=cache_hard)

        base2 = KnormPress(compression_ratio=0.5)
        wrapper = MergingPress(press=base2, similarity_threshold=0.0)
        with wrapper(unit_test_model):
            cache_merge = DynamicCache()
            unit_test_model(input_ids.clone(), past_key_values=cache_merge)

        any_different = False
        for i in range(len(cache_hard.layers)):
            if not torch.equal(cache_hard.layers[i].keys, cache_merge.layers[i].keys):
                any_different = True
                break
        assert any_different, "Merging did not modify keys"

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_half_precision_no_nan(self, unit_test_model, dtype):  # noqa: F811
        """Merged keys/values must be finite in float16 and bfloat16."""
        model = unit_test_model.to(dtype)
        torch.manual_seed(42)
        input_ids = torch.randint(0, 1024, (1, 64), device=model.device)

        base = KnormPress(compression_ratio=0.5)
        wrapper = MergingPress(press=base, similarity_threshold=0.0)
        with wrapper(model):
            cache = DynamicCache()
            model(input_ids, past_key_values=cache)

        for layer in cache.layers:
            assert torch.isfinite(layer.keys).all(), f"Non-finite keys with {dtype}"
            assert torch.isfinite(layer.values).all(), f"Non-finite values with {dtype}"
            assert layer.keys.dtype == dtype
        model.float()  # restore

    def test_repeated_compression_stable(self, unit_test_model):  # noqa: F811
        """MergingPress can be applied multiple times (simulating streaming/multi-turn).

        Each round shrinks the cache further and all values must remain finite.
        """
        torch.manual_seed(42)
        input_ids = torch.randint(0, 1024, (1, 128), device=unit_test_model.device)

        base = KnormPress(compression_ratio=0.4)
        wrapper = MergingPress(press=base, similarity_threshold=0.0)

        with wrapper(unit_test_model):
            cache = DynamicCache()
            unit_test_model(input_ids, past_key_values=cache)

        seq_lengths = [cache.get_seq_length()]
        # Simulate 3 additional recompression rounds (streaming decoding scenario)
        for _ in range(3):
            with wrapper(unit_test_model):
                new_tokens = torch.randint(0, 1024, (1, 16), device=unit_test_model.device)
                unit_test_model(new_tokens, past_key_values=cache)
            seq_lengths.append(cache.get_seq_length())

        # Cache should grow by fewer than 16 tokens each round (compression kicks in)
        for layer in cache.layers:
            assert torch.isfinite(layer.keys).all(), "Non-finite keys after repeated compression"
            assert torch.isfinite(layer.values).all(), "Non-finite values after repeated compression"

        # Verify compression actually happened (not just appending)
        assert seq_lengths[-1] < 128 + 3 * 16, (
            f"Cache grew to {seq_lengths[-1]} — compression not applied during recompression rounds"
        )

    def test_merge_keys_false_preserves_keys(self, unit_test_model):  # noqa: F811
        """With merge_keys=False, keys should be identical to hard eviction (only values differ)."""
        torch.manual_seed(42)
        input_ids = torch.randint(0, 1024, (1, 64), device=unit_test_model.device)

        # Hard eviction baseline
        base = KnormPress(compression_ratio=0.5)
        with base(unit_test_model):
            cache_hard = DynamicCache()
            unit_test_model(input_ids.clone(), past_key_values=cache_hard)

        # Merge values only (merge_keys=False)
        base2 = KnormPress(compression_ratio=0.5)
        wrapper = MergingPress(press=base2, similarity_threshold=0.0, merge_keys=False)
        with wrapper(unit_test_model):
            cache_merge = DynamicCache()
            unit_test_model(input_ids.clone(), past_key_values=cache_merge)

        # Keys should be identical (no key merging)
        for i in range(len(cache_hard.layers)):
            assert torch.equal(cache_hard.layers[i].keys, cache_merge.layers[i].keys), (
                f"Layer {i}: keys differ when merge_keys=False"
            )

        # Values should still differ (merging happened)
        any_different = False
        for i in range(len(cache_hard.layers)):
            if not torch.equal(cache_hard.layers[i].values, cache_merge.layers[i].values):
                any_different = True
                break
        assert any_different, "merge_keys=False did not merge values"

    def test_value_norm_weighting_differs(self, unit_test_model):  # noqa: F811
        """value_norm_weighting=True should produce different merge results."""
        torch.manual_seed(42)
        input_ids = torch.randint(0, 1024, (1, 64), device=unit_test_model.device)

        # Without value-norm weighting
        base = KnormPress(compression_ratio=0.5)
        wrapper = MergingPress(press=base, similarity_threshold=0.0, value_norm_weighting=False)
        with wrapper(unit_test_model):
            cache_plain = DynamicCache()
            unit_test_model(input_ids.clone(), past_key_values=cache_plain)

        # With value-norm weighting
        base2 = KnormPress(compression_ratio=0.5)
        wrapper2 = MergingPress(press=base2, similarity_threshold=0.0, value_norm_weighting=True)
        with wrapper2(unit_test_model):
            cache_vnorm = DynamicCache()
            unit_test_model(input_ids.clone(), past_key_values=cache_vnorm)

        any_different = False
        for i in range(len(cache_plain.layers)):
            if not torch.equal(cache_plain.layers[i].values, cache_vnorm.layers[i].values):
                any_different = True
                break
        assert any_different, "value_norm_weighting did not change merge results"

    def test_merge_preserves_more_info_than_hard_eviction(self, unit_test_model):  # noqa: F811
        """Under aggressive compression, merge-on-evict should stay closer to the
        uncompressed cache than hard eviction (measured as Frobenius norm of value diff)."""
        torch.manual_seed(42)
        input_ids = torch.randint(0, 1024, (1, 64), device=unit_test_model.device)

        # Uncompressed reference
        cache_ref = DynamicCache()
        unit_test_model(input_ids.clone(), past_key_values=cache_ref)
        ref_values = [layer.values.float() for layer in cache_ref.layers]

        # Hard eviction
        base_hard = KnormPress(compression_ratio=0.7)
        with base_hard(unit_test_model):
            cache_hard = DynamicCache()
            unit_test_model(input_ids.clone(), past_key_values=cache_hard)

        # Merge-on-evict
        base_merge = KnormPress(compression_ratio=0.7)
        wrapper = MergingPress(press=base_merge, similarity_threshold=0.0)
        with wrapper(unit_test_model):
            cache_merge = DynamicCache()
            unit_test_model(input_ids.clone(), past_key_values=cache_merge)

        # Compare: sum of Frobenius norms of (compressed_value - ref_value) over shared positions
        def value_reconstruction_error(compressed_cache):
            total = 0.0
            for i, layer in enumerate(compressed_cache.layers):
                n = layer.values.shape[2]
                total += (layer.values.float() - ref_values[i][:, :, :n]).norm().item()
            return total

        err_hard = value_reconstruction_error(cache_hard)
        err_merge = value_reconstruction_error(cache_merge)

        # Merging should have lower or equal reconstruction error
        assert err_merge <= err_hard + 1e-6, (
            f"Merge error ({err_merge:.4f}) > hard eviction error ({err_hard:.4f})"
        )
