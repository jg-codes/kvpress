# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from transformers import DynamicCache, QuantizedCache
from transformers.utils import is_optimum_quanto_available

from kvpress import KnormPress, SnapKVPress
from kvpress.presses.merging_press import MergingAdaKVPress, MergingPress
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

        assert cache_hard.get_seq_length() == cache_merge.get_seq_length() == 32

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

        def max_value_diff(cache_a, cache_b):
            return max(
                (cache_a.layers[i].values - cache_b.layers[i].values).abs().max().item()
                for i in range(len(cache_a.layers))
            )

        diff_lo = max_value_diff(cache_lo, cache_hard)
        diff_hi = max_value_diff(cache_hi, cache_hard)

        assert diff_hi <= diff_lo, f"High-threshold diff ({diff_hi}) > low-threshold diff ({diff_lo})"

    def test_keys_are_modified_by_merge(self, unit_test_model):  # noqa: F811
        """With merge_keys=True, keys should also be modified by the score-weighted blending."""
        torch.manual_seed(42)
        input_ids = torch.randint(0, 1024, (1, 64), device=unit_test_model.device)

        base = KnormPress(compression_ratio=0.5)
        with base(unit_test_model):
            cache_hard = DynamicCache()
            unit_test_model(input_ids.clone(), past_key_values=cache_hard)

        base2 = KnormPress(compression_ratio=0.5)
        wrapper = MergingPress(press=base2, similarity_threshold=0.0, merge_keys=True)
        with wrapper(unit_test_model):
            cache_merge = DynamicCache()
            unit_test_model(input_ids.clone(), past_key_values=cache_merge)

        any_different = False
        for i in range(len(cache_hard.layers)):
            if not torch.equal(cache_hard.layers[i].keys, cache_merge.layers[i].keys):
                any_different = True
                break
        assert any_different, "Merging did not modify keys"

    def test_default_preserves_keys(self, unit_test_model):  # noqa: F811
        """Default merge_keys=False should not modify keys (preserves RoPE)."""
        torch.manual_seed(42)
        input_ids = torch.randint(0, 1024, (1, 64), device=unit_test_model.device)

        base = KnormPress(compression_ratio=0.5)
        with base(unit_test_model):
            cache_hard = DynamicCache()
            unit_test_model(input_ids.clone(), past_key_values=cache_hard)

        base2 = KnormPress(compression_ratio=0.5)
        wrapper = MergingPress(press=base2)  # defaults: merge_keys=False, value_norm_weighting=True
        with wrapper(unit_test_model):
            cache_merge = DynamicCache()
            unit_test_model(input_ids.clone(), past_key_values=cache_merge)

        for i in range(len(cache_hard.layers)):
            assert torch.equal(cache_hard.layers[i].keys, cache_merge.layers[i].keys), (
                f"Layer {i}: default merge_keys=False should not modify keys"
            )

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
        model.float()

    def test_repeated_compression_stable(self, unit_test_model):  # noqa: F811
        """MergingPress can be applied multiple times (simulating streaming/multi-turn)."""
        torch.manual_seed(42)
        input_ids = torch.randint(0, 1024, (1, 128), device=unit_test_model.device)

        base = KnormPress(compression_ratio=0.4)
        wrapper = MergingPress(press=base, similarity_threshold=0.0)

        with wrapper(unit_test_model):
            cache = DynamicCache()
            unit_test_model(input_ids, past_key_values=cache)

        seq_lengths = [cache.get_seq_length()]
        for _ in range(3):
            with wrapper(unit_test_model):
                new_tokens = torch.randint(0, 1024, (1, 16), device=unit_test_model.device)
                unit_test_model(new_tokens, past_key_values=cache)
            seq_lengths.append(cache.get_seq_length())

        for layer in cache.layers:
            assert torch.isfinite(layer.keys).all(), "Non-finite keys after repeated compression"
            assert torch.isfinite(layer.values).all(), "Non-finite values after repeated compression"

        assert seq_lengths[-1] < 128 + 3 * 16, (
            f"Cache grew to {seq_lengths[-1]} — compression not applied during recompression rounds"
        )

    def test_merge_keys_false_preserves_keys(self, unit_test_model):  # noqa: F811
        """With merge_keys=False, keys should be identical to hard eviction."""
        torch.manual_seed(42)
        input_ids = torch.randint(0, 1024, (1, 64), device=unit_test_model.device)

        base = KnormPress(compression_ratio=0.5)
        with base(unit_test_model):
            cache_hard = DynamicCache()
            unit_test_model(input_ids.clone(), past_key_values=cache_hard)

        base2 = KnormPress(compression_ratio=0.5)
        wrapper = MergingPress(press=base2, similarity_threshold=0.0, merge_keys=False)
        with wrapper(unit_test_model):
            cache_merge = DynamicCache()
            unit_test_model(input_ids.clone(), past_key_values=cache_merge)

        for i in range(len(cache_hard.layers)):
            assert torch.equal(cache_hard.layers[i].keys, cache_merge.layers[i].keys), (
                f"Layer {i}: keys differ when merge_keys=False"
            )

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

        base = KnormPress(compression_ratio=0.5)
        wrapper = MergingPress(press=base, similarity_threshold=0.0, value_norm_weighting=False)
        with wrapper(unit_test_model):
            cache_plain = DynamicCache()
            unit_test_model(input_ids.clone(), past_key_values=cache_plain)

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
        """Merge-on-evict should stay closer to the uncompressed cache than hard eviction."""
        torch.manual_seed(42)
        input_ids = torch.randint(0, 1024, (1, 64), device=unit_test_model.device)

        cache_ref = DynamicCache()
        unit_test_model(input_ids.clone(), past_key_values=cache_ref)
        ref_values = [layer.values.float() for layer in cache_ref.layers]

        base_hard = KnormPress(compression_ratio=0.7)
        with base_hard(unit_test_model):
            cache_hard = DynamicCache()
            unit_test_model(input_ids.clone(), past_key_values=cache_hard)

        base_merge = KnormPress(compression_ratio=0.7)
        wrapper = MergingPress(press=base_merge, similarity_threshold=0.0)
        with wrapper(unit_test_model):
            cache_merge = DynamicCache()
            unit_test_model(input_ids.clone(), past_key_values=cache_merge)

        def value_reconstruction_error(compressed_cache):
            total = 0.0
            for i, layer in enumerate(compressed_cache.layers):
                n = layer.values.shape[2]
                total += (layer.values.float() - ref_values[i][:, :, :n]).norm().item()
            return total

        err_hard = value_reconstruction_error(cache_hard)
        err_merge = value_reconstruction_error(cache_merge)

        assert err_merge <= err_hard + 1e-6, (
            f"Merge error ({err_merge:.4f}) > hard eviction error ({err_hard:.4f})"
        )

    def test_batch_size_greater_than_one(self, unit_test_model):  # noqa: F811
        """The nonzero().reshape() partition must work correctly for batch_size > 1."""
        torch.manual_seed(42)
        input_ids = torch.randint(0, 1024, (2, 64), device=unit_test_model.device)

        base = KnormPress(compression_ratio=0.5)
        wrapper = MergingPress(press=base, similarity_threshold=0.0)
        with wrapper(unit_test_model):
            cache = DynamicCache()
            unit_test_model(input_ids, past_key_values=cache)

        assert cache.get_seq_length() == 32
        for layer in cache.layers:
            assert layer.keys.shape[0] == 2, "Batch dimension lost"
            assert torch.isfinite(layer.keys).all()
            assert torch.isfinite(layer.values).all()

    def test_best_config_merge_keys_false_vnorm_true(self, unit_test_model):  # noqa: F811
        """Combined merge_keys=False + value_norm_weighting=True should produce
        different values than either flag alone."""
        torch.manual_seed(42)
        input_ids = torch.randint(0, 1024, (1, 64), device=unit_test_model.device)

        def run_with(mk, vnw):
            base = KnormPress(compression_ratio=0.5)
            wrapper = MergingPress(press=base, similarity_threshold=0.0, merge_keys=mk, value_norm_weighting=vnw)
            with wrapper(unit_test_model):
                cache = DynamicCache()
                unit_test_model(input_ids.clone(), past_key_values=cache)
            return cache

        cache_both = run_with(mk=False, vnw=True)
        cache_vnorm_only = run_with(mk=True, vnw=True)
        cache_nokeys_only = run_with(mk=False, vnw=False)

        keys_differ = any(
            not torch.equal(cache_both.layers[i].keys, cache_vnorm_only.layers[i].keys)
            for i in range(len(cache_both.layers))
        )
        assert keys_differ, "merge_keys=False should produce different keys from merge_keys=True"

        vals_differ = any(
            not torch.equal(cache_both.layers[i].values, cache_nokeys_only.layers[i].values)
            for i in range(len(cache_both.layers))
        )
        assert vals_differ, "value_norm_weighting should change values"

    def test_max_merge_per_token_validation(self):
        with pytest.raises(AssertionError, match="non-negative"):
            MergingPress(press=KnormPress(compression_ratio=0.5), max_merge_per_token=-1)

    def test_max_merge_per_token_changes_output(self, unit_test_model):  # noqa: F811
        """Capping merges per survivor should produce different values than uncapped."""
        torch.manual_seed(42)
        input_ids = torch.randint(0, 1024, (1, 64), device=unit_test_model.device)

        base1 = KnormPress(compression_ratio=0.5)
        wrap_uncapped = MergingPress(press=base1, similarity_threshold=0.0, max_merge_per_token=0)
        with wrap_uncapped(unit_test_model):
            cache_uncapped = DynamicCache()
            unit_test_model(input_ids.clone(), past_key_values=cache_uncapped)

        base2 = KnormPress(compression_ratio=0.5)
        wrap_capped = MergingPress(press=base2, similarity_threshold=0.0, max_merge_per_token=1)
        with wrap_capped(unit_test_model):
            cache_capped = DynamicCache()
            unit_test_model(input_ids.clone(), past_key_values=cache_capped)

        any_different = False
        for i in range(len(cache_uncapped.layers)):
            if not torch.equal(cache_uncapped.layers[i].values, cache_capped.layers[i].values):
                any_different = True
                break
        assert any_different, "max_merge_per_token=1 should differ from uncapped"

    def test_max_merge_cap_reduces_dilution(self, unit_test_model):  # noqa: F811
        """With a strict cap, merged values should stay closer to the original survivors."""
        torch.manual_seed(42)
        input_ids = torch.randint(0, 1024, (1, 64), device=unit_test_model.device)

        # Reference: hard eviction (no merge at all)
        base_hard = KnormPress(compression_ratio=0.7)
        with base_hard(unit_test_model):
            cache_hard = DynamicCache()
            unit_test_model(input_ids.clone(), past_key_values=cache_hard)

        # Uncapped merging
        base1 = KnormPress(compression_ratio=0.7)
        wrap1 = MergingPress(press=base1, similarity_threshold=0.0, max_merge_per_token=0)
        with wrap1(unit_test_model):
            cache_uncapped = DynamicCache()
            unit_test_model(input_ids.clone(), past_key_values=cache_uncapped)

        # Capped merging — values should be closer to hard eviction
        base2 = KnormPress(compression_ratio=0.7)
        wrap2 = MergingPress(press=base2, similarity_threshold=0.0, max_merge_per_token=1)
        with wrap2(unit_test_model):
            cache_capped = DynamicCache()
            unit_test_model(input_ids.clone(), past_key_values=cache_capped)

        def max_value_diff(cache_a, cache_b):
            return max(
                (cache_a.layers[i].values - cache_b.layers[i].values).abs().max().item()
                for i in range(len(cache_a.layers))
            )

        diff_uncapped = max_value_diff(cache_uncapped, cache_hard)
        diff_capped = max_value_diff(cache_capped, cache_hard)

        assert diff_capped <= diff_uncapped + 1e-6, (
            f"Capped diff ({diff_capped}) > uncapped diff ({diff_uncapped})"
        )

    def test_high_compression_short_sequence(self, unit_test_model):  # noqa: F811
        """Very high compression on a short sequence must not crash."""
        torch.manual_seed(42)
        input_ids = torch.randint(0, 1024, (1, 8), device=unit_test_model.device)
        base = KnormPress(compression_ratio=0.9)
        wrapper = MergingPress(press=base, similarity_threshold=0.0)
        with wrapper(unit_test_model):
            cache = DynamicCache()
            unit_test_model(input_ids, past_key_values=cache)

        input_ids2 = torch.randint(0, 1024, (1, 16), device=unit_test_model.device)
        base2 = KnormPress(compression_ratio=0.9)
        wrapper2 = MergingPress(press=base2, similarity_threshold=0.0)
        with wrapper2(unit_test_model):
            cache2 = DynamicCache()
            unit_test_model(input_ids2, past_key_values=cache2)
        seq_len = cache2.get_seq_length()
        assert seq_len >= 1, f"Cache is empty after high compression on 16 tokens: {seq_len}"
        for layer in cache2.layers:
            assert torch.isfinite(layer.keys).all()
            assert torch.isfinite(layer.values).all()

    @pytest.mark.skipif(not is_optimum_quanto_available(), reason="Optimum Quanto is not available")
    def test_quantized_cache_compatibility(self, unit_test_model):  # noqa: F811
        """MergingPress should work with QuantizedCache (dequant → merge → requant)."""
        torch.manual_seed(42)
        input_ids = torch.randint(0, 1024, (1, 64), device=unit_test_model.device)

        base = KnormPress(compression_ratio=0.5)
        wrapper = MergingPress(press=base, similarity_threshold=0.0)
        cache = QuantizedCache(backend="quanto", config=unit_test_model.config, nbits=4)
        with wrapper(unit_test_model):
            unit_test_model(input_ids, past_key_values=cache)

        assert cache.get_seq_length() == 32
        for layer in cache.layers:
            assert torch.isfinite(layer.keys).all(), "Non-finite keys with QuantizedCache"
            assert torch.isfinite(layer.values).all(), "Non-finite values with QuantizedCache"

    def test_score_weighting_changes_output(self, unit_test_model):  # noqa: F811
        """score_weighting=True should produce different merge results."""
        torch.manual_seed(42)
        input_ids = torch.randint(0, 1024, (1, 64), device=unit_test_model.device)

        base1 = KnormPress(compression_ratio=0.5)
        wrap_plain = MergingPress(press=base1, score_weighting=False)
        with wrap_plain(unit_test_model):
            cache_plain = DynamicCache()
            unit_test_model(input_ids.clone(), past_key_values=cache_plain)

        base2 = KnormPress(compression_ratio=0.5)
        wrap_score = MergingPress(press=base2, score_weighting=True)
        with wrap_score(unit_test_model):
            cache_score = DynamicCache()
            unit_test_model(input_ids.clone(), past_key_values=cache_score)

        any_different = any(
            not torch.equal(cache_plain.layers[i].values, cache_score.layers[i].values)
            for i in range(len(cache_plain.layers))
        )
        assert any_different, "score_weighting did not change merge results"

    def test_adaptive_threshold_changes_output(self, unit_test_model):  # noqa: F811
        """adaptive_threshold=True should produce different merge results from fixed threshold."""
        torch.manual_seed(42)
        input_ids = torch.randint(0, 1024, (1, 64), device=unit_test_model.device)

        base1 = KnormPress(compression_ratio=0.5)
        wrap_fixed = MergingPress(press=base1, adaptive_threshold=False)
        with wrap_fixed(unit_test_model):
            cache_fixed = DynamicCache()
            unit_test_model(input_ids.clone(), past_key_values=cache_fixed)

        base2 = KnormPress(compression_ratio=0.5)
        wrap_adaptive = MergingPress(press=base2, adaptive_threshold=True)
        with wrap_adaptive(unit_test_model):
            cache_adaptive = DynamicCache()
            unit_test_model(input_ids.clone(), past_key_values=cache_adaptive)

        any_different = any(
            not torch.equal(cache_fixed.layers[i].values, cache_adaptive.layers[i].values)
            for i in range(len(cache_fixed.layers))
        )
        assert any_different, "adaptive_threshold did not change merge results"

    def test_adaptive_threshold_no_nan(self, unit_test_model):  # noqa: F811
        """adaptive_threshold should produce finite values."""
        torch.manual_seed(42)
        input_ids = torch.randint(0, 1024, (1, 64), device=unit_test_model.device)

        base = KnormPress(compression_ratio=0.7)
        wrapper = MergingPress(press=base, adaptive_threshold=True)
        with wrapper(unit_test_model):
            cache = DynamicCache()
            unit_test_model(input_ids, past_key_values=cache)

        for layer in cache.layers:
            assert torch.isfinite(layer.keys).all(), "Non-finite keys with adaptive_threshold"
            assert torch.isfinite(layer.values).all(), "Non-finite values with adaptive_threshold"


class TestMergingAdaKVPress:
    def test_requires_scorer_press(self):
        with pytest.raises(AssertionError, match="requires a ScorerPress"):
            MergingAdaKVPress(press="not_a_press")

    def test_compression_ratio_delegation(self):
        base = KnormPress(compression_ratio=0.3)
        wrapper = MergingAdaKVPress(press=base)
        assert wrapper.compression_ratio == 0.3
        wrapper.compression_ratio = 0.6
        assert base.compression_ratio == 0.6

    def test_zero_compression_is_identity(self, unit_test_model):  # noqa: F811
        base = KnormPress(compression_ratio=0.0)
        wrapper = MergingAdaKVPress(press=base)
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
        wrapper = MergingAdaKVPress(press=base)
        with wrapper(unit_test_model):
            input_ids = torch.randint(0, 1024, (1, 64), device=unit_test_model.device)
            cache = DynamicCache()
            unit_test_model(input_ids, past_key_values=cache)
            # Per-head budgets vary, so seq_length is max_budget (could be >= n_kept_uniform)
            assert cache.get_seq_length() > 0
            for layer in cache.layers:
                assert torch.isfinite(layer.keys).all()
                assert torch.isfinite(layer.values).all()

    def test_differs_from_uniform_merging(self, unit_test_model):  # noqa: F811
        """MergingAdaKV should differ from uniform MergingPress due to per-head budgets."""
        torch.manual_seed(42)
        input_ids = torch.randint(0, 1024, (1, 64), device=unit_test_model.device)

        base1 = KnormPress(compression_ratio=0.5)
        wrap_uniform = MergingPress(press=base1)
        with wrap_uniform(unit_test_model):
            cache_uniform = DynamicCache()
            unit_test_model(input_ids.clone(), past_key_values=cache_uniform)

        base2 = KnormPress(compression_ratio=0.5)
        wrap_adakv = MergingAdaKVPress(press=base2)
        with wrap_adakv(unit_test_model):
            cache_adakv = DynamicCache()
            unit_test_model(input_ids.clone(), past_key_values=cache_adakv)

        # At least one difference should exist (head budgets differ from uniform)
        any_different = any(
            not torch.equal(cache_uniform.layers[i].values, cache_adakv.layers[i].values)
            for i in range(len(cache_uniform.layers))
        )
        assert any_different, "MergingAdaKV produced identical values to uniform MergingPress"

    def test_half_precision_no_nan(self, unit_test_model):  # noqa: F811
        """MergingAdaKV should produce finite values in bfloat16."""
        model = unit_test_model.to(torch.bfloat16)
        torch.manual_seed(42)
        input_ids = torch.randint(0, 1024, (1, 64), device=model.device)

        base = KnormPress(compression_ratio=0.5)
        wrapper = MergingAdaKVPress(press=base)
        with wrapper(model):
            cache = DynamicCache()
            model(input_ids, past_key_values=cache)

        for layer in cache.layers:
            assert torch.isfinite(layer.keys).all(), "Non-finite keys in MergingAdaKV bfloat16"
            assert torch.isfinite(layer.values).all(), "Non-finite values in MergingAdaKV bfloat16"
        model.float()

    def test_alpha_safeguard_bounds(self):
        with pytest.raises(AssertionError):
            MergingAdaKVPress(press=KnormPress(compression_ratio=0.5), alpha_safeguard=-0.1)
        with pytest.raises(AssertionError):
            MergingAdaKVPress(press=KnormPress(compression_ratio=0.5), alpha_safeguard=1.1)

    def test_diagnostics(self, unit_test_model):  # noqa: F811
        """collect_diagnostics=True should record per-head diagnostics."""
        torch.manual_seed(42)
        input_ids = torch.randint(0, 1024, (1, 64), device=unit_test_model.device)

        base = KnormPress(compression_ratio=0.5)
        wrapper = MergingAdaKVPress(press=base, collect_diagnostics=True)
        with wrapper(unit_test_model):
            cache = DynamicCache()
            unit_test_model(input_ids, past_key_values=cache)

        assert len(wrapper.diagnostics) > 0, "No diagnostics collected"
