from dataclasses import dataclass

import pytest
import torch
from transformers import DynamicCache

import kvpress.presses.merging_press as merging_module
from kvpress import CriticalKVPress, MergingPress, SnapKVPress
from kvpress.presses.scorer_press import ScorerPress
from tests.fixtures import unit_test_model  # noqa: F401


@dataclass
class FixedScorePress(ScorerPress):
    compression_ratio: float = 0.5
    fixed_scores: torch.Tensor | None = None

    def score(self, module, hidden_states, keys, values, attentions, kwargs):
        return self.fixed_scores.to(keys.device)


def test_merging_press_requires_scorer_press():
    with pytest.raises(AssertionError, match="ScorerPress"):
        MergingPress(press=None)


def test_merging_press_delegates_compression_ratio():
    inner = FixedScorePress(compression_ratio=0.2, fixed_scores=torch.ones(1, 1, 4))
    press = MergingPress(press=inner)
    assert press.compression_ratio == 0.2
    press.compression_ratio = 0.5
    assert inner.compression_ratio == 0.5


def test_merging_press_threshold_blocks_dissimilar_tokens():
    keys = torch.eye(4).unsqueeze(0).unsqueeze(0)
    values = torch.ones(1, 1, 4, 4)
    scores = torch.tensor([[[10.0, 9.0, 2.0, 1.0]]])
    press = MergingPress(
        press=FixedScorePress(compression_ratio=0.5, fixed_scores=scores),
        similarity_threshold=0.8,
        value_merge_mode="sum",
    )

    kept_values = values[:, :, :2].clone()
    _, merged_values = press.compress(module=None, hidden_states=None, keys=keys, values=values, attentions=None, kwargs={})

    assert press._n_merged == 0
    assert press._n_dropped == 2
    assert torch.allclose(merged_values, kept_values)


def test_merging_press_chunked_cosine_matches_full():
    keys = torch.randn(1, 2, 32, 8)
    values = torch.randn(1, 2, 32, 8)
    scores = torch.arange(32, dtype=torch.float).unsqueeze(0).unsqueeze(0).expand(1, 2, -1)

    full_press = MergingPress(press=FixedScorePress(compression_ratio=0.5, fixed_scores=scores))
    chunked_press = MergingPress(press=FixedScorePress(compression_ratio=0.5, fixed_scores=scores.clone()))

    original_limit = merging_module._MAX_SIM_PAIRS
    try:
        merging_module._MAX_SIM_PAIRS = 1 << 30
        full_keys, full_values = full_press.compress(None, None, keys, values, None, {})

        merging_module._MAX_SIM_PAIRS = 4
        chunked_keys, chunked_values = chunked_press.compress(None, None, keys, values, None, {})
    finally:
        merging_module._MAX_SIM_PAIRS = original_limit

    assert torch.allclose(full_keys, chunked_keys, atol=1e-5)
    assert torch.allclose(full_values, chunked_values, atol=1e-5)


def test_merging_press_runs_on_unit_model(unit_test_model):  # noqa: F811
    press = MergingPress(press=SnapKVPress(compression_ratio=0.5), similarity_threshold=0.8)
    with press(unit_test_model):
        input_ids = torch.randint(0, 1024, (1, 128), device=unit_test_model.device)
        unit_test_model(input_ids, past_key_values=DynamicCache()).past_key_values


def test_merging_press_runs_with_criticalkv_scoring(unit_test_model):  # noqa: F811
    press = MergingPress(
        press=CriticalKVPress(press=SnapKVPress(compression_ratio=0.5)),
        similarity_threshold=0.8,
    )
    with press(unit_test_model):
        input_ids = torch.randint(0, 1024, (1, 128), device=unit_test_model.device)
        unit_test_model(input_ids, past_key_values=DynamicCache()).past_key_values