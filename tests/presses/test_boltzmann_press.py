# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from transformers import DynamicCache

from kvpress import BoltzmannPress, KnormPress, MergingPress
from tests.fixtures import unit_test_model  # noqa: F401


def test_boltzmann_runs_and_compresses(unit_test_model):  # noqa: F811
    torch.manual_seed(0)
    input_ids = torch.randint(0, 1024, (1, 128), device=unit_test_model.device)

    press = BoltzmannPress(compression_ratio=0.5, window_size=8, kernel_size=3)
    with press(unit_test_model):
        cache = DynamicCache()
        unit_test_model(input_ids, past_key_values=cache)

    assert cache.get_seq_length() == 64


def test_boltzmann_differs_from_knorm(unit_test_model):  # noqa: F811
    torch.manual_seed(0)
    input_ids = torch.randint(0, 1024, (1, 128), device=unit_test_model.device)

    with KnormPress(compression_ratio=0.5)(unit_test_model):
        cache_knorm = DynamicCache()
        unit_test_model(input_ids.clone(), past_key_values=cache_knorm)

    with BoltzmannPress(compression_ratio=0.5, window_size=8, kernel_size=3)(unit_test_model):
        cache_boltz = DynamicCache()
        unit_test_model(input_ids.clone(), past_key_values=cache_boltz)

    assert cache_knorm.get_seq_length() == cache_boltz.get_seq_length() == 64
    any_diff = any(
        not torch.equal(cache_knorm.layers[i].keys, cache_boltz.layers[i].keys)
        for i in range(len(cache_knorm.layers))
    )
    assert any_diff, "BoltzmannPress selected identical tokens to KnormPress"


def test_merging_boltzmann_composition(unit_test_model):  # noqa: F811
    torch.manual_seed(0)
    input_ids = torch.randint(0, 1024, (1, 128), device=unit_test_model.device)

    wrapper = MergingPress(press=BoltzmannPress(compression_ratio=0.5, window_size=8, kernel_size=3))
    with wrapper(unit_test_model):
        cache = DynamicCache()
        unit_test_model(input_ids, past_key_values=cache)

    assert cache.get_seq_length() == 64
