#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Leaderboard evaluation for MergingPress variants
# Matches upstream leaderboard.sh: RULER 4096, Qwen/Qwen3-8B
# Run on a single GPU (sequential) or set NGPUS=4 for parallel

set -euo pipefail

dataset="ruler"
data_dir="4096"
model="Qwen/Qwen3-8B"
output_dir="./results_lb"
NGPUS="${NGPUS:-1}"

# ── Helper: run across GPUs (round-robin) or sequential ──────────────
run_press() {
    local press="$1"
    shift
    local extra_args=("$@")

    if (( NGPUS >= 4 )); then
        python evaluate.py --dataset $dataset --data_dir $data_dir --model $model \
            --press_name "$press" --compression_ratio 0.25  --output_dir $output_dir --device "cuda:0" "${extra_args[@]}" &
        python evaluate.py --dataset $dataset --data_dir $data_dir --model $model \
            --press_name "$press" --compression_ratio 0.50  --output_dir $output_dir --device "cuda:1" "${extra_args[@]}" &
        python evaluate.py --dataset $dataset --data_dir $data_dir --model $model \
            --press_name "$press" --compression_ratio 0.75  --output_dir $output_dir --device "cuda:2" "${extra_args[@]}" &
        python evaluate.py --dataset $dataset --data_dir $data_dir --model $model \
            --press_name "$press" --compression_ratio 0.875 --output_dir $output_dir --device "cuda:3" "${extra_args[@]}" &
        wait
    else
        for cr in 0.25 0.50 0.75 0.875; do
            echo ">>> $press  CR=$cr"
            python evaluate.py --dataset $dataset --data_dir $data_dir --model $model \
                --press_name "$press" --compression_ratio "$cr" --output_dir $output_dir --device "cuda:0" "${extra_args[@]}"
        done
    fi
}

# ── Baselines (for head-to-head comparison) ──────────────────────────
echo "=== Baselines ==="
# no_press (skip if already exists)
if [ ! -d "$output_dir/ruler/4096/${model}/no_press" ]; then
    python evaluate.py --dataset $dataset --data_dir $data_dir --model $model \
        --press_name no_press --compression_ratio 0.00 --output_dir $output_dir --device "cuda:0"
fi

# Raw scorers that we wrap (for orthogonality comparison)
for press in knorm snapkv expected_attention tova observed_attention compactor; do
    echo "--- baseline: $press ---"
    run_press "$press"
done

# AdaKV-wrapped baselines
for press in adakv_snapkv adakv_compactor; do
    echo "--- baseline: $press ---"
    run_press "$press"
done

# Query-aware baselines
for press in snapkv adakv_snapkv; do
    echo "--- baseline (query-aware): $press ---"
    run_press "$press" --query_aware
done

# ── MergingPress variants (our contribution) ─────────────────────────
echo ""
echo "=== MergingPress variants ==="

# Core MergingPress wrapping different scorers
for press in \
    merging_vonorm_knorm \
    merging_vonorm_snapkv \
    merging_expected_attention \
    merging_tova \
    merging_observed_attention \
    merging_compactor \
; do
    echo "--- $press ---"
    run_press "$press"
done

# MergingPress wrapping KVzap (the big test!)
for press in merging_kvzap_mlp merging_kvzap_linear; do
    echo "--- $press ---"
    run_press "$press"
done

# MergingAdaKVPress variants (adaptive head allocation + merge)
for press in \
    merging_adakv_knorm \
    merging_adakv_snapkv \
    merging_adakv_expected_attention \
    merging_adakv_kvzap_mlp \
; do
    echo "--- $press ---"
    run_press "$press"
done

# Best config variants (score weighting, adaptive threshold, critical)
for press in \
    merging_score_snapkv \
    merging_adaptive_snapkv \
    merging_vonorm_critical_snapkv \
    merging_adakv_critical_snapkv \
; do
    echo "--- $press ---"
    run_press "$press"
done

# ── Query-aware MergingPress ─────────────────────────────────────────
echo ""
echo "=== Query-aware MergingPress ==="
for press in merging_vonorm_snapkv merging_adakv_snapkv; do
    echo "--- $press (query-aware) ---"
    run_press "$press" --query_aware
done

echo ""
echo "=== Done! Results in $output_dir ==="
echo "Total press variants evaluated: ~22 + baselines"
echo "Submit: fork https://huggingface.co/spaces/nvidia/kvpress-leaderboard"
echo "        copy $output_dir/* into benchmark/ and PR"
