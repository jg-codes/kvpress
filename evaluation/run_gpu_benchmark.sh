#!/usr/bin/env bash
# GPU benchmark for MergingPress paper results.
# Run on a single L4 GPU via GCloud Spot.
#
# evaluate.py has skip-if-exists logic: if predictions.csv + metrics.json
# already exist for a (press, cr, model, dataset) combo, the run is skipped.
# This makes the script safe to re-run after spot preemption.
#
# Usage:
#   cd evaluation
#   bash run_gpu_benchmark.sh [--model MODEL] [--dataset DATASET] [--data-dir DATADIR] [--fraction FRAC]
#
# Defaults to Qwen/Qwen3-8B on RULER-4096 (matches official leaderboard.sh).
# Results are saved to ./results/<model>/<dataset>/<press>/<cr>/

set -euo pipefail

# ── Defaults (override via CLI flags) ──
MODEL="${MODEL:-Qwen/Qwen3-8B}"
DATASET="${DATASET:-ruler}"
DATA_DIR="${DATA_DIR:-4096}"
FRACTION="${FRACTION:-1.0}"
DEVICE="${DEVICE:-cuda:0}"

# Parse optional CLI overrides
while [[ $# -gt 0 ]]; do
  case $1 in
    --model)     MODEL="$2";     shift 2 ;;
    --dataset)   DATASET="$2";   shift 2 ;;
    --data-dir)  DATA_DIR="$2";  shift 2 ;;
    --fraction)  FRACTION="$2";  shift 2 ;;
    --device)    DEVICE="$2";    shift 2 ;;
    *) echo "Unknown flag: $1"; exit 1 ;;
  esac
done

# ── Presses to evaluate ──
# Baselines
PRESSES=(
  "snapkv"
  "critical_snapkv"
  # Our MergingPress contribution (merge_keys=False, value_norm_weighting=True)
  "merging_vonorm_snapkv"
  "merging_vonorm_critical_snapkv"
)

COMPRESSION_RATIOS=(0.75)

echo "========================================="
echo "  GPU Benchmark: MergingPress evaluation"
echo "========================================="
echo "Model:    $MODEL"
echo "Dataset:  $DATASET (data_dir=$DATA_DIR)"
echo "Fraction: $FRACTION"
echo "Device:   $DEVICE"
echo "Presses:  ${PRESSES[*]}"
echo "CRs:      ${COMPRESSION_RATIOS[*]}"
echo "========================================="

TOTAL=0
DONE=0

# Count total runs
for press in "${PRESSES[@]}"; do
  if [[ "$press" == "no_press" ]]; then
    TOTAL=$((TOTAL + 1))
  else
    TOTAL=$((TOTAL + ${#COMPRESSION_RATIOS[@]}))
  fi
done

echo "Total runs: $TOTAL"
echo ""

# ── Run evaluations ──
for press in "${PRESSES[@]}"; do
  if [[ "$press" == "no_press" ]]; then
    DONE=$((DONE + 1))
    echo "[$DONE/$TOTAL] press=$press (baseline, cr=0.0)"
    python evaluate.py \
      --model "$MODEL" \
      --dataset "$DATASET" \
      --data_dir "$DATA_DIR" \
      --press_name "$press" \
      --compression_ratio 0.0 \
      --fraction "$FRACTION" \
      --device "$DEVICE" \
      --output_dir "./results"
  else
    for cr in "${COMPRESSION_RATIOS[@]}"; do
      DONE=$((DONE + 1))
      echo "[$DONE/$TOTAL] press=$press  cr=$cr"
      python evaluate.py \
        --model "$MODEL" \
        --dataset "$DATASET" \
        --data_dir "$DATA_DIR" \
        --press_name "$press" \
        --compression_ratio "$cr" \
        --fraction "$FRACTION" \
        --device "$DEVICE" \
        --output_dir "./results"
    done
  fi
done

echo ""
echo "========================================="
echo "  All $TOTAL runs completed."
echo "  Results in: ./results/"
echo "========================================="
