#!/usr/bin/env bash
# Quick comparison: MergingPress vs CAMPress vs baselines
# Runs fraction=0.03 (~195 samples, 15/task) on RULER-4096, Qwen3-8B
# Expected: ~3-5 min per variant, ~50 min total
set -euo pipefail

cd "$(dirname "$0")"

MODEL="Qwen/Qwen3-8B"
DATASET="ruler"
DATA_DIR="4096"
FRACTION="0.03"
SEED="42"
OUTPUT_DIR="./quick_comparison_results"
CR="0.75"

# Decoding press params (matching CaM's PR table format)
CI="8"
TS="2048"

mkdir -p "$OUTPUT_DIR"

log() { echo "$(date '+%H:%M:%S') | $*"; }

run_prefill() {
    local name="$1"
    log "START prefill: $name (cr=$CR)"
    python evaluate.py \
        --dataset "$DATASET" --data_dir "$DATA_DIR" \
        --model "$MODEL" --device cuda:0 \
        --press_name "$name" --compression_ratio "$CR" \
        --fraction "$FRACTION" --seed "$SEED" \
        --output_dir "$OUTPUT_DIR" 2>&1 | tail -3
    log "DONE  prefill: $name"
}

run_decoding() {
    local name="$1"
    log "START decoding: $name (ci=$CI, ts=$TS)"
    python evaluate.py \
        --dataset "$DATASET" --data_dir "$DATA_DIR" \
        --model "$MODEL" --device cuda:0 \
        --press_name "$name" \
        --compression_interval "$CI" --target_size "$TS" \
        --fraction "$FRACTION" --seed "$SEED" \
        --output_dir "$OUTPUT_DIR" 2>&1 | tail -3
    log "DONE  decoding: $name"
}

log "=== Quick Comparison: MergingPress vs CAMPress vs Baselines ==="
log "Model: $MODEL | Dataset: $DATASET/$DATA_DIR | Fraction: $FRACTION | Seed: $SEED"
log ""

# --- Baselines ---
log "--- Baselines ---"
run_prefill "no_press"
run_prefill "snapkv"
run_prefill "critical_snapkv"

# --- MergingPress prefill: threshold sweep ---
log "--- MergingPress prefill (threshold sweep) ---"
run_prefill "merging_vonorm_snapkv"       # threshold=0.0 (default)
run_prefill "merging_vonorm_snapkv_t03"   # threshold=0.3
run_prefill "merging_vonorm_snapkv_t05"   # threshold=0.5
run_prefill "merging_vonorm_snapkv_t07"   # threshold=0.7

# --- Decoding comparison ---
log "--- Decoding comparison (ci=$CI, ts=$TS) ---"
run_decoding "decoding_knorm"
run_decoding "cam_knorm"
run_decoding "merging_decoding_knorm"

log ""
log "=== All variants complete ==="
log "Results in: $OUTPUT_DIR"
log ""

# --- Collect and display results ---
log "=== Collecting metrics ==="
python3 << 'PYEOF'
import json, os, sys
from pathlib import Path

results_dir = Path("./quick_comparison_results")
all_metrics = {}

for d in sorted(results_dir.rglob("metrics.json")):
    with open(d) as f:
        data = json.load(f)
    # Extract press name from directory path
    parts = d.parent.name.split("__")
    press_name = parts[3] if len(parts) > 3 else d.parent.name
    cr_or_ts = parts[4] if len(parts) > 4 else ""
    label = f"{press_name}"
    if cr_or_ts:
        label += f" ({cr_or_ts})"
    all_metrics[label] = data

if not all_metrics:
    print("No metrics found!")
    sys.exit(1)

# Get all task names from first result
tasks = sorted(next(iter(all_metrics.values())).keys())

# Print header
header = f"{'Metric':<25}" + "".join(f"{k:<22}" for k in all_metrics)
print(header)
print("-" * len(header))

# Print per-task scores
averages = {}
for label in all_metrics:
    vals = []
    for t in tasks:
        v = all_metrics[label].get(t, {})
        if isinstance(v, dict):
            v = v.get("string_match", v.get("rouge1", 0))
        vals.append(v)
    averages[label] = sum(vals) / len(vals) if vals else 0

for t in tasks:
    row = f"{t:<25}"
    for label in all_metrics:
        v = all_metrics[label].get(t, {})
        if isinstance(v, dict):
            v = v.get("string_match", v.get("rouge1", 0))
        row += f"{v:<22.2f}"
    print(row)

print("-" * len(header))
row = f"{'Average':<25}"
for label in all_metrics:
    row += f"{averages[label]:<22.2f}"
print(row)

# Save combined results
combined = {"variants": all_metrics, "averages": averages}
with open(results_dir / "comparison_summary.json", "w") as f:
    json.dump(combined, f, indent=2)
print(f"\nSaved to {results_dir / 'comparison_summary.json'}")
PYEOF
