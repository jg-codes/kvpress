#!/usr/bin/env bash
# Multi-CR benchmark: proves MergingPress improves any ScorerPress it wraps.
#
# Scientific goal: "merge-on-evict is a general technique that improves any
# scorer it wraps, across compression ratios."
#
# Matrix: 7 prefill variants × 4 CRs = 28 runs + no_press baseline + 3 decoding
# Two-phase: smoke (fraction=0.001, ~7 samples) → full (fraction=0.10, ~650 samples)
# Bootstrap 95% CI computed at collect step.
#
# Usage:
#   ./multi_cr_benchmark.sh           # runs both phases
#   ./multi_cr_benchmark.sh smoke     # smoke phase only
#   ./multi_cr_benchmark.sh full      # full phase only (skip smoke)
#   ./multi_cr_benchmark.sh collect   # collect results only (no new runs)
set -euo pipefail

cd "$(dirname "$0")"

MODEL="Qwen/Qwen3-8B"
DATASET="ruler"
DATA_DIR="4096"
SEED="42"
OUTPUT_DIR="./multi_cr_results"

# Compression ratios for full multi-CR sweep
CRS=(0.25 0.50 0.75 0.875)

# Decoding press params
CI="8"
TS="2048"

# Phase fractions
SMOKE_FRACTION="0.001"
FULL_FRACTION="0.10"

# Prefill variants to test at all CRs
# Groups:   [baselines]            [merging wrapping each scorer]
PREFILL_VARIANTS=(
    "knorm"
    "snapkv"
    "critical_snapkv"
    "merging_vonorm_knorm"
    "merging_vonorm_snapkv"
    "merging_vonorm_critical_snapkv"
)

# Decoding variants (run at no specific CR — use compression_interval/target_size)
DECODING_VARIANTS=(
    "decoding_knorm"
    "cam_knorm"
    "merging_decoding_knorm"
)

PHASE="${1:-all}"  # all | smoke | full | collect

mkdir -p "$OUTPUT_DIR"

log() { echo "$(date '+%H:%M:%S') | $*"; }

run_prefill() {
    local name="$1" cr="$2" fraction="$3" outdir="$4"
    log "  START  $name  cr=$cr  frac=$fraction"
    python evaluate.py \
        --dataset "$DATASET" --data_dir "$DATA_DIR" \
        --model "$MODEL" --device cuda:0 \
        --press_name "$name" --compression_ratio "$cr" \
        --fraction "$fraction" --seed "$SEED" \
        --output_dir "$outdir" 2>&1 | tail -3
    log "  DONE   $name  cr=$cr"
}

run_decoding() {
    local name="$1" fraction="$2" outdir="$3"
    log "  START  $name  (decoding ci=$CI ts=$TS)  frac=$fraction"
    python evaluate.py \
        --dataset "$DATASET" --data_dir "$DATA_DIR" \
        --model "$MODEL" --device cuda:0 \
        --press_name "$name" \
        --compression_interval "$CI" --target_size "$TS" \
        --fraction "$fraction" --seed "$SEED" \
        --output_dir "$outdir" 2>&1 | tail -3
    log "  DONE   $name  (decoding)"
}

run_phase() {
    local phase_name="$1" fraction="$2"
    local outdir="$OUTPUT_DIR/${phase_name}"
    mkdir -p "$outdir"
    log "=== Phase: $phase_name (fraction=$fraction) ==="

    # no_press ceiling (only needs one CR — compression is ignored)
    log "--- no_press baseline ---"
    run_prefill "no_press" "0.75" "$fraction" "$outdir"

    # Prefill matrix: all variants × all CRs
    log "--- Prefill matrix: ${#PREFILL_VARIANTS[@]} variants × ${#CRS[@]} CRs ---"
    for variant in "${PREFILL_VARIANTS[@]}"; do
        for cr in "${CRS[@]}"; do
            run_prefill "$variant" "$cr" "$fraction" "$outdir"
        done
    done

    # Decoding variants (single run each, no CR loop)
    log "--- Decoding variants ---"
    for variant in "${DECODING_VARIANTS[@]}"; do
        run_decoding "$variant" "$fraction" "$outdir"
    done

    log "=== Phase $phase_name complete. Results in: $outdir ==="
}

# ── Phase dispatch ────────────────────────────────────────────────────────────
case "$PHASE" in
    smoke)
        run_phase "smoke" "$SMOKE_FRACTION"
        log "Smoke passed — review above, then run: $0 full"
        ;;
    full)
        run_phase "full" "$FULL_FRACTION"
        "$0" collect
        ;;
    all)
        run_phase "smoke" "$SMOKE_FRACTION"
        log "Smoke complete — starting full run"
        run_phase "full" "$FULL_FRACTION"
        "$0" collect
        ;;
    collect)
        log "=== Collecting results + bootstrap CIs ==="
        ;;
    *)
        echo "Usage: $0 [smoke|full|all|collect]" >&2; exit 1
        ;;
esac

[[ "$PHASE" == "smoke" ]] && exit 0

# ── Results collection with bootstrap 95% CI ─────────────────────────────────
python3 << 'PYEOF'
import json, sys
from pathlib import Path
import random

RESULTS_DIR = Path("./multi_cr_results/full")
BOOTSTRAP_N = 1000
SEED = 42
random.seed(SEED)

def load_metrics(results_dir: Path) -> dict:
    """Returns {variant_label: {task: score, ...}}"""
    all_metrics = {}
    for d in sorted(results_dir.rglob("metrics.json")):
        with open(d) as f:
            data = json.load(f)
        parts = d.parent.name.split("__")
        # directory structure: dataset__data_dir__model--name__press_name__cr[__extra]
        press_name = parts[3] if len(parts) > 3 else "unknown"
        cr_field = parts[4] if len(parts) > 4 else ""
        label = f"{press_name} ({cr_field})" if cr_field else press_name
        all_metrics[label] = data
    return all_metrics

def flatten_score(val) -> float:
    """Extract numeric score from dict or float."""
    if isinstance(val, dict):
        return val.get("string_match", val.get("rouge1", val.get("f1", 0.0)))
    return float(val) if val is not None else 0.0

def compute_mean_and_ci(scores: list[float], n: int = BOOTSTRAP_N) -> tuple[float, float, float]:
    """Returns (mean, ci_lo, ci_hi) via bootstrap resampling."""
    mean = sum(scores) / len(scores)
    boot_means = []
    for _ in range(n):
        sample = random.choices(scores, k=len(scores))
        boot_means.append(sum(sample) / len(sample))
    boot_means.sort()
    lo = boot_means[int(0.025 * n)]
    hi = boot_means[int(0.975 * n)]
    return mean, lo, hi

if not RESULTS_DIR.exists():
    print(f"Results dir not found: {RESULTS_DIR}")
    sys.exit(1)

all_metrics = load_metrics(RESULTS_DIR)
if not all_metrics:
    print("No metrics.json files found in results dir.")
    sys.exit(1)

# Build table: {press_name: {cr: {mean, ci_lo, ci_hi, per_task}}}
table = {}
for label, task_metrics in all_metrics.items():
    tasks = sorted(task_metrics.keys())
    scores = [flatten_score(task_metrics[t]) for t in tasks]
    mean, lo, hi = compute_mean_and_ci(scores)
    table[label] = {
        "mean": round(mean, 2),
        "ci_lo": round(lo, 2),
        "ci_hi": round(hi, 2),
        "n_tasks": len(tasks),
        "per_task": {t: round(flatten_score(task_metrics[t]), 2) for t in tasks},
    }

# Print summary table
print("\n=== Multi-CR Benchmark Results (RULER-4096, fraction=0.10) ===")
print(f"Bootstrap 95% CI over {BOOTSTRAP_N} resamples of task scores\n")
header = f"{'Variant':<42} {'Mean':>6}  {'95% CI':>15}  {'n_tasks':>7}"
print(header)
print("-" * len(header))
for label, r in sorted(table.items(), key=lambda x: -x[1]["mean"]):
    ci_str = f"[{r['ci_lo']:.2f}, {r['ci_hi']:.2f}]"
    print(f"{label:<42} {r['mean']:>6.2f}  {ci_str:>15}  {r['n_tasks']:>7}")

# Save full results
out_path = Path("./multi_cr_results/results_summary.json")
with open(out_path, "w") as f:
    json.dump(table, f, indent=2)
print(f"\nFull results saved to: {out_path}")
PYEOF
