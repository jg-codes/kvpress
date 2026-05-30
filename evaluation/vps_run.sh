#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# MergingPress Cross-Scorer Evaluation — VPS Runner
# ═══════════════════════════════════════════════════════════════
#
# Runs bare vs merging comparisons for multiple scorers on CPU.
# Designed for wahlbot VPS (6 vCPU, 7.7GB RAM, no GPU).
#
# Prerequisites:
#   cd /root/kvpress-eval
#   git fetch origin pr/merging-press
#   git checkout pr/merging-press
#   pip install -e ".[dev]"
#
# Usage:
#   # Run in background (survives SSH disconnect):
#   nohup bash evaluation/vps_run.sh > vps_eval.log 2>&1 &
#
#   # Monitor progress:
#   tail -f vps_eval.log
#
#   # After completion, analyze:
#   cd evaluation && python cross_scorer_eval.py --analyze
# ═══════════════════════════════════════════════════════════════

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

MODEL="Qwen/Qwen2.5-1.5B-Instruct"
FRACTION=0.05
OUTPUT="./results_cross_scorer"
DATA_DIR=4096

echo "═══════════════════════════════════════════════════════════"
echo " MergingPress Cross-Scorer Evaluation"
echo " Model:    $MODEL"
echo " Fraction: $FRACTION ($(echo "$FRACTION * 1300" | bc | cut -d. -f1) samples approx)"
echo " Output:   $OUTPUT"
echo " Started:  $(date)"
echo "═══════════════════════════════════════════════════════════"

# Verify kvpress is importable and has MergingPress
python -c "from kvpress import MergingPress; print('MergingPress available')" || {
    echo "ERROR: MergingPress not found. Ensure pr/merging-press branch is checked out."
    echo "  git checkout pr/merging-press && pip install -e ."
    exit 1
}

# ─── Configuration ───
# Format: "press_name compression_ratio"
# evaluate.py skips configs where results already exist, so this is resumable.
CONFIGS=(
    "no_press 0.0"
    "knorm 0.25"
    "merging_knorm 0.25"
    "knorm 0.50"
    "merging_knorm 0.50"
    "snapkv 0.25"
    "merging_snapkv 0.25"
    "snapkv 0.50"
    "merging_snapkv 0.50"
)

TOTAL=${#CONFIGS[@]}
COUNT=0
START_TIME=$(date +%s)

for config in "${CONFIGS[@]}"; do
    read -r PRESS CR <<< "$config"
    COUNT=$((COUNT + 1))

    ELAPSED=$(( $(date +%s) - START_TIME ))
    if [ "$COUNT" -gt 1 ]; then
        AVG=$(( ELAPSED / (COUNT - 1) ))
        ETA=$(( AVG * (TOTAL - COUNT + 1) / 60 ))
    else
        ETA="?"
    fi

    echo ""
    echo "────────────────────────────────────────────────────────"
    echo "[$COUNT/$TOTAL] $PRESS @ CR=$CR  (elapsed: $((ELAPSED/60))m, ETA: ${ETA}m)"
    echo "────────────────────────────────────────────────────────"

    python evaluate.py \
        --model "$MODEL" \
        --dataset ruler \
        --data_dir "$DATA_DIR" \
        --press_name "$PRESS" \
        --compression_ratio "$CR" \
        --fraction "$FRACTION" \
        --output_dir "$OUTPUT" \
    || echo "WARNING: $PRESS @ CR=$CR failed (exit $?), continuing..."
done

TOTAL_TIME=$(( ($(date +%s) - START_TIME) / 60 ))
echo ""
echo "═══════════════════════════════════════════════════════════"
echo " All evaluations complete in ${TOTAL_TIME} minutes"
echo " Finished: $(date)"
echo ""
echo " Analyze results:"
echo "   python cross_scorer_eval.py --analyze --extra-dirs $OUTPUT"
echo "═══════════════════════════════════════════════════════════"
