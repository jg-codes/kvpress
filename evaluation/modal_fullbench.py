# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Full RULER benchmark for MergingPress best variants on Modal.

Runs 4 variants in PARALLEL on separate L4 GPUs:
  1. snapkv                         — baseline
  2. merging_vonorm_snapkv           — our best (merge_keys=F, vnorm=T)
  3. critical_snapkv                 — CriticalKV scoring baseline (SOTA scorer)
  4. merging_vonorm_critical_snapkv  — our best wrapping SOTA scorer

Usage:
    modal run evaluation/modal_fullbench.py
"""

import json

import modal

# ---------------------------------------------------------------------------
# Modal image
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("packaging", "setuptools", "wheel")
    .pip_install(
        "torch>=2.3",
        "transformers>=4.48",
        "datasets",
        "pandas",
        "numpy",
        "fire",
        "pyyaml",
        "tqdm",
        "accelerate",
        gpu="L4",
    )
    .pip_install("kvpress @ git+https://github.com/jg-codes/kvpress.git@feature/merging-press")
    .pip_install("jieba", "bert_score", "fuzzywuzzy", "python-Levenshtein", "nltk", "rouge")
    .run_commands(
        "git clone --branch feature/merging-press --depth 1 https://github.com/jg-codes/kvpress.git /eval_repo",
    )
)

app = modal.App("kvpress-fullbench", image=image)

# ---------------------------------------------------------------------------
# Experiment config
# ---------------------------------------------------------------------------
MODEL = "Qwen/Qwen3-8B"
DATASET = "ruler"
DATA_DIR = "4096"
FRACTION = 1.0  # Full dataset: 13 tasks × 500 = 6500 samples
SEED = 42
CR = 0.75

VARIANTS = [
    ("snapkv", "SnapKV baseline"),
    ("merging_vonorm_snapkv", "MergingPress(SnapKV) merge_keys=F vnorm=T"),
    ("critical_snapkv", "CriticalKV(SnapKV) baseline"),
    ("merging_vonorm_critical_snapkv", "MergingPress(CriticalKV(SnapKV)) merge_keys=F vnorm=T"),
]


@app.function(
    gpu="L4",
    timeout=7200,  # 2h per variant (generous)
    memory=32768,
    scaledown_window=2,
)
def run_variant(press_name: str, label: str) -> dict:
    """Run a single evaluation variant and return metrics."""
    import glob
    import os
    import sys

    sys.path.insert(0, "/eval_repo/evaluation")
    os.chdir("/eval_repo/evaluation")

    from evaluate import EvaluationConfig, EvaluationRunner

    config = EvaluationConfig(
        dataset=DATASET,
        data_dir=DATA_DIR,
        model=MODEL,
        device="cuda:0",
        press_name=press_name,
        compression_ratio=CR,
        fraction=FRACTION,
        seed=SEED,
        output_dir=f"/results/{press_name}",
    )

    print(f"[{press_name}] Starting evaluation: {FRACTION*100:.0f}% of {DATASET}-{DATA_DIR}")
    runner = EvaluationRunner(config)
    runner.run_evaluation()
    print(f"[{press_name}] Evaluation complete")

    # Read metrics
    metrics_files = glob.glob(f"/results/{press_name}/**/metrics.json", recursive=True)
    if metrics_files:
        with open(metrics_files[0]) as f:
            metrics = json.load(f)
    else:
        metrics = {"error": "no metrics.json found"}

    return {"press_name": press_name, "label": label, "metrics": metrics}


@app.local_entrypoint()
def main():
    """Launch all variants in parallel, collect and display results."""
    print(f"\n{'='*80}")
    print(f"MergingPress FULL Benchmark on Modal")
    print(f"Model: {MODEL} | Dataset: {DATASET}-{DATA_DIR} | CR: {CR}")
    print(f"Fraction: {FRACTION} (full dataset) | Seed: {SEED}")
    print(f"Variants: {len(VARIANTS)}")
    print(f"{'='*80}\n")

    results = list(run_variant.starmap(VARIANTS, return_exceptions=True))

    # Display results table
    print(f"\n{'='*110}")
    print(f"{'Variant':<55} {'Avg':>6}  | Task scores")
    print(f"{'-'*110}")

    rows = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            print(f"{VARIANTS[i][1]:<55} ERROR: {r}")
            continue
        m = r["metrics"]
        if "error" in m:
            print(f"{r['label']:<55} ERROR: {m['error']}")
            continue
        task_scores = {k: v["string_match"] for k, v in m.items() if isinstance(v, dict) and "string_match" in v}
        avg = sum(task_scores.values()) / len(task_scores) if task_scores else 0
        rows.append((avg, r["label"], r["press_name"], task_scores))

    rows.sort(key=lambda x: -x[0])

    for avg, label, press_name, task_scores in rows:
        scores_str = " ".join(f"{k}={v:.1f}" for k, v in sorted(task_scores.items()))
        print(f"{label:<55} {avg:>6.2f}  | {scores_str}")

    print(f"{'='*110}")

    # Save full JSON results
    output = {
        "config": {
            "model": MODEL,
            "dataset": DATASET,
            "data_dir": DATA_DIR,
            "compression_ratio": CR,
            "fraction": FRACTION,
            "seed": SEED,
        },
        "results": [
            r if not isinstance(r, Exception) else {"error": str(r), "variant": VARIANTS[i][0]}
            for i, r in enumerate(results)
        ],
    }

    print("\n--- Raw JSON ---")
    print(json.dumps(output, indent=2))

    import pathlib

    out_path = pathlib.Path("evaluation/fullbench_results.json")
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {out_path}")
