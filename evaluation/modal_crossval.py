# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Fast cross-validation of MergingPress variants on Modal.

Runs all variants in PARALLEL on separate GPUs, same fraction+seed for fair comparison.
Results are collected and printed as a comparison table.

Usage:
    modal run evaluation/modal_crossval.py
"""

import json

import modal

# ---------------------------------------------------------------------------
# Modal setup
# ---------------------------------------------------------------------------
KVPRESS_COMMIT = "d558e7e"  # current feature/merging-press HEAD

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

app = modal.App("kvpress-crossval", image=image)

# ---------------------------------------------------------------------------
# Experiment matrix
# ---------------------------------------------------------------------------
MODEL = "Qwen/Qwen3-8B"
DATASET = "ruler"
DATA_DIR = "4096"
FRACTION = 0.03  # ~195 samples (15 per RULER task) — fast directional results
SEED = 42
CR = 0.75

# All variants to compare on the SAME subsample
VARIANTS = [
    # Baseline: plain scorer, no merging
    ("snapkv", "Plain SnapKV (no merge)"),
    # Original: merge keys + values (merge_keys=True, default)
    ("merging_snapkv", "MergingPress(SnapKV) merge_keys=T"),
    # New: values-only merge
    ("merging_vo_snapkv", "MergingPress(SnapKV) merge_keys=F"),
    # New: values-only + value-norm weighting
    ("merging_vonorm_snapkv", "MergingPress(SnapKV) merge_keys=F vnorm=T"),
    # Also test with ExpectedAttention as inner scorer
    ("expected_attention_plain", "Plain ExpectedAttention (no merge)"),
    ("merging_expected_attention", "MergingPress(EA) merge_keys=T"),
    ("merging_vo_expected_attention", "MergingPress(EA) merge_keys=F"),
    ("merging_vonorm_expected_attention", "MergingPress(EA) merge_keys=F vnorm=T"),
]


@app.function(
    gpu="L4",
    timeout=3600,
    memory=32768,
    scaledown_window=2,
)
def run_variant(press_name: str, label: str) -> dict:
    """Run a single evaluation variant and return metrics + label."""
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

    runner = EvaluationRunner(config)
    runner.run_evaluation()

    # Read metrics
    import glob

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
    print("MergingPress Cross-Validation on Modal")
    print(f"Model: {MODEL} | Dataset: {DATASET}-{DATA_DIR} | CR: {CR} | Fraction: {FRACTION} | Seed: {SEED}")
    print(f"Variants: {len(VARIANTS)}")
    print(f"{'='*80}\n")

    # Launch ALL variants in parallel via Modal .starmap()
    # return_exceptions=True: one OOM won't cancel the rest
    results = list(run_variant.starmap(VARIANTS, return_exceptions=True))

    # Compute averages and build table
    print(f"\n{'='*100}")
    print(f"{'Variant':<45} {'Avg':>6}  | Task scores")
    print(f"{'-'*100}")

    rows = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            print(f"{VARIANTS[i][1]:<45} ERROR: {r}")
            continue
        m = r["metrics"]
        if "error" in m:
            print(f"{r['label']:<45} ERROR: {m['error']}")
            continue
        task_scores = {k: v["string_match"] for k, v in m.items() if isinstance(v, dict) and "string_match" in v}
        avg = sum(task_scores.values()) / len(task_scores) if task_scores else 0
        rows.append((avg, r["label"], r["press_name"], task_scores))

    # Sort by average descending
    rows.sort(key=lambda x: -x[0])

    for avg, label, press_name, task_scores in rows:
        scores_str = " ".join(f"{k}={v:.0f}" for k, v in sorted(task_scores.items()))
        print(f"{label:<45} {avg:>6.2f}  | {scores_str}")

    print(f"{'='*100}")

    # Also dump raw JSON for later analysis
    print("\n--- Raw JSON ---")
    output = {
        "config": {
            "model": MODEL,
            "dataset": DATASET,
            "data_dir": DATA_DIR,
            "cr": CR,
            "fraction": FRACTION,
            "seed": SEED,
        },
        "results": [
            r if not isinstance(r, Exception) else {"error": str(r), "variant": VARIANTS[i][0]}
            for i, r in enumerate(results)
        ],
    }
    print(json.dumps(output, indent=2))

    # Save locally
    import pathlib

    out_path = pathlib.Path("evaluation/crossval_results.json")
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {out_path}")
