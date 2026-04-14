# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Multi-CR smoke test on Modal — validates all variants run correctly.

Runs each (variant × CR) combination on a separate L4 GPU in parallel.
fraction=0.001 (~7 samples total) for fast sanity check.

Usage:
    modal run evaluation/modal_smoke.py
    modal run evaluation/modal_smoke.py --fraction 0.10   # full run
"""

import json

import modal

# ---------------------------------------------------------------------------
# Modal setup
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
    .pip_install("kvpress @ git+https://github.com/jg-codes/kvpress.git@merging-press")
    .pip_install("jieba", "bert_score", "fuzzywuzzy", "python-Levenshtein", "nltk", "rouge")
    .run_commands(
        "git clone --branch merging-press --depth 1 https://github.com/jg-codes/kvpress.git /eval_repo",
    )
)

app = modal.App("kvpress-multi-cr", image=image)

# ---------------------------------------------------------------------------
# Experiment matrix
# ---------------------------------------------------------------------------
MODEL = "Qwen/Qwen3-8B"
DATASET = "ruler"
DATA_DIR = "4096"
SEED = 42

# Prefill variants: [baselines] + [merging wrapping each scorer]
PREFILL_VARIANTS = [
    "knorm",
    "snapkv",
    "critical_snapkv",
    "merging_vonorm_knorm",
    "merging_vonorm_snapkv",
    "merging_vonorm_critical_snapkv",
]

CRS = [0.25, 0.50, 0.75, 0.875]


def build_jobs(fraction: float) -> list[tuple[str, float, float]]:
    """Build (press_name, cr, fraction) tuples for all runs."""
    jobs = []
    # no_press baseline (cr is ignored)
    jobs.append(("no_press", 0.75, fraction))
    # Prefill matrix
    for variant in PREFILL_VARIANTS:
        for cr in CRS:
            jobs.append((variant, cr, fraction))
    return jobs


@app.function(
    gpu="L4",
    timeout=3600,
    memory=32768,
    scaledown_window=2,
)
def run_one(press_name: str, cr: float, fraction: float) -> dict:
    """Run a single (variant, CR) evaluation and return metrics."""
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
        compression_ratio=cr,
        fraction=fraction,
        seed=SEED,
        output_dir=f"/results/{press_name}__{cr:.2f}",
    )

    runner = EvaluationRunner(config)
    runner.run_evaluation()

    metrics_files = glob.glob(f"/results/{press_name}__{cr:.2f}/**/metrics.json", recursive=True)
    if metrics_files:
        with open(metrics_files[0]) as f:
            metrics = json.load(f)
    else:
        metrics = {"error": "no metrics.json found"}

    return {"press_name": press_name, "cr": cr, "metrics": metrics}


def flatten_score(val) -> float:
    if isinstance(val, dict):
        return val.get("string_match", val.get("rouge1", val.get("f1", 0.0)))
    return float(val) if val is not None else 0.0


def compute_ci(scores: list[float], n: int = 1000, seed: int = 42) -> tuple[float, float, float]:
    """Bootstrap 95% CI over task scores."""
    import random

    random.seed(seed)
    mean = sum(scores) / len(scores) if scores else 0.0
    boot = sorted(
        sum(random.choices(scores, k=len(scores))) / len(scores)
        for _ in range(n)
    )
    return mean, boot[int(0.025 * n)], boot[int(0.975 * n)]


@app.local_entrypoint()
def main(fraction: float = 0.001):
    """Launch all (variant × CR) jobs in parallel, collect and display."""
    jobs = build_jobs(fraction)

    print(f"\n{'='*90}")
    print(f"Multi-CR Benchmark — Modal")
    print(f"Model: {MODEL} | Dataset: {DATASET}-{DATA_DIR} | Fraction: {fraction} | Seed: {SEED}")
    print(f"Jobs: {len(jobs)} ({len(PREFILL_VARIANTS)} variants × {len(CRS)} CRs + no_press)")
    print(f"{'='*90}\n")

    results = list(run_one.starmap(jobs, return_exceptions=True))

    # Build table: {(press_name, cr): {mean, ci_lo, ci_hi, per_task}}
    table = {}
    errors = []
    for i, r in enumerate(results):
        press_name, cr, _ = jobs[i]
        label = f"{press_name} (cr={cr:.2f})" if press_name != "no_press" else "no_press"

        if isinstance(r, Exception):
            errors.append((label, str(r)))
            continue
        m = r["metrics"]
        if "error" in m:
            errors.append((label, m["error"]))
            continue

        tasks = sorted(m.keys())
        scores = [flatten_score(m[t]) for t in tasks]
        mean, lo, hi = compute_ci(scores)
        table[label] = {
            "press_name": press_name,
            "cr": cr,
            "mean": round(mean, 2),
            "ci_lo": round(lo, 2),
            "ci_hi": round(hi, 2),
            "n_tasks": len(tasks),
            "per_task": {t: round(flatten_score(m[t]), 2) for t in tasks},
        }

    # Print grouped by scorer
    print(f"\n{'='*90}")
    print(f"{'Variant':<45} {'Mean':>6}  {'95% CI':>15}  {'n':>3}")
    print(f"{'-'*90}")

    # Group: no_press first, then by CR within each scorer pair
    for label, r in sorted(table.items(), key=lambda x: (-x[1]["mean"],)):
        ci_str = f"[{r['ci_lo']:.1f}, {r['ci_hi']:.1f}]"
        print(f"{label:<45} {r['mean']:>6.1f}  {ci_str:>15}  {r['n_tasks']:>3}")

    if errors:
        print(f"\n--- Errors ({len(errors)}) ---")
        for label, err in errors:
            print(f"  {label}: {err}")

    print(f"{'='*90}")

    # --- Comparison table: baseline vs merging for each scorer ---
    print(f"\n{'='*90}")
    print("Scorer-Pair Comparison (merging delta over baseline)")
    print(f"{'-'*90}")
    print(f"{'Scorer':<15} {'CR':>5} {'Baseline':>10} {'+ Merging':>10} {'Delta':>8} {'Significant?':>13}")
    print(f"{'-'*90}")

    pairs = [
        ("knorm", "merging_vonorm_knorm"),
        ("snapkv", "merging_vonorm_snapkv"),
        ("critical_snapkv", "merging_vonorm_critical_snapkv"),
    ]
    for base_name, merge_name in pairs:
        for cr in CRS:
            base_label = f"{base_name} (cr={cr:.2f})"
            merge_label = f"{merge_name} (cr={cr:.2f})"
            if base_label in table and merge_label in table:
                b = table[base_label]
                m = table[merge_label]
                delta = m["mean"] - b["mean"]
                # Non-overlapping CIs = likely significant
                sig = "YES" if m["ci_lo"] > b["ci_hi"] or b["ci_lo"] > m["ci_hi"] else "no"
                print(f"{base_name:<15} {cr:>5.2f} {b['mean']:>10.1f} {m['mean']:>10.1f} {delta:>+8.1f} {sig:>13}")

    print(f"{'='*90}")

    # Save JSON
    import pathlib

    output = {
        "config": {
            "model": MODEL,
            "dataset": DATASET,
            "data_dir": DATA_DIR,
            "fraction": fraction,
            "seed": SEED,
            "crs": CRS,
        },
        "table": table,
        "errors": errors,
    }
    out_path = pathlib.Path(f"evaluation/multi_cr_results_f{fraction}.json")
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {out_path}")
