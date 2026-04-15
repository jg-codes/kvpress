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
    # Pin torch to cu124 wheels — avoids CUDA driver mismatch on Modal fleet
    .run_commands(
        "pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "transformers>=4.48",
        "datasets",
        "pandas",
        "numpy",
        "fire",
        "pyyaml",
        "tqdm",
        "accelerate",
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
    "merging_knorm",
    "merging_snapkv",
    "merging_critical_snapkv",
    "merging_adakv_snapkv",
    "merging_cam_knorm",
]

# Hyperparameter sweep variants (not in registry — define inline in build_jobs if needed)
SWEEP_VARIANTS = [
    "snapkv",
    "merging_snapkv",
]

CRS = [0.25, 0.50, 0.75, 0.875]


def build_jobs(fraction: float, sweep: bool = False) -> list[tuple[str, float, float]]:
    """Build (press_name, cr, fraction) tuples for all runs."""
    variants = SWEEP_VARIANTS if sweep else PREFILL_VARIANTS
    jobs = []
    # no_press baseline (cr is ignored)
    jobs.append(("no_press", 0.75, fraction))
    for variant in variants:
        for cr in CRS:
            jobs.append((variant, cr, fraction))
    return jobs


@app.function(
    gpu="A100",
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
    boot = sorted(sum(random.choices(scores, k=len(scores))) / len(scores) for _ in range(n))
    return mean, boot[int(0.025 * n)], boot[int(0.975 * n)]


def paired_bootstrap_ci(
    base_tasks: dict[str, float],
    merge_tasks: dict[str, float],
    n: int = 10000,
    seed: int = 42,
) -> dict:
    """Paired bootstrap 95% CI for delta (merging − baseline) over shared tasks.

    Returns dict with mean_delta, ci_lo, ci_hi, p_value (two-sided),
    sign_test_p, and n_tasks.
    """
    import random

    shared = sorted(set(base_tasks) & set(merge_tasks))
    deltas = [merge_tasks[t] - base_tasks[t] for t in shared]
    k = len(deltas)
    if k == 0:
        return {"mean_delta": 0, "ci_lo": 0, "ci_hi": 0, "p_value": 1.0, "sign_test_p": 1.0, "n_tasks": 0}

    observed = sum(deltas) / k

    random.seed(seed)
    boot_means = sorted(sum(random.choices(deltas, k=k)) / k for _ in range(n))
    ci_lo = boot_means[int(0.025 * n)]
    ci_hi = boot_means[int(0.975 * n)]

    # Bootstrap p-value: fraction of resamples on wrong side of zero
    if observed >= 0:
        p = 2 * sum(1 for b in boot_means if b <= 0) / n
    else:
        p = 2 * sum(1 for b in boot_means if b >= 0) / n
    p = min(p, 1.0)

    # Sign test: under H0 (no effect), positive deltas ~ Binomial(k, 0.5)
    pos = sum(1 for d in deltas if d > 0)
    neg = sum(1 for d in deltas if d < 0)
    n_nonzero = pos + neg
    if n_nonzero > 0:
        # Two-sided: P(X >= max(pos,neg)) under Binom(n_nonzero, 0.5)
        from math import comb

        tail_count = max(pos, neg)
        sign_p = 2 * sum(comb(n_nonzero, i) for i in range(tail_count, n_nonzero + 1)) / (2**n_nonzero)
        sign_p = min(sign_p, 1.0)
    else:
        sign_p = 1.0

    return {
        "mean_delta": round(observed, 2),
        "ci_lo": round(ci_lo, 2),
        "ci_hi": round(ci_hi, 2),
        "p_value": round(p, 4),
        "sign_test_p": round(sign_p, 4),
        "n_tasks": k,
        "positive": sum(1 for d in deltas if d > 0),
        "negative": sum(1 for d in deltas if d < 0),
        "tied": sum(1 for d in deltas if d == 0),
    }


@app.local_entrypoint()
def main(fraction: float = 0.001, sweep: bool = False):
    """Launch all (variant × CR) jobs in parallel, collect and display."""
    jobs = build_jobs(fraction, sweep=sweep)
    variants = SWEEP_VARIANTS if sweep else PREFILL_VARIANTS
    mode = "SWEEP" if sweep else "Multi-CR"

    print(f"\n{'='*100}")
    print(f"{mode} Benchmark — Modal")
    print(f"Model: {MODEL} | Dataset: {DATASET}-{DATA_DIR} | Fraction: {fraction} | Seed: {SEED}")
    print(f"Jobs: {len(jobs)} ({len(variants)} variants × {len(CRS)} CRs + no_press)")
    print(f"{'='*100}\n")

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
    # Detect all merging wrapper pairs dynamically
    pairs = []
    seen_bases = set()
    MERGE_PREFIX = "merging_"
    for label, r in table.items():
        pn = r["press_name"]
        if (
            pn.startswith(MERGE_PREFIX)
            and not pn.startswith("merging_adakv_")
            and not pn.startswith("merging_cam_")
            and not pn.startswith("merging_decoding_")
        ):
            base = pn[len(MERGE_PREFIX) :]
            if base not in seen_bases:
                pairs.append((base, pn))
                seen_bases.add(base)
    # Fallback to known pairs if detection fails
    if not pairs:
        pairs = [
            ("knorm", "merging_knorm"),
            ("snapkv", "merging_snapkv"),
            ("critical_snapkv", "merging_critical_snapkv"),
        ]

    print(f"\n{'='*100}")
    print("Paired Bootstrap Comparison (merging delta over baseline)")
    print(f"{'-'*100}")
    print(
        f"{'Scorer':<20} {'CR':>5} {'Base':>6} {'Merg':>6}"
        f" {'Delta':>7} {'95% CI':>16} {'p_boot':>7} {'p_sign':>7} {'W/L/T':>7}"
    )
    print(f"{'-'*100}")

    paired_results = []
    for base_name, merge_name in pairs:
        for cr in CRS:
            base_label = f"{base_name} (cr={cr:.2f})"
            merge_label = f"{merge_name} (cr={cr:.2f})"
            if base_label in table and merge_label in table:
                b = table[base_label]
                m = table[merge_label]
                pr = paired_bootstrap_ci(b["per_task"], m["per_task"])
                paired_results.append(
                    {"scorer": base_name, "cr": cr, **pr, "base_mean": b["mean"], "merge_mean": m["mean"]}
                )
                ci_str = f"[{pr['ci_lo']:+.1f}, {pr['ci_hi']:+.1f}]"
                sig_mark = "*" if pr["p_value"] < 0.05 else " "
                wlt = f"{pr['positive']}/{pr['negative']}/{pr['tied']}"
                print(
                    f"{base_name:<20} {cr:>5.2f} {b['mean']:>6.1f}"
                    f" {m['mean']:>6.1f} {pr['mean_delta']:>+7.1f}"
                    f" {ci_str:>16} {pr['p_value']:>7.4f}{sig_mark}"
                    f"{pr['sign_test_p']:>7.4f} {wlt:>7}"
                )

    print(f"{'-'*100}")
    print("  * = p < 0.05 (paired bootstrap, B=10000)")
    print(f"{'='*100}")

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
        "paired": paired_results,
        "errors": errors,
    }
    suffix = "_sweep" if sweep else ""
    out_path = pathlib.Path(f"evaluation/multi_cr_results_f{fraction}{suffix}.json")
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {out_path}")
