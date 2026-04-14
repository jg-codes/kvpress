# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Fast ablation sweep on Modal — compares merge weighting strategies.

Uses a small model (Llama-3.2-1B or Qwen2.5-0.5B) for rapid iteration.
Full RULER-4096 at f=1.0 takes ~45-60 min on A100 with 0.5B model.

Usage:
    modal run evaluation/modal_ablation.py                          # quick smoke (f=0.01)
    modal run evaluation/modal_ablation.py --fraction 0.10          # 10% subsample
    modal run evaluation/modal_ablation.py --fraction 1.0           # full dataset
    modal run evaluation/modal_ablation.py --model Qwen/Qwen2.5-0.5B-Instruct
"""

import json

import modal

# ---------------------------------------------------------------------------
# Modal setup — same image as modal_smoke.py but installs from latest branch
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("packaging", "setuptools", "wheel")
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

app = modal.App("kvpress-ablation", image=image)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATASET = "ruler"
DATA_DIR = "4096"
SEED = 42

# Ablation matrix: each variant name → registry key
# The baselines are the raw scorers; the variants test different merge weighting combos.
ABLATION_VARIANTS = [
    # Baselines (no merging)
    "knorm",
    "snapkv",
    "critical_snapkv",
    # AdaKV baselines (head-wise adaptive, no merging)
    "adakv_snapkv",
    # Default config: vonorm only (merge_keys=False, value_norm_weighting=True)
    "merging_vonorm_knorm",
    "merging_vonorm_snapkv",
    "merging_vonorm_critical_snapkv",
    # Score-weighted: vonorm + score_weighting
    "merging_score_knorm",
    "merging_score_snapkv",
    "merging_score_critical_snapkv",
    # Ablation: similarity-only (no vonorm, no score)
    "merging_simonly_knorm",
    "merging_simonly_snapkv",
    # Adaptive threshold (25th percentile similarity gate)
    "merging_adaptive_knorm",
    "merging_adaptive_snapkv",
    # MergingAdaKVPress: adaptive head-wise budgets + merge-on-evict
    "merging_adakv_knorm",
    "merging_adakv_snapkv",
    "merging_adakv_snapkv_score",
    "merging_adakv_critical_snapkv",
]

CRS = [0.25, 0.50, 0.75, 0.875]


def build_jobs(fraction: float, model: str) -> list[tuple[str, float, float, str]]:
    """Build (press_name, cr, fraction, model) tuples."""
    jobs = [("no_press", 0.75, fraction, model)]
    for variant in ABLATION_VARIANTS:
        for cr in CRS:
            jobs.append((variant, cr, fraction, model))
    return jobs


@app.function(
    gpu="A100",
    timeout=1800,
    memory=32768,
    scaledown_window=2,
)
def run_one(press_name: str, cr: float, fraction: float, model: str) -> dict:
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
        model=model,
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


def paired_bootstrap_ci(
    base_tasks: dict[str, float], merge_tasks: dict[str, float],
    n: int = 10000, seed: int = 42,
) -> dict:
    """Paired bootstrap 95% CI for delta (merging − baseline) over shared tasks."""
    import random
    from math import comb

    shared = sorted(set(base_tasks) & set(merge_tasks))
    deltas = [merge_tasks[t] - base_tasks[t] for t in shared]
    k = len(deltas)
    if k == 0:
        return {"mean_delta": 0, "ci_lo": 0, "ci_hi": 0, "p_value": 1.0,
                "sign_test_p": 1.0, "n_tasks": 0}

    observed = sum(deltas) / k

    random.seed(seed)
    boot_means = sorted(
        sum(random.choices(deltas, k=k)) / k for _ in range(n)
    )
    ci_lo = boot_means[int(0.025 * n)]
    ci_hi = boot_means[int(0.975 * n)]

    if observed >= 0:
        p = 2 * sum(1 for b in boot_means if b <= 0) / n
    else:
        p = 2 * sum(1 for b in boot_means if b >= 0) / n
    p = min(p, 1.0)

    pos = sum(1 for d in deltas if d > 0)
    neg = sum(1 for d in deltas if d < 0)
    n_nonzero = pos + neg
    if n_nonzero > 0:
        tail_count = max(pos, neg)
        sign_p = 2 * sum(comb(n_nonzero, i) for i in range(tail_count, n_nonzero + 1)) / (2 ** n_nonzero)
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
        "positive": pos,
        "negative": neg,
        "tied": sum(1 for d in deltas if d == 0),
    }


@app.local_entrypoint()
def main(fraction: float = 0.01, model: str = "meta-llama/Llama-3.2-1B-Instruct"):
    """Launch ablation sweep: baselines × vonorm × score × simonly."""
    jobs = build_jobs(fraction, model)

    model_short = model.split("/")[-1]
    print(f"\n{'='*100}")
    print(f"Ablation Sweep — Modal")
    print(f"Model: {model} | Dataset: {DATASET}-{DATA_DIR} | Fraction: {fraction} | Seed: {SEED}")
    print(f"Jobs: {len(jobs)} ({len(ABLATION_VARIANTS)} variants × {len(CRS)} CRs + no_press)")
    print(f"{'='*100}\n")

    results = list(run_one.starmap(jobs, return_exceptions=True))

    table = {}
    errors = []
    for i, r in enumerate(results):
        press_name, cr, _, _ = jobs[i]
        label = f"{press_name} (cr={cr:.2f})" if press_name != "no_press" else "no_press"

        if isinstance(r, Exception):
            errors.append({"variant": press_name, "compression_ratio": cr, "error": str(r)})
            continue
        m = r["metrics"]
        if "error" in m:
            errors.append({"variant": press_name, "compression_ratio": cr, "error": m["error"]})
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

    # --- Print results ---
    print(f"\n{'='*90}")
    print(f"{'Variant':<50} {'Mean':>6}  {'95% CI':>15}  {'n':>3}")
    print(f"{'-'*90}")

    for label, r in sorted(table.items(), key=lambda x: (x[1]["cr"], -x[1]["mean"])):
        ci_str = f"[{r['ci_lo']:.1f}, {r['ci_hi']:.1f}]"
        print(f"{label:<50} {r['mean']:>6.1f}  {ci_str:>15}  {r['n_tasks']:>3}")

    if errors:
        print(f"\n--- Errors ({len(errors)}) ---")
        for e in errors:
            print(f"  {e['variant']} CR={e['compression_ratio']}: {e['error']}")

    # --- Paired comparisons: each merge variant vs its base scorer ---
    MERGE_PREFIXES = {
        "merging_vonorm_": "vonorm",
        "merging_score_": "score",
        "merging_simonly_": "simonly",
        "merging_adaptive_": "adaptive",
        "merging_adakv_": "adakv",
    }
    SCORERS = ["knorm", "snapkv", "critical_snapkv"]

    print(f"\n{'='*115}")
    print("Paired Bootstrap: merge variant delta over baseline scorer")
    print(f"{'-'*115}")
    print(f"{'Config':<18} {'Scorer':<20} {'CR':>5} {'Base':>6} {'Merg':>6} {'Δ':>6} {'95% CI':>16} {'p':>7} {'W/L/T':>7}")
    print(f"{'-'*115}")

    paired_results = []
    for prefix, config_name in MERGE_PREFIXES.items():
        for scorer in SCORERS:
            merge_name = f"{prefix}{scorer}"
            for cr in CRS:
                base_label = f"{scorer} (cr={cr:.2f})"
                merge_label = f"{merge_name} (cr={cr:.2f})"
                if base_label in table and merge_label in table:
                    b = table[base_label]
                    m = table[merge_label]
                    pr = paired_bootstrap_ci(b["per_task"], m["per_task"])
                    paired_results.append({
                        "config": config_name, "scorer": scorer, "cr": cr, **pr,
                        "base_mean": b["mean"], "merge_mean": m["mean"],
                    })
                    ci_str = f"[{pr['ci_lo']:+.1f}, {pr['ci_hi']:+.1f}]"
                    sig = "*" if pr["p_value"] < 0.05 else " "
                    wlt = f"{pr['positive']}/{pr['negative']}/{pr['tied']}"
                    print(f"{config_name:<18} {scorer:<20} {cr:>5.2f} {b['mean']:>6.1f} {m['mean']:>6.1f} {pr['mean_delta']:>+6.1f} {ci_str:>16} {pr['p_value']:>6.4f}{sig} {wlt:>7}")

    # --- Head-to-head: vonorm vs score (same scorer, same CR) ---
    print(f"\n{'='*115}")
    print("Head-to-Head: vonorm vs score (are scores additive?)")
    print(f"{'-'*115}")
    print(f"{'Scorer':<20} {'CR':>5} {'vnorm':>6} {'score':>6} {'Δ':>6} {'95% CI':>16} {'p':>7}")
    print(f"{'-'*115}")

    h2h_results = []
    for scorer in SCORERS:
        for cr in CRS:
            v_label = f"merging_vonorm_{scorer} (cr={cr:.2f})"
            s_label = f"merging_score_{scorer} (cr={cr:.2f})"
            if v_label in table and s_label in table:
                v = table[v_label]
                s = table[s_label]
                pr = paired_bootstrap_ci(v["per_task"], s["per_task"])
                h2h_results.append({
                    "scorer": scorer, "cr": cr, **pr,
                    "vonorm_mean": v["mean"], "score_mean": s["mean"],
                })
                ci_str = f"[{pr['ci_lo']:+.1f}, {pr['ci_hi']:+.1f}]"
                sig = "*" if pr["p_value"] < 0.05 else " "
                print(f"{scorer:<20} {cr:>5.2f} {v['mean']:>6.1f} {s['mean']:>6.1f} {pr['mean_delta']:>+6.1f} {ci_str:>16} {pr['p_value']:>6.4f}{sig}")

    # --- Head-to-head: MergingAdaKV vs AdaKV (does merge help on top of adaptive allocation?) ---
    print(f"\n{'='*115}")
    print("Head-to-Head: MergingAdaKV vs AdaKV (does merging add value to adaptive heads?)")
    print(f"{'-'*115}")
    print(f"{'Comparison':<30} {'CR':>5} {'AdaKV':>6} {'MrgAda':>6} {'Δ':>6} {'95% CI':>16} {'p':>7}")
    print(f"{'-'*115}")

    adakv_h2h = []
    adakv_pairs = [("adakv_snapkv", "merging_adakv_snapkv")]
    for base_name, merge_name in adakv_pairs:
        for cr in CRS:
            b_label = f"{base_name} (cr={cr:.2f})"
            m_label = f"{merge_name} (cr={cr:.2f})"
            if b_label in table and m_label in table:
                b = table[b_label]
                m = table[m_label]
                pr = paired_bootstrap_ci(b["per_task"], m["per_task"])
                adakv_h2h.append({
                    "base": base_name, "merge": merge_name, "cr": cr, **pr,
                    "base_mean": b["mean"], "merge_mean": m["mean"],
                })
                ci_str = f"[{pr['ci_lo']:+.1f}, {pr['ci_hi']:+.1f}]"
                sig = "*" if pr["p_value"] < 0.05 else " "
                print(f"{base_name} vs {merge_name:<15} {cr:>5.2f} {b['mean']:>6.1f} {m['mean']:>6.1f} {pr['mean_delta']:>+6.1f} {ci_str:>16} {pr['p_value']:>6.4f}{sig}")

    print(f"{'='*115}")

    # Save JSON
    import pathlib

    output = {
        "config": {
            "model": model,
            "dataset": DATASET,
            "data_dir": DATA_DIR,
            "fraction": fraction,
            "seed": SEED,
            "crs": CRS,
            "variants": ABLATION_VARIANTS,
        },
        "table": table,
        "paired_vs_baseline": paired_results,
        "head_to_head_vonorm_vs_score": h2h_results,
        "head_to_head_adakv_vs_merging_adakv": adakv_h2h,
        "errors": errors,
    }
    out_path = pathlib.Path(f"evaluation/ablation_{model_short}_f{fraction}.json")
    out_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to {out_path}")
