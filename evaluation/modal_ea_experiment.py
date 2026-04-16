# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Targeted experiment: MergingPress(ExpectedAttentionPress) vs bare ExpectedAttentionPress
on RULER-4096, Qwen3-8B, fraction=0.1 (~650 samples).

Usage:
    modal run evaluation/modal_ea_experiment.py
    modal run evaluation/modal_ea_experiment.py --fraction 1.0   # full dataset (~$50, 6h)
"""

import json
import os
import pathlib

import modal

_hf_token = os.environ.get("HF_TOKEN", "")
_secrets = [modal.Secret.from_dict({"HF_TOKEN": _hf_token})] if _hf_token else []

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
    .pip_install("kvpress @ git+https://github.com/jg-codes/kvpress.git@pr/merging-press")
    .pip_install("jieba", "bert_score", "fuzzywuzzy", "python-Levenshtein", "nltk", "rouge")
    .run_commands(
        "git clone --branch pr/merging-press --depth 1 https://github.com/jg-codes/kvpress.git /eval_repo",
    )
)

app = modal.App("kvpress-ea-experiment", image=image)
results_vol = modal.Volume.from_name("kvpress-ea-results", create_if_missing=True)

DATASET = "ruler"
DATA_DIR = "4096"
MODEL = "Qwen/Qwen3-8B"
CRS = [0.25, 0.50, 0.75, 0.875]

# Experiment presses
PRESSES = [
    "merging_expected_attention",   # MergingPress(ExpectedAttentionPress(epsilon=1e-2))
    "expected_attention_bare",      # ExpectedAttentionPress(epsilon=1e-2)  — no AdaKV wrapper
    "expected_attention",           # AdaKV(ExpectedAttentionPress(epsilon=1e-2)) — leaderboard baseline
]


@app.function(
    gpu="A100",
    timeout=14400,  # 4 hours
    memory=65536,
    scaledown_window=2,
    secrets=_secrets,
    volumes={"/results": results_vol},
)
def run_one(press_name: str, cr: float, fraction: float = 0.1) -> dict:
    """Run a single (press, CR) evaluation, injecting registry entries at runtime."""
    import glob
    import os
    import sys

    sys.path.insert(0, "/eval_repo/evaluation")
    os.chdir("/eval_repo/evaluation")

    # Inject our custom press configs into the registry before evaluate imports it
    from evaluate_registry import PRESS_REGISTRY
    from kvpress import ExpectedAttentionPress, MergingPress

    if "merging_expected_attention" not in PRESS_REGISTRY:
        PRESS_REGISTRY["merging_expected_attention"] = MergingPress(ExpectedAttentionPress(epsilon=1e-2))
    if "expected_attention_bare" not in PRESS_REGISTRY:
        PRESS_REGISTRY["expected_attention_bare"] = ExpectedAttentionPress(epsilon=1e-2)

    from evaluate import EvaluationConfig, EvaluationRunner

    output_tag = f"{press_name}__{cr:.3f}__f{fraction:.3f}"
    config = EvaluationConfig(
        dataset=DATASET,
        data_dir=DATA_DIR,
        model=MODEL,
        device="cuda:0",
        press_name=press_name,
        compression_ratio=cr,
        fraction=fraction,
        seed=42,
        output_dir=f"/results/{output_tag}",
    )

    runner = EvaluationRunner(config)
    runner.run_evaluation()

    results_vol.commit()

    result_files = {}
    base = f"/results/{output_tag}"
    for path in glob.glob(f"{base}/**/*", recursive=True):
        if os.path.isfile(path):
            rel = os.path.relpath(path, "/results")
            with open(path) as f:
                result_files[rel] = f.read()

    metrics_files = glob.glob(f"{base}/**/metrics.json", recursive=True)
    metrics = {}
    if metrics_files:
        with open(metrics_files[0]) as f:
            metrics = json.load(f)

    return {
        "press_name": press_name,
        "cr": cr,
        "fraction": fraction,
        "metrics": metrics,
        "files": result_files,
    }


def flatten_score(val) -> float:
    if isinstance(val, dict):
        return val.get("string_match", val.get("rouge1", val.get("f1", 0.0)))
    return float(val) if val is not None else 0.0


@app.local_entrypoint()
def main(fraction: float = 0.1):
    """Launch EA experiment on Modal."""
    jobs = [("no_press", 0.0, fraction)]
    for press in PRESSES:
        for cr in CRS:
            jobs.append((press, cr, fraction))

    frac_label = f"fraction={fraction}" if fraction < 1.0 else "full dataset"
    print(f"\n{'='*80}")
    print("ExpectedAttention Experiment — Modal A100")
    print(f"Model: {MODEL} | Dataset: {DATASET}-{DATA_DIR} | {frac_label}")
    print(f"Presses: {len(PRESSES)} | Jobs: {len(jobs)}")
    print(f"{'='*80}\n")

    results = list(run_one.starmap(jobs, return_exceptions=True))

    # Save results locally
    output_dir = pathlib.Path("evaluation/results_ea")
    output_dir.mkdir(parents=True, exist_ok=True)

    table = {}
    errors = []
    for i, r in enumerate(results):
        press_name, cr, _ = jobs[i]
        label = f"{press_name} (cr={cr:.3f})" if press_name != "no_press" else "no_press"

        if isinstance(r, Exception):
            errors.append({"variant": press_name, "cr": cr, "error": str(r)})
            continue
        if "error" in r.get("metrics", {}):
            errors.append({"variant": press_name, "cr": cr, "error": r["metrics"]["error"]})
            continue

        for rel_path, content in r.get("files", {}).items():
            file_path = output_dir / rel_path.split("/", 1)[-1] if "/" in rel_path else output_dir / rel_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)

        m = r["metrics"]
        tasks = sorted(m.keys())
        scores = [flatten_score(m[t]) for t in tasks]
        mean = sum(scores) / len(scores) if scores else 0.0
        table[label] = {"press_name": press_name, "cr": cr, "mean": round(mean, 2), "n_tasks": len(tasks)}

    # Print summary table
    print(f"\n{'='*80}")
    print(f"{'Variant':<50} {'Mean':>6}  {'n':>3}")
    print(f"{'-'*80}")
    for label, r in sorted(table.items(), key=lambda x: (x[1]["cr"], -x[1]["mean"])):
        print(f"{label:<50} {r['mean']:>6.1f}  {r['n_tasks']:>3}")

    # Delta comparison
    print(f"\n{'='*80}")
    print("MergingPress(EA) vs bare EA vs AdaKV(EA)")
    print(f"{'-'*80}")
    print(f"{'CR':>5} {'Merging(EA)':>12} {'Bare EA':>10} {'AdaKV(EA)':>10} {'Δ vs bare':>10} {'Δ vs AdaKV':>11}")
    print(f"{'-'*80}")

    for cr in CRS:
        m_label = f"merging_expected_attention (cr={cr:.3f})"
        b_label = f"expected_attention_bare (cr={cr:.3f})"
        a_label = f"expected_attention (cr={cr:.3f})"
        m_val = table.get(m_label, {}).get("mean", "—")
        b_val = table.get(b_label, {}).get("mean", "—")
        a_val = table.get(a_label, {}).get("mean", "—")

        d_bare = f"{m_val - b_val:+.1f}" if isinstance(m_val, float) and isinstance(b_val, float) else "—"
        d_ada = f"{m_val - a_val:+.1f}" if isinstance(m_val, float) and isinstance(a_val, float) else "—"
        print(f"{cr:>5.3f} {m_val:>12} {b_val:>10} {a_val:>10} {d_bare:>10} {d_ada:>11}")

    if errors:
        print(f"\n--- Errors ({len(errors)}) ---")
        for e in errors:
            print(f"  {e['variant']} CR={e['cr']}: {e['error']}")

    # Save summary JSON
    summary = {"table": table, "errors": errors, "fraction": fraction}
    summary_path = output_dir / "ea_experiment_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nResults saved to {output_dir}/")
