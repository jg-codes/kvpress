# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Quick experiment: MergingPress(AdaKVPress(...)) compositions vs baselines
on RULER-4096, Qwen3-8B, fraction=0.05 (~33 samples).

Usage:
    modal run evaluation/modal_adakv_experiment.py
    modal run evaluation/modal_adakv_experiment.py --fraction 0.1
"""

import json
import os
import pathlib

import modal

_hf_token = os.environ.get("HF_TOKEN", "")
_secrets = [modal.Secret.from_dict({"HF_TOKEN": _hf_token})] if _hf_token else []

BRANCH = "dev/merging-base-press"

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
    .pip_install(f"kvpress @ git+https://github.com/jg-codes/kvpress.git@{BRANCH}")
    .pip_install("jieba", "bert_score", "fuzzywuzzy", "python-Levenshtein", "nltk", "rouge")
    .run_commands(
        f"git clone --branch {BRANCH} --depth 1 https://github.com/jg-codes/kvpress.git /eval_repo",
    )
)

app = modal.App("kvpress-adakv-experiment", image=image)
results_vol = modal.Volume.from_name("kvpress-adakv-results", create_if_missing=True)

DATASET = "ruler"
DATA_DIR = "4096"
MODEL = "Qwen/Qwen3-8B"
CRS = [0.25, 0.50, 0.75]

# Presses to compare
PRESSES = [
    # New compositions: MergingPress(AdaKVPress(...))
    "merging_adakv_ea",       # MergingPress(AdaKVPress(ExpectedAttentionPress))
    "merging_adakv_snapkv",   # MergingPress(AdaKVPress(SnapKVPress))
    "merging_adakv_knorm",    # MergingPress(AdaKVPress(KnormPress))
    # Baselines: plain AdaKV wrapping
    "expected_attention",     # AdaKVPress(ExpectedAttentionPress)
    "adakv_snapkv",           # AdaKVPress(SnapKVPress)
    # Existing MergingPress (uniform budget)
    "merging_expected_attention",
    "merging_snapkv",
    "merging_knorm",
]


@app.function(
    gpu="A100",
    timeout=14400,
    memory=65536,
    scaledown_window=2,
    secrets=_secrets,
    volumes={"/results": results_vol},
)
def run_one(press_name: str, cr: float, fraction: float = 0.05) -> dict:
    """Run a single (press, CR) evaluation."""
    import glob
    import os
    import sys

    sys.path.insert(0, "/eval_repo/evaluation")
    os.chdir("/eval_repo/evaluation")

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
def main(fraction: float = 0.05):
    """Launch AdaKV wrapping experiment on Modal."""
    jobs = [("no_press", 0.0, fraction)]
    for press in PRESSES:
        for cr in CRS:
            jobs.append((press, cr, fraction))

    frac_label = f"fraction={fraction}" if fraction < 1.0 else "full dataset"
    print(f"\n{'='*80}")
    print("MergingPress(AdaKV) Experiment — Modal A100")
    print(f"Model: {MODEL} | Dataset: {DATASET}-{DATA_DIR} | {frac_label}")
    print(f"Presses: {len(PRESSES)} | CRs: {CRS} | Jobs: {len(jobs)}")
    print(f"{'='*80}\n")

    results = list(run_one.starmap(jobs, return_exceptions=True))

    output_dir = pathlib.Path("evaluation/results_adakv")
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
    print(f"{'Variant':<55} {'Mean':>6}  {'n':>3}")
    print(f"{'-'*80}")
    for label, r in sorted(table.items(), key=lambda x: (x[1]["cr"], -x[1]["mean"])):
        print(f"{label:<55} {r['mean']:>6.1f}  {r['n_tasks']:>3}")

    # Delta: MergingPress(AdaKV(X)) vs AdaKV(X) vs MergingPress(X)
    print(f"\n{'='*80}")
    print("Comparison: MergingPress(AdaKV(X)) vs AdaKV(X) vs MergingPress(X)")
    print(f"{'-'*80}")
    scorers = [
        ("EA", "merging_adakv_ea", "expected_attention", "merging_expected_attention"),
        ("SnapKV", "merging_adakv_snapkv", "adakv_snapkv", "merging_snapkv"),
        ("Knorm", "merging_adakv_knorm", None, "merging_knorm"),
    ]
    for scorer_name, merge_adakv, adakv_base, merge_base in scorers:
        print(f"\n  {scorer_name}:")
        print(f"  {'CR':>5} {'Merging(AdaKV)':>14} {'AdaKV':>10} {'Merging':>10} {'Δ vs AdaKV':>11} {'Δ vs Merging':>13}")
        for cr in CRS:
            ma = table.get(f"{merge_adakv} (cr={cr:.3f})", {}).get("mean", None)
            ab = table.get(f"{adakv_base} (cr={cr:.3f})", {}).get("mean", None) if adakv_base else None
            mb = table.get(f"{merge_base} (cr={cr:.3f})", {}).get("mean", None)

            ma_s = f"{ma:.1f}" if ma is not None else "—"
            ab_s = f"{ab:.1f}" if ab is not None else "—"
            mb_s = f"{mb:.1f}" if mb is not None else "—"
            d_ada = f"{ma - ab:+.1f}" if ma is not None and ab is not None else "—"
            d_merge = f"{ma - mb:+.1f}" if ma is not None and mb is not None else "—"
            print(f"  {cr:>5.3f} {ma_s:>14} {ab_s:>10} {mb_s:>10} {d_ada:>11} {d_merge:>13}")

    if errors:
        print(f"\n--- Errors ({len(errors)}) ---")
        for e in errors:
            print(f"  {e['variant']} CR={e['cr']}: {e['error']}")

    summary = {"table": table, "errors": errors, "fraction": fraction}
    summary_path = output_dir / "adakv_experiment_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nResults saved to {output_dir}/")
