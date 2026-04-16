# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Comprehensive MergingPress experiment including KVzap variants.
Tests all MergingPress compositions vs their baselines on RULER-4096.

Presses tested:
  - kvzap_mlp_head       : KVzapPress(mlp)                    — bare scorer
  - kvzap_mlp_layer      : AdaKVPress(KVzapPress(mlp))        — adaptive per-layer
  - merging_kvzap_mlp    : MergingPress(KVzapPress(mlp))      — uniform merge
  - merging_adakv_kvzap  : MergingPress(AdaKVPress(KVzapPress))— adaptive merge
  - kvzap_mlp            : DMSPress(KVzapPress(mlp))          — DMS wrapping (leaderboard)
  - merging_adakv_ea     : MergingPress(AdaKVPress(EA))        — best from prior run
  - merging_adakv_snapkv : MergingPress(AdaKVPress(SnapKV))
  - merging_adakv_knorm  : MergingPress(AdaKVPress(Knorm))
  + no_press baseline

Usage:
    modal run evaluation/modal_kvzap_experiment.py
    modal run evaluation/modal_kvzap_experiment.py --fraction 0.1
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
    # skorch + sklearn for KVzap MLP surrogate
    .pip_install("skorch", "scikit-learn")
    .run_commands(
        f"git clone --branch {BRANCH} --depth 1 https://github.com/jg-codes/kvpress.git /eval_repo",
    )
)

app = modal.App("kvpress-kvzap-experiment", image=image)
results_vol = modal.Volume.from_name("kvpress-kvzap-results", create_if_missing=True)

DATASET = "ruler"
DATA_DIR = "4096"
MODEL = "Qwen/Qwen3-8B"
CRS = [0.25, 0.50, 0.75]

# All presses to compare — KVzap-centric + AdaKV compositions
PRESSES = [
    # KVzap variants (all wrapping styles)
    "kvzap_mlp_head",         # KVzapPress(mlp) — bare scorer, uniform per-head CR
    "kvzap_mlp_layer",        # AdaKVPress(KVzapPress(mlp)) — adaptive per-layer budget
    "merging_kvzap_mlp",      # MergingPress(KVzapPress(mlp)) — uniform merge-on-evict
    "merging_adakv_kvzap",    # MergingPress(AdaKVPress(KVzapPress(mlp))) — adaptive merge
    "kvzap_mlp",              # DMSPress(KVzapPress(mlp)) — leaderboard DMS variant
    # MergingPress(AdaKV) compositions (other scorers for comparison)
    "merging_adakv_ea",       # MergingPress(AdaKVPress(ExpectedAttentionPress))
    "merging_adakv_snapkv",   # MergingPress(AdaKVPress(SnapKVPress))
    "merging_adakv_knorm",    # MergingPress(AdaKVPress(KnormPress))
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
    """Launch KVzap + MergingPress experiment on Modal."""
    jobs = [("no_press", 0.0, fraction)]
    for press in PRESSES:
        for cr in CRS:
            jobs.append((press, cr, fraction))

    frac_label = f"fraction={fraction}" if fraction < 1.0 else "full dataset"
    print(f"\n{'='*80}")
    print("KVzap + MergingPress Comprehensive Experiment — Modal A100")
    print(f"Model: {MODEL} | Dataset: {DATASET}-{DATA_DIR} | {frac_label}")
    print(f"Presses: {len(PRESSES)} | CRs: {CRS} | Jobs: {len(jobs)}")
    print(f"{'='*80}\n")

    results = list(run_one.starmap(jobs, return_exceptions=True))

    output_dir = pathlib.Path("evaluation/results_kvzap")
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
        table[label] = {
            "press_name": press_name,
            "cr": cr,
            "mean": round(mean, 2),
            "n_tasks": len(tasks),
            "per_task": {t: round(flatten_score(m[t]), 2) for t in tasks},
        }

    # ── Summary table ──────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"{'Variant':<55} {'Mean':>6}  {'n':>3}")
    print(f"{'-'*80}")
    for label, r in sorted(table.items(), key=lambda x: (x[1]["cr"], -x[1]["mean"])):
        print(f"{label:<55} {r['mean']:>6.1f}  {r['n_tasks']:>3}")

    # ── Per-task breakdown for KVzap variants ──────────────────────────
    kvzap_presses = ["kvzap_mlp_head", "kvzap_mlp_layer", "merging_kvzap_mlp", "merging_adakv_kvzap", "kvzap_mlp"]
    all_tasks = set()
    for label, r in table.items():
        all_tasks.update(r.get("per_task", {}).keys())
    all_tasks = sorted(all_tasks)

    if all_tasks:
        print(f"\n{'='*80}")
        print("KVzap Per-Task Breakdown (CR=0.50)")
        print(f"{'-'*80}")
        header = f"{'Task':<30}"
        for pn in kvzap_presses:
            short = pn.replace("kvzap_", "").replace("merging_", "m_").replace("adakv_", "a_")
            header += f" {short:>10}"
        print(header)
        for task in all_tasks:
            row = f"{task:<30}"
            for pn in kvzap_presses:
                label_05 = f"{pn} (cr=0.500)"
                val = table.get(label_05, {}).get("per_task", {}).get(task, None)
                row += f" {val:>10.1f}" if val is not None else f" {'—':>10}"
            print(row)

    # ── Comparison: wrapping strategies ────────────────────────────────
    print(f"\n{'='*80}")
    print("KVzap Wrapping Comparison")
    print(f"{'-'*80}")
    print(f"  {'CR':>5} {'Bare':>8} {'AdaKV':>8} {'Merging':>8} {'M(AdaKV)':>9} {'DMS':>8} | {'M(A) vs Bare':>13} {'M(A) vs DMS':>12}")
    for cr in CRS:
        bare = table.get(f"kvzap_mlp_head (cr={cr:.3f})", {}).get("mean")
        adakv = table.get(f"kvzap_mlp_layer (cr={cr:.3f})", {}).get("mean")
        merge = table.get(f"merging_kvzap_mlp (cr={cr:.3f})", {}).get("mean")
        m_adakv = table.get(f"merging_adakv_kvzap (cr={cr:.3f})", {}).get("mean")
        dms = table.get(f"kvzap_mlp (cr={cr:.3f})", {}).get("mean")

        def fmt(v):
            return f"{v:.1f}" if v is not None else "—"

        d_bare = f"{m_adakv - bare:+.1f}" if m_adakv is not None and bare is not None else "—"
        d_dms = f"{m_adakv - dms:+.1f}" if m_adakv is not None and dms is not None else "—"
        print(f"  {cr:>5.3f} {fmt(bare):>8} {fmt(adakv):>8} {fmt(merge):>8} {fmt(m_adakv):>9} {fmt(dms):>8} | {d_bare:>13} {d_dms:>12}")

    # ── Cross-scorer comparison at CR=0.50 ─────────────────────────────
    print(f"\n{'='*80}")
    print("MergingPress(AdaKV(X)) Scorer Comparison — CR=0.50")
    print(f"{'-'*80}")
    merge_adakv_scorers = [
        ("EA", "merging_adakv_ea"),
        ("SnapKV", "merging_adakv_snapkv"),
        ("Knorm", "merging_adakv_knorm"),
        ("KVzap", "merging_adakv_kvzap"),
    ]
    for scorer_name, pn in merge_adakv_scorers:
        val = table.get(f"{pn} (cr=0.500)", {}).get("mean")
        val_s = f"{val:.1f}" if val is not None else "—"
        print(f"  {scorer_name:<12} {val_s:>8}")

    if errors:
        print(f"\n--- Errors ({len(errors)}) ---")
        for e in errors:
            print(f"  {e['variant']} CR={e['cr']}: {e['error']}")

    summary = {"table": table, "errors": errors, "fraction": fraction}
    summary_path = output_dir / "kvzap_experiment_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nResults saved to {output_dir}/")
