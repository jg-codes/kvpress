# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
MergingPress(DMSPress(KVzapPress)) v2 — perturbation-bound gating sweep.

Tests whether perturbation_gate fixes qa_1 regression (-2.1pp) observed in v1.
Sweeps pg=0.5/1.0/2.0 at t=-3, plus t=-2 for more aggressive compression.

Usage:
    modal run --detach evaluation/modal_dms_merging_v2.py                    # f=0.1 quick
    modal run --detach evaluation/modal_dms_merging_v2.py --fraction 1.0     # full
"""

import json
import os
import pathlib

import modal

_hf_token = os.environ.get("HF_TOKEN", "")
_secrets = [modal.Secret.from_dict({"HF_TOKEN": _hf_token})] if _hf_token else []

BRANCH = "dev/merging-hook-composition"

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
    # Install kvpress from dev branch with perturbation_gate support
    .run_commands(
        f"pip install --no-cache-dir 'kvpress[kvzap] @ git+https://github.com/jg-codes/kvpress.git@{BRANCH}'"
    )
    .pip_install("jieba", "bert_score", "fuzzywuzzy", "python-Levenshtein", "nltk", "rouge")
    .pip_install("skorch", "scikit-learn")
    .run_commands(
        f"git clone --branch {BRANCH} --depth 1 https://github.com/jg-codes/kvpress.git /eval_repo",
    )
)

app = modal.App("kvpress-dms-merging-v2", image=image)
results_vol = modal.Volume.from_name("kvpress-dms-merging-v2-results", create_if_missing=True)

DATASET = "ruler"
DATA_DIR = "4096"
MODEL = "Qwen/Qwen3-8B"

# ── Evaluation configs ──────────────────────────────────────────────────
#
# v1 findings: at t=-3, merge default gives +0.40pp mean but qa_1 regresses -2.1pp.
# Hypothesis: qa_1 answer tokens have high ||v|| and low similarity to survivors.
# perturbation_gate skips merges where ||v_i|| * (1-w)/(1+w) > gate.
#
CONFIGS = [
    # ── Baselines (reuse from v1 for consistency) ──
    {"name": "no_press", "threshold": None, "merge_params": None},
    {"name": "dms_bare_t-3", "threshold": -3, "merge_params": None},
    {"name": "dms_bare_t-4", "threshold": -4, "merge_params": None},
    # ── Current best from v1 ──
    {"name": "m_dms_t-3_default", "threshold": -3, "merge_params": {}},
    # ── Perturbation gate sweep at t=-3 ──
    {"name": "m_dms_t-3_pg0.5", "threshold": -3, "merge_params": {"perturbation_gate": 0.5}},
    {"name": "m_dms_t-3_pg1.0", "threshold": -3, "merge_params": {"perturbation_gate": 1.0}},
    {"name": "m_dms_t-3_pg2.0", "threshold": -3, "merge_params": {"perturbation_gate": 2.0}},
    # ── Combined: perturbation gate + merge_fraction ──
    {"name": "m_dms_t-3_pg1.0_mf0.75", "threshold": -3,
     "merge_params": {"perturbation_gate": 1.0, "merge_fraction": 0.75}},
    # ── Aggressive threshold: more room for merging ──
    {"name": "dms_bare_t-2", "threshold": -2, "merge_params": None},
    {"name": "m_dms_t-2_pg1.0", "threshold": -2, "merge_params": {"perturbation_gate": 1.0}},
]


def _build_press(config: dict):
    """Construct press from config dict."""
    from kvpress import DMSPress, KVzapPress, MergingPress

    if config["name"] == "no_press":
        return None

    threshold = config["threshold"]
    merge_params = config["merge_params"]

    # Bare DMSPress(KVzapPress) baseline
    dms = DMSPress(press=KVzapPress(model_type="mlp"), threshold=threshold, sliding_window_size=128)

    if merge_params is None:
        return dms

    # MergingPress(DMSPress(KVzapPress)) with optional perturbation gating
    return MergingPress(
        dms,
        similarity_threshold=merge_params.get("similarity_threshold", 0.0),
        merge_fraction=merge_params.get("merge_fraction", 1.0),
        value_norm_weighting=merge_params.get("value_norm_weighting", True),
        merge_keys=merge_params.get("merge_keys", False),
        max_merge_per_token=merge_params.get("max_merge_per_token", 0),
        perturbation_gate=merge_params.get("perturbation_gate", 0.0),
    )


def _flatten(val) -> float:
    if isinstance(val, dict):
        return val.get("string_match", val.get("rouge1", val.get("f1", 0.0)))
    return float(val) if val is not None else 0.0


@app.function(
    gpu="A100",
    timeout=25200,
    memory=65536,
    scaledown_window=2,
    secrets=_secrets,
    volumes={"/results": results_vol},
)
def run_one(config: dict, fraction: float = 0.10) -> dict:
    """Run a single config: load model, run inference, return metrics + timing."""
    import os
    import random
    import sys
    import time

    import numpy as np
    import pandas as pd
    import torch
    from datasets import load_dataset
    from tqdm import tqdm
    from transformers import pipeline as hf_pipeline

    sys.path.insert(0, "/eval_repo/evaluation")
    os.chdir("/eval_repo/evaluation")

    import kvpress  # noqa: F401 — registers kv-press-text-generation pipeline

    from benchmarks.ruler.calculate_metrics import calculate_metrics as ruler_scorer

    name = config["name"]

    # ── Deterministic seeds ─────────────────────────────────────────
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # ── Load model ──────────────────────────────────────────────────
    t_model = time.monotonic()
    pipe = hf_pipeline(
        "kv-press-text-generation",
        model=MODEL,
        device_map="auto",
        trust_remote_code=True,
    )
    pipe.model.eval()
    model_load_s = round(time.monotonic() - t_model, 1)

    # ── Load dataset ────────────────────────────────────────────────
    t_data = time.monotonic()
    df = load_dataset("simonjegou/ruler", data_dir=DATA_DIR, split="test").to_pandas()
    if fraction < 1.0:
        df = df.sample(frac=fraction, random_state=42)
    data_load_s = round(time.monotonic() - t_data, 1)

    # ── Build press ─────────────────────────────────────────────────
    press = _build_press(config)
    df["predicted_answer"] = None
    df["compression_ratio"] = 0.0

    # ── Inference with timing ───────────────────────────────────────
    df_grouped = df.groupby("context")
    n_contexts = df["context"].nunique()

    torch.cuda.empty_cache()
    t_infer = time.monotonic()

    with torch.inference_mode():
        for context, df_group in tqdm(df_grouped, total=n_contexts, desc=name):
            questions = df_group["question"].to_list()
            max_new_tokens = df_group["max_new_tokens"].iloc[0]
            answer_prefix = df_group["answer_prefix"].iloc[0]

            output = pipe(
                context,
                questions=questions,
                answer_prefix=answer_prefix,
                press=press,
                max_new_tokens=max_new_tokens,
            )
            df.loc[df_group.index, "predicted_answer"] = output["answers"]

            if press is not None and hasattr(press, "compression_ratio"):
                try:
                    cr_val = press.compression_ratio
                except (AssertionError, AttributeError):
                    cr_val = 0.0
                df.loc[df_group.index, "compression_ratio"] = cr_val

            torch.cuda.empty_cache()

    infer_s = round(time.monotonic() - t_infer, 1)

    # ── Metrics ─────────────────────────────────────────────────────
    mean_cr = float(df["compression_ratio"].mean())
    metrics = ruler_scorer(df)
    tasks = sorted(metrics.keys())
    scores = [_flatten(metrics[t]) for t in tasks]
    mean_score = sum(scores) / len(scores) if scores else 0.0

    # ── Save to volume ──────────────────────────────────────────────
    cfg_dir = f"/results/{name}"
    os.makedirs(cfg_dir, exist_ok=True)
    df[list(set(df.columns) - {"context"})].to_csv(f"{cfg_dir}/predictions.csv", index=False)
    with open(f"{cfg_dir}/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(f"{cfg_dir}/timing.json", "w") as f:
        json.dump({
            "model_load_seconds": model_load_s,
            "data_load_seconds": data_load_s,
            "inference_seconds": infer_s,
        }, f, indent=2)
    results_vol.commit()

    return {
        "name": name,
        "threshold": config["threshold"],
        "merge_params": config["merge_params"],
        "mean_score": round(mean_score, 2),
        "mean_compression_ratio": round(mean_cr, 4),
        "model_load_seconds": model_load_s,
        "inference_seconds": infer_s,
        "per_task": {t: round(_flatten(metrics[t]), 2) for t in tasks},
        "n_samples": len(df),
    }


@app.local_entrypoint()
def main(fraction: float = 0.10):
    """Launch perturbation-gate sweep on Modal — all configs in parallel."""
    n_baselines = sum(1 for c in CONFIGS if c["merge_params"] is None)
    n_merging = len(CONFIGS) - n_baselines

    print(f"\n{'='*90}")
    print("MergingPress(DMSPress(KVzapPress)) v2 — Perturbation-Bound Gating Sweep")
    print(f"Model: {MODEL} | Dataset: {DATASET}-{DATA_DIR} | f={fraction}")
    print(f"Configs: {len(CONFIGS)} ({n_baselines} baselines + {n_merging} merging variants)")
    print(f"Execution: parallel starmap on {len(CONFIGS)} A100s")
    print(f"Branch: {BRANCH}")
    print(f"{'='*90}\n")

    for c in CONFIGS:
        mp = c["merge_params"]
        mp_str = f"MergingPress({mp})" if mp is not None else "bare"
        print(f"  {c['name']:<42} threshold={str(c['threshold']):<5} {mp_str}")
    print()

    # Launch all configs in parallel
    starmap_args = [(c, fraction) for c in CONFIGS]
    results_raw = list(run_one.starmap(starmap_args, return_exceptions=True))

    # ── Collect results ──────────────────────────────────────────────
    results = []
    errors = []
    for i, r in enumerate(results_raw):
        if isinstance(r, Exception):
            errors.append({"name": CONFIGS[i]["name"], "error": str(r)})
        else:
            results.append(r)

    # ── Summary table ────────────────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"{'Name':<42} {'Score':>6} {'CR':>7} {'Infer':>7} {'Load':>6} {'Thr':>5}")
    print(f"{'-'*90}")

    baselines = {}
    for r in sorted(results, key=lambda x: (x["threshold"] or 0, x["merge_params"] is not None)):
        t_str = f"{r['threshold']}" if r["threshold"] is not None else "-"
        cr_str = f"{r['mean_compression_ratio']:.2%}" if r["mean_compression_ratio"] > 0 else "0%"
        print(f"{r['name']:<42} {r['mean_score']:>6.2f} {cr_str:>7} "
              f"{r['inference_seconds']:>6.0f}s {r['model_load_seconds']:>5.0f}s {t_str:>5}")

        if r["merge_params"] is None and r["threshold"] is not None:
            baselines[r["threshold"]] = r

    # ── Delta table: MergingPress vs bare DMSPress ──────────────────
    merge_results = [r for r in results if r["merge_params"] is not None]
    if merge_results and baselines:
        print(f"\n{'='*90}")
        print("MergingPress vs bare DMSPress(KVzapPress)")
        print(f"{'-'*90}")
        print(f"{'Variant':<42} {'Thr':>4} {'Base':>6} {'Merge':>6} {'Delta':>6} {'Time+':>7}")
        print(f"{'-'*90}")

        for r in sorted(merge_results, key=lambda x: (x["threshold"], x["name"])):
            base = baselines.get(r["threshold"])
            if base:
                delta = r["mean_score"] - base["mean_score"]
                time_delta = r["inference_seconds"] - base["inference_seconds"]
                sign = "+" if delta >= 0 else ""
                print(f"{r['name']:<42} {r['threshold']:>4} {base['mean_score']:>6.2f} "
                      f"{r['mean_score']:>6.2f} {sign}{delta:>5.2f} {time_delta:>+6.0f}s")

    # ── Per-task breakdown for qa_1 (the regression we're fixing) ────
    if merge_results:
        print(f"\n{'='*90}")
        print("qa_1 scores (the regression target)")
        print(f"{'-'*90}")
        for r in sorted(results, key=lambda x: (x["threshold"] or 0, x["name"])):
            qa1 = r["per_task"].get("qa_1", 0)
            marker = ""
            if r["merge_params"] is not None and r["threshold"] in baselines:
                delta = qa1 - baselines[r["threshold"]]["per_task"].get("qa_1", 0)
                marker = f"  ({'+' if delta >= 0 else ''}{delta:.1f}pp)"
            print(f"  {r['name']:<42} qa_1={qa1:>6.2f}{marker}")

    # ── Per-task breakdown for best merging variant ──────────────────
    if merge_results:
        best = max(merge_results, key=lambda x: x["mean_score"])
        base = baselines.get(best["threshold"])
        if base:
            print(f"\n{'='*90}")
            print(f"Per-task: {best['name']} vs {base['name']}")
            print(f"{'-'*90}")
            print(f"{'Task':<25} {'Base':>6} {'Merge':>6} {'Delta':>6}")
            print(f"{'-'*90}")
            for task in sorted(best["per_task"].keys()):
                b_score = base["per_task"].get(task, 0)
                m_score = best["per_task"][task]
                delta = m_score - b_score
                sign = "+" if delta >= 0 else ""
                marker = " ***" if abs(delta) >= 3.0 else ""
                print(f"{task:<25} {b_score:>6.1f} {m_score:>6.1f} {sign}{delta:>5.1f}{marker}")

    if errors:
        print(f"\n{'='*90}")
        print(f"ERRORS ({len(errors)}):")
        for e in errors:
            print(f"  {e['name']}: {e['error'][:200]}")

    # Save all results locally
    out_dir = pathlib.Path("evaluation/results_dms_merging_v2")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.json", "w") as f:
        json.dump({"configs": CONFIGS, "results": results, "errors": errors}, f, indent=2)
    print(f"\nResults saved to {out_dir}/results.json")
