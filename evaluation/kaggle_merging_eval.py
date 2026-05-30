#!/usr/bin/env python
"""
MergingPress perturbation-gate sweep on Kaggle T4x2.

Self-contained: collects baselines (bare DMSPress) and merging variants
in the same run, so deltas are always valid regardless of inner scorer.

Results saved to /kaggle/working/results_merging.json.
"""

import gc
import json
import os
import random
import re
import sys
import time

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import pipeline as hf_pipeline

# ── Install kvpress from PR branch ─────────────────────────────────────
os.system("pip install -q --force-reinstall 'kvpress @ git+https://github.com/jg-codes/kvpress.git@pr/merging-press'")

import kvpress  # noqa: F401 — registers kv-press-text-generation pipeline
from kvpress import DMSPress, KnormPress, MergingPress

# Try KVzapPress, fall back to KnormPress
try:
    from kvpress import KVzapPress
    _test_press = KVzapPress(model_type="mlp")
    INNER_PRESS_CLS = lambda: KVzapPress(model_type="mlp")
    INNER_NAME = "kvzap_mlp"
    print("Inner scorer: KVzapPress(mlp)")
except Exception as e:
    print(f"KVzapPress unavailable ({e}), using KnormPress")
    INNER_PRESS_CLS = lambda: KnormPress()
    INNER_NAME = "knorm"

# ── Config ──────────────────────────────────────────────────────────────
MODEL = "Qwen/Qwen3-8B"
DATASET_NAME = "simonjegou/ruler"
DATA_DIR = "4096"
FRACTION = 0.10
OUTPUT_DIR = "/kaggle/working"

# Each config: name, threshold, merge_params (None = bare DMS baseline)
CONFIGS = [
    # Baselines: bare DMSPress (no merging)
    {"name": f"bare_dms_{INNER_NAME}_t-3", "threshold": -3, "merge_params": None},
    {"name": f"bare_dms_{INNER_NAME}_t-2", "threshold": -2, "merge_params": None},

    # Merging: default (no perturbation gate)
    {"name": "m_dms_t-3_default", "threshold": -3, "merge_params": {}},

    # Perturbation gate sweep at t=-3
    {"name": "m_dms_t-3_pg0.5", "threshold": -3, "merge_params": {"perturbation_gate": 0.5}},
    {"name": "m_dms_t-3_pg1.0", "threshold": -3, "merge_params": {"perturbation_gate": 1.0}},
    {"name": "m_dms_t-3_pg2.0", "threshold": -3, "merge_params": {"perturbation_gate": 2.0}},

    # Combined: perturbation gate + merge_fraction
    {"name": "m_dms_t-3_pg1.0_mf0.75", "threshold": -3,
     "merge_params": {"perturbation_gate": 1.0, "merge_fraction": 0.75}},

    # Aggressive threshold: more room for merging to recover
    {"name": "m_dms_t-2_pg1.0", "threshold": -2, "merge_params": {"perturbation_gate": 1.0}},
]


# ── RULER scorer (inlined to avoid eval_repo dependency) ────────────────
def string_match_part(preds, refs):
    return round(sum(max(1.0 if r.lower() in p.lower() else 0.0 for r in ref)
                     for p, ref in zip(preds, refs)) / len(preds) * 100, 2)

def string_match_all(preds, refs):
    return round(sum(sum(1.0 if r.lower() in p.lower() else 0.0 for r in ref) / len(ref)
                     for p, ref in zip(preds, refs)) / len(preds) * 100, 2)

def ruler_scorer(df):
    np_pattern = re.compile(r"[\x00-\x1f]")
    df["predicted_answer"] = df["predicted_answer"].apply(lambda x: np_pattern.sub("", x.strip()).strip())
    scores = {}
    for task, df_task in df.groupby("task"):
        fn = string_match_part if task.split("_")[0] == "qa" else string_match_all
        scores[task] = {"string_match": fn(df_task["predicted_answer"].tolist(), df_task["answer"].tolist())}
    return scores


def flatten(val):
    if isinstance(val, dict):
        return val.get("string_match", val.get("rouge1", val.get("f1", 0.0)))
    return float(val) if val is not None else 0.0


def build_press(config):
    threshold = config["threshold"]
    merge_params = config["merge_params"]
    inner = INNER_PRESS_CLS()
    dms = DMSPress(press=inner, threshold=threshold, sliding_window_size=128)

    if merge_params is None:
        # Bare DMS baseline — no merging
        return dms

    return MergingPress(
        dms,
        similarity_threshold=merge_params.get("similarity_threshold", 0.0),
        merge_fraction=merge_params.get("merge_fraction", 1.0),
        value_norm_weighting=merge_params.get("value_norm_weighting", True),
        merge_keys=merge_params.get("merge_keys", False),
        max_merge_per_token=merge_params.get("max_merge_per_token", 0),
        perturbation_gate=merge_params.get("perturbation_gate", 0.0),
    )


def run_one(config, pipe, df_template):
    """Run a single config, return metrics dict."""
    name = config["name"]
    press = build_press(config)
    df = df_template.copy()
    df["predicted_answer"] = None
    df["compression_ratio"] = 0.0

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
            torch.cuda.empty_cache()

    infer_s = round(time.monotonic() - t_infer, 1)

    metrics = ruler_scorer(df)
    tasks = sorted(metrics.keys())
    scores = [flatten(metrics[t]) for t in tasks]
    mean_score = sum(scores) / len(scores) if scores else 0.0

    return {
        "name": name,
        "threshold": config["threshold"],
        "merge_params": config["merge_params"],
        "mean_score": round(mean_score, 2),
        "inference_seconds": infer_s,
        "per_task": {t: round(flatten(metrics[t]), 2) for t in tasks},
        "n_samples": len(df),
        "inner_press": INNER_NAME,
    }


def main():
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    print(f"GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        mem = props.total_memory / 1e9
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)} — {mem:.1f} GB (sm_{props.major}{props.minor})")
        if props.major < 7:
            print(f"ERROR: GPU {i} has compute capability {props.major}.{props.minor} < 7.0")
            sys.exit(1)

    # ── Load model once ─────────────────────────────────────────────
    print(f"\nLoading {MODEL}...")
    t_model = time.monotonic()

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    pipe = hf_pipeline(
        "kv-press-text-generation",
        model=MODEL,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    pipe.model.eval()
    model_load_s = round(time.monotonic() - t_model, 1)
    print(f"Model loaded in {model_load_s}s")

    # ── Load dataset ────────────────────────────────────────────────
    df_full = load_dataset(DATASET_NAME, data_dir=DATA_DIR, split="test").to_pandas()
    df_sample = df_full.sample(frac=FRACTION, random_state=42)
    print(f"Dataset: {len(df_sample)} samples (f={FRACTION})")

    # ── Run configs sequentially ────────────────────────────────────
    results = []
    baselines = {}  # threshold -> result dict for bare DMS
    errors = []

    for i, config in enumerate(CONFIGS):
        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(CONFIGS)}] {config['name']}")
        print(f"{'='*70}")
        try:
            result = run_one(config, pipe, df_sample)
            results.append(result)

            # Track baselines
            if config["merge_params"] is None:
                baselines[config["threshold"]] = result

            # Print immediate comparison
            base = baselines.get(config["threshold"])
            if base and config["merge_params"] is not None:
                delta = result["mean_score"] - base["mean_score"]
                qa1 = result["per_task"].get("qa_1", 0)
                qa1_base = base["per_task"].get("qa_1", 0)
                print(f"  Mean: {result['mean_score']:.2f} (base {base['mean_score']:.2f}, "
                      f"delta {'+' if delta >= 0 else ''}{delta:.2f})")
                print(f"  qa_1: {qa1:.2f} (base {qa1_base:.2f}, "
                      f"delta {'+' if (qa1-qa1_base) >= 0 else ''}{qa1-qa1_base:.2f})")
            else:
                print(f"  Mean: {result['mean_score']:.2f}")

            # Save incremental results
            with open(f"{OUTPUT_DIR}/results_merging.json", "w") as f:
                json.dump({"results": results, "baselines": {str(k): v for k, v in baselines.items()},
                           "errors": errors, "inner_press": INNER_NAME}, f, indent=2)

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            errors.append({"name": config["name"], "error": str(e)})

        gc.collect()
        torch.cuda.empty_cache()

    # ── Final summary ───────────────────────────────────────────────
    print(f"\n{'='*90}")
    print(f"{'Name':<42} {'Score':>6} {'Infer':>7} {'qa_1':>6} {'fwe':>6} {'cwe':>6}")
    print(f"{'-'*90}")

    for r in sorted(results, key=lambda x: (x["threshold"], x["name"])):
        print(f"{r['name']:<42} {r['mean_score']:>6.2f} {r['inference_seconds']:>6.0f}s "
              f"{r['per_task'].get('qa_1', 0):>6.2f} "
              f"{r['per_task'].get('fwe', 0):>6.2f} "
              f"{r['per_task'].get('cwe', 0):>6.2f}")

    print(f"\n{'='*90}")
    print("Delta vs bare DMSPress:")
    print(f"{'Name':<42} {'Thr':>4} {'Base':>6} {'Merge':>6} {'Delta':>6} {'qa1_d':>6}")
    print(f"{'-'*90}")
    for r in sorted(results, key=lambda x: (x["threshold"], x["name"])):
        if r["merge_params"] is None:
            continue
        base = baselines.get(r["threshold"])
        if base:
            d = r["mean_score"] - base["mean_score"]
            qa1_d = r["per_task"].get("qa_1", 0) - base["per_task"].get("qa_1", 0)
            print(f"{r['name']:<42} {r['threshold']:>4} {base['mean_score']:>6.2f} "
                  f"{r['mean_score']:>6.2f} {'+' if d >= 0 else ''}{d:>5.2f} "
                  f"{'+' if qa1_d >= 0 else ''}{qa1_d:>5.2f}")

    if errors:
        print(f"\nErrors: {errors}")

    print(f"\nInner press: {INNER_NAME}")
    print(f"Results saved to {OUTPUT_DIR}/results_merging.json")


if __name__ == "__main__":
    main()
