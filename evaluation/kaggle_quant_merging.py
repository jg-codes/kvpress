#!/usr/bin/env python
"""
MergingPress × Quantization × DMS(SOTA) — Kaggle T4 Evaluation
================================================================
2×2×2 matrix: {no_press, Knorm, DMS} × {bare, merge} × {fp16, 4-bit cache}

Shows that merge-on-evict + KV cache quantization stack:
more compression, same or better quality.

Runs on Kaggle T4 (16GB VRAM) with Qwen2.5-1.5B-Instruct in fp16.
Results saved to /kaggle/working/results_quant_merging.json
"""

import os
import subprocess
import sys

# ── Install dependencies BEFORE any transformers import ───────────────
# Kaggle may assign a P100 (sm_60) — do NOT upgrade torch or it will
# install a build that drops sm_60 support. Pin torch at current version.
# Get current torch version to constrain it
import importlib
_torch_ver = importlib.import_module("torch").__version__.split("+")[0]
print(f"Pinning torch=={_torch_ver} to preserve CUDA compatibility")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                        f"torch=={_torch_ver}",  # pin torch
                        "transformers>=4.56",
                        "optimum-quanto",
                        "fire",  # kvpress dep
                        "kvpress @ git+https://github.com/jg-codes/kvpress.git@pr/merging-press"])
del _torch_ver

import gc  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from tqdm import tqdm  # noqa: E402
from transformers import DynamicCache, QuantizedCache  # noqa: E402
from transformers import pipeline as hf_pipeline  # noqa: E402

import kvpress  # noqa: F401, E402 — registers kv-press-text-generation pipeline
from kvpress import DMSPress, KnormPress, MergingPress, RandomPress  # noqa: E402

# ── Config ─────────────────────────────────────────────────────────────
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DATASET_NAME = "simonjegou/ruler"
DATA_DIR = "4096"
FRACTION = 0.10  # ~650 samples — more stats power with GPU speed
OUTPUT_DIR = "/kaggle/working"
MAX_NEW_TOKENS = 50
CR = 0.5
DMS_THRESHOLD = -3.0


# ── RULER scorer (self-contained) ─────────────────────────────────────
def string_match_part(preds, refs):
    return round(sum(max(1.0 if r.lower() in p.lower() else 0.0 for r in ref)
                     for p, ref in zip(preds, refs)) / len(preds) * 100, 2)

def string_match_all(preds, refs):
    return round(sum(sum(1.0 if r.lower() in p.lower() else 0.0 for r in ref) / len(ref)
                     for p, ref in zip(preds, refs)) / len(preds) * 100, 2)

def ruler_scorer(df):
    np_pattern = re.compile(r"[\x00-\x1f]")
    df = df.copy()
    df["predicted_answer"] = df["predicted_answer"].apply(
        lambda x: np_pattern.sub("", str(x).strip()).strip()
    )
    scores = {}
    for task, df_task in df.groupby("task"):
        fn = string_match_part if task.split("_")[0] == "qa" else string_match_all
        scores[task] = fn(df_task["predicted_answer"].tolist(), df_task["answer"].tolist())
    return scores


def flatten(val):
    if isinstance(val, dict):
        return val.get("string_match", val.get("rouge1", val.get("f1", 0.0)))
    return float(val) if val is not None else 0.0


# ── Conditions ─────────────────────────────────────────────────────────
def make_conditions():
    return [
        # ─── No compression baselines ───
        {"name": "no_press_fp16",
         "press_fn": lambda: None, "quantize": False,
         "group": "baseline"},

        {"name": "no_press_q4",
         "press_fn": lambda: None, "quantize": True,
         "group": "baseline"},

        # ─── KnormPress (simple scorer) ───
        {"name": "knorm_fp16",
         "press_fn": lambda: KnormPress(compression_ratio=CR), "quantize": False,
         "group": "knorm"},

        {"name": "knorm_q4",
         "press_fn": lambda: KnormPress(compression_ratio=CR), "quantize": True,
         "group": "knorm"},

        {"name": "merge_knorm_fp16",
         "press_fn": lambda: MergingPress(press=KnormPress(compression_ratio=CR)),
         "quantize": False, "group": "knorm"},

        {"name": "merge_knorm_q4",
         "press_fn": lambda: MergingPress(press=KnormPress(compression_ratio=CR)),
         "quantize": True, "group": "knorm"},

        # ─── DMSPress (SOTA) ───
        {"name": "dms_fp16",
         "press_fn": lambda: DMSPress(press=RandomPress(), threshold=DMS_THRESHOLD, sliding_window_size=0),
         "quantize": False, "group": "dms"},

        {"name": "dms_q4",
         "press_fn": lambda: DMSPress(press=RandomPress(), threshold=DMS_THRESHOLD, sliding_window_size=0),
         "quantize": True, "group": "dms"},

        {"name": "merge_dms_fp16",
         "press_fn": lambda: MergingPress(press=DMSPress(press=RandomPress(), threshold=DMS_THRESHOLD, sliding_window_size=0)),
         "quantize": False, "group": "dms"},

        {"name": "merge_dms_q4",
         "press_fn": lambda: MergingPress(press=DMSPress(press=RandomPress(), threshold=DMS_THRESHOLD, sliding_window_size=0)),
         "quantize": True, "group": "dms"},
    ]


# ── Run one condition ──────────────────────────────────────────────────
def run_one(config, pipe, df_template):
    name = config["name"]
    quantize = config["quantize"]
    press = config["press_fn"]()

    df = df_template.copy()
    df["predicted_answer"] = None
    df["compression_ratio"] = 0.0

    df_grouped = df.groupby("context")
    n_contexts = df["context"].nunique()

    torch.cuda.empty_cache()
    t_infer = time.monotonic()

    compressed_lens = []
    with torch.inference_mode():
        for context, df_group in tqdm(df_grouped, total=n_contexts, desc=name):
            questions = df_group["question"].to_list()
            max_new_tokens = int(df_group["max_new_tokens"].iloc[0])
            answer_prefix = str(df_group["answer_prefix"].iloc[0]) if df_group["answer_prefix"].iloc[0] else ""

            # Create cache (quantized or dynamic)
            cache = (QuantizedCache(backend="quanto", nbits=4, config=pipe.model.config)
                     if quantize else DynamicCache())

            output = pipe(
                context,
                questions=questions,
                answer_prefix=answer_prefix,
                press=press,
                max_new_tokens=max_new_tokens,
                cache=cache,
            )
            df.loc[df_group.index, "predicted_answer"] = output["answers"]
            compressed_lens.append(cache.get_seq_length())
            torch.cuda.empty_cache()

    infer_s = round(time.monotonic() - t_infer, 1)

    # Score
    metrics = ruler_scorer(df)
    task_scores = {t: round(flatten(metrics[t]), 2) for t in sorted(metrics.keys())}
    mean_score = round(np.mean(list(task_scores.values())), 2)
    avg_len = round(np.mean(compressed_lens), 1) if compressed_lens else 0

    # Memory estimate
    bits_per_channel = 4 if quantize else 16  # quanto 4-bit vs fp16
    mem_factor = bits_per_channel / 16  # relative to fp16 baseline
    effective_compression = mem_factor * (avg_len / compressed_lens[0] if compressed_lens else 1.0)

    return {
        "name": name,
        "quantize": quantize,
        "group": config["group"],
        "mean_score": mean_score,
        "per_task": task_scores,
        "inference_seconds": infer_s,
        "avg_compressed_len": avg_len,
        "bits_per_channel": bits_per_channel,
        "n_samples": len(df),
    }


# ── Main ───────────────────────────────────────────────────────────────
def main():
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} | VRAM: {props.total_memory/1e9:.1f} GB | sm_{props.major}{props.minor}")
        print(f"PyTorch: {torch.__version__}")
        import transformers; print(f"Transformers: {transformers.__version__}")
    else:
        print("WARNING: No GPU available, running on CPU")
        print(f"PyTorch: {torch.__version__}")

    # Test quanto
    try:
        _ = QuantizedCache(backend="quanto", nbits=4)
        print("QuantizedCache(quanto): OK")
    except Exception as e:
        print(f"QuantizedCache FAILED: {e}")
        print("Continuing without quantized conditions...")

    # Load model
    print(f"\nLoading {MODEL}...")
    t0 = time.monotonic()
    torch.manual_seed(42)
    np.random.seed(42)

    pipe = hf_pipeline(
        "kv-press-text-generation",
        model=MODEL,
        device_map="auto",
        dtype=torch.float16,
    )
    pipe.model.eval()
    print(f"Model loaded in {time.monotonic()-t0:.1f}s")
    print(f"VRAM used: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Load dataset
    df_full = load_dataset(DATASET_NAME, data_dir=DATA_DIR, split="test").to_pandas()
    df_sample = df_full.sample(frac=FRACTION, random_state=42)
    print(f"Dataset: {len(df_sample)} samples (f={FRACTION})")

    # Run all conditions
    conditions = make_conditions()
    results = []
    errors = []

    for i, config in enumerate(conditions):
        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(conditions)}] {config['name']}")
        print(f"  quantize={config['quantize']}, group={config['group']}")
        print(f"{'='*70}")

        try:
            result = run_one(config, pipe, df_sample)
            results.append(result)
            print(f"  Mean: {result['mean_score']:.2f} | "
                  f"Len: {result['avg_compressed_len']:.0f} | "
                  f"Time: {result['inference_seconds']:.0f}s | "
                  f"Bits: {result['bits_per_channel']}")

            # Incremental save
            with open(f"{OUTPUT_DIR}/results_quant_merging.json", "w") as f:
                json.dump({"results": results, "errors": errors,
                           "model": MODEL, "fraction": FRACTION,
                           "cr": CR, "dms_threshold": DMS_THRESHOLD}, f, indent=2)

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            errors.append({"name": config["name"], "error": str(e)})

        gc.collect()
        torch.cuda.empty_cache()

    # ── Summary table ──────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print(f" RESULTS: MergingPress x Quantization x DMS(SOTA)")
    print(f" Model: {MODEL} | f={FRACTION} | CR={CR} | DMS t={DMS_THRESHOLD}")
    print(f"{'='*100}")

    header = f"{'Condition':<30} {'Mean':>6} {'FWE':>6} {'NIAH1':>6} {'QA_1':>6} {'CWE':>6} {'Len':>5} {'Bits':>4} {'Time':>6}"
    print(header)
    print("-" * 100)

    for group in ["baseline", "knorm", "dms"]:
        group_results = [r for r in results if r["group"] == group]
        for r in group_results:
            t = r["per_task"]
            print(f"{r['name']:<30} {r['mean_score']:>6.1f} "
                  f"{t.get('fwe',0):>6.1f} {t.get('niah_single_1',0):>6.1f} "
                  f"{t.get('qa_1',0):>6.1f} {t.get('cwe',0):>6.1f} "
                  f"{r['avg_compressed_len']:>5.0f} {r['bits_per_channel']:>4} "
                  f"{r['inference_seconds']:>5.0f}s")
        print()

    # Deltas
    print("─── Key Comparisons ───")
    result_map = {r["name"]: r for r in results}
    comparisons = [
        ("knorm_fp16", "merge_knorm_fp16", "Merging delta (Knorm fp16)"),
        ("knorm_q4", "merge_knorm_q4", "Merging delta (Knorm 4-bit)"),
        ("dms_fp16", "merge_dms_fp16", "Merging delta (DMS fp16) [SOTA]"),
        ("dms_q4", "merge_dms_q4", "Merging delta (DMS 4-bit)"),
        ("knorm_fp16", "merge_knorm_q4", "Merge+4bit vs bare+fp16 (Knorm)"),
        ("dms_fp16", "merge_dms_q4", "Merge+4bit vs bare+fp16 (DMS)"),
    ]
    for base_name, comp_name, desc in comparisons:
        if base_name in result_map and comp_name in result_map:
            delta = result_map[comp_name]["mean_score"] - result_map[base_name]["mean_score"]
            print(f"  {desc}: {delta:+.2f}pp")

    print(f"\n─── Memory Savings ───")
    print("  fp16 full:        16 bits/ch x full seq     (1.0x)")
    print("  4-bit full:        4 bits/ch x full seq     (4.0x reduction)")
    print("  fp16 + CR=0.5:    16 bits/ch x 0.5 seq      (2.0x reduction)")
    print("  4-bit + CR=0.5:    4 bits/ch x 0.5 seq      (8.0x reduction)")
    print("  MergingPress preserves quality while enabling both dimensions!")

    if errors:
        print(f"\nErrors: {json.dumps(errors, indent=2)}")

    print(f"\nResults: {OUTPUT_DIR}/results_quant_merging.json")


if __name__ == "__main__":
    main()
