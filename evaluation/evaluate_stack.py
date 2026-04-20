#!/usr/bin/env python
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Stack evaluation: BoltzmannPress x MergingPress x Quantization vs SOTA.

Press axis    : {no_press, boltzmann, merge_boltzmann, merge_dms_t-3}
Cache axis    : {fp16, quanto-4bit, quanto-2bit}  (TurboQuant deferred)
CR axis       : {0.5, 0.75, 0.875}

Fixed for each run: one model, one dataset, one fraction.

Usage:
    # local smoke (tiny model, few samples)
    python evaluate_stack.py --model Qwen/Qwen2.5-0.5B-Instruct --fraction 0.005

    # full run (invoke from Modal wrapper)
    python evaluate_stack.py --model Qwen/Qwen2.5-7B-Instruct --fraction 0.10
"""

import argparse
import gc
import json
import os
import re
import time
from typing import Callable, Optional

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import DynamicCache, QuantizedCache
from transformers import pipeline as hf_pipeline

import kvpress  # noqa: F401 — registers kv-press-text-generation pipeline
from kvpress import BoltzmannPress, DMSPress, MergingPress, RandomPress


DMS_THRESHOLD = -3.0  # SOTA operating point (per CLAUDE.md findings)


# ── Scorers (self-contained RULER) ─────────────────────────────────────
def _string_match_part(preds, refs):
    return round(sum(max(1.0 if r.lower() in p.lower() else 0.0 for r in ref)
                     for p, ref in zip(preds, refs)) / len(preds) * 100, 2)


def _string_match_all(preds, refs):
    return round(sum(sum(1.0 if r.lower() in p.lower() else 0.0 for r in ref) / len(ref)
                     for p, ref in zip(preds, refs)) / len(preds) * 100, 2)


def ruler_score(df):
    cc = re.compile(r"[\x00-\x1f]")
    df = df.copy()
    df["predicted_answer"] = df["predicted_answer"].apply(
        lambda x: cc.sub("", str(x).strip()).strip()
    )
    scores = {}
    for task, df_task in df.groupby("task"):
        fn = _string_match_part if str(task).split("_")[0] == "qa" else _string_match_all
        scores[str(task)] = fn(df_task["predicted_answer"].tolist(), df_task["answer"].tolist())
    return scores


def _flatten(val):
    if isinstance(val, dict):
        return val.get("string_match", val.get("rouge1", val.get("f1", 0.0)))
    return float(val) if val is not None else 0.0


# ── Condition matrix ───────────────────────────────────────────────────
def make_conditions(crs: list[float], caches: list[Optional[int]]) -> list[dict]:
    """Build condition list.

    crs    : list of compression ratios for Boltzmann/MergeBoltzmann (e.g. [0.5, 0.75, 0.875])
    caches : list of cache configs, each either None (fp16) or an int nbits (4 or 2)
    """
    conditions: list[dict] = []

    def cache_tag(c):
        return "fp16" if c is None else f"q{c}"

    # Baselines — no press, every cache (CR axis doesn't apply)
    for cache in caches:
        conditions.append({
            "name": f"no_press_{cache_tag(cache)}",
            "press_fn": lambda: None,
            "cache_nbits": cache,
            "cr": 0.0,
            "group": "baseline",
        })

    # Boltzmann (our scorer) × CR × cache
    for cr in crs:
        for cache in caches:
            conditions.append({
                "name": f"boltzmann_cr{cr}_{cache_tag(cache)}",
                "press_fn": lambda _cr=cr: BoltzmannPress(compression_ratio=_cr),
                "cache_nbits": cache,
                "cr": cr,
                "group": "boltzmann",
            })

    # MergingPress(Boltzmann) × CR × cache — our main compression vehicle
    for cr in crs:
        for cache in caches:
            conditions.append({
                "name": f"merge_boltzmann_cr{cr}_{cache_tag(cache)}",
                "press_fn": lambda _cr=cr: MergingPress(press=BoltzmannPress(compression_ratio=_cr)),
                "cache_nbits": cache,
                "cr": cr,
                "group": "merge_boltzmann",
            })

    # DMSPress(t=-3) SOTA control — threshold not CR, so single CR entry per cache
    for cache in caches:
        conditions.append({
            "name": f"dms_t-3_{cache_tag(cache)}",
            "press_fn": lambda: DMSPress(press=RandomPress(), threshold=DMS_THRESHOLD, sliding_window_size=0),
            "cache_nbits": cache,
            "cr": float("nan"),
            "group": "dms",
        })

    # MergingPress(DMS_t=-3) — shows +0.4pp merging recovery on top of SOTA
    for cache in caches:
        conditions.append({
            "name": f"merge_dms_t-3_{cache_tag(cache)}",
            "press_fn": lambda: MergingPress(press=DMSPress(press=RandomPress(),
                                                            threshold=DMS_THRESHOLD,
                                                            sliding_window_size=0)),
            "cache_nbits": cache,
            "cr": float("nan"),
            "group": "merge_dms",
        })

    return conditions


# ── Single condition runner ────────────────────────────────────────────
def run_one(config: dict, pipe, df_template) -> dict:
    name = config["name"]
    cache_nbits = config["cache_nbits"]
    press = config["press_fn"]()

    df = df_template.copy()
    df["predicted_answer"] = None

    df_grouped = df.groupby("context")
    n_contexts = df["context"].nunique()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    t_infer = time.monotonic()

    compressed_lens: list[int] = []
    input_lens: list[int] = []

    with torch.inference_mode():
        for context, df_group in tqdm(df_grouped, total=n_contexts, desc=name):
            questions = df_group["question"].to_list()
            max_new_tokens = int(df_group["max_new_tokens"].iloc[0])
            answer_prefix = str(df_group["answer_prefix"].iloc[0]) if df_group["answer_prefix"].iloc[0] else ""

            if cache_nbits is None:
                cache = DynamicCache()
            else:
                cache = QuantizedCache(backend="quanto", nbits=cache_nbits, config=pipe.model.config)

            output = pipe(
                context,
                questions=questions,
                answer_prefix=answer_prefix,
                press=press,
                max_new_tokens=max_new_tokens,
                cache=cache,
            )
            df.loc[df_group.index, "predicted_answer"] = output["answers"]
            compressed_lens.append(int(cache.get_seq_length()))
            if not input_lens:
                input_lens.append(int(cache.get_seq_length()))  # first context used as baseline marker
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    infer_s = round(time.monotonic() - t_infer, 1)

    metrics = ruler_score(df)
    task_scores = {t: round(_flatten(v), 2) for t, v in metrics.items()}
    mean_score = round(float(np.mean(list(task_scores.values()))), 2) if task_scores else 0.0
    avg_len = round(float(np.mean(compressed_lens)), 1) if compressed_lens else 0.0

    bits_per_ch = 16 if cache_nbits is None else cache_nbits
    quant_factor = 16.0 / bits_per_ch

    return {
        "name": name,
        "group": config["group"],
        "cache_nbits": cache_nbits,
        "cr": config["cr"],
        "mean_score": mean_score,
        "per_task": task_scores,
        "inference_seconds": infer_s,
        "avg_compressed_len": avg_len,
        "bits_per_channel": bits_per_ch,
        "quantization_factor": round(quant_factor, 3),
        "n_samples": len(df),
    }


# ── Main ───────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--dataset", default="simonjegou/ruler")
    ap.add_argument("--data_dir", default="4096", help="RULER sequence length bucket")
    ap.add_argument("--fraction", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output_dir", default="./results_stack")
    ap.add_argument("--crs", default="0.5,0.75,0.875")
    ap.add_argument("--caches", default="fp16,q4,q2", help="Comma-separated: fp16,q4,q2")
    ap.add_argument("--only_groups", default="",
                    help="If set, only run these comma-sep groups: baseline,boltzmann,merge_boltzmann,dms,merge_dms")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "results.json")

    crs = [float(x) for x in args.crs.split(",") if x.strip()]
    cache_map = {"fp16": None, "q4": 4, "q2": 2}
    caches = [cache_map[c.strip()] for c in args.caches.split(",") if c.strip()]
    only_groups = {g.strip() for g in args.only_groups.split(",") if g.strip()}

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} | VRAM: {props.total_memory/1e9:.1f} GB | sm_{props.major}{props.minor}")
    else:
        print("WARNING: No GPU — slow run, use tiny model + fraction")

    import transformers
    print(f"torch={torch.__version__} transformers={transformers.__version__}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Sanity check: optimum-quanto installed
    if any(c is not None for c in caches):
        try:
            import optimum.quanto  # noqa: F401
            print("optimum-quanto: OK")
        except ImportError:
            print("optimum-quanto not installed — dropping quantized cache conditions")
            caches = [c for c in caches if c is None]

    # Load model (once, reused across all conditions)
    print(f"\nLoading {args.model}...")
    t0 = time.monotonic()
    pipe = hf_pipeline(
        "kv-press-text-generation",
        model=args.model,
        device_map="auto",
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    )
    pipe.model.eval()
    print(f"Model loaded in {time.monotonic()-t0:.1f}s")
    if torch.cuda.is_available():
        print(f"VRAM used: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Load dataset
    df_full = load_dataset(args.dataset, data_dir=args.data_dir, split="test").to_pandas()
    df = df_full.sample(frac=args.fraction, random_state=args.seed) if args.fraction < 1.0 else df_full
    print(f"Dataset: {len(df)} samples (f={args.fraction})")

    conditions = make_conditions(crs, caches)
    if only_groups:
        conditions = [c for c in conditions if c["group"] in only_groups]
    print(f"Conditions: {len(conditions)}")

    results: list[dict] = []
    errors: list[dict] = []

    for i, config in enumerate(conditions):
        print(f"\n{'='*80}\n[{i+1}/{len(conditions)}] {config['name']}  group={config['group']}  "
              f"nbits={config['cache_nbits']}  cr={config['cr']}\n{'='*80}")
        try:
            result = run_one(config, pipe, df)
            results.append(result)
            print(f"  mean={result['mean_score']:.2f}  len={result['avg_compressed_len']:.0f}  "
                  f"bits={result['bits_per_channel']}  t={result['inference_seconds']:.0f}s")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            errors.append({"name": config["name"], "error": str(e)})

        # Incremental save (crash-safe)
        with open(out_path, "w") as f:
            json.dump({
                "model": args.model,
                "dataset": args.dataset,
                "data_dir": args.data_dir,
                "fraction": args.fraction,
                "seed": args.seed,
                "crs": crs,
                "caches": args.caches,
                "dms_threshold": DMS_THRESHOLD,
                "results": results,
                "errors": errors,
            }, f, indent=2)

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summary
    print(f"\n{'='*100}\n SUMMARY — {args.model} | {args.dataset}-{args.data_dir} | "
          f"f={args.fraction}\n{'='*100}")
    hdr = f"{'Condition':<38} {'Mean':>6} {'FWE':>6} {'NIAH1':>6} {'QA1':>6} {'CWE':>6} "\
          f"{'Len':>5} {'bits':>4} {'t(s)':>6}"
    print(hdr)
    print("-" * 100)
    for g in ["baseline", "boltzmann", "merge_boltzmann", "dms", "merge_dms"]:
        for r in [x for x in results if x["group"] == g]:
            t = r["per_task"]
            print(f"{r['name']:<38} {r['mean_score']:>6.1f} "
                  f"{t.get('fwe',0):>6.1f} {t.get('niah_single_1',0):>6.1f} "
                  f"{t.get('qa_1',0):>6.1f} {t.get('cwe',0):>6.1f} "
                  f"{r['avg_compressed_len']:>5.0f} {r['bits_per_channel']:>4} "
                  f"{r['inference_seconds']:>5.0f}")
        print()

    # Effective compression vs fp16 no-press
    rmap = {r["name"]: r for r in results}
    baseline = rmap.get("no_press_fp16")
    if baseline is not None:
        print("─── Effective compression vs fp16 no_press ───")
        for r in results:
            if r["name"] == "no_press_fp16":
                continue
            len_factor = baseline["avg_compressed_len"] / r["avg_compressed_len"] if r["avg_compressed_len"] else 1.0
            eff = r["quantization_factor"] * len_factor
            delta = r["mean_score"] - baseline["mean_score"]
            print(f"  {r['name']:<38} {eff:>6.2f}x  Δmean={delta:+.2f}pp")

    if errors:
        print(f"\nErrors: {json.dumps(errors, indent=2)}")
    print(f"\nResults: {out_path}")


if __name__ == "__main__":
    main()
