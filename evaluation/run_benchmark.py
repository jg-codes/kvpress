#!/usr/bin/env python3
"""
Focused benchmark: Compare KV cache compression methods on LongBench.

Compares: SnapKV, AdaKV+SnapKV, VmaxAdaKV+SnapKV, CriticalAdaKV+SnapKV,
          VmaxCriticalAdaKV+SnapKV at multiple compression ratios.

Runs on CPU with Qwen2-0.5B-Instruct — designed for the wahlbot VPS.
"""

import json
import os
import time
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from transformers import pipeline

from kvpress import (
    AdaKVPress,
    CriticalAdaKVPress,
    SnapKVPress,
    VmaxAdaKVPress,
    VmaxCriticalAdaKVPress,
)


# --- Configuration ---
MODEL = "Qwen/Qwen2-0.5B-Instruct"
DATASET = "Xnhyacinth/LongBench"
DEVICE = "cpu"
FRACTION = 0.1          # 10% of dataset for speed
SEED = 42
MAX_NEW_TOKENS = 64
MAX_CONTEXT_LENGTH = 3800  # Keep context short for CPU feasibility
OUTPUT_DIR = Path("/srv/algo-lab/benchmark_results")

COMPRESSION_RATIOS = [0.5, 0.7]

# Subset of LongBench tasks with shorter contexts
TASKS = [
    "qasper",
    "multifieldqa_en",
    "trec",
    "triviaqa",
    "samsum",
]


def build_presses():
    """Build the press configurations to compare."""
    return {
        "snapkv": SnapKVPress(),
        "adakv_snapkv": AdaKVPress(SnapKVPress()),
        "vmax_adakv_snapkv": VmaxAdaKVPress(SnapKVPress()),
        "critical_adakv_snapkv": CriticalAdaKVPress(SnapKVPress()),
        "vmax_critical_adakv_snapkv": VmaxCriticalAdaKVPress(SnapKVPress()),
    }


def rouge_l(prediction: str, reference: str) -> float:
    """Simple ROUGE-L F1 implementation."""
    pred_tokens = prediction.lower().split()
    ref_tokens = reference.lower().split()
    if not pred_tokens or not ref_tokens:
        return 0.0

    # LCS via DP
    m, n = len(pred_tokens), len(ref_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == ref_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs_len = dp[m][n]

    if lcs_len == 0:
        return 0.0
    precision = lcs_len / m
    recall = lcs_len / n
    return 2 * precision * recall / (precision + recall)


def f1_score(prediction: str, reference: str) -> float:
    """Token-level F1."""
    pred_tokens = set(prediction.lower().split())
    ref_tokens = set(reference.lower().split())
    if not pred_tokens or not ref_tokens:
        return 0.0
    common = pred_tokens & ref_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def score_prediction(predicted: str, answers: list[str], task: str) -> float:
    """Score a prediction against reference answers."""
    if task in ("trec",):
        # Classification: exact match
        return max(float(predicted.strip().lower().startswith(a.lower())) for a in answers)
    elif task in ("samsum",):
        return max(rouge_l(predicted, a) for a in answers)
    else:
        return max(f1_score(predicted, a) for a in answers)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Load model once
    print(f"Loading model {MODEL}...")
    pipe = pipeline(
        "kv-press-text-generation",
        model=MODEL,
        device=DEVICE,
        trust_remote_code=True,
    )
    pipe.model.eval()
    print(f"Model loaded. RAM: {torch.cuda.memory_allocated() / 1e9:.1f}GB" if torch.cuda.is_available()
          else "Model loaded (CPU mode)")

    all_results = []

    for task in TASKS:
        print(f"\n{'='*60}")
        print(f"Task: {task}")
        print(f"{'='*60}")

        # Load dataset
        try:
            ds = load_dataset(DATASET, task, split="test")
        except Exception as e:
            print(f"  Failed to load {task}: {e}")
            continue

        df = ds.to_pandas()

        # Sample
        if FRACTION < 1.0:
            n_samples = max(3, int(len(df) * FRACTION))
            df = df.sample(n=min(n_samples, len(df)), random_state=SEED)
        print(f"  Samples: {len(df)}")

        # Baseline: no compression
        print(f"  Running no_press baseline...")
        baseline_scores = []
        t0 = time.time()
        for _, row in tqdm(df.iterrows(), total=len(df), desc="no_press"):
            context = row["context"]
            question = row.get("input", "")
            answers = list(row["answers"]) if hasattr(row["answers"], '__iter__') and not isinstance(row["answers"], str) else [str(row["answers"])]

            try:
                output = pipe(
                    context,
                    questions=[question],
                    answer_prefix="",
                    press=None,
                    max_new_tokens=MAX_NEW_TOKENS,
                    max_context_length=MAX_CONTEXT_LENGTH,
                )
                predicted = output["answers"][0]
            except Exception as e:
                print(f"    Error: {e}")
                predicted = ""

            score = score_prediction(predicted, answers, task)
            baseline_scores.append(score)

        baseline_mean = np.mean(baseline_scores)
        baseline_time = time.time() - t0
        print(f"  no_press: score={baseline_mean:.4f} time={baseline_time:.1f}s")

        all_results.append({
            "task": task,
            "press": "no_press",
            "compression_ratio": 0.0,
            "score": float(baseline_mean),
            "time_s": baseline_time,
            "n_samples": len(df),
        })

        # Run each press at each compression ratio
        presses = build_presses()
        for press_name, press in presses.items():
            for cr in COMPRESSION_RATIOS:
                press.compression_ratio = cr
                scores = []
                t0 = time.time()

                for _, row in tqdm(df.iterrows(), total=len(df),
                                   desc=f"{press_name} cr={cr}"):
                    context = row["context"]
                    question = row.get("input", "")
                    answers = list(row["answers"]) if hasattr(row["answers"], '__iter__') and not isinstance(row["answers"], str) else [str(row["answers"])]

                    try:
                        output = pipe(
                            context,
                            questions=[question],
                            answer_prefix="",
                            press=press,
                            max_new_tokens=MAX_NEW_TOKENS,
                            max_context_length=MAX_CONTEXT_LENGTH,
                        )
                        predicted = output["answers"][0]
                    except Exception as e:
                        print(f"    Error ({press_name}, cr={cr}): {e}")
                        predicted = ""

                    score = score_prediction(predicted, answers, task)
                    scores.append(score)

                mean_score = np.mean(scores)
                elapsed = time.time() - t0
                print(f"  {press_name} cr={cr}: score={mean_score:.4f} "
                      f"(Δ={mean_score - baseline_mean:+.4f}) time={elapsed:.1f}s")

                all_results.append({
                    "task": task,
                    "press": press_name,
                    "compression_ratio": cr,
                    "score": float(mean_score),
                    "delta_vs_baseline": float(mean_score - baseline_mean),
                    "time_s": elapsed,
                    "n_samples": len(df),
                })

        # Save intermediate results
        results_file = OUTPUT_DIR / "benchmark_results.json"
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nIntermediate results saved to {results_file}")

    # Final summary
    print(f"\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")

    results_df = pd.DataFrame(all_results)
    if len(results_df) > 0:
        # Average across tasks
        summary = results_df.groupby(["press", "compression_ratio"]).agg(
            mean_score=("score", "mean"),
            total_time=("time_s", "sum"),
        ).reset_index()
        print(summary.to_string(index=False))

        summary_file = OUTPUT_DIR / "summary.csv"
        summary.to_csv(summary_file, index=False)
        print(f"\nSummary saved to {summary_file}")

    results_file = OUTPUT_DIR / "benchmark_results.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Full results saved to {results_file}")


if __name__ == "__main__":
    main()
