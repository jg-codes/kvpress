"""
Distractor token diagnostic analysis — M(DMS(KVzap)) on qa_1 + qa_2 samples.

Hypothesis: compression can surpass no_press on qa_2 by evicting "attention
distractor" tokens.  Evidence: M(DMS) default gets 59.09 vs no_press 56.82
on qa_2 in EXP-06.  Meanwhile qa_1 regresses -2.13pp — comparing both tasks
reveals what makes merging help vs hurt.

This script:
1. Loads qa_1 + qa_2 samples from RULER-4096
2. Runs no_press, bare DMS, and M(DMS) with diagnostics=True
3. Saves per-sample: predictions, correctness, eviction positions, merge stats,
   DMS importance scores, position distribution
4. Saves aggregate analysis: flip table, merge_better vs merge_worse profiles,
   DMS score distributions, position heatmap data

Usage:
    modal run --detach evaluation/modal_distractor_analysis.py
"""

import json
import os
import pathlib
import re

import modal

_hf_token = os.environ.get("HF_TOKEN", "")
_secrets = [modal.Secret.from_dict({"HF_TOKEN": _hf_token})] if _hf_token else []

COMMIT = "654a19796ebc8d8638b7b81e14a60fa7e139e1ec"  # diagnostics + DMS score logging

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "packaging", "setuptools", "wheel",
        "torch==2.5.1",
        "transformers>=4.48,<5.0",
        "datasets", "pandas", "numpy", "tqdm", "accelerate",
        f"kvpress @ git+https://github.com/jg-codes/kvpress.git@{COMMIT}",
        gpu="A100",
    )
)

app = modal.App("kvpress-distractor-analysis-v2", image=image)
results_vol = modal.Volume.from_name("kvpress-distractor-analysis-v2-results", create_if_missing=True)

MODEL = "Qwen/Qwen3-8B"
DATA_DIR = "4096"
TASKS = ["qa_1", "qa_2"]
THRESHOLD = -3


def _string_match_part(pred: str, refs: list[str]) -> float:
    """RULER qa scorer: 1.0 if any reference appears in prediction."""
    return max(1.0 if r.lower() in pred.lower() else 0.0 for r in refs)


def _build_press(mode: str):
    """Build press for a given mode: 'no_press', 'bare_dms', 'merge_dms'."""
    if mode == "no_press":
        return None

    from kvpress import DMSPress, KVzapPress, MergingPress

    dms = DMSPress(press=KVzapPress(model_type="mlp"), threshold=THRESHOLD, sliding_window_size=128)

    if mode == "bare_dms":
        return dms

    if mode == "merge_dms":
        return MergingPress(dms, diagnostics=True)

    raise ValueError(f"Unknown mode: {mode}")


def _summarize_diagnostics(diag_log: list[dict], n_context_tokens: int) -> dict:
    """Compress per-layer diagnostics into a JSON-serializable summary."""
    from collections import Counter

    all_evict_positions = set()
    all_merge_target_positions = set()
    total_evicted = 0
    total_merged = 0
    all_similarities = []
    all_evict_vnorms = []
    all_keep_vnorms = []
    all_dms_scores_evicted = []
    all_dms_scores_kept = []
    per_layer = []

    for layer_entry in diag_log:
        layer_idx = layer_entry["layer_idx"]
        layer_evict_positions = set()
        layer_sims = []
        layer_dms_evict = []
        layer_dms_keep = []

        for hd in layer_entry["per_head"]:
            evict_pos = hd["evict_positions"].tolist()
            layer_evict_positions.update(evict_pos)
            all_evict_positions.update(evict_pos)

            merged_mask = hd["merged_mask"]
            if merged_mask.any():
                merge_targets = hd["merge_targets"][merged_mask].tolist()
                all_merge_target_positions.update(merge_targets)
                sims = hd["similarities"][merged_mask].tolist()
                layer_sims.extend(sims)
                all_similarities.extend(sims)

            all_evict_vnorms.extend(hd["evict_value_norms"].tolist())
            all_keep_vnorms.extend(hd["keep_value_norms"].tolist())

            # DMS scores (from Change 2)
            if "dms_scores_evicted" in hd:
                vals = hd["dms_scores_evicted"].tolist()
                layer_dms_evict.extend(vals)
                all_dms_scores_evicted.extend(vals)
            if "dms_scores_kept" in hd:
                vals = hd["dms_scores_kept"].tolist()
                layer_dms_keep.extend(vals)
                all_dms_scores_kept.extend(vals)

        total_evicted += layer_entry["n_evicted_total"]
        total_merged += layer_entry["n_merged_total"]

        per_layer.append({
            "layer_idx": layer_idx,
            "n_evicted": layer_entry["n_evicted_total"],
            "n_merged": layer_entry["n_merged_total"],
            "n_unique_positions_evicted": len(layer_evict_positions),
            "mean_similarity": round(sum(layer_sims) / len(layer_sims), 4) if layer_sims else None,
            "mean_dms_score_evicted": round(sum(layer_dms_evict) / len(layer_dms_evict), 4) if layer_dms_evict else None,
            "mean_dms_score_kept": round(sum(layer_dms_keep) / len(layer_dms_keep), 4) if layer_dms_keep else None,
        })

    # Position frequency: how many layers evict each position
    position_layer_count = Counter()
    for layer_entry in diag_log:
        layer_positions = set()
        for hd in layer_entry["per_head"]:
            layer_positions.update(hd["evict_positions"].tolist())
        for pos in layer_positions:
            position_layer_count[pos] += 1

    # Top-50 most frequently evicted positions
    top_evicted = position_layer_count.most_common(50)

    # Position distribution: 10-bin histogram normalized by context length
    n_bins = 10
    position_histogram = [0] * n_bins
    for pos, count in position_layer_count.items():
        bin_idx = min(int(pos / n_context_tokens * n_bins), n_bins - 1)
        position_histogram[bin_idx] += count
    total_pos_count = sum(position_histogram)
    if total_pos_count > 0:
        position_histogram = [round(c / total_pos_count, 4) for c in position_histogram]

    # DMS score stats
    def _score_stats(scores):
        if not scores:
            return None
        import statistics
        return {
            "mean": round(statistics.mean(scores), 4),
            "std": round(statistics.stdev(scores), 4) if len(scores) > 1 else 0,
            "min": round(min(scores), 4),
            "max": round(max(scores), 4),
            "median": round(statistics.median(scores), 4),
            "score_margin_mean": round(THRESHOLD - statistics.mean(scores), 4),
        }

    return {
        "total_evicted": total_evicted,
        "total_merged": total_merged,
        "merge_rate": round(total_merged / total_evicted, 4) if total_evicted > 0 else 0,
        "n_unique_evict_positions": len(all_evict_positions),
        "n_unique_merge_targets": len(all_merge_target_positions),
        "mean_similarity": round(sum(all_similarities) / len(all_similarities), 4) if all_similarities else None,
        "mean_evict_value_norm": round(sum(all_evict_vnorms) / len(all_evict_vnorms), 4) if all_evict_vnorms else None,
        "mean_keep_value_norm": round(sum(all_keep_vnorms) / len(all_keep_vnorms), 4) if all_keep_vnorms else None,
        "dms_scores_evicted": _score_stats(all_dms_scores_evicted),
        "dms_scores_kept": _score_stats(all_dms_scores_kept),
        "position_histogram": position_histogram,
        "top_evicted_positions": top_evicted,
        "per_layer_summary": per_layer,
    }


@app.function(
    gpu="A100",
    timeout=14400,
    memory=65536,
    scaledown_window=2,
    secrets=_secrets,
    volumes={"/results": results_vol},
)
def run_analysis():
    """Run distractor token analysis on qa_1 + qa_2 samples."""
    import gc
    import random
    import time

    import numpy as np
    import torch
    from datasets import load_dataset
    from transformers import pipeline as hf_pipeline

    import kvpress
    print(f"kvpress loaded from: {kvpress.__file__}")

    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load model
    print(f"\nLoading {MODEL}...")
    t0 = time.monotonic()
    pipe = hf_pipeline(
        "kv-press-text-generation",
        model=MODEL,
        device_map="auto",
        trust_remote_code=True,
    )
    pipe.model.eval()
    tokenizer = pipe.tokenizer
    print(f"Model loaded in {time.monotonic() - t0:.1f}s")

    # Load dataset — filter to qa_1 + qa_2
    df_full = load_dataset("simonjegou/ruler", data_dir=DATA_DIR, split="test").to_pandas()
    df_qa = df_full[df_full["task"].isin(TASKS)].copy()
    print(f"qa samples: {len(df_qa)} ({', '.join(f'{t}: {(df_qa.task==t).sum()}' for t in TASKS)})")

    np_pat = re.compile(r"[\x00-\x1f]")
    modes = ["no_press", "bare_dms", "merge_dms"]
    sample_results = []

    t_start = time.monotonic()

    for sample_idx, (idx, row) in enumerate(df_qa.iterrows()):
        context = row["context"]
        question = row["question"]
        answer = row["answer"]
        max_new_tokens = row["max_new_tokens"]
        answer_prefix = row["answer_prefix"]
        task = row["task"]

        # Tokenize context for position mapping
        context_tokens = tokenizer.encode(context, add_special_tokens=False)
        token_texts = [tokenizer.decode([t]) for t in context_tokens]
        n_tokens = len(context_tokens)

        sample_data = {
            "sample_idx": sample_idx,
            "dataset_idx": int(idx),
            "task": task,
            "n_context_tokens": n_tokens,
            "question": question,
            "answers": answer,
            "predictions": {},
            "correct": {},
            "diagnostics": None,
        }

        for mode in modes:
            press = _build_press(mode)

            if press is not None and hasattr(press, "clear_diagnostics"):
                press.clear_diagnostics()

            torch.cuda.empty_cache()

            with torch.inference_mode():
                output = pipe(
                    context,
                    questions=[question],
                    answer_prefix=answer_prefix,
                    press=press,
                    max_new_tokens=max_new_tokens,
                )

            pred = np_pat.sub("", output["answers"][0].strip()).strip()
            correct = _string_match_part(pred, answer)

            sample_data["predictions"][mode] = pred
            sample_data["correct"][mode] = correct

            # Collect diagnostics from merge_dms
            if mode == "merge_dms" and hasattr(press, "get_diagnostics"):
                diag = press.get_diagnostics()
                if diag:
                    summary = _summarize_diagnostics(diag, n_tokens)

                    # Map top evicted positions to token text
                    top_pos_with_text = []
                    for pos, count in summary["top_evicted_positions"]:
                        if pos < len(token_texts):
                            top_pos_with_text.append({
                                "position": pos,
                                "rel_position": round(pos / n_tokens, 4),
                                "n_layers_evicted": count,
                                "token_text": token_texts[pos],
                            })
                    summary["top_evicted_with_text"] = top_pos_with_text

                    sample_data["diagnostics"] = summary

            gc.collect()
            torch.cuda.empty_cache()

        # Classify flip category
        c = sample_data["correct"]
        if c["merge_dms"] > c["no_press"]:
            flip = "merge_better"
        elif c["merge_dms"] < c["no_press"]:
            flip = "merge_worse"
        elif c["merge_dms"] == 1.0:
            flip = "same_correct"
        else:
            flip = "same_wrong"
        sample_data["flip_category"] = flip

        # Log
        p = sample_data["predictions"]
        d = sample_data["diagnostics"]
        elapsed = time.monotonic() - t_start
        eta = elapsed / (sample_idx + 1) * (len(df_qa) - sample_idx - 1)

        print(f"\n[{sample_idx+1}/{len(df_qa)}] {task} | {flip.upper()} | {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")
        print(f"  Q: {question[:80]}...")
        print(f"  no_press:  {c['no_press']:.0f}  '{p['no_press'][:60]}'")
        print(f"  bare_dms:  {c['bare_dms']:.0f}  '{p['bare_dms'][:60]}'")
        print(f"  merge_dms: {c['merge_dms']:.0f}  '{p['merge_dms'][:60]}'")
        if d:
            print(f"  evict: {d['total_evicted']} total, {d['total_merged']} merged ({d['merge_rate']:.1%})")
            print(f"  mean sim: {d['mean_similarity']}, vnorm evict/keep: {d['mean_evict_value_norm']}/{d['mean_keep_value_norm']}")
            if d.get("dms_scores_evicted"):
                ds = d["dms_scores_evicted"]
                print(f"  DMS scores evicted: mean={ds['mean']}, median={ds['median']}, margin={ds['score_margin_mean']}")
            if d.get("position_histogram"):
                print(f"  position dist (10 bins): {d['position_histogram']}")
            if d.get("top_evicted_with_text", [])[:5]:
                print(f"  top evicted tokens: {[t['token_text'] for t in d['top_evicted_with_text'][:5]]}")

        sample_results.append(sample_data)

    # ── Aggregate analysis per task ──
    print(f"\n{'='*80}")
    print(f"AGGREGATE RESULTS")
    print(f"{'='*80}")

    analysis = {"model": MODEL, "threshold": THRESHOLD, "per_task": {}}

    for task in TASKS:
        task_samples = [s for s in sample_results if s["task"] == task]
        n = len(task_samples)
        if n == 0:
            continue

        accuracy = {mode: round(sum(s["correct"][mode] for s in task_samples) / n * 100, 2) for mode in modes}
        flip_counts = {}
        for cat in ["merge_better", "merge_worse", "same_correct", "same_wrong"]:
            flip_counts[cat] = sum(1 for s in task_samples if s["flip_category"] == cat)

        # Aggregate diagnostics by flip category
        def _agg_diag(samples):
            diag_samples = [s for s in samples if s.get("diagnostics")]
            if not diag_samples:
                return None
            merge_rates = [s["diagnostics"]["merge_rate"] for s in diag_samples]
            sims = [s["diagnostics"]["mean_similarity"] for s in diag_samples if s["diagnostics"]["mean_similarity"]]
            evict_vn = [s["diagnostics"]["mean_evict_value_norm"] for s in diag_samples if s["diagnostics"]["mean_evict_value_norm"]]
            keep_vn = [s["diagnostics"]["mean_keep_value_norm"] for s in diag_samples if s["diagnostics"]["mean_keep_value_norm"]]
            dms_evict = [s["diagnostics"]["dms_scores_evicted"]["mean"] for s in diag_samples if s["diagnostics"].get("dms_scores_evicted")]
            dms_kept = [s["diagnostics"]["dms_scores_kept"]["mean"] for s in diag_samples if s["diagnostics"].get("dms_scores_kept")]
            pos_hists = [s["diagnostics"]["position_histogram"] for s in diag_samples if s["diagnostics"].get("position_histogram")]

            _mean = lambda xs: round(sum(xs) / len(xs), 4) if xs else None
            mean_pos_hist = None
            if pos_hists:
                mean_pos_hist = [round(sum(h[i] for h in pos_hists) / len(pos_hists), 4) for i in range(10)]

            return {
                "n_samples": len(diag_samples),
                "mean_merge_rate": _mean(merge_rates),
                "mean_similarity": _mean(sims),
                "mean_evict_vnorm": _mean(evict_vn),
                "mean_keep_vnorm": _mean(keep_vn),
                "mean_dms_score_evicted": _mean(dms_evict),
                "mean_dms_score_kept": _mean(dms_kept),
                "mean_position_histogram": mean_pos_hist,
            }

        better = [s for s in task_samples if s["flip_category"] == "merge_better"]
        worse = [s for s in task_samples if s["flip_category"] == "merge_worse"]

        task_analysis = {
            "n_samples": n,
            "accuracy": accuracy,
            "flip_counts": flip_counts,
            "merge_better_profile": _agg_diag(better),
            "merge_worse_profile": _agg_diag(worse),
            "all_samples_profile": _agg_diag(task_samples),
        }
        analysis["per_task"][task] = task_analysis

        print(f"\n--- {task} (n={n}) ---")
        print(f"Accuracy: no_press={accuracy['no_press']}%, bare_dms={accuracy['bare_dms']}%, merge_dms={accuracy['merge_dms']}%")
        print(f"Flips: {flip_counts}")
        if task_analysis["merge_better_profile"]:
            print(f"Merge-better profile: {task_analysis['merge_better_profile']}")
        if task_analysis["merge_worse_profile"]:
            print(f"Merge-worse profile:  {task_analysis['merge_worse_profile']}")
        if task_analysis["all_samples_profile"]:
            ap = task_analysis["all_samples_profile"]
            print(f"Overall DMS: evicted={ap.get('mean_dms_score_evicted')}, kept={ap.get('mean_dms_score_kept')}")
            if ap.get("mean_position_histogram"):
                print(f"Position dist: {ap['mean_position_histogram']}")

    # Strip per-layer details to reduce JSON size
    for s in sample_results:
        if s.get("diagnostics"):
            s["diagnostics"].pop("per_layer_summary", None)

    output = {
        "analysis": analysis,
        "samples": sample_results,
    }

    with open("/results/distractor_analysis.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    results_vol.commit()

    total_time = time.monotonic() - t_start
    print(f"\nTotal inference time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"Results saved to Modal volume")
    return output


@app.local_entrypoint()
def main():
    results = run_analysis.remote()
    out = pathlib.Path("evaluation/results_distractor")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "distractor_analysis.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nLocal copy saved to {out}/distractor_analysis.json")
