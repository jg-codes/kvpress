"""
Analyze distractor token diagnostic results from Modal pilot run.

Reads distractor_analysis.json and produces:
- Per-task flip tables (merge_better / merge_worse / same_correct / same_wrong)
- Eviction profile comparison: merge_better vs merge_worse
- DMS score distribution analysis
- Position distribution analysis (lost-in-the-middle)
- Value norm ratio analysis
- Token category breakdown for flipped samples
- At f=1.0: McNemar's test + bootstrap 95% CI

Usage:
    python evaluation/analyze_distractor_results.py [path/to/distractor_analysis.json]

    Default: evaluation/results_distractor/distractor_analysis.json
"""

import json
import sys
from collections import Counter
from pathlib import Path


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def print_flip_table(samples: list[dict], task: str):
    """Print per-sample flip analysis."""
    cats = Counter(s["flip_category"] for s in samples)
    n = len(samples)

    print(f"\n{'='*60}")
    print(f"FLIP TABLE: {task} (n={n})")
    print(f"{'='*60}")
    print(f"  merge_better  (0→1): {cats.get('merge_better', 0):3d}  ({cats.get('merge_better', 0)/n*100:5.1f}%)")
    print(f"  merge_worse   (1→0): {cats.get('merge_worse', 0):3d}  ({cats.get('merge_worse', 0)/n*100:5.1f}%)")
    print(f"  same_correct  (1→1): {cats.get('same_correct', 0):3d}  ({cats.get('same_correct', 0)/n*100:5.1f}%)")
    print(f"  same_wrong    (0→0): {cats.get('same_wrong', 0):3d}  ({cats.get('same_wrong', 0)/n*100:5.1f}%)")
    print(f"  net effect: {cats.get('merge_better', 0) - cats.get('merge_worse', 0):+d} samples")


def print_accuracy_table(samples: list[dict], task: str):
    """Print accuracy comparison."""
    n = len(samples)
    modes = ["no_press", "bare_dms", "merge_dms"]
    accs = {m: sum(s["correct"][m] for s in samples) / n * 100 for m in modes}

    print(f"\n  Accuracy:")
    for m in modes:
        print(f"    {m:12s}: {accs[m]:5.1f}%  ({int(accs[m]*n/100)}/{n})")
    print(f"    M(DMS) vs no_press: {accs['merge_dms'] - accs['no_press']:+.2f}pp")
    print(f"    M(DMS) vs bare_dms: {accs['merge_dms'] - accs['bare_dms']:+.2f}pp")


def print_eviction_profile(samples: list[dict], label: str):
    """Print aggregate eviction diagnostics for a set of samples."""
    diag_samples = [s for s in samples if s.get("diagnostics")]
    if not diag_samples:
        print(f"  {label}: no diagnostics available")
        return

    print(f"\n  {label} (n={len(diag_samples)}):")

    # Merge rate
    merge_rates = [s["diagnostics"]["merge_rate"] for s in diag_samples]
    print(f"    merge_rate: {_mean(merge_rates):.3f}")

    # Similarity
    sims = [s["diagnostics"]["mean_similarity"] for s in diag_samples if s["diagnostics"]["mean_similarity"] is not None]
    if sims:
        print(f"    mean_similarity: {_mean(sims):.4f}")

    # Value norms
    evn = [s["diagnostics"]["mean_evict_value_norm"] for s in diag_samples if s["diagnostics"]["mean_evict_value_norm"] is not None]
    kvn = [s["diagnostics"]["mean_keep_value_norm"] for s in diag_samples if s["diagnostics"]["mean_keep_value_norm"] is not None]
    if evn and kvn:
        ratio = _mean(evn) / _mean(kvn) if _mean(kvn) > 0 else float("inf")
        print(f"    value_norm evict/keep: {_mean(evn):.4f} / {_mean(kvn):.4f}  (ratio: {ratio:.3f})")

    # DMS scores
    dms_e = [s["diagnostics"]["dms_scores_evicted"]["mean"] for s in diag_samples if s["diagnostics"].get("dms_scores_evicted")]
    dms_k = [s["diagnostics"]["dms_scores_kept"]["mean"] for s in diag_samples if s["diagnostics"].get("dms_scores_kept")]
    if dms_e and dms_k:
        print(f"    DMS score evicted/kept: {_mean(dms_e):.4f} / {_mean(dms_k):.4f}")
        margin = [s["diagnostics"]["dms_scores_evicted"]["score_margin_mean"] for s in diag_samples if s["diagnostics"].get("dms_scores_evicted")]
        if margin:
            print(f"    score_margin (threshold - score): {_mean(margin):.4f}")

    # Position distribution
    pos_hists = [s["diagnostics"]["position_histogram"] for s in diag_samples if s["diagnostics"].get("position_histogram")]
    if pos_hists:
        mean_hist = [sum(h[i] for h in pos_hists) / len(pos_hists) for i in range(len(pos_hists[0]))]
        # Find peak bin
        peak_bin = max(range(len(mean_hist)), key=lambda i: mean_hist[i])
        labels = [f"{i*10}-{(i+1)*10}%" for i in range(10)]
        print(f"    position dist: {[f'{v:.3f}' for v in mean_hist]}")
        print(f"    peak eviction zone: {labels[peak_bin]} of context")


def print_flipped_sample_details(samples: list[dict], flip_cat: str, task: str, max_show: int = 5):
    """Print detailed info for specific flipped samples."""
    flipped = [s for s in samples if s["flip_category"] == flip_cat]
    if not flipped:
        return

    print(f"\n  --- {flip_cat.upper()} samples ({task}) ---")
    for s in flipped[:max_show]:
        print(f"  Sample {s['sample_idx']} (dataset_idx={s['dataset_idx']}):")
        print(f"    Q: {s['question'][:100]}...")
        print(f"    Answers: {s['answers'][:3]}")
        print(f"    no_press: '{s['predictions']['no_press'][:80]}'")
        print(f"    merge_dms: '{s['predictions']['merge_dms'][:80]}'")

        d = s.get("diagnostics")
        if d:
            print(f"    evicted: {d['total_evicted']}, merged: {d['total_merged']} ({d['merge_rate']:.1%})")
            if d.get("dms_scores_evicted"):
                ds = d["dms_scores_evicted"]
                print(f"    DMS evicted: mean={ds['mean']}, median={ds['median']}, margin={ds['score_margin_mean']}")
            if d.get("top_evicted_with_text"):
                tokens = d["top_evicted_with_text"][:10]
                print(f"    top evicted tokens:")
                for t in tokens:
                    print(f"      pos={t['position']} ({t.get('rel_position', '?')}) "
                          f"layers={t['n_layers_evicted']} text='{t['token_text']}'")


def mcnemar_test(samples: list[dict]):
    """McNemar's test for M(DMS) vs no_press (only meaningful at f=1.0)."""
    n = len(samples)
    if n < 30:
        print(f"\n  McNemar's test: SKIPPED (n={n} < 30, use f=1.0 for statistical power)")
        return

    # Contingency table
    b = sum(1 for s in samples if s["correct"]["merge_dms"] > s["correct"]["no_press"])  # merge_better
    c = sum(1 for s in samples if s["correct"]["merge_dms"] < s["correct"]["no_press"])  # merge_worse

    if b + c == 0:
        print(f"\n  McNemar's test: no discordant pairs")
        return

    # McNemar's chi-squared (with continuity correction)
    chi2 = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0

    # Approximate p-value from chi2(1)
    import math
    p_value = math.erfc(math.sqrt(chi2 / 2))

    print(f"\n  McNemar's test (M(DMS) vs no_press):")
    print(f"    merge_better (b): {b}")
    print(f"    merge_worse (c):  {c}")
    print(f"    chi2 (corrected): {chi2:.3f}")
    print(f"    p-value (approx): {p_value:.4f}")
    print(f"    {'SIGNIFICANT' if p_value < 0.05 else 'NOT significant'} at α=0.05")


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "evaluation/results_distractor/distractor_analysis.json"

    if not Path(path).exists():
        print(f"File not found: {path}")
        print("Run the Modal analysis first, or provide the correct path.")
        sys.exit(1)

    data = load_results(path)
    analysis = data.get("analysis", {})
    samples = data.get("samples", [])

    print(f"Model: {analysis.get('model', '?')}")
    print(f"Threshold: {analysis.get('threshold', '?')}")
    print(f"Total samples: {len(samples)}")

    tasks = sorted(set(s["task"] for s in samples))

    for task in tasks:
        task_samples = [s for s in samples if s["task"] == task]

        print_flip_table(task_samples, task)
        print_accuracy_table(task_samples, task)

        # Eviction profiles by flip category
        better = [s for s in task_samples if s["flip_category"] == "merge_better"]
        worse = [s for s in task_samples if s["flip_category"] == "merge_worse"]
        same = [s for s in task_samples if s["flip_category"] in ("same_correct", "same_wrong")]

        print_eviction_profile(better, "merge_better eviction profile")
        print_eviction_profile(worse, "merge_worse eviction profile")
        print_eviction_profile(task_samples, "all samples eviction profile")

        # Detailed look at flipped samples
        print_flipped_sample_details(task_samples, "merge_better", task)
        print_flipped_sample_details(task_samples, "merge_worse", task)

        # Statistical test (meaningful at f=1.0)
        mcnemar_test(task_samples)

    # Cross-task comparison
    if len(tasks) > 1:
        print(f"\n{'='*60}")
        print(f"CROSS-TASK COMPARISON")
        print(f"{'='*60}")
        for task in tasks:
            task_samples = [s for s in samples if s["task"] == task]
            diag = [s for s in task_samples if s.get("diagnostics")]
            if diag:
                dms_e = [s["diagnostics"]["dms_scores_evicted"]["mean"] for s in diag if s["diagnostics"].get("dms_scores_evicted")]
                vnorm_e = [s["diagnostics"]["mean_evict_value_norm"] for s in diag if s["diagnostics"]["mean_evict_value_norm"] is not None]
                vnorm_k = [s["diagnostics"]["mean_keep_value_norm"] for s in diag if s["diagnostics"]["mean_keep_value_norm"] is not None]
                mr = [s["diagnostics"]["merge_rate"] for s in diag]
                print(f"  {task}: dms_evict={_mean(dms_e):.4f}, vnorm_ratio={_mean(vnorm_e)/_mean(vnorm_k):.3f}, merge_rate={_mean(mr):.3f}")

    print(f"\n{'='*60}")
    print("HYPOTHESIS ASSESSMENT")
    print(f"{'='*60}")
    print("Check:")
    print("  1. qa_2 merge_better samples: are evicted tokens distractors (low DMS, low vnorm)?")
    print("  2. qa_1 merge_worse samples: are evicted tokens answer-bearing (high vnorm)?")
    print("  3. Position distribution: are evictions concentrated mid-context (lost-in-middle)?")
    print("  4. DMS score margin: are evicted tokens near-threshold (borderline) or far below (obvious junk)?")


if __name__ == "__main__":
    main()
