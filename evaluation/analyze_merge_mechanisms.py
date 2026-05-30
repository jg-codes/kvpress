"""
Phase 2: Mechanism analysis — WHY does merging help/hurt?

Mines the EXP-07 distractor_analysis.json (n=500 qa_1 + 500 qa_2) to understand:
  2A. Prediction text comparison: what changes when merging makes things worse?
  2B. Distribution analysis: do any sample features distinguish help vs hurt?
  2C. Merge-target concentration: are survivors being diluted by too many merges?
  2D. Summary: synthesize findings into testable mechanism hypothesis.

Usage:
    python evaluation/analyze_merge_mechanisms.py
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

DATA_PATH = Path("evaluation/results_distractor/distractor_analysis.json")


def load_data():
    with open(DATA_PATH) as f:
        return json.load(f)


# ── 2A: Prediction Text Comparison ──────────────────────────────────────

def analyze_prediction_changes(samples: list[dict], task: str):
    """Compare bare_dms vs merge_dms predictions for merge_worse and merge_better."""
    task_samples = [s for s in samples if s["task"] == task]

    for flip_cat in ["merge_worse", "merge_better"]:
        flipped = [s for s in task_samples if s["flip_category"] == flip_cat]
        if not flipped:
            continue

        print(f"\n{'=' * 70}")
        print(f"  {task} / {flip_cat} (n={len(flipped)})")
        print(f"{'=' * 70}")

        change_types = Counter()

        for s in flipped:
            bare = s["predictions"]["bare_dms"].strip()
            merge = s["predictions"]["merge_dms"].strip()
            no_p = s["predictions"]["no_press"].strip()
            answers = s["answers"]
            question = s["question"]

            # Classify the change
            if bare.lower() == merge.lower():
                change_type = "IDENTICAL_TEXT"
            elif bare.rstrip(".,:;!? ") == merge.rstrip(".,:;!? "):
                change_type = "PUNCTUATION_ONLY"
            elif any(a.lower() in bare.lower() for a in answers) and not any(
                a.lower() in merge.lower() for a in answers
            ):
                change_type = "LOST_ANSWER" if flip_cat == "merge_worse" else "FOUND_ANSWER"
            elif not any(a.lower() in bare.lower() for a in answers) and any(
                a.lower() in merge.lower() for a in answers
            ):
                change_type = "FOUND_ANSWER" if flip_cat == "merge_worse" else "LOST_ANSWER"
            elif len(merge) > len(bare) * 1.5:
                change_type = "LONGER_OUTPUT"
            elif len(merge) < len(bare) * 0.5:
                change_type = "SHORTER_OUTPUT"
            else:
                change_type = "ENTITY_SUBSTITUTION"

            change_types[change_type] += 1

            # Print details for first 10
            if sum(change_types.values()) <= 10 or change_type in (
                "IDENTICAL_TEXT",
                "PUNCTUATION_ONLY",
            ):
                print(f"\n  [{change_type}] sample {s['sample_idx']}")
                print(f"    Q: {question[:90]}...")
                print(f"    Answers: {answers}")
                print(f"    no_press:  '{no_p[:80]}'")
                print(f"    bare_dms:  '{bare[:80]}'")
                print(f"    merge_dms: '{merge[:80]}'")

        print(f"\n  Change type summary ({task}/{flip_cat}):")
        for ct, count in change_types.most_common():
            pct = count / len(flipped) * 100
            print(f"    {ct}: {count} ({pct:.0f}%)")


# ── 2B: Distribution Analysis by Flip Category ─────────────────────────

def analyze_distributions(samples: list[dict], task: str):
    """Compare diagnostic metrics across flip categories."""
    task_samples = [s for s in samples if s["task"] == task and s.get("diagnostics")]

    categories = ["merge_better", "merge_worse", "same_correct", "same_wrong"]
    metrics = [
        "mean_similarity",
        "mean_evict_value_norm",
        "mean_keep_value_norm",
        "n_context_tokens",
        "merge_rate",
    ]

    print(f"\n{'=' * 70}")
    print(f"  {task}: Diagnostic distributions by flip category")
    print(f"{'=' * 70}")

    # Header
    header = f"{'Metric':<25}"
    for cat in categories:
        n = sum(1 for s in task_samples if s["flip_category"] == cat)
        header += f" {cat[:12]:>12}({n})"
    print(header)
    print("-" * 80)

    for metric in metrics:
        row = f"{metric:<25}"
        for cat in categories:
            cat_samples = [s for s in task_samples if s["flip_category"] == cat]
            if not cat_samples:
                row += f" {'—':>16}"
                continue

            if metric == "n_context_tokens":
                vals = [s[metric] for s in cat_samples]
            elif metric == "merge_rate":
                vals = [s["diagnostics"][metric] for s in cat_samples]
            else:
                vals = [
                    s["diagnostics"][metric]
                    for s in cat_samples
                    if s["diagnostics"].get(metric) is not None
                ]

            if vals:
                mean = sum(vals) / len(vals)
                row += f" {mean:>16.4f}"
            else:
                row += f" {'—':>16}"
        print(row)

    # Value norm ratio (evict/keep)
    row = f"{'vnorm_ratio (e/k)':<25}"
    for cat in categories:
        cat_samples = [s for s in task_samples if s["flip_category"] == cat]
        ratios = []
        for s in cat_samples:
            d = s["diagnostics"]
            if d.get("mean_evict_value_norm") and d.get("mean_keep_value_norm"):
                ratios.append(d["mean_evict_value_norm"] / d["mean_keep_value_norm"])
        if ratios:
            row += f" {sum(ratios)/len(ratios):>16.4f}"
        else:
            row += f" {'—':>16}"
    print(row)

    # DMS score margin
    row = f"{'dms_score_margin':<25}"
    for cat in categories:
        cat_samples = [s for s in task_samples if s["flip_category"] == cat]
        margins = []
        for s in cat_samples:
            d = s["diagnostics"]
            if d.get("dms_scores_evicted") and d["dms_scores_evicted"].get("score_margin_mean"):
                margins.append(d["dms_scores_evicted"]["score_margin_mean"])
        if margins:
            row += f" {sum(margins)/len(margins):>16.4f}"
        else:
            row += f" {'—':>16}"
    print(row)


# ── 2C: Merge-Target Concentration ─────────────────────────────────────

def analyze_concentration(samples: list[dict], task: str):
    """Compute merge-target concentration ratio per flip category."""
    task_samples = [s for s in samples if s["task"] == task and s.get("diagnostics")]

    print(f"\n{'=' * 70}")
    print(f"  {task}: Merge-target concentration")
    print(f"{'=' * 70}")
    print(
        f"  Ratio = n_unique_merge_targets / n_unique_evict_positions"
    )
    print(f"  <1.0 = some survivors absorb multiple evicted tokens")
    print(f"  >1.0 = more targets than evict positions (multi-layer effect)")
    print()

    categories = ["merge_better", "merge_worse", "same_correct", "same_wrong"]

    for cat in categories:
        cat_samples = [s for s in task_samples if s["flip_category"] == cat]
        if not cat_samples:
            continue

        ratios = []
        total_evict = []
        total_merged = []
        for s in cat_samples:
            d = s["diagnostics"]
            if d["n_unique_evict_positions"] > 0:
                ratios.append(d["n_unique_merge_targets"] / d["n_unique_evict_positions"])
            total_evict.append(d["total_evicted"])
            total_merged.append(d["total_merged"])

        mean_ratio = sum(ratios) / len(ratios) if ratios else 0
        mean_evict = sum(total_evict) / len(total_evict) if total_evict else 0
        mean_merged = sum(total_merged) / len(total_merged) if total_merged else 0

        print(
            f"  {cat:<15} (n={len(cat_samples):>3}): "
            f"ratio={mean_ratio:.4f}  "
            f"evicted={mean_evict:.0f}  "
            f"merged={mean_merged:.0f}  "
            f"merge_rate={mean_merged/mean_evict:.1%}" if mean_evict > 0 else ""
        )

    # Per-sample correlation: does concentration predict flip?
    worse = [s for s in task_samples if s["flip_category"] == "merge_worse"]
    better = [s for s in task_samples if s["flip_category"] == "merge_better"]

    if worse and better:
        worse_ratios = [
            s["diagnostics"]["n_unique_merge_targets"] / s["diagnostics"]["n_unique_evict_positions"]
            for s in worse
            if s["diagnostics"]["n_unique_evict_positions"] > 0
        ]
        better_ratios = [
            s["diagnostics"]["n_unique_merge_targets"] / s["diagnostics"]["n_unique_evict_positions"]
            for s in better
            if s["diagnostics"]["n_unique_evict_positions"] > 0
        ]
        print(f"\n  Concentration difference (worse vs better):")
        print(f"    worse  mean={sum(worse_ratios)/len(worse_ratios):.4f} (n={len(worse_ratios)})")
        print(f"    better mean={sum(better_ratios)/len(better_ratios):.4f} (n={len(better_ratios)})")


# ── 2D: Cross-Task Comparison ───────────────────────────────────────────

def cross_task_comparison(samples: list[dict]):
    """Compare diagnostic profiles between qa_1 and qa_2."""
    print(f"\n{'=' * 70}")
    print(f"  Cross-task diagnostic comparison (qa_1 vs qa_2)")
    print(f"{'=' * 70}")

    metrics = [
        "mean_similarity",
        "mean_evict_value_norm",
        "mean_keep_value_norm",
        "n_unique_evict_positions",
        "n_unique_merge_targets",
        "total_evicted",
        "total_merged",
    ]

    for task in ["qa_1", "qa_2"]:
        task_samples = [s for s in samples if s["task"] == task and s.get("diagnostics")]
        print(f"\n  {task} (n={len(task_samples)}):")
        for metric in metrics:
            vals = [s["diagnostics"].get(metric, s.get(metric)) for s in task_samples]
            vals = [v for v in vals if v is not None]
            if vals:
                mean_v = sum(vals) / len(vals)
                print(f"    {metric:<30}: {mean_v:.4f}")


# ── 2E: Scoring Artifact Check ──────────────────────────────────────────

def check_scoring_artifacts(samples: list[dict], task: str):
    """Check if flip categories are genuine or scoring edge cases."""
    task_samples = [s for s in samples if s["task"] == task]

    print(f"\n{'=' * 70}")
    print(f"  {task}: Scoring artifact check")
    print(f"{'=' * 70}")

    # Check merge_worse: is bare_dms really correct and merge_dms really wrong?
    worse = [s for s in task_samples if s["flip_category"] == "merge_worse"]

    artifacts = 0
    genuine = 0
    edge_cases = 0

    for s in worse:
        bare = s["predictions"]["bare_dms"].strip()
        merge = s["predictions"]["merge_dms"].strip()
        answers = s["answers"]

        # Check if bare_dms matches but merge_dms is very close
        bare_matches = [a for a in answers if a.lower() in bare.lower()]
        merge_matches = [a for a in answers if a.lower() in merge.lower()]

        # Punctuation-only difference
        bare_clean = re.sub(r"[^\w\s]", "", bare.lower()).strip()
        merge_clean = re.sub(r"[^\w\s]", "", merge.lower()).strip()

        if bare_clean == merge_clean:
            artifacts += 1
        elif len(bare_matches) > 0 and len(merge_matches) == 0:
            # Genuine: bare has answer, merge doesn't
            # Check if merge is "close" (shares most words)
            bare_words = set(bare.lower().split())
            merge_words = set(merge.lower().split())
            overlap = len(bare_words & merge_words) / max(len(bare_words | merge_words), 1)
            if overlap > 0.8:
                edge_cases += 1
            else:
                genuine += 1
        else:
            edge_cases += 1

    total = len(worse)
    print(f"\n  merge_worse samples (n={total}):")
    print(f"    Genuine behavioral change: {genuine} ({genuine/total*100:.0f}%)")
    print(f"    Scoring edge case (close): {edge_cases} ({edge_cases/total*100:.0f}%)")
    print(f"    Punctuation artifact:      {artifacts} ({artifacts/total*100:.0f}%)")

    # Same analysis for merge_better
    better = [s for s in task_samples if s["flip_category"] == "merge_better"]
    if better:
        b_artifacts = 0
        b_genuine = 0
        b_edge = 0
        for s in better:
            bare = s["predictions"]["bare_dms"].strip()
            merge = s["predictions"]["merge_dms"].strip()

            bare_clean = re.sub(r"[^\w\s]", "", bare.lower()).strip()
            merge_clean = re.sub(r"[^\w\s]", "", merge.lower()).strip()

            if bare_clean == merge_clean:
                b_artifacts += 1
            else:
                answers = s["answers"]
                merge_matches = [a for a in answers if a.lower() in merge.lower()]
                bare_matches = [a for a in answers if a.lower() in bare.lower()]
                if len(merge_matches) > 0 and len(bare_matches) == 0:
                    b_genuine += 1
                else:
                    b_edge += 1

        bt = len(better)
        print(f"\n  merge_better samples (n={bt}):")
        print(f"    Genuine behavioral change: {b_genuine} ({b_genuine/bt*100:.0f}%)")
        print(f"    Scoring edge case (close): {b_edge} ({b_edge/bt*100:.0f}%)")
        print(f"    Punctuation artifact:      {b_artifacts} ({b_artifacts/bt*100:.0f}%)")


# ── Isolate Merging vs DMS Eviction Effect ──────────────────────────────

def isolate_merging_effect(samples: list[dict], task: str):
    """Separate merging's effect from DMS eviction's effect.

    The McNemar test in EXP-07 compared M(DMS) vs no_press. But to isolate
    merging, we need M(DMS) vs bare_DMS. The 'merge_worse' category
    conflates both DMS and merging regressions.
    """
    task_samples = [s for s in samples if s["task"] == task]
    n = len(task_samples)

    print(f"\n{'=' * 70}")
    print(f"  {task}: Isolating merging vs DMS eviction (n={n})")
    print(f"{'=' * 70}")

    # Accuracy per mode
    for mode in ["no_press", "bare_dms", "merge_dms"]:
        acc = sum(s["correct"][mode] for s in task_samples) / n * 100
        print(f"  {mode}: {acc:.1f}%")

    # M(DMS) vs no_press flips (what EXP-07 measured)
    m_vs_np = {"merge_helps": 0, "merge_hurts": 0, "same": 0}
    for s in task_samples:
        if s["correct"]["merge_dms"] > s["correct"]["no_press"]:
            m_vs_np["merge_helps"] += 1
        elif s["correct"]["merge_dms"] < s["correct"]["no_press"]:
            m_vs_np["merge_hurts"] += 1
        else:
            m_vs_np["same"] += 1
    print(f"\n  M(DMS) vs no_press (what McNemar measured):")
    print(f"    helps={m_vs_np['merge_helps']}  hurts={m_vs_np['merge_hurts']}  net={m_vs_np['merge_helps'] - m_vs_np['merge_hurts']}")

    # M(DMS) vs bare_DMS flips (isolates MERGING)
    m_vs_bare = {"merge_helps": 0, "merge_hurts": 0, "same": 0}
    for s in task_samples:
        if s["correct"]["merge_dms"] > s["correct"]["bare_dms"]:
            m_vs_bare["merge_helps"] += 1
        elif s["correct"]["merge_dms"] < s["correct"]["bare_dms"]:
            m_vs_bare["merge_hurts"] += 1
        else:
            m_vs_bare["same"] += 1
    print(f"\n  M(DMS) vs bare_DMS (isolates MERGING effect):")
    print(f"    helps={m_vs_bare['merge_helps']}  hurts={m_vs_bare['merge_hurts']}  net={m_vs_bare['merge_helps'] - m_vs_bare['merge_hurts']}")

    # bare_DMS vs no_press flips (isolates DMS EVICTION)
    dms_vs_np = {"dms_helps": 0, "dms_hurts": 0, "same": 0}
    for s in task_samples:
        if s["correct"]["bare_dms"] > s["correct"]["no_press"]:
            dms_vs_np["dms_helps"] += 1
        elif s["correct"]["bare_dms"] < s["correct"]["no_press"]:
            dms_vs_np["dms_hurts"] += 1
        else:
            dms_vs_np["same"] += 1
    print(f"\n  bare_DMS vs no_press (isolates DMS EVICTION effect):")
    print(f"    helps={dms_vs_np['dms_helps']}  hurts={dms_vs_np['dms_hurts']}  net={dms_vs_np['dms_helps'] - dms_vs_np['dms_hurts']}")

    # Decomposition: which merge_worse cases are DMS-caused vs merge-caused?
    merge_worse_vs_np = [s for s in task_samples if s["flip_category"] == "merge_worse"]
    dms_caused = 0
    merge_caused = 0
    both_wrong = 0
    for s in merge_worse_vs_np:
        bare_correct = s["correct"]["bare_dms"] > 0
        merge_correct = s["correct"]["merge_dms"] > 0
        if not bare_correct and not merge_correct:
            dms_caused += 1  # DMS already broke it, merging didn't change outcome
        elif bare_correct and not merge_correct:
            merge_caused += 1  # bare_dms correct but merging broke it
        else:
            both_wrong += 1

    print(f"\n  Decomposition of merge_worse (n={len(merge_worse_vs_np)}):")
    print(f"    DMS-caused (bare also wrong):      {dms_caused} ({dms_caused/len(merge_worse_vs_np)*100:.0f}%)")
    print(f"    Merge-caused (bare was correct):    {merge_caused} ({merge_caused/len(merge_worse_vs_np)*100:.0f}%)")

    # McNemar for M(DMS) vs bare_DMS
    a = m_vs_bare["merge_helps"]
    b = m_vs_bare["merge_hurts"]
    if a + b > 0:
        chi2 = (abs(a - b) - 1) ** 2 / (a + b) if (a + b) > 0 else 0
        print(f"\n  McNemar M(DMS) vs bare_DMS: chi2={chi2:.2f} (a={a}, b={b})")
        print(f"  {'SIGNIFICANT' if chi2 > 3.84 else 'NOT SIGNIFICANT'} at p<0.05")


# ── Main ────────────────────────────────────────────────────────────────

def main():
    print("Loading EXP-07 data...")
    data = load_data()
    samples = data["samples"]
    print(f"  {len(samples)} samples loaded")
    print(f"  Tasks: {Counter(s['task'] for s in samples)}")
    print(f"  Flips: {Counter(s['flip_category'] for s in samples)}")

    print("\n" + "=" * 70)
    print("  PHASE 2A: PREDICTION TEXT COMPARISON")
    print("=" * 70)
    for task in ["qa_1", "qa_2"]:
        analyze_prediction_changes(samples, task)

    print("\n" + "=" * 70)
    print("  PHASE 2E: SCORING ARTIFACT CHECK")
    print("=" * 70)
    for task in ["qa_1", "qa_2"]:
        check_scoring_artifacts(samples, task)

    print("\n" + "=" * 70)
    print("  PHASE 2B: DISTRIBUTION ANALYSIS BY FLIP CATEGORY")
    print("=" * 70)
    for task in ["qa_1", "qa_2"]:
        analyze_distributions(samples, task)

    print("\n" + "=" * 70)
    print("  PHASE 2C: MERGE-TARGET CONCENTRATION")
    print("=" * 70)
    for task in ["qa_1", "qa_2"]:
        analyze_concentration(samples, task)

    print("\n" + "=" * 70)
    print("  PHASE 2D: CROSS-TASK COMPARISON")
    print("=" * 70)
    cross_task_comparison(samples)

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print("\n" + "=" * 70)
    print("  ISOLATING MERGING vs DMS EVICTION")
    print("=" * 70)
    for task in ["qa_1", "qa_2"]:
        isolate_merging_effect(samples, task)

    print("""
Key questions answered:
  1. Are merge_worse flips genuine behavioral changes or scoring artifacts?
  2. Do any diagnostic metrics distinguish merge_better from merge_worse?
  3. Is merge-target concentration correlated with outcome?
  4. Do qa_1 and qa_2 have systematically different diagnostic profiles?
  5. How much of the regression is from merging vs from DMS eviction?

Next steps (if findings warrant):
  - Phase 3A: Add value_cosine_distance diagnostic (GPU) to test whether
    evicted FWE tokens have low value distinctiveness (merge is lossless)
    and evicted QA tokens have high value distinctiveness (merge destroys signal).
""")


if __name__ == "__main__":
    main()
