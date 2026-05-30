#!/usr/bin/env python3
"""
Cross-Scorer MergingPress Evaluation
=====================================
Answers: "Does merge-on-evict improve over hard eviction across different scorers?"

Two modes:
  --analyze    : Mine existing results (no compute needed)
  --run        : Run 1.5B evals (VPS/CPU) then analyze

Examples:
  # Analyze existing 8B results (local machine):
  cd evaluation/
  python cross_scorer_eval.py --analyze

  # Run 1.5B evals on VPS then analyze:
  python cross_scorer_eval.py --run --analyze

  # Custom model/fraction:
  python cross_scorer_eval.py --run --model Qwen/Qwen2.5-1.5B-Instruct --fraction 0.10

  # Only run specific CRs (faster):
  python cross_scorer_eval.py --run --crs 0.50
"""

import argparse
import json
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════

# (bare_press_name, merge_press_name, family_label)
PRESS_PAIRS = [
    ("knorm", "merging_knorm", "KnormPress"),
    ("snapkv", "merging_snapkv", "SnapKV"),
    ("adakv_snapkv", "merging_adakv_snapkv", "AdaKV+SnapKV"),
    ("expected_attention_bare", "merging_expected_attention", "ExpectedAttention"),
    ("kvzap_mlp_head", "merging_kvzap_mlp", "KVZap(head)"),
    ("kvzap_mlp_layer", "merging_adakv_kvzap", "KVZap(AdaKV)"),
]

# Pairs available in the standard evaluate_registry.py (safe to --run)
RUNNABLE_PAIRS = [
    ("knorm", "merging_knorm", "KnormPress"),
    ("snapkv", "merging_snapkv", "SnapKV"),
    ("adakv_snapkv", "merging_adakv_snapkv", "AdaKV+SnapKV"),
]

RULER_TASKS = [
    "cwe", "fwe",
    "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
    "niah_multiquery", "niah_multivalue",
    "niah_single_1", "niah_single_2", "niah_single_3",
    "qa_1", "qa_2", "vt",
]

TASK_CATEGORIES = {
    "retrieval": [
        "fwe", "niah_multikey_1", "niah_multikey_2", "niah_multikey_3",
        "niah_multiquery", "niah_multivalue",
        "niah_single_1", "niah_single_2", "niah_single_3",
    ],
    "qa": ["qa_1", "qa_2"],
    "linguistic": ["cwe", "vt"],
}


# ═══════════════════════════════════════════════════════════════
# Result Discovery
# ═══════════════════════════════════════════════════════════════

def parse_dir_name(dir_name: str) -> Optional[dict]:
    """Parse evaluation result directory name into components.

    Format: {dataset}__{data_dir}__{model}__{press_name}__{cr}[__fractionX.XXX]
    """
    parts = dir_name.split("__")
    if len(parts) < 5:
        return None

    try:
        cr = float(parts[4])
    except ValueError:
        return None

    fraction = None
    for p in parts[5:]:
        if p.startswith("fraction"):
            try:
                fraction = float(p.replace("fraction", ""))
            except ValueError:
                pass

    return {
        "dataset": parts[0],
        "data_dir": parts[1],
        "model": parts[2].replace("--", "/"),
        "press_name": parts[3],
        "cr": cr,
        "fraction": fraction,
    }


def load_metrics(path: Path) -> dict:
    """Load metrics.json and flatten to {task: score}."""
    with open(path) as f:
        raw = json.load(f)
    flat = {}
    for task, val in raw.items():
        if isinstance(val, dict):
            flat[task] = val.get("string_match", next(iter(val.values())))
        else:
            flat[task] = val
    return flat


def discover_results(search_dirs: List[Path]) -> dict:
    """Scan directories for metrics.json, return structured results.

    Returns: {(model, press_name, cr, fraction, data_dir): {task: score}}
    """
    results = {}
    seen_dirs = set()

    for base in search_dirs:
        if not base.exists() or base.resolve() in seen_dirs:
            continue
        seen_dirs.add(base.resolve())

        for metrics_file in base.rglob("metrics.json"):
            rel = metrics_file.relative_to(base)
            parts = list(rel.parts)

            if len(parts) < 2:
                continue

            # Walk up past numbered subdirs (reruns like /1/, /2/)
            dir_name = parts[-2]
            if dir_name.isdigit() and len(parts) >= 3:
                dir_name = parts[-3]

            parsed = parse_dir_name(dir_name)
            if not parsed or parsed["dataset"] != "ruler":
                continue

            key = (
                parsed["model"],
                parsed["press_name"],
                parsed["cr"],
                parsed["fraction"],
                parsed["data_dir"],
            )
            if key not in results:
                results[key] = load_metrics(metrics_file)

    return results


# ═══════════════════════════════════════════════════════════════
# Analysis
# ═══════════════════════════════════════════════════════════════

def find_pairs(results: dict) -> List[dict]:
    """Match bare/merge pairs and compute per-task deltas."""
    pairs = []

    for bare_name, merge_name, family in PRESS_PAIRS:
        for key, bare_metrics in results.items():
            model, press, cr, frac, data_dir = key
            if press != bare_name:
                continue

            merge_key = (model, merge_name, cr, frac, data_dir)
            if merge_key not in results:
                continue

            merge_metrics = results[merge_key]
            common = sorted(set(bare_metrics) & set(merge_metrics) & set(RULER_TASKS))
            if not common:
                continue

            deltas = {t: merge_metrics[t] - bare_metrics[t] for t in common}

            pairs.append({
                "family": family,
                "model": model,
                "cr": cr,
                "fraction": frac,
                "data_dir": data_dir,
                "bare": {t: bare_metrics[t] for t in common},
                "merge": {t: merge_metrics[t] for t in common},
                "deltas": deltas,
                "mean_bare": sum(bare_metrics[t] for t in common) / len(common),
                "mean_merge": sum(merge_metrics[t] for t in common) / len(common),
                "mean_delta": sum(deltas.values()) / len(deltas),
                "n_better": sum(1 for d in deltas.values() if d > 0.5),
                "n_worse": sum(1 for d in deltas.values() if d < -0.5),
                "n_neutral": sum(1 for d in deltas.values() if abs(d) <= 0.5),
            })

            # Category-level deltas
            for cat, tasks in TASK_CATEGORIES.items():
                cat_deltas = [deltas[t] for t in tasks if t in deltas]
                if cat_deltas:
                    pairs[-1][f"mean_{cat}"] = sum(cat_deltas) / len(cat_deltas)

    return pairs


# ═══════════════════════════════════════════════════════════════
# Output
# ═══════════════════════════════════════════════════════════════

def _sign(v):
    return "+" if v >= 0 else ""


def print_summary(pairs: List[dict]):
    """Cross-scorer summary table."""
    print("\n" + "=" * 100)
    print("CROSS-SCORER SUMMARY: merge - bare delta (pp)")
    print("  Positive = merging helps. 'Better/Worse/Neutral' = task count with |delta| > 0.5pp")
    print("=" * 100)

    for model in sorted(set(p["model"] for p in pairs)):
        mp = sorted(
            [p for p in pairs if p["model"] == model],
            key=lambda p: (p["family"], p["cr"]),
        )
        print(f"\n── {model} ──")
        print(
            f"{'Scorer':<20} {'CR':>5} {'f':>5} {'Bare':>7} {'Merge':>7} "
            f"{'Mean Δ':>8} {'Retr Δ':>8} {'QA Δ':>8} "
            f"{'↑':>3} {'↓':>3} {'=':>3}"
        )
        print("-" * 95)

        prev_family = None
        for p in mp:
            if p["family"] != prev_family and prev_family is not None:
                print()
            prev_family = p["family"]

            retr = p.get("mean_retrieval", 0)
            qa = p.get("mean_qa", 0)
            f_str = f"{p['fraction']:.2f}" if p.get("fraction") else "1.0"
            print(
                f"{p['family']:<20} {p['cr']:>5.2f} {f_str:>5} "
                f"{p['mean_bare']:>7.2f} {p['mean_merge']:>7.2f} "
                f"{_sign(p['mean_delta'])}{p['mean_delta']:>6.2f} "
                f"{_sign(retr)}{retr:>6.2f} "
                f"{_sign(qa)}{qa:>6.2f} "
                f"{p['n_better']:>3} {p['n_worse']:>3} {p['n_neutral']:>3}"
            )


def print_per_task(pairs: List[dict]):
    """Per-task deltas across all scorers."""
    if not pairs:
        return

    print("\n" + "=" * 140)
    print("PER-TASK DELTAS (pp): merge - bare")
    print("=" * 140)

    for model in sorted(set(p["model"] for p in pairs)):
        mp = sorted(
            [p for p in pairs if p["model"] == model and "deltas" in p],
            key=lambda p: (p["family"], p["cr"]),
        )
        if not mp:
            continue

        print(f"\n── {model} ──")
        tasks = sorted(set().union(*(p["deltas"].keys() for p in mp)))
        # Short labels that remain unique
        _abbr_map = {
            "cwe": "cwe", "fwe": "fwe", "vt": "vt",
            "qa_1": "qa_1", "qa_2": "qa_2",
            "niah_multikey_1": "mk1", "niah_multikey_2": "mk2",
            "niah_multikey_3": "mk3", "niah_multiquery": "mq",
            "niah_multivalue": "mv", "niah_single_1": "s1",
            "niah_single_2": "s2", "niah_single_3": "s3",
        }
        abbr = {t: _abbr_map.get(t, t[:7]) for t in tasks}

        hdr = f"{'Scorer':<18} {'CR':>4}"
        for t in tasks:
            hdr += f" {abbr[t]:>8}"
        hdr += f" {'MEAN':>7}"
        print(hdr)
        print("-" * len(hdr))

        for p in mp:
            row = f"{p['family'][:18]:<18} {p['cr']:>4.2f}"
            for t in tasks:
                d = p["deltas"].get(t)
                if d is None:
                    row += f" {'---':>8}"
                else:
                    row += f" {_sign(d)}{d:>6.1f}"
            row += f" {_sign(p['mean_delta'])}{p['mean_delta']:>5.1f}"
            print(row)


def print_consistency(pairs: List[dict]):
    """Per-task consistency across all scorer x CR combinations."""
    if not pairs:
        return

    print("\n" + "=" * 80)
    print("TASK CONSISTENCY: how often does merging help each task?")
    print("  Counted across all scorer x CR combinations")
    print("=" * 80)

    counts = defaultdict(lambda: {"better": 0, "worse": 0, "neutral": 0, "total": 0})
    for p in pairs:
        for task, delta in p["deltas"].items():
            counts[task]["total"] += 1
            if delta > 0.5:
                counts[task]["better"] += 1
            elif delta < -0.5:
                counts[task]["worse"] += 1
            else:
                counts[task]["neutral"] += 1

    print(
        f"\n{'Task':<18} {'↑ Better':>10} {'↓ Worse':>10} "
        f"{'= Neutral':>10} {'Total':>7} {'%Better':>9}"
    )
    print("-" * 68)

    for task in RULER_TASKS:
        if task not in counts:
            continue
        c = counts[task]
        pct = 100 * c["better"] / c["total"] if c["total"] > 0 else 0
        print(
            f"{task:<18} {c['better']:>10} {c['worse']:>10} "
            f"{c['neutral']:>10} {c['total']:>7} {pct:>8.0f}%"
        )

    total_b = sum(c["better"] for c in counts.values())
    total_w = sum(c["worse"] for c in counts.values())
    total_n = sum(c["neutral"] for c in counts.values())
    total_t = total_b + total_w + total_n
    print("-" * 68)
    print(
        f"{'OVERALL':<18} {total_b:>10} {total_w:>10} "
        f"{total_n:>10} {total_t:>7} {100 * total_b / total_t:>8.0f}%"
    )


def print_cr_scaling(pairs: List[dict]):
    """Show how merge benefit scales with compression ratio."""
    if not pairs:
        return

    print("\n" + "=" * 70)
    print("CR SCALING: does merging benefit increase with compression?")
    print("=" * 70)

    for model in sorted(set(p["model"] for p in pairs)):
        mp = [p for p in pairs if p["model"] == model]
        families = sorted(set(p["family"] for p in mp))

        print(f"\n── {model} ──")
        print(f"{'Scorer':<20}", end="")
        crs = sorted(set(p["cr"] for p in mp))
        for cr in crs:
            print(f" {'CR=' + str(cr):>10}", end="")
        print()
        print("-" * (20 + 11 * len(crs)))

        for fam in families:
            print(f"{fam:<20}", end="")
            for cr in crs:
                match = [p for p in mp if p["family"] == fam and p["cr"] == cr]
                if match:
                    # Pick highest fraction (most samples) if duplicates
                    best = max(match, key=lambda p: p.get("fraction") or 1.0)
                    d = best["mean_delta"]
                    print(f" {_sign(d)}{d:>8.2f}pp", end="")
                else:
                    print(f" {'---':>10}", end="")
            print()


# ═══════════════════════════════════════════════════════════════
# Runner (VPS mode)
# ═══════════════════════════════════════════════════════════════

def run_evaluations(eval_dir: Path, model: str, fraction: float,
                    crs: List[float], output_dir: str):
    """Run evaluate.py for bare/merge pairs."""
    configs = [("no_press", 0.0)]
    for bare, merge, _ in RUNNABLE_PAIRS:
        for cr in crs:
            configs.append((bare, cr))
            configs.append((merge, cr))

    total = len(configs)
    print(f"\n{'=' * 60}")
    print(f"Running {total} configurations")
    print(f"  Model:    {model}")
    print(f"  Fraction: {fraction}")
    print(f"  CRs:      {crs}")
    print(f"  Output:   {output_dir}")
    print(f"{'=' * 60}")

    t0 = time.time()
    for i, (press, cr) in enumerate(configs, 1):
        elapsed = time.time() - t0
        eta = (elapsed / max(i - 1, 1)) * (total - i + 1) if i > 1 else 0
        print(
            f"\n[{i}/{total}] {press} @ CR={cr}"
            f"  (elapsed: {elapsed / 60:.0f}m, ETA: {eta / 60:.0f}m)"
        )

        cmd = [
            sys.executable, str(eval_dir / "evaluate.py"),
            "--model", model,
            "--dataset", "ruler",
            "--data_dir", "4096",
            "--press_name", press,
            "--compression_ratio", str(cr),
            "--fraction", str(fraction),
            "--output_dir", output_dir,
        ]

        result = subprocess.run(cmd, cwd=str(eval_dir))
        if result.returncode != 0:
            print(f"  WARNING: returned exit code {result.returncode}")

    total_time = time.time() - t0
    print(f"\nAll evaluations done in {total_time / 60:.1f} minutes")


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

def build_search_dirs(eval_dir: Path, extra_dirs: List[str]) -> List[Path]:
    """Build list of directories to scan for results."""
    dirs = []

    # Standard result dirs under evaluation/
    for name in sorted(eval_dir.iterdir()):
        if name.is_dir() and name.name.startswith("results"):
            dirs.append(name)
            # Also check subdirs (e.g. results_targeted/ada/)
            for sub in sorted(name.iterdir()):
                if sub.is_dir():
                    dirs.append(sub)

    # Extra dirs from CLI
    for d in extra_dirs:
        p = Path(d)
        if p.exists():
            dirs.append(p)

    return dirs


def main():
    p = argparse.ArgumentParser(
        description="Cross-scorer MergingPress evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--run", action="store_true", help="Run evaluations (VPS mode)")
    p.add_argument("--analyze", action="store_true", help="Analyze existing results")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--fraction", type=float, default=0.05)
    p.add_argument("--crs", nargs="+", type=float, default=[0.25, 0.50])
    p.add_argument("--output-dir", default="./results_cross_scorer")
    p.add_argument("--extra-dirs", nargs="*", default=[], help="Extra dirs to scan")
    p.add_argument("--save", default="cross_scorer_summary.json")
    args = p.parse_args()

    eval_dir = Path(__file__).parent

    if not args.run and not args.analyze:
        args.analyze = True

    # ── Run evaluations ──
    if args.run:
        run_evaluations(eval_dir, args.model, args.fraction, args.crs, args.output_dir)
        args.analyze = True

    # ── Analyze ──
    if args.analyze:
        search = build_search_dirs(eval_dir, args.extra_dirs + [args.output_dir])
        print(f"\nScanning {len(search)} directories for results...")

        all_results = discover_results(search)
        print(f"Discovered {len(all_results)} result sets")

        # Show what models/presses were found
        models = sorted(set(k[0] for k in all_results))
        presses = sorted(set(k[1] for k in all_results))
        print(f"Models: {models}")
        print(f"Presses: {presses}")

        pairs = find_pairs(all_results)
        print(f"Matched {len(pairs)} bare/merge pairs")

        if not pairs:
            print("\nNo bare/merge pairs found!")
            print("Check that result directories contain matched pairs.")
            print("Expected directory format: ruler__4096__Model__press_name__CR__fractionX.XXX/")
            return

        print_summary(pairs)
        print_per_task(pairs)
        print_consistency(pairs)
        print_cr_scaling(pairs)

        # Save JSON
        save_path = eval_dir / args.save
        summary = {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_pairs": len(pairs),
            "n_models": len(set(p["model"] for p in pairs)),
            "n_scorers": len(set(p["family"] for p in pairs)),
            "pairs": [
                {k: v for k, v in p.items() if k not in ("bare", "merge")}
                for p in pairs
            ],
        }
        with open(save_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nSaved summary to {save_path}")


if __name__ == "__main__":
    main()
