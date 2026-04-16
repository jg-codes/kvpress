# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Parameter sweep for MergingPress with KVzap — finding the merge_fraction /
similarity_threshold combo that beats bare KVzap.

Sweeps merge_fraction × similarity_threshold × CR for M(KVzap) and M(AdaKV(KVzap)).
Also runs the bare baselines for direct comparison.

Usage:
    modal run evaluation/modal_merge_sweep.py
    modal run evaluation/modal_merge_sweep.py --fraction 0.1
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
    # Force rebuild with merge_fraction changes
    .run_commands(f"pip install --no-cache-dir 'kvpress @ git+https://github.com/jg-codes/kvpress.git@{BRANCH}'")
    .pip_install("jieba", "bert_score", "fuzzywuzzy", "python-Levenshtein", "nltk", "rouge")
    .pip_install("skorch", "scikit-learn")
    .run_commands(
        f"git clone --branch {BRANCH} --depth 1 https://github.com/jg-codes/kvpress.git /eval_repo",
    )
)

app = modal.App("kvpress-merge-sweep", image=image)
results_vol = modal.Volume.from_name("kvpress-merge-sweep-results", create_if_missing=True)

DATASET = "ruler"
DATA_DIR = "4096"
MODEL = "Qwen/Qwen3-8B"

# ── Sweep grid ──────────────────────────────────────────────────────────
CRS = [0.50, 0.75, 0.875]
# One targeted combo: merge only top 75% by similarity, drop dissimilar tokens
MERGE_FRACTIONS = [0.75]
SIMILARITY_THRESHOLDS = [0.5]
# Focus: bare KVzap vs M(KVzap) only
PRESS_ARCHETYPES = ["m_kvzap"]
BASELINES = ["kvzap_bare"]


def _build_press(archetype: str, merge_fraction: float, similarity_threshold: float):
    """Construct press instance from archetype + sweep params."""
    from kvpress import AdaKVPress, KVzapPress, MergingPress

    if archetype == "m_kvzap":
        return MergingPress(
            KVzapPress(model_type="mlp"),
            merge_fraction=merge_fraction,
            similarity_threshold=similarity_threshold,
        )
    elif archetype == "m_adakv_kvzap":
        return MergingPress(
            AdaKVPress(KVzapPress(model_type="mlp")),
            merge_fraction=merge_fraction,
            similarity_threshold=similarity_threshold,
        )
    raise ValueError(f"Unknown archetype: {archetype}")


def _build_baseline(name: str):
    """Construct baseline press (no merge params)."""
    from kvpress import AdaKVPress, KVzapPress

    if name == "kvzap_bare":
        return KVzapPress(model_type="mlp")
    elif name == "adakv_kvzap":
        return AdaKVPress(KVzapPress(model_type="mlp"))
    raise ValueError(f"Unknown baseline: {name}")


@app.function(
    gpu="A100",
    timeout=14400,
    memory=65536,
    scaledown_window=2,
    secrets=_secrets,
    volumes={"/results": results_vol},
)
def run_one(
    tag: str,
    press_archetype: str,
    cr: float,
    merge_fraction: float,
    similarity_threshold: float,
    is_baseline: bool,
    fraction: float = 0.10,
) -> dict:
    """Run a single sweep point."""
    import glob
    import os
    import sys
    import time

    sys.path.insert(0, "/eval_repo/evaluation")
    os.chdir("/eval_repo/evaluation")

    from evaluate import EvaluationConfig, EvaluationRunner
    from evaluate_registry import PRESS_REGISTRY

    # Build and inject the custom press into the registry
    dynamic_name = f"__sweep__{tag}"
    if is_baseline:
        PRESS_REGISTRY[dynamic_name] = _build_baseline(press_archetype)
    elif press_archetype == "no_press":
        dynamic_name = "no_press"
    else:
        PRESS_REGISTRY[dynamic_name] = _build_press(
            press_archetype, merge_fraction, similarity_threshold
        )

    config = EvaluationConfig(
        dataset=DATASET,
        data_dir=DATA_DIR,
        model=MODEL,
        device="cuda:0",
        press_name=dynamic_name,
        compression_ratio=cr,
        fraction=fraction,
        seed=42,
        output_dir=f"/results/{tag}",
    )

    runner = EvaluationRunner(config)
    t0 = time.monotonic()
    runner.run_evaluation()
    elapsed = round(time.monotonic() - t0, 1)
    results_vol.commit()

    # Collect metrics
    metrics_files = glob.glob(f"/results/{tag}/**/metrics.json", recursive=True)
    metrics = {}
    if metrics_files:
        with open(metrics_files[0]) as f:
            metrics = json.load(f)

    return {
        "tag": tag,
        "press_archetype": press_archetype,
        "cr": cr,
        "merge_fraction": merge_fraction,
        "similarity_threshold": similarity_threshold,
        "is_baseline": is_baseline,
        "metrics": metrics,
        "elapsed_seconds": elapsed,
    }


def flatten_score(val) -> float:
    if isinstance(val, dict):
        return val.get("string_match", val.get("rouge1", val.get("f1", 0.0)))
    return float(val) if val is not None else 0.0


@app.local_entrypoint()
def main(fraction: float = 0.10):
    """Launch MergingPress parameter sweep on Modal."""
    jobs = []

    # no_press baseline
    jobs.append({
        "tag": "no_press__0.000",
        "press_archetype": "no_press",
        "cr": 0.0,
        "merge_fraction": 1.0,
        "similarity_threshold": 0.0,
        "is_baseline": True,
        "fraction": fraction,
    })

    # Bare baselines at each CR
    for bl in BASELINES:
        for cr in CRS:
            tag = f"{bl}__cr{cr:.3f}"
            jobs.append({
                "tag": tag,
                "press_archetype": bl,
                "cr": cr,
                "merge_fraction": 1.0,
                "similarity_threshold": 0.0,
                "is_baseline": True,
                "fraction": fraction,
            })

    # Sweep grid: archetype × merge_fraction × sim_threshold × CR
    for arch in PRESS_ARCHETYPES:
        for mf in MERGE_FRACTIONS:
            for st in SIMILARITY_THRESHOLDS:
                for cr in CRS:
                    tag = f"{arch}__mf{mf:.2f}_st{st:.2f}__cr{cr:.3f}"
                    jobs.append({
                        "tag": tag,
                        "press_archetype": arch,
                        "cr": cr,
                        "merge_fraction": mf,
                        "similarity_threshold": st,
                        "is_baseline": False,
                        "fraction": fraction,
                    })

    n_baselines = 1 + len(BASELINES) * len(CRS)
    n_sweep = len(PRESS_ARCHETYPES) * len(MERGE_FRACTIONS) * len(SIMILARITY_THRESHOLDS) * len(CRS)
    print(f"\n{'='*90}")
    print("MergingPress Parameter Sweep — Modal A100")
    print(f"Model: {MODEL} | Dataset: {DATASET}-{DATA_DIR} | f={fraction}")
    print(f"Baselines: {n_baselines} | Sweep points: {n_sweep} | Total jobs: {len(jobs)}")
    print(f"Grid: archetypes={PRESS_ARCHETYPES}")
    print(f"       merge_fraction={MERGE_FRACTIONS}")
    print(f"       similarity_threshold={SIMILARITY_THRESHOLDS}")
    print(f"       CRs={CRS}")
    print(f"{'='*90}\n")

    # Prepare starmap args
    starmap_args = [
        (j["tag"], j["press_archetype"], j["cr"], j["merge_fraction"],
         j["similarity_threshold"], j["is_baseline"], j["fraction"])
        for j in jobs
    ]

    results = list(run_one.starmap(starmap_args, return_exceptions=True))

    # ── Collect & display ────────────────────────────────────────────────
    output_dir = pathlib.Path("evaluation/results_sweep")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []
    errors = []
    for i, r in enumerate(results):
        j = jobs[i]
        if isinstance(r, Exception):
            errors.append({"tag": j["tag"], "error": str(r)})
            continue
        if "error" in r.get("metrics", {}):
            errors.append({"tag": j["tag"], "error": r["metrics"]["error"]})
            continue

        m = r["metrics"]
        tasks = sorted(m.keys())
        scores = [flatten_score(m[t]) for t in tasks]
        mean = sum(scores) / len(scores) if scores else 0.0
        r["mean"] = round(mean, 2)
        r["per_task"] = {t: round(flatten_score(m[t]), 2) for t in tasks}
        all_results.append(r)

    # Save raw results
    with open(output_dir / "sweep_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # ── Summary table ────────────────────────────────────────────────────
    print(f"\n{'='*100}")
    print(f"{'Tag':<50} {'CR':>5} {'MF':>5} {'ST':>5} {'Mean':>6} {'Time':>6}")
    print(f"{'-'*100}")

    # Sort: baselines first, then by (archetype, cr, -mean)
    for r in sorted(all_results, key=lambda x: (
        not x["is_baseline"], x["press_archetype"], x["cr"], -x["mean"]
    )):
        t = r.get("elapsed_seconds")
        t_str = f"{t:.0f}s" if t else "—"
        mf_str = f"{r['merge_fraction']:.2f}" if not r["is_baseline"] else "—"
        st_str = f"{r['similarity_threshold']:.2f}" if not r["is_baseline"] else "—"
        print(f"{r['tag']:<50} {r['cr']:>5.3f} {mf_str:>5} {st_str:>5} {r['mean']:>6.1f} {t_str:>6}")

    # ── CR=0.75 comparison (the critical compression level) ──────────────
    cr75 = [r for r in all_results if abs(r["cr"] - 0.75) < 0.01]
    if cr75:
        print(f"\n{'='*90}")
        print("CR=0.75 Comparison (sorted by mean score)")
        print(f"{'-'*90}")
        for r in sorted(cr75, key=lambda x: -x["mean"]):
            bl = " [baseline]" if r["is_baseline"] else ""
            print(f"  {r['tag']:<50} mean={r['mean']:>6.1f}{bl}")

    # ── CR=0.875 comparison ──────────────────────────────────────────────
    cr875 = [r for r in all_results if abs(r["cr"] - 0.875) < 0.01]
    if cr875:
        print(f"\n{'='*90}")
        print("CR=0.875 Comparison (sorted by mean score)")
        print(f"{'-'*90}")
        for r in sorted(cr875, key=lambda x: -x["mean"]):
            bl = " [baseline]" if r["is_baseline"] else ""
            print(f"  {r['tag']:<50} mean={r['mean']:>6.1f}{bl}")

    if errors:
        print(f"\n⚠ {len(errors)} errors:")
        for e in errors:
            print(f"  {e['tag']}: {e['error'][:100]}")

    print(f"\nResults saved to {output_dir / 'sweep_results.json'}")
