# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Leaderboard evaluation on Modal — runs MergingPress variants on RULER 4096 / Qwen3-8B.

Produces results in the exact format expected by
https://huggingface.co/spaces/nvidia/kvpress-leaderboard

Usage:
    modal run evaluation/modal_leaderboard.py                              # all merging variants
    modal run evaluation/modal_leaderboard.py --presses merging_snapkv,merging_kvzap_mlp
    modal run evaluation/modal_leaderboard.py --include-baselines          # also run unwrapped scorers
"""

import json

# ---------------------------------------------------------------------------
# Modal image — needs enough VRAM for Qwen3-8B (~16 GB fp16, fits A100-40 GB)
# ---------------------------------------------------------------------------
import os
import pathlib

import modal

_hf_token = os.environ.get("HF_TOKEN", "")
_secrets = [modal.Secret.from_dict({"HF_TOKEN": _hf_token})] if _hf_token else []

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
    .pip_install("kvpress @ git+https://github.com/jg-codes/kvpress.git@merging-press")
    .pip_install("jieba", "bert_score", "fuzzywuzzy", "python-Levenshtein", "nltk", "rouge")
    .run_commands(
        "git clone --branch merging-press --depth 1 https://github.com/jg-codes/kvpress.git /eval_repo",
    )
)

app = modal.App("kvpress-leaderboard", image=image)

# Persistent volume — survives container restarts / orchestrator disconnects
results_vol = modal.Volume.from_name("kvpress-leaderboard-results", create_if_missing=True)

# ---------------------------------------------------------------------------
# Leaderboard config (must match upstream leaderboard.sh)
# ---------------------------------------------------------------------------
DATASET = "ruler"
DATA_DIR = "4096"
MODEL = "Qwen/Qwen3-8B"
CRS = [0.25, 0.50, 0.75, 0.875]

# Our MergingPress variants to submit
MERGING_PRESSES = [
    # Core MergingPress wrapping different scorers
    "merging_knorm",
    "merging_snapkv",
    "merging_critical_snapkv",
    "merging_expected_attention",
    "merging_kvzap_mlp",
    # MergingAdaKVPress
    "merging_adakv_snapkv",
    # PrefillDecoding stacking: MergingPress + CAMPress
    "merging_cam_knorm",
]

# Baselines (the unwrapped scorers, for comparison)
BASELINE_PRESSES = [
    "knorm",
    "snapkv",
    "critical_snapkv",
    "expected_attention",
    "fastkvzip",
    "cam_knorm",
    "cam_streaming_llm",
    "adakv_snapkv",
]


def build_jobs(presses: list[str]) -> list[tuple[str, float]]:
    """Build (press_name, cr) tuples for leaderboard eval."""
    jobs = [("no_press", 0.0)]
    for press in presses:
        for cr in CRS:
            jobs.append((press, cr))
    return jobs


# Need A100-40GB for Qwen3-8B in fp16
@app.function(
    gpu="A100",
    timeout=25200,  # 7 hours — RULER-4096 fraction=1.0 takes 3-6h on A100
    memory=65536,
    scaledown_window=2,
    secrets=_secrets,
    volumes={"/results": results_vol},
)
def run_one(press_name: str, cr: float, fraction: float = 1.0) -> dict:
    """Run a single (variant, CR) leaderboard evaluation."""
    import glob
    import os
    import sys

    sys.path.insert(0, "/eval_repo/evaluation")
    os.chdir("/eval_repo/evaluation")

    from evaluate import EvaluationConfig, EvaluationRunner

    output_tag = f"{press_name}__{cr:.3f}"
    config = EvaluationConfig(
        dataset=DATASET,
        data_dir=DATA_DIR,
        model=MODEL,
        device="cuda:0",
        press_name=press_name,
        compression_ratio=cr,
        fraction=fraction,
        seed=42,
        output_dir=f"/results/{output_tag}",
    )

    runner = EvaluationRunner(config)
    runner.run_evaluation()

    # Persist results to volume so they survive orchestrator disconnects
    results_vol.commit()

    # Collect ALL output files for leaderboard submission
    result_files = {}
    base = f"/results/{output_tag}"
    for path in glob.glob(f"{base}/**/*", recursive=True):
        if os.path.isfile(path):
            rel = os.path.relpath(path, "/results")
            with open(path) as f:
                result_files[rel] = f.read()

    # Also get metrics for summary
    metrics_files = glob.glob(f"{base}/**/metrics.json", recursive=True)
    metrics = {}
    if metrics_files:
        with open(metrics_files[0]) as f:
            metrics = json.load(f)

    return {
        "press_name": press_name,
        "cr": cr,
        "metrics": metrics,
        "files": result_files,
    }


def flatten_score(val) -> float:
    if isinstance(val, dict):
        return val.get("string_match", val.get("rouge1", val.get("f1", 0.0)))
    return float(val) if val is not None else 0.0


@app.local_entrypoint()
def main(
    presses: str = "",
    include_baselines: bool = False,
    fraction: float = 1.0,
):
    """Launch leaderboard evaluation on Modal."""
    if presses:
        press_list = [p.strip() for p in presses.split(",")]
    else:
        press_list = list(MERGING_PRESSES)

    if include_baselines:
        press_list = BASELINE_PRESSES + press_list

    jobs = build_jobs(press_list)
    # Add fraction as third element to each job tuple
    jobs_with_fraction = [(p, cr, fraction) for p, cr in jobs]

    frac_label = f"fraction={fraction}" if fraction < 1.0 else "full dataset"
    print(f"\n{'='*100}")
    print("KVPress Leaderboard Evaluation — Modal A100")
    print(f"Model: {MODEL} | Dataset: {DATASET}-{DATA_DIR} | {frac_label}")
    print(f"Presses: {len(press_list)} | Jobs: {len(jobs)} ({len(press_list)} × {len(CRS)} CRs + no_press)")
    print(f"{'='*100}\n")

    results = list(run_one.starmap(jobs_with_fraction, return_exceptions=True))

    # Save result files to local disk (leaderboard submission format)
    output_dir = pathlib.Path("evaluation/results_lb")
    output_dir.mkdir(parents=True, exist_ok=True)

    table = {}
    errors = []
    for i, r in enumerate(results):
        press_name, cr = jobs[i]
        label = f"{press_name} (cr={cr:.3f})" if press_name != "no_press" else "no_press"

        if isinstance(r, Exception):
            errors.append({"variant": press_name, "cr": cr, "error": str(r)})
            continue
        if "error" in r.get("metrics", {}):
            errors.append({"variant": press_name, "cr": cr, "error": r["metrics"]["error"]})
            continue

        # Save all files for leaderboard submission
        for rel_path, content in r.get("files", {}).items():
            # Strip the output_tag prefix to match leaderboard structure
            # Input: "press_name__0.250/ruler/4096/Qwen/Qwen3-8B/press_name/0.25/..."
            # We just save under results_lb/ preserving the inner structure
            file_path = output_dir / rel_path.split("/", 1)[-1] if "/" in rel_path else output_dir / rel_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)

        m = r["metrics"]
        tasks = sorted(m.keys())
        scores = [flatten_score(m[t]) for t in tasks]
        mean = sum(scores) / len(scores) if scores else 0.0
        table[label] = {"press_name": press_name, "cr": cr, "mean": round(mean, 2), "n_tasks": len(tasks)}

    # Print summary table
    print(f"\n{'='*70}")
    print(f"{'Variant':<45} {'Mean':>6}  {'n':>3}")
    print(f"{'-'*70}")

    for label, r in sorted(table.items(), key=lambda x: (x[1]["cr"], -x[1]["mean"])):
        print(f"{label:<45} {r['mean']:>6.1f}  {r['n_tasks']:>3}")

    if errors:
        print(f"\n--- Errors ({len(errors)}) ---")
        for e in errors:
            print(f"  {e['variant']} CR={e['cr']}: {e['error']}")

    # Comparison: each merging variant vs its base scorer
    print(f"\n{'='*70}")
    print("Orthogonality: MergingPress improvement over base scorer")
    print(f"{'-'*70}")
    print(f"{'Merging Variant':<35} {'CR':>5} {'Base':>6} {'Mrg':>6} {'Δ':>6}")
    print(f"{'-'*70}")

    WRAP_MAP = {
        "merging_knorm": "knorm",
        "merging_snapkv": "snapkv",
        "merging_critical_snapkv": "critical_snapkv",
        "merging_expected_attention": "expected_attention",
        "merging_kvzap_mlp": "kvzap_mlp",
        "merging_adakv_snapkv": "adakv_snapkv",
        "merging_cam_knorm": "cam_knorm",
    }

    for merge_name in press_list:
        base_name = WRAP_MAP.get(merge_name)
        if not base_name:
            continue
        for cr in CRS:
            m_label = f"{merge_name} (cr={cr:.3f})"
            b_label = f"{base_name} (cr={cr:.3f})"
            if m_label in table and b_label in table:
                delta = table[m_label]["mean"] - table[b_label]["mean"]
                sig = "+" if delta > 0 else ""
                b_mean = table[b_label]["mean"]
                m_mean = table[m_label]["mean"]
                print(
                    f"{merge_name:<35} {cr:>5.3f}"
                    f" {b_mean:>6.1f} {m_mean:>6.1f} {sig}{delta:>5.1f}"
                )

    print(f"\n{'='*70}")
    print(f"Results saved to {output_dir}/")
    print("To submit: fork huggingface.co/spaces/nvidia/kvpress-leaderboard")
    print(f"           copy {output_dir}/* into benchmark/ and create PR")
    print(f"{'='*70}")
