# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Targeted leaderboard + speed evaluation on Modal A100.

Runs only two MergingPress variants (kvzap_mlp and cam stacking)
plus their baselines, with both RULER quality eval and speed profiling.

Usage:
    modal run evaluation/modal_targeted_eval.py
    modal run evaluation/modal_targeted_eval.py --speed-only
    modal run evaluation/modal_targeted_eval.py --leaderboard-only
"""

import json
import os
import pathlib
import time

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
        "transformers>=4.56.0,<5.3",
        "datasets",
        "pandas",
        "numpy",
        "fire",
        "pyyaml",
        "tqdm",
        "accelerate",
    )
    .pip_install(
        "kvpress @ git+https://github.com/jg-codes/kvpress.git@merging-press",
        force_build=True,
    )
    .pip_install("jieba", "bert_score", "fuzzywuzzy", "python-Levenshtein", "nltk", "rouge")
    .run_commands(
        "git clone --branch merging-press --depth 1 https://github.com/jg-codes/kvpress.git /eval_repo",
        force_build=True,
    )
)

app = modal.App("kvpress-targeted-eval", image=image)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATASET = "ruler"
DATA_DIR = "4096"
MODEL = "Qwen/Qwen3-8B"
CRS = [0.25, 0.50, 0.75, 0.875]

# Only merging variants — baselines taken from HF leaderboard
MERGING_PRESSES = ["merging_kvzap_mlp", "merging_cam_knorm"]

# AdaKV fraction benchmark (f=0.1): paired comparison of merge vs non-merge
ADA_PRESSES = [("merging_adakv_snapkv", "adakv_snapkv")]
ADA_FRACTION = 0.1

# Leaderboard baselines (NVIDIA/kvpress-leaderboard HF Space, Qwen3-8B RULER-4096)
LEADERBOARD_BASELINES = {
    "no_press": {0.0: 95.3},
    "knorm": {0.25: 87.2, 0.50: 68.3, 0.75: 32.6, 0.875: 8.9},
}

SPEED_PRESSES = [
    "no_press",
    "merging_kvzap_mlp",
    "merging_cam_knorm",
    "merging_adakv_snapkv",
    "adakv_snapkv",
    "knorm",
]
SPEED_CRS = [0.25, 0.50, 0.75]
N_GENERATE = 50  # tokens to generate for speed test
N_WARMUP = 1
N_RUNS = 3


def build_leaderboard_jobs(presses):
    jobs = []
    for p in presses:
        for cr in CRS:
            jobs.append((p, cr))
    return jobs


def build_ada_jobs():
    """Build jobs for AdaKV fraction benchmark: both merging and baseline."""
    jobs = []
    for merge_name, base_name in ADA_PRESSES:
        for cr in CRS:
            jobs.append((merge_name, cr))
            jobs.append((base_name, cr))
    return jobs


# ---------------------------------------------------------------------------
# Leaderboard eval (one press × CR)
# ---------------------------------------------------------------------------
@app.function(gpu="A100", timeout=3600, memory=65536, scaledown_window=2, secrets=_secrets)
def run_leaderboard(press_name: str, cr: float) -> dict:
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
        fraction=1.0,
        seed=42,
        output_dir=f"/results/{output_tag}",
    )

    runner = EvaluationRunner(config)
    runner.run_evaluation()

    result_files = {}
    base = f"/results/{output_tag}"
    for path in glob.glob(f"{base}/**/*", recursive=True):
        if os.path.isfile(path):
            rel = os.path.relpath(path, "/results")
            with open(path) as f:
                result_files[rel] = f.read()

    metrics_files = glob.glob(f"{base}/**/metrics.json", recursive=True)
    metrics = {}
    if metrics_files:
        with open(metrics_files[0]) as f:
            metrics = json.load(f)

    return {"press_name": press_name, "cr": cr, "metrics": metrics, "files": result_files}


# ---------------------------------------------------------------------------
# Fraction eval (one press × CR, configurable fraction for quick comparison)
# ---------------------------------------------------------------------------
@app.function(gpu="A100", timeout=3600, memory=65536, scaledown_window=2, secrets=_secrets)
def run_fraction_eval(press_name: str, cr: float, fraction: float = 0.1) -> dict:
    import glob
    import os
    import sys

    sys.path.insert(0, "/eval_repo/evaluation")
    os.chdir("/eval_repo/evaluation")

    from evaluate import EvaluationConfig, EvaluationRunner

    output_tag = f"{press_name}__{cr:.3f}__f{fraction}"
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

    result_files = {}
    base = f"/results/{output_tag}"
    for path in glob.glob(f"{base}/**/*", recursive=True):
        if os.path.isfile(path):
            rel = os.path.relpath(path, "/results")
            with open(path) as f:
                result_files[rel] = f.read()

    metrics_files = glob.glob(f"{base}/**/metrics.json", recursive=True)
    metrics = {}
    if metrics_files:
        with open(metrics_files[0]) as f:
            metrics = json.load(f)

    return {
        "press_name": press_name,
        "cr": cr,
        "fraction": fraction,
        "metrics": metrics,
        "files": result_files,
    }


# ---------------------------------------------------------------------------
# Speed eval (one press × CR) — measures prefill + generation latency
# ---------------------------------------------------------------------------
@app.function(gpu="A100", timeout=1800, memory=65536, scaledown_window=2, secrets=_secrets)
def run_speed(press_name: str, cr: float) -> dict:
    import gc

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from kvpress import KVPressTextGenerationPipeline

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        torch_dtype=torch.float16,
        device_map="cuda:0",
    )

    # Build a ~4096-token context
    context = "The quick brown fox jumps over the lazy dog. " * 500
    input_ids = tokenizer.encode(context, return_tensors="pt", truncation=True, max_length=4096)
    context_len = input_ids.shape[1]

    # Get press from registry
    import sys

    sys.path.insert(0, "/eval_repo/evaluation")
    from evaluate_registry import PRESS_REGISTRY

    press = PRESS_REGISTRY.get(press_name)
    if press is not None and hasattr(press, "compression_ratio"):
        press.compression_ratio = cr

    # Handle PrefillDecodingPress CR delegation
    from kvpress import PrefillDecodingPress

    if isinstance(press, PrefillDecodingPress):
        if press.prefilling_press is not None and hasattr(press.prefilling_press, "compression_ratio"):
            press.prefilling_press.compression_ratio = cr

    pipe = KVPressTextGenerationPipeline(model=model, tokenizer=tokenizer, press=press)

    # Warmup
    for _ in range(N_WARMUP):
        pipe(
            tokenizer.decode(input_ids[0]),
            max_new_tokens=5,
            return_full_text=False,
        )
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()

    # Timed runs
    prefill_times = []
    gen_times = []
    peak_mems = []

    for _ in range(N_RUNS):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        output = pipe(
            tokenizer.decode(input_ids[0]),
            max_new_tokens=N_GENERATE,
            return_full_text=False,
        )
        torch.cuda.synchronize()
        total = time.perf_counter() - t0

        peak_mem = torch.cuda.max_memory_allocated() / 1e9

        # Approximate: prefill is bulk of first-token time, gen is the rest
        # We measure total and tokens/sec for now
        # Pipeline output can be [{"generated_text": ...}] or {"generated_text": ...}
        if isinstance(output, list):
            gen_text = output[0]["generated_text"]
        elif isinstance(output, dict):
            gen_text = output["generated_text"]
        else:
            gen_text = str(output)
        gen_tokens = len(tokenizer.encode(gen_text))
        prefill_times.append(total)  # total includes both
        gen_times.append(gen_tokens)
        peak_mems.append(peak_mem)

    avg_time = sum(prefill_times) / len(prefill_times)
    avg_tokens = sum(gen_times) / len(gen_times)
    avg_mem = sum(peak_mems) / len(peak_mems)

    return {
        "press_name": press_name,
        "cr": cr,
        "context_len": context_len,
        "n_generate": N_GENERATE,
        "avg_total_time_s": round(avg_time, 3),
        "avg_tokens_generated": round(avg_tokens, 1),
        "tokens_per_sec": round(avg_tokens / avg_time, 1) if avg_time > 0 else 0,
        "peak_memory_gb": round(avg_mem, 2),
        "n_runs": N_RUNS,
    }


def flatten_score(val):
    if isinstance(val, dict):
        return val.get("string_match", val.get("rouge1", val.get("f1", 0.0)))
    return float(val) if val is not None else 0.0


@app.local_entrypoint()
def main(
    speed_only: bool = False,
    leaderboard_only: bool = False,
    ada_only: bool = False,
):
    output_dir = pathlib.Path("evaluation/results_targeted")
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Leaderboard ---
    if not speed_only and not ada_only:
        jobs = build_leaderboard_jobs(MERGING_PRESSES)
        print(f"\n{'='*80}")
        print(f"Leaderboard Eval: {len(jobs)} jobs | Model: {MODEL} | RULER-{DATA_DIR}")
        print(f"Presses: {MERGING_PRESSES} (baselines from HF leaderboard)")
        print(f"{'='*80}\n")

        results = list(run_leaderboard.starmap(jobs, return_exceptions=True))

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

            for rel_path, content in r.get("files", {}).items():
                file_path = (
                    output_dir / rel_path.split("/", 1)[-1] if "/" in rel_path else output_dir / rel_path
                )
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(content)

            m = r["metrics"]
            tasks = sorted(m.keys())
            scores = [flatten_score(m[t]) for t in tasks]
            mean = sum(scores) / len(scores) if scores else 0.0
            table[label] = {"press_name": press_name, "cr": cr, "mean": round(mean, 2)}

        print(f"\n{'Variant':<45} {'Mean':>6}")
        print(f"{'-'*55}")
        for label, r in sorted(table.items(), key=lambda x: (x[1]["cr"], -x[1]["mean"])):
            print(f"{label:<45} {r['mean']:>6.1f}")

        if errors:
            print(f"\n--- Errors ({len(errors)}) ---")
            for e in errors:
                print(f"  {e['variant']} CR={e['cr']}: {e['error']}")

        # Delta vs leaderboard baselines
        # merging_kvzap_mlp wraps KVzapPress — compare vs knorm (best simple baseline)
        # merging_cam_knorm wraps MergingPress+CAMPress+KnormPress — compare vs knorm
        print(f"\n{'Merging':<30} {'CR':>5} {'knorm':>6} {'Mrg':>6} {'Delta':>6}")
        print(f"{'-'*55}")
        for merge_name in MERGING_PRESSES:
            for cr in CRS:
                m_label = f"{merge_name} (cr={cr:.3f})"
                knorm_mean = LEADERBOARD_BASELINES.get("knorm", {}).get(cr, None)
                if m_label in table and knorm_mean is not None:
                    delta = table[m_label]["mean"] - knorm_mean
                    sig = "+" if delta > 0 else ""
                    print(
                        f"{merge_name:<30} {cr:>5.3f}"
                        f" {knorm_mean:>6.1f}"
                        f" {table[m_label]['mean']:>6.1f}"
                        f" {sig}{delta:>5.1f}"
                    )

        (output_dir / "leaderboard_summary.json").write_text(json.dumps(table, indent=2))

    # --- AdaKV fraction benchmark ---
    if ada_only or (not speed_only and not leaderboard_only):
        ada_jobs = build_ada_jobs()
        print(f"\n{'='*80}")
        print(
            f"AdaKV Fraction Eval: {len(ada_jobs)} jobs | "
            f"Model: {MODEL} | RULER-{DATA_DIR} | f={ADA_FRACTION}"
        )
        print(f"Pairs: {ADA_PRESSES}")
        print(f"{'='*80}\n")

        ada_results = list(
            run_fraction_eval.starmap(
                [(p, cr, ADA_FRACTION) for p, cr in ada_jobs],
                return_exceptions=True,
            )
        )

        ada_table = {}
        ada_errors = []
        for i, r in enumerate(ada_results):
            press_name, cr = ada_jobs[i]
            label = f"{press_name} (cr={cr:.3f})"

            if isinstance(r, Exception):
                ada_errors.append({"variant": press_name, "cr": cr, "error": str(r)})
                continue
            if "error" in r.get("metrics", {}):
                ada_errors.append(
                    {"variant": press_name, "cr": cr, "error": r["metrics"]["error"]}
                )
                continue

            for rel_path, content in r.get("files", {}).items():
                file_path = (
                    output_dir
                    / "ada"
                    / (rel_path.split("/", 1)[-1] if "/" in rel_path else rel_path)
                )
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(content)

            m = r["metrics"]
            tasks = sorted(m.keys())
            scores = [flatten_score(m[t]) for t in tasks]
            mean = sum(scores) / len(scores) if scores else 0.0
            ada_table[label] = {
                "press_name": press_name,
                "cr": cr,
                "mean": round(mean, 2),
                "per_task": {t: flatten_score(m[t]) for t in tasks},
            }

        # Print AdaKV results
        print(f"\n{'Variant':<45} {'Mean':>6}")
        print(f"{'-'*55}")
        for label, r in sorted(
            ada_table.items(), key=lambda x: (x[1]["cr"], -x[1]["mean"])
        ):
            print(f"{label:<45} {r['mean']:>6.1f}")

        # Paired comparison: merging vs baseline
        print(f"\n{'Merging':<30} {'CR':>5} {'Base':>6} {'Mrg':>6} {'Δ':>6} {'Δ%rel':>7}")
        print(f"{'-'*60}")
        for merge_name, base_name in ADA_PRESSES:
            for cr in CRS:
                m_label = f"{merge_name} (cr={cr:.3f})"
                b_label = f"{base_name} (cr={cr:.3f})"
                if m_label in ada_table and b_label in ada_table:
                    m_mean = ada_table[m_label]["mean"]
                    b_mean = ada_table[b_label]["mean"]
                    delta = m_mean - b_mean
                    rel_pct = (delta / b_mean * 100) if b_mean > 0 else 0.0
                    sig = "+" if delta > 0 else ""
                    print(
                        f"{merge_name:<30} {cr:>5.3f}"
                        f" {b_mean:>6.1f} {m_mean:>6.1f}"
                        f" {sig}{delta:>5.1f} {sig}{rel_pct:>6.1f}%"
                    )

        if ada_errors:
            print(f"\n--- AdaKV Errors ({len(ada_errors)}) ---")
            for e in ada_errors:
                print(f"  {e['variant']} CR={e['cr']}: {e['error']}")

        (output_dir / "ada_summary.json").write_text(json.dumps(ada_table, indent=2))

    # --- Speed ---
    if not leaderboard_only and not ada_only:
        speed_jobs = []
        for p in SPEED_PRESSES:
            if p == "no_press":
                speed_jobs.append((p, 0.0))
            else:
                for cr in SPEED_CRS:
                    speed_jobs.append((p, cr))

        print(f"\n{'='*80}")
        print(f"Speed Eval: {len(speed_jobs)} jobs | Model: {MODEL} | Context: 4096")
        print(f"{'='*80}\n")

        speed_results = list(run_speed.starmap(speed_jobs, return_exceptions=True))

        print(f"\n{'Press':<30} {'CR':>5} {'Time(s)':>8} {'Tok/s':>7} {'Mem(GB)':>8}")
        print(f"{'-'*65}")

        speed_table = []
        for i, r in enumerate(speed_results):
            press_name, cr = speed_jobs[i]
            if isinstance(r, Exception):
                print(f"{press_name:<30} {cr:>5.2f} ERROR: {r}")
                continue
            print(
                f"{r['press_name']:<30} {r['cr']:>5.2f}"
                f" {r['avg_total_time_s']:>8.3f}"
                f" {r['tokens_per_sec']:>7.1f}"
                f" {r['peak_memory_gb']:>8.2f}"
            )
            speed_table.append(r)

        # Compute overhead vs no_press
        no_press_time = None
        for r in speed_table:
            if r["press_name"] == "no_press":
                no_press_time = r["avg_total_time_s"]
                break

        if no_press_time:
            print(f"\n{'Press':<30} {'CR':>5} {'Overhead':>10}")
            print(f"{'-'*50}")
            for r in speed_table:
                if r["press_name"] == "no_press":
                    continue
                overhead = (r["avg_total_time_s"] / no_press_time - 1) * 100
                print(f"{r['press_name']:<30} {r['cr']:>5.2f} {overhead:>+9.1f}%")

        (output_dir / "speed_results.json").write_text(json.dumps(speed_table, indent=2))

    print(f"\n{'='*80}")
    print(f"All results saved to {output_dir}/")
    print(f"{'='*80}")
