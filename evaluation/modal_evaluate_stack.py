# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Modal A100 wrapper for evaluate_stack.py — full stack benchmark.

Press axis    : {no_press, boltzmann, merge_boltzmann, dms_t-3, merge_dms_t-3}
Cache axis    : {fp16, quanto-4bit, quanto-2bit}
CR axis       : {0.5, 0.75, 0.875}   (applies to Boltzmann/MergeBoltzmann only)

Conditions run in parallel via .starmap() across A100 instances.
Each condition writes to the shared Modal volume /results/stack/<name>/.

Usage:
    # smoke (few samples, fp16 only, two groups)
    modal run --detach evaluation/modal_evaluate_stack.py::smoke

    # full RULER-4096 run
    modal run --detach evaluation/modal_evaluate_stack.py --fraction 0.10

    # restrict to specific groups (faster/cheaper)
    modal run --detach evaluation/modal_evaluate_stack.py --fraction 0.10 --only-groups "baseline,boltzmann,merge_boltzmann"
"""

import json
import os
import pathlib

import modal

_hf_token = os.environ.get("HF_TOKEN", "")
_secrets = [modal.Secret.from_dict({"HF_TOKEN": _hf_token})] if _hf_token else []

BRANCH = "dev/boltzmann-stack"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11"
    )
    .apt_install("git")
    .env({"CUDA_HOME": "/usr/local/cuda"})
    .pip_install("packaging", "setuptools", "wheel")
    .run_commands(
        "pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "transformers>=4.56",
        "datasets",
        "pandas",
        "numpy",
        "fire",
        "pyyaml",
        "tqdm",
        "accelerate",
        "optimum-quanto",
    )
    .run_commands(
        f"pip install --no-cache-dir 'kvpress @ git+https://github.com/jg-codes/kvpress.git@{BRANCH}'"
    )
)

app = modal.App("kvpress-stack-eval", image=image)
results_vol = modal.Volume.from_name("kvpress-stack-results", create_if_missing=True)

MODEL = "Qwen/Qwen3-8B"
DATASET = "simonjegou/ruler"
DATA_DIR = "4096"
DMS_THRESHOLD = -3.0


# ─────────── remote condition runner ──────────────────────────────────
@app.function(
    gpu="A100",
    timeout=25200,
    memory=65536,
    scaledown_window=2,
    secrets=_secrets,
    volumes={"/results": results_vol},
)
def run_condition(config: dict, fraction: float) -> dict:
    import gc
    import re
    import time

    import numpy as np
    import torch
    from datasets import load_dataset
    from tqdm import tqdm
    from transformers import DynamicCache, QuantizedCache
    from transformers import pipeline as hf_pipeline

    import kvpress  # noqa: F401
    from kvpress import BoltzmannPress, DMSPress, KVzapPress, MergingPress

    # ─── build press from config ──────────────────────────────────────
    # DMS inner scorer is KVzapPress(mlp) — RandomPress would give [0,1] scores
    # that never cross threshold=-3 (zero eviction bug); matches prior eval scripts.
    def build_press(cfg):
        g, cr = cfg["group"], cfg["cr"]
        if g == "baseline":
            return None
        if g == "boltzmann":
            return BoltzmannPress(compression_ratio=cr)
        if g == "merge_boltzmann":
            return MergingPress(press=BoltzmannPress(compression_ratio=cr))
        if g == "dms":
            return DMSPress(press=KVzapPress(model_type="mlp"),
                            threshold=DMS_THRESHOLD, sliding_window_size=128)
        if g == "merge_dms":
            return MergingPress(press=DMSPress(press=KVzapPress(model_type="mlp"),
                                               threshold=DMS_THRESHOLD, sliding_window_size=128))
        raise ValueError(f"Unknown group: {g}")

    # ─── scorer (self-contained RULER) ────────────────────────────────
    def score_ruler(df):
        cc = re.compile(r"[\x00-\x1f]")
        df = df.copy()
        df["predicted_answer"] = df["predicted_answer"].apply(
            lambda x: cc.sub("", str(x).strip()).strip())
        scores = {}

        def sm_part(preds, refs):
            return round(sum(max(1.0 if r.lower() in p.lower() else 0.0 for r in ref)
                             for p, ref in zip(preds, refs)) / len(preds) * 100, 2)

        def sm_all(preds, refs):
            return round(sum(sum(1.0 if r.lower() in p.lower() else 0.0 for r in ref) / len(ref)
                             for p, ref in zip(preds, refs)) / len(preds) * 100, 2)

        for task, df_task in df.groupby("task"):
            fn = sm_part if str(task).split("_")[0] == "qa" else sm_all
            scores[str(task)] = fn(df_task["predicted_answer"].tolist(), df_task["answer"].tolist())
        return scores

    name = config["name"]
    cache_nbits = config["cache_nbits"]

    # deterministic seeds
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # model
    t_model = time.monotonic()
    pipe = hf_pipeline(
        "kv-press-text-generation",
        model=MODEL,
        device_map="auto",
        dtype=torch.float16,
        trust_remote_code=True,
    )
    pipe.model.eval()
    model_load_s = round(time.monotonic() - t_model, 1)
    vram_after_load = torch.cuda.memory_allocated() / 1e9

    # data
    df = load_dataset(DATASET, data_dir=DATA_DIR, split="test").to_pandas()
    if fraction < 1.0:
        df = df.sample(frac=fraction, random_state=42)

    press = build_press(config)
    df["predicted_answer"] = None

    df_grouped = df.groupby("context")
    n_contexts = df["context"].nunique()

    peak_vram_gb = 0.0
    compressed_lens: list[int] = []

    torch.cuda.empty_cache()
    t_infer = time.monotonic()

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
            peak_vram_gb = max(peak_vram_gb, torch.cuda.max_memory_allocated() / 1e9)
            torch.cuda.empty_cache()

    infer_s = round(time.monotonic() - t_infer, 1)

    metrics = score_ruler(df)
    def flat(v):
        if isinstance(v, dict):
            return v.get("string_match", v.get("rouge1", v.get("f1", 0.0)))
        return float(v) if v is not None else 0.0
    task_scores = {t: round(flat(v), 2) for t, v in metrics.items()}
    mean_score = round(float(np.mean(list(task_scores.values()))), 2) if task_scores else 0.0
    avg_len = round(float(np.mean(compressed_lens)), 1) if compressed_lens else 0.0

    bits_per_ch = 16 if cache_nbits is None else cache_nbits
    quant_factor = 16.0 / bits_per_ch

    # persist per-condition
    cfg_dir = f"/results/stack/{name}"
    os.makedirs(cfg_dir, exist_ok=True)
    df[list(set(df.columns) - {"context"})].to_csv(f"{cfg_dir}/predictions.csv", index=False)
    with open(f"{cfg_dir}/metrics.json", "w") as f:
        json.dump(task_scores, f, indent=2)
    with open(f"{cfg_dir}/run_meta.json", "w") as f:
        json.dump({
            "model": MODEL,
            "dataset": f"{DATASET}::{DATA_DIR}",
            "fraction": fraction,
            "model_load_seconds": model_load_s,
            "inference_seconds": infer_s,
            "vram_after_load_gb": round(vram_after_load, 2),
            "peak_vram_gb": round(peak_vram_gb, 2),
            "avg_compressed_len": avg_len,
            "bits_per_channel": bits_per_ch,
            "quantization_factor": quant_factor,
            "n_samples": len(df),
        }, f, indent=2)
    results_vol.commit()

    del pipe
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "name": name,
        "group": config["group"],
        "cache_nbits": cache_nbits,
        "cr": config["cr"],
        "mean_score": mean_score,
        "per_task": task_scores,
        "model_load_seconds": model_load_s,
        "inference_seconds": infer_s,
        "peak_vram_gb": round(peak_vram_gb, 2),
        "avg_compressed_len": avg_len,
        "bits_per_channel": bits_per_ch,
        "quantization_factor": round(quant_factor, 3),
        "n_samples": len(df),
    }


# ─────────── condition builder (duplicated to avoid Modal import quirks) ──
def _make_conditions(crs, caches):
    conds = []

    def tag(c):
        return "fp16" if c is None else f"q{c}"

    for c in caches:
        conds.append({"name": f"no_press_{tag(c)}", "group": "baseline",
                      "cache_nbits": c, "cr": 0.0})
    for cr in crs:
        for c in caches:
            conds.append({"name": f"boltzmann_cr{cr}_{tag(c)}", "group": "boltzmann",
                          "cache_nbits": c, "cr": cr})
    for cr in crs:
        for c in caches:
            conds.append({"name": f"merge_boltzmann_cr{cr}_{tag(c)}", "group": "merge_boltzmann",
                          "cache_nbits": c, "cr": cr})
    for c in caches:
        conds.append({"name": f"dms_t-3_{tag(c)}", "group": "dms",
                      "cache_nbits": c, "cr": float("nan")})
    for c in caches:
        conds.append({"name": f"merge_dms_t-3_{tag(c)}", "group": "merge_dms",
                      "cache_nbits": c, "cr": float("nan")})
    return conds


# ─────────── entrypoints ──────────────────────────────────────────────
@app.local_entrypoint()
def main(
    fraction: float = 0.10,
    crs: str = "0.5,0.75,0.875",
    caches: str = "fp16,q4,q2",
    only_groups: str = "",
):
    """Launch the full stack sweep on Modal (parallel A100s)."""
    crs_list = [float(x) for x in crs.split(",") if x.strip()]
    cache_map = {"fp16": None, "q4": 4, "q2": 2}
    caches_list = [cache_map[c.strip()] for c in caches.split(",") if c.strip()]

    conds = _make_conditions(crs_list, caches_list)
    if only_groups:
        keep = {g.strip() for g in only_groups.split(",") if g.strip()}
        conds = [c for c in conds if c["group"] in keep]

    print(f"\n{'='*90}\nStack sweep — {MODEL} | {DATASET}::{DATA_DIR} | f={fraction}")
    print(f"CRs: {crs_list} | caches: {caches} | only_groups: {only_groups or 'all'}")
    print(f"Conditions: {len(conds)}\n{'='*90}")
    for c in conds:
        print(f"  {c['name']:<42} group={c['group']:<18} nbits={c['cache_nbits']} cr={c['cr']}")

    print("\nDispatching...")
    args_list = [(c, fraction) for c in conds]
    raw = list(run_condition.starmap(args_list, return_exceptions=True))

    results, errors = [], []
    for i, r in enumerate(raw):
        if isinstance(r, Exception):
            errors.append({"name": conds[i]["name"], "error": str(r)})
        else:
            results.append(r)

    summary_path = pathlib.Path("results_stack_modal.json")
    summary_path.write_text(json.dumps({
        "model": MODEL, "dataset": f"{DATASET}::{DATA_DIR}", "fraction": fraction,
        "crs": crs_list, "caches": caches, "dms_threshold": DMS_THRESHOLD,
        "results": results, "errors": errors,
    }, indent=2))

    # Quick summary
    print(f"\n{'='*100}\nRESULTS (n={len(results)} ok, {len(errors)} errors)\n{'='*100}")
    print(f"{'Condition':<42} {'Mean':>6} {'Len':>6} {'bits':>5} {'t(s)':>6} {'VRAM':>6}")
    print("-" * 100)
    for r in sorted(results, key=lambda x: (x["group"], -x["mean_score"])):
        print(f"{r['name']:<42} {r['mean_score']:>6.1f} {r['avg_compressed_len']:>6.0f} "
              f"{r['bits_per_channel']:>5} {r['inference_seconds']:>5.0f} "
              f"{r['peak_vram_gb']:>5.1f}")

    # Pareto: effective compression vs fp16 no_press
    rmap = {r["name"]: r for r in results}
    base = rmap.get("no_press_fp16")
    if base is not None:
        print(f"\n─── Effective compression vs fp16 no_press (Δmean in pp) ───")
        for r in results:
            if r["name"] == "no_press_fp16":
                continue
            len_factor = base["avg_compressed_len"] / r["avg_compressed_len"] if r["avg_compressed_len"] else 1.0
            eff = r["quantization_factor"] * len_factor
            delta = r["mean_score"] - base["mean_score"]
            print(f"  {r['name']:<42} {eff:>6.2f}x  Δ={delta:+.2f}pp")

    if errors:
        print(f"\nErrors:\n{json.dumps(errors, indent=2)}")

    print(f"\nSummary saved: {summary_path}")
    print("Per-condition artifacts: Modal volume 'kvpress-stack-results' (/results/stack/)")


@app.local_entrypoint()
def smoke_probe2(fraction: float = 0.005):
    """Step-B isolation: compose-failure locator + DMS comparison.

    merge_boltzmann_cr0.875_fp16 → merge+aggressive CR, NO quant (quant the culprit?)
    merge_boltzmann_cr0.5_q4     → full stack at milder CR (dose test)
    merge_dms_t-3_fp16           → SOTA reference: MergingPress(DMS) no quant
    merge_dms_t-3_q4             → SOTA reference: MergingPress(DMS) with quanto-4bit
    """
    conds = [
        {"name": "merge_boltzmann_cr0.875_fp16", "group": "merge_boltzmann",
         "cache_nbits": None, "cr": 0.875},
        {"name": "merge_boltzmann_cr0.5_q4", "group": "merge_boltzmann",
         "cache_nbits": 4, "cr": 0.5},
        {"name": "merge_dms_t-3_fp16", "group": "merge_dms",
         "cache_nbits": None, "cr": float("nan")},
        {"name": "merge_dms_t-3_q4", "group": "merge_dms",
         "cache_nbits": 4, "cr": float("nan")},
    ]
    print(f"Smoke probe2: {len(conds)} conditions × f={fraction} on {MODEL}")
    raw = list(run_condition.starmap([(c, fraction) for c in conds], return_exceptions=True))
    for c, r in zip(conds, raw):
        if isinstance(r, Exception):
            print(f"  ✗ {c['name']}: {r}")
        else:
            print(f"  ✓ {c['name']}: mean={r['mean_score']:.1f}  len={r['avg_compressed_len']:.0f}  "
                  f"t={r['inference_seconds']:.0f}s  peak_vram={r['peak_vram_gb']:.1f}GB")


@app.local_entrypoint()
def smoke_ablate(fraction: float = 0.005):
    """Step-A isolation: localize which pillar broke the q4 stack.

    no_press_q4         → does optimum-quanto 4-bit alone work on RULER?
    boltzmann_cr0.875_fp16 → does Boltzmann survive 8x pruning without quant/merge?
    """
    conds = [
        # Re-baseline on Qwen3-8B (prior smokes were Qwen2.5-7B)
        {"name": "no_press_fp16", "group": "baseline",
         "cache_nbits": None, "cr": 0.0},
        {"name": "boltzmann_cr0.5_fp16", "group": "boltzmann",
         "cache_nbits": None, "cr": 0.5},
        # Isolation probes
        {"name": "no_press_q4", "group": "baseline",
         "cache_nbits": 4, "cr": 0.0},
        {"name": "boltzmann_cr0.875_fp16", "group": "boltzmann",
         "cache_nbits": None, "cr": 0.875},
    ]
    print(f"Smoke ablate: {len(conds)} conditions × f={fraction} on {MODEL}")
    raw = list(run_condition.starmap([(c, fraction) for c in conds], return_exceptions=True))
    for c, r in zip(conds, raw):
        if isinstance(r, Exception):
            print(f"  ✗ {c['name']}: {r}")
        else:
            print(f"  ✓ {c['name']}: mean={r['mean_score']:.1f}  len={r['avg_compressed_len']:.0f}  "
                  f"t={r['inference_seconds']:.0f}s  peak_vram={r['peak_vram_gb']:.1f}GB")


@app.local_entrypoint()
def smoke_q4(fraction: float = 0.005):
    """Run ONLY the merge_boltzmann_cr0.875_q4 condition (stack composition check)."""
    conds = [
        {"name": "merge_boltzmann_cr0.875_q4", "group": "merge_boltzmann",
         "cache_nbits": 4, "cr": 0.875},
    ]
    print(f"Smoke q4-only: {len(conds)} condition × f={fraction} on {MODEL}")
    raw = list(run_condition.starmap([(c, fraction) for c in conds], return_exceptions=True))
    for c, r in zip(conds, raw):
        if isinstance(r, Exception):
            print(f"  ✗ {c['name']}: {r}")
        else:
            print(f"  ✓ {c['name']}: mean={r['mean_score']:.1f}  len={r['avg_compressed_len']:.0f}  "
                  f"t={r['inference_seconds']:.0f}s  peak_vram={r['peak_vram_gb']:.1f}GB")


@app.local_entrypoint()
def smoke(fraction: float = 0.005):
    """Smoke test: 3 conditions × tiny sample (≈30 samples) to validate plumbing on 7B."""
    conds = [
        {"name": "no_press_fp16", "group": "baseline", "cache_nbits": None, "cr": 0.0},
        {"name": "boltzmann_cr0.5_fp16", "group": "boltzmann", "cache_nbits": None, "cr": 0.5},
        {"name": "merge_boltzmann_cr0.875_q4", "group": "merge_boltzmann",
         "cache_nbits": 4, "cr": 0.875},
    ]
    print(f"Smoke: {len(conds)} conditions × f={fraction} on {MODEL}")
    raw = list(run_condition.starmap([(c, fraction) for c in conds], return_exceptions=True))
    for c, r in zip(conds, raw):
        if isinstance(r, Exception):
            print(f"  ✗ {c['name']}: {r}")
        else:
            print(f"  ✓ {c['name']}: mean={r['mean_score']:.1f}  len={r['avg_compressed_len']:.0f}  "
                  f"t={r['inference_seconds']:.0f}s  peak_vram={r['peak_vram_gb']:.1f}GB")
