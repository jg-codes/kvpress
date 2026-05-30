"""
Minimal MergingPress perturbation-gate eval — single A100, sequential configs.

Uses KVzapPress(mlp) as inner scorer with pinned transformers<5.0.
Self-contained RULER scoring (no /eval_repo).

Usage:
    modal run evaluation/modal_dms_pg_minimal.py
    modal run --detach evaluation/modal_dms_pg_minimal.py
"""

import json
import os
import pathlib
import re

import modal

_hf_token = os.environ.get("HF_TOKEN", "")
_secrets = [modal.Secret.from_dict({"HF_TOKEN": _hf_token})] if _hf_token else []

COMMIT = "6a9a3d782e8edfe4aa04c014495d0d371c13109f"  # pr/merging-press + KVzapConfig fix

# Single image layer: everything installed together, no caching issues
# Pin transformers<5.0 so KVzapPress(mlp) works (KVzapConfig breaks on >=5.x)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "packaging", "setuptools", "wheel",
        "torch==2.5.1",
        "transformers>=4.48,<5.0",
        "datasets", "pandas", "numpy", "tqdm", "accelerate",
        f"kvpress @ git+https://github.com/jg-codes/kvpress.git@{COMMIT}",
        gpu="A100",  # build with CUDA
    )
    # Verify perturbation_gate exists and transformers version is pinned
    .run_commands(
        "python -c 'from kvpress import MergingPress; assert \"perturbation_gate\" in MergingPress.__dataclass_fields__, \"MISSING perturbation_gate\"; print(\"OK: perturbation_gate\")'",
        "python -c 'from kvpress import KVzapPress; print(f\"OK: KVzapPress importable\")'",
        "python -c 'import transformers; v = transformers.__version__; print(f\"transformers {v}\"); assert int(v.split(\".\")[0]) < 5, f\"Need <5.0, got {v}\"'",
    )
)

app = modal.App("kvpress-dms-pg-kvzap-v4", image=image)
results_vol = modal.Volume.from_name("kvpress-dms-pg-kvzap-v4-results", create_if_missing=True)

MODEL = "Qwen/Qwen3-8B"
DATA_DIR = "4096"
FRACTION = 0.10

CONFIGS = [
    {"name": "bare_dms_kvzap_t-3", "threshold": -3, "merge_params": None},
    {"name": "m_dms_kvzap_t-3_default", "threshold": -3, "merge_params": {}},
    {"name": "m_dms_kvzap_t-3_pg1.0", "threshold": -3, "merge_params": {"perturbation_gate": 1.0}},
]


# ── Inlined RULER scorer (no eval_repo dependency) ────────────────────
def _string_match_part(preds, refs):
    return round(sum(max(1.0 if r.lower() in p.lower() else 0.0 for r in ref)
                     for p, ref in zip(preds, refs)) / len(preds) * 100, 2)

def _string_match_all(preds, refs):
    return round(sum(sum(1.0 if r.lower() in p.lower() else 0.0 for r in ref) / len(ref)
                     for p, ref in zip(preds, refs)) / len(preds) * 100, 2)

def ruler_scorer(df):
    np_pat = re.compile(r"[\x00-\x1f]")
    df = df.copy()
    df["predicted_answer"] = df["predicted_answer"].apply(lambda x: np_pat.sub("", x.strip()).strip())
    scores = {}
    for task, df_task in df.groupby("task"):
        fn = _string_match_part if task.split("_")[0] == "qa" else _string_match_all
        scores[task] = {"string_match": fn(df_task["predicted_answer"].tolist(), df_task["answer"].tolist())}
    return scores


def _build_press(config: dict):
    from kvpress import DMSPress, KVzapPress, MergingPress
    print(f"DEBUG MergingPress fields: {list(MergingPress.__dataclass_fields__.keys())}")

    threshold = config["threshold"]
    merge_params = config["merge_params"]
    dms = DMSPress(press=KVzapPress(model_type="mlp"), threshold=threshold, sliding_window_size=128)

    if merge_params is None:
        return dms

    return MergingPress(
        dms,
        similarity_threshold=merge_params.get("similarity_threshold", 0.0),
        merge_fraction=merge_params.get("merge_fraction", 1.0),
        value_norm_weighting=merge_params.get("value_norm_weighting", True),
        merge_keys=merge_params.get("merge_keys", False),
        max_merge_per_token=merge_params.get("max_merge_per_token", 0),
        perturbation_gate=merge_params.get("perturbation_gate", 0.0),
    )


def _flatten(val) -> float:
    if isinstance(val, dict):
        return val.get("string_match", val.get("rouge1", val.get("f1", 0.0)))
    return float(val) if val is not None else 0.0


@app.function(
    gpu="A100",
    timeout=14400,
    memory=65536,
    scaledown_window=2,
    secrets=_secrets,
    volumes={"/results": results_vol},
)
def run_all(fraction: float = FRACTION):
    """Run all configs sequentially on a single A100 (model loaded once)."""
    import gc
    import random
    import time

    import numpy as np
    import pandas as pd
    import torch
    from datasets import load_dataset
    from tqdm import tqdm
    from transformers import pipeline as hf_pipeline

    import kvpress  # registers kv-press-text-generation pipeline
    print(f"kvpress loaded from: {kvpress.__file__}")

    torch.manual_seed(42); np.random.seed(42); random.seed(42)

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"  {props.total_memory / 1e9:.1f} GB, sm_{props.major}{props.minor}")

    # Load model ONCE
    print(f"\nLoading {MODEL}...")
    t_model = time.monotonic()
    pipe = hf_pipeline(
        "kv-press-text-generation",
        model=MODEL,
        device_map="auto",
        trust_remote_code=True,
    )
    pipe.model.eval()
    model_load_s = round(time.monotonic() - t_model, 1)
    print(f"Model loaded in {model_load_s}s")

    # Load dataset ONCE
    df_full = load_dataset("simonjegou/ruler", data_dir=DATA_DIR, split="test").to_pandas()
    df_sample = df_full.sample(frac=fraction, random_state=42)
    print(f"Dataset: {len(df_sample)} samples (f={fraction})")

    all_results = []

    for i, config in enumerate(CONFIGS):
        name = config["name"]
        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(CONFIGS)}] {name}")
        print(f"{'='*70}")

        try:
            press = _build_press(config)
            df = df_sample.copy()
            df["predicted_answer"] = None

            df_grouped = df.groupby("context")
            n_contexts = df["context"].nunique()

            torch.cuda.empty_cache()
            t_infer = time.monotonic()

            with torch.inference_mode():
                for context, df_group in tqdm(df_grouped, total=n_contexts, desc=name):
                    questions = df_group["question"].to_list()
                    max_new_tokens = df_group["max_new_tokens"].iloc[0]
                    answer_prefix = df_group["answer_prefix"].iloc[0]
                    output = pipe(
                        context, questions=questions, answer_prefix=answer_prefix,
                        press=press, max_new_tokens=max_new_tokens,
                    )
                    df.loc[df_group.index, "predicted_answer"] = output["answers"]
                    torch.cuda.empty_cache()

            infer_s = round(time.monotonic() - t_infer, 1)
            metrics = ruler_scorer(df)
            tasks = sorted(metrics.keys())
            scores = [_flatten(metrics[t]) for t in tasks]
            mean_score = sum(scores) / len(scores) if scores else 0.0

            result = {
                "name": name,
                "threshold": config["threshold"],
                "merge_params": config["merge_params"],
                "mean_score": round(mean_score, 2),
                "inference_seconds": infer_s,
                "per_task": {t: round(_flatten(metrics[t]), 2) for t in tasks},
                "n_samples": len(df),
            }
            all_results.append(result)

            print(f"  Mean: {mean_score:.2f} | Infer: {infer_s}s")
            for t in tasks:
                print(f"    {t}: {_flatten(metrics[t]):.2f}")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            all_results.append({"name": name, "error": str(e)})

        gc.collect(); torch.cuda.empty_cache()

    # Save results
    with open("/results/results.json", "w") as f:
        json.dump({"configs": CONFIGS, "results": all_results, "model": MODEL,
                    "fraction": fraction, "model_load_seconds": model_load_s}, f, indent=2)
    results_vol.commit()

    # Summary
    print(f"\n{'='*90}")
    print(f"{'Name':<35} {'Mean':>6} {'Infer':>7} {'qa_1':>6} {'fwe':>6} {'cwe':>6}")
    print(f"{'-'*90}")
    for r in all_results:
        if "error" in r:
            print(f"{r['name']:<35} ERROR: {r['error'][:40]}")
            continue
        print(f"{r['name']:<35} {r['mean_score']:>6.2f} {r['inference_seconds']:>6.0f}s "
              f"{r['per_task'].get('qa_1', 0):>6.2f} "
              f"{r['per_task'].get('fwe', 0):>6.2f} "
              f"{r['per_task'].get('cwe', 0):>6.2f}")

    print(f"\nModel load: {model_load_s}s")
    return all_results


@app.local_entrypoint()
def main():
    results = run_all.remote()
    out = pathlib.Path("evaluation/results_pg_minimal")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nLocal copy saved to {out}/results.json")
