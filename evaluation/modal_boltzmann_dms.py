"""
Boltzmann × DMS composition smoke — DMSPress(BoltzmannPress) vs DMSPress(KVzap).

Tests H_B: DMSPress with Boltzmann as inner scorer > DMSPress with KVzap at equal CR.
Reuses the same image recipe as modal_dms_pg_minimal.py (KVzap fix already cherry-picked).

Usage:
    modal run --detach evaluation/modal_boltzmann_dms.py
"""

import json
import os
import pathlib
import re

import modal

_hf_token = os.environ.get("HF_TOKEN", "")
_secrets = [modal.Secret.from_dict({"HF_TOKEN": _hf_token})] if _hf_token else []

# Cherry-picked KVzap fix onto dev/boltzmann-stack (HEAD = 7df9779)
COMMIT = "7df9779a99b5631ac35185bdb29ea77b64764975"

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
    .run_commands(
        "python -c 'from kvpress import BoltzmannPress, KVzapPress, DMSPress; print(\"OK: imports\")'",
        "python -c 'import transformers; v = transformers.__version__; assert int(v.split(\".\")[0]) < 5; print(f\"OK: transformers {v}\")'",
    )
)

app = modal.App("kvpress-boltz-dms-smoke", image=image)
results_vol = modal.Volume.from_name("kvpress-boltz-dms-results", create_if_missing=True)

MODEL = "Qwen/Qwen3-8B"
DATA_DIR = "4096"
FRACTION = 0.01  # n=~65, fast smoke

CONFIGS = [
    {"name": "dms_kvzap_t-3", "scorer": "kvzap", "threshold": -3.0},
    {"name": "dms_boltz_t-8", "scorer": "boltz", "threshold": -8.0},
    {"name": "dms_boltz_t-7", "scorer": "boltz", "threshold": -7.0},
    {"name": "dms_boltz_t-6", "scorer": "boltz", "threshold": -6.0},
    {"name": "dms_boltz_t-5", "scorer": "boltz", "threshold": -5.0},
]


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
        scores[task] = fn(df_task["predicted_answer"].tolist(), df_task["answer"].tolist())
    return scores


def _build_press(config: dict):
    from kvpress import BoltzmannPress, DMSPress, KVzapPress
    if config["scorer"] == "kvzap":
        inner = KVzapPress(model_type="mlp")
    elif config["scorer"] == "boltz":
        inner = BoltzmannPress()
    else:
        raise ValueError(config["scorer"])
    return DMSPress(press=inner, threshold=config["threshold"], sliding_window_size=128)


@app.function(
    gpu="A100",
    timeout=7200,
    memory=65536,
    scaledown_window=2,
    secrets=_secrets,
    volumes={"/results": results_vol},
)
def run_all(fraction: float = FRACTION):
    import gc
    import random
    import time

    import numpy as np
    import torch
    from datasets import load_dataset
    from tqdm import tqdm
    from transformers import pipeline as hf_pipeline

    import kvpress  # registers kv-press-text-generation pipeline
    print(f"kvpress: {kvpress.__file__}")

    torch.manual_seed(42); np.random.seed(42); random.seed(42)

    print(f"Loading {MODEL}...")
    t_model = time.monotonic()
    pipe = hf_pipeline(
        "kv-press-text-generation", model=MODEL,
        device_map="auto", trust_remote_code=True,
    )
    pipe.model.eval()
    model_load_s = round(time.monotonic() - t_model, 1)
    print(f"Model loaded in {model_load_s}s")

    df_full = load_dataset("simonjegou/ruler", data_dir=DATA_DIR, split="test").to_pandas()
    df_sample = df_full.sample(frac=fraction, random_state=42)
    print(f"Dataset: {len(df_sample)} samples (f={fraction})")

    all_results = []

    for i, config in enumerate(CONFIGS):
        name = config["name"]
        print(f"\n{'='*70}\n[{i+1}/{len(CONFIGS)}] {name}\n{'='*70}")

        try:
            press = _build_press(config)
            df = df_sample.copy()
            df["predicted_answer"] = None
            df_grouped = df.groupby("context")
            n_contexts = df["context"].nunique()

            torch.cuda.empty_cache()
            t_infer = time.monotonic()

            compression_ratios_sampled = []
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
                    # Record CR from this forward pass (DMSPress populates after each call)
                    try:
                        compression_ratios_sampled.append(press.compression_ratio)
                    except Exception:
                        pass
                    torch.cuda.empty_cache()

            infer_s = round(time.monotonic() - t_infer, 1)
            metrics = ruler_scorer(df)
            tasks = sorted(metrics.keys())
            scores = [metrics[t] for t in tasks]
            mean_score = sum(scores) / len(scores) if scores else 0.0
            mean_cr = sum(compression_ratios_sampled) / len(compression_ratios_sampled) if compression_ratios_sampled else 0.0

            result = {
                "name": name,
                "scorer": config["scorer"],
                "threshold": config["threshold"],
                "mean_score": round(mean_score, 2),
                "mean_compression_ratio": round(mean_cr, 4),
                "inference_seconds": infer_s,
                "per_task": {t: round(metrics[t], 2) for t in tasks},
                "n_samples": len(df),
            }
            all_results.append(result)

            print(f"  Mean: {mean_score:.2f} | CR: {mean_cr:.3f} | Infer: {infer_s}s")
            for t in tasks:
                print(f"    {t}: {metrics[t]:.2f}")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            all_results.append({"name": name, "error": str(e)})

        gc.collect(); torch.cuda.empty_cache()

    with open("/results/results.json", "w") as f:
        json.dump({"configs": CONFIGS, "results": all_results, "model": MODEL,
                    "fraction": fraction, "model_load_seconds": model_load_s}, f, indent=2)
    results_vol.commit()

    print(f"\n{'='*90}")
    print(f"{'Name':<22} {'Scorer':<8} {'Thr':>6} {'CR':>6} {'Mean':>6} {'fwe':>6} {'niah_mk1':>8} {'qa_1':>6}")
    print(f"{'-'*90}")
    for r in all_results:
        if "error" in r:
            print(f"{r['name']:<22} ERROR: {r['error'][:40]}")
            continue
        pt = r["per_task"]
        print(f"{r['name']:<22} {r['scorer']:<8} {r['threshold']:>6.1f} "
              f"{r['mean_compression_ratio']:>6.3f} {r['mean_score']:>6.2f} "
              f"{pt.get('fwe', 0):>6.2f} {pt.get('niah_multikey_1', 0):>8.2f} "
              f"{pt.get('qa_1', 0):>6.2f}")

    print(f"\nModel load: {model_load_s}s")
    return all_results


@app.local_entrypoint()
def main():
    results = run_all.remote()
    out = pathlib.Path("evaluation/results_boltz_dms")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nLocal copy saved to {out}/results.json")
