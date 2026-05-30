"""
Long-context MergingPress evaluation — tests H_A1/H_A2: do retrieval gains amplify at long context?

Compares: no_press, bare DMS(KVzap) t=-3, MergingPress(DMS) t=-3 across RULER context lengths.

Supports both simonjegou/ruler (4K, 8K, 16K) and langq1225/qwen_ruler (32K, Qwen-tokenized).

Usage:
    modal run --detach evaluation/modal_merging_longctx.py                      # default: 32K
    modal run --detach evaluation/modal_merging_longctx.py --ctx-len 16384
    modal run --detach evaluation/modal_merging_longctx.py --ctx-len 32768 --fraction 0.10
"""

import json
import os
import pathlib
import re

import modal

_hf_token = os.environ.get("HF_TOKEN", "")
_secrets = [modal.Secret.from_dict({"HF_TOKEN": _hf_token})] if _hf_token else []

# dev/boltzmann-stack HEAD with KVzap fix (same commit as working 4K smokes)
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
        "python -c 'from kvpress import DMSPress, KVzapPress, MergingPress; print(\"OK: imports\")'",
        "python -c 'import transformers; v = transformers.__version__; assert int(v.split(\".\")[0]) < 5; print(f\"OK: transformers {v}\")'",
    )
)

app = modal.App("kvpress-merge-longctx", image=image)
results_vol = modal.Volume.from_name("kvpress-merge-longctx-results", create_if_missing=True)

MODEL = "Qwen/Qwen3-8B"

# Map context length → HF dataset + data_dir subpath.
# langq1225/qwen_ruler is Qwen-tokenized (matches our model); simonjegou/ruler otherwise.
DATASET_FOR_CTX = {
    "4096":  ("simonjegou/ruler",   "4096"),
    "8192":  ("simonjegou/ruler",   "8192"),
    "16384": ("simonjegou/ruler",   "16384"),
    "32768": ("langq1225/qwen_ruler", "32768"),
}

CONFIGS = [
    {"name": "bare_dms_t-3",            "press": "dms",       "threshold": -3},
    {"name": "bare_dms_t-1",            "press": "dms",       "threshold": -1},
    {"name": "merge_dms_t-3_f1.0",      "press": "merge_dms", "threshold": -3, "merge_fraction": 1.0},
    {"name": "merge_dms_t-3_f0.5",      "press": "merge_dms", "threshold": -3, "merge_fraction": 0.5},
    {"name": "merge_dms_t-1_f1.0",      "press": "merge_dms", "threshold": -1, "merge_fraction": 1.0},
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
    from kvpress import DMSPress, KVzapPress, MergingPress
    kind = config["press"]
    if kind == "none":
        return None
    if kind == "dms":
        return DMSPress(press=KVzapPress(model_type="mlp"), threshold=config["threshold"], sliding_window_size=128)
    if kind == "merge_dms":
        dms = DMSPress(press=KVzapPress(model_type="mlp"), threshold=config["threshold"], sliding_window_size=128)
        merge_fraction = config.get("merge_fraction", 1.0)
        return MergingPress(dms, merge_fraction=merge_fraction)  # merge_keys=False, similarity_threshold=0 per defaults
    raise ValueError(f"unknown press kind: {kind}")


@app.function(
    gpu="A100-80GB",  # need headroom for 32K KV cache
    timeout=14400,
    memory=65536,
    scaledown_window=2,
    secrets=_secrets,
    volumes={"/results": results_vol},
)
def run_longctx(ctx_len: str = "32768", fraction: float = 0.05):
    import gc
    import random
    import time

    import numpy as np
    import torch
    from datasets import load_dataset
    from tqdm import tqdm
    from transformers import pipeline as hf_pipeline

    import kvpress
    print(f"kvpress: {kvpress.__file__}")

    torch.manual_seed(42); np.random.seed(42); random.seed(42)

    if ctx_len not in DATASET_FOR_CTX:
        raise ValueError(f"ctx_len {ctx_len} not in {list(DATASET_FOR_CTX.keys())}")
    hf_dataset, data_dir = DATASET_FOR_CTX[ctx_len]

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"  {props.total_memory / 1e9:.1f} GB, sm_{props.major}{props.minor}")

    print(f"\nLoading {MODEL}...")
    t_model = time.monotonic()
    pipe = hf_pipeline(
        "kv-press-text-generation", model=MODEL,
        device_map="auto", trust_remote_code=True,
    )
    pipe.model.eval()
    model_load_s = round(time.monotonic() - t_model, 1)
    vram_after_load = round(torch.cuda.memory_allocated() / 1e9, 2)
    print(f"Model loaded in {model_load_s}s | VRAM after load: {vram_after_load} GB")

    print(f"\nLoading dataset {hf_dataset} data_dir={data_dir}")
    df_full = load_dataset(hf_dataset, data_dir=data_dir, split="test").to_pandas()
    df_sample = df_full.sample(frac=fraction, random_state=42)
    print(f"Dataset: {len(df_sample)} samples (f={fraction})")

    all_results = []

    for i, config in enumerate(CONFIGS):
        name = config["name"]
        print(f"\n{'='*70}\n[{i+1}/{len(CONFIGS)}] {name} @ ctx_len={ctx_len}\n{'='*70}")

        try:
            press = _build_press(config)
            df = df_sample.copy()
            df["predicted_answer"] = None
            df_grouped = df.groupby("context")
            n_contexts = df["context"].nunique()

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            t_infer = time.monotonic()

            compression_ratios = []
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
                    if press is not None:
                        try:
                            compression_ratios.append(press.compression_ratio)
                        except Exception:
                            pass
                    torch.cuda.empty_cache()

            infer_s = round(time.monotonic() - t_infer, 1)
            peak_vram = round(torch.cuda.max_memory_allocated() / 1e9, 2)
            metrics = ruler_scorer(df)
            tasks = sorted(metrics.keys())
            scores = [metrics[t] for t in tasks]
            mean_score = sum(scores) / len(scores) if scores else 0.0
            mean_cr = sum(compression_ratios) / len(compression_ratios) if compression_ratios else 0.0

            result = {
                "name": name,
                "press_kind": config["press"],
                "threshold": config.get("threshold"),
                "mean_score": round(mean_score, 2),
                "mean_compression_ratio": round(mean_cr, 4),
                "peak_vram_gb": peak_vram,
                "inference_seconds": infer_s,
                "per_task": {t: round(metrics[t], 2) for t in tasks},
                "n_samples": len(df),
                "ctx_len": ctx_len,
                "dataset": hf_dataset,
            }
            all_results.append(result)

            # Per-condition artifact
            cond_dir = f"/results/longctx/{ctx_len}/{name}"
            os.makedirs(cond_dir, exist_ok=True)
            with open(f"{cond_dir}/metrics.json", "w") as f:
                json.dump(result, f, indent=2)

            print(f"  Mean: {mean_score:.2f} | CR: {mean_cr:.3f} | Peak VRAM: {peak_vram}GB | Infer: {infer_s}s")
            for t in tasks:
                print(f"    {t}: {metrics[t]:.2f}")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            all_results.append({"name": name, "error": str(e), "ctx_len": ctx_len})

        gc.collect(); torch.cuda.empty_cache()

    # Aggregate
    agg_path = f"/results/longctx/{ctx_len}/aggregate.json"
    os.makedirs(os.path.dirname(agg_path), exist_ok=True)
    with open(agg_path, "w") as f:
        json.dump({"model": MODEL, "ctx_len": ctx_len, "dataset": hf_dataset,
                    "fraction": fraction, "model_load_seconds": model_load_s,
                    "vram_after_load_gb": vram_after_load,
                    "configs": CONFIGS, "results": all_results}, f, indent=2)
    results_vol.commit()

    # Paired deltas vs baselines
    print(f"\n{'='*100}")
    print(f"ctx_len={ctx_len}  dataset={hf_dataset}  fraction={fraction}")
    print(f"{'Name':<28} {'Mean':>6} {'CR':>6} {'VRAM':>6} {'fwe':>6} {'niah_mk1':>8} {'niah_mv':>7} {'qa_1':>6} {'qa_2':>6}")
    print(f"{'-'*100}")
    for r in all_results:
        if "error" in r:
            print(f"{r['name']:<28} ERROR: {r['error'][:40]}")
            continue
        pt = r["per_task"]
        print(f"{r['name']:<28} {r['mean_score']:>6.2f} {r['mean_compression_ratio']:>6.3f} "
              f"{r['peak_vram_gb']:>5.1f}G {pt.get('fwe', 0):>6.2f} "
              f"{pt.get('niah_multikey_1', 0):>8.2f} {pt.get('niah_multivalue', 0):>7.2f} "
              f"{pt.get('qa_1', 0):>6.2f} {pt.get('qa_2', 0):>6.2f}")

    # Paired deltas: each merge_* vs its matched-threshold bare_dms
    by_name = {r["name"]: r for r in all_results if "error" not in r}
    pairs = [
        ("bare_dms_t-3", "merge_dms_t-3_f1.0"),
        ("bare_dms_t-3", "merge_dms_t-3_f0.5"),
        ("bare_dms_t-1", "merge_dms_t-1_f1.0"),
    ]
    for bare_name, merge_name in pairs:
        if bare_name in by_name and merge_name in by_name:
            bare = by_name[bare_name]
            merge = by_name[merge_name]
            print(f"\nDelta {merge_name} vs {bare_name}:")
            for t in sorted(bare["per_task"].keys()):
                d = merge["per_task"].get(t, 0) - bare["per_task"].get(t, 0)
                print(f"  {t}: {d:+.2f}")
            d_mean = merge["mean_score"] - bare["mean_score"]
            print(f"  MEAN: {d_mean:+.2f}")

    print(f"\nModel load: {model_load_s}s")
    return all_results


@app.local_entrypoint()
def main(ctx_len: str = "32768", fraction: float = 0.05):
    results = run_longctx.remote(ctx_len=ctx_len, fraction=fraction)
    out = pathlib.Path(f"evaluation/results_longctx/{ctx_len}")
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nLocal copy saved to {out}/results.json")
