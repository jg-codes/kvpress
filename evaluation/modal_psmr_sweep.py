"""
PSMR (Position-Similarity Merge Routing) staged evaluation on RULER-4096.

Stage 1: Threshold sweep — find DMS threshold where PSMR has leverage.
  Runs no_press + bare DMS + default M(DMS) at thresholds {-3.0, -3.5, -4.0}
  on {qa_1, fwe} at f=0.05 (~25 samples/task).
  M(DMS) runs with diagnostics to measure survivor density per head.

Stage 2: PSMR validation — compare routing/key-merge variants at selected threshold.
  Configs: {bare_dms, merge_val_only, merge_val_pos, merge_full_pos}
  on {qa_1, fwe} at f=0.05.
  Decision: if merge_full_pos > merge_val_only on either task -> Stage 3.

Usage:
    modal run --detach evaluation/modal_psmr_sweep.py --stage 1
    modal run --detach evaluation/modal_psmr_sweep.py --stage 2 --threshold -3.5
"""

import json
import os
import pathlib
import re

import modal

_hf_token = os.environ.get("HF_TOKEN", "")
_secrets = [modal.Secret.from_dict({"HF_TOKEN": _hf_token})] if _hf_token else []

COMMIT = "5865a4f4a501a6cb2b397e336992a85ec9aa641c"  # dev/psmr: PSMR commit

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
        "python -c '"
        "from kvpress import MergingPress; "
        "fields = list(MergingPress.__dataclass_fields__.keys()); "
        "assert \"position_sigma\" in fields, f\"MISSING position_sigma in {fields}\"; "
        "print(\"OK: PSMR params present\")'",
        "python -c 'import transformers; v = transformers.__version__; "
        "print(f\"transformers {v}\"); assert int(v.split(\".\")[0]) < 5'",
    )
)

app = modal.App("kvpress-psmr-sweep", image=image)
results_vol = modal.Volume.from_name("kvpress-psmr-sweep-results", create_if_missing=True)

MODEL = "Qwen/Qwen3-8B"
DATA_DIR = "4096"
TASKS = ["qa_1", "fwe"]
FRACTION = 0.05


# ── RULER scoring ────────────────────────────────────────────────────────

def _string_match_part(preds, refs):
    return round(sum(max(1.0 if r.lower() in p.lower() else 0.0 for r in ref)
                     for p, ref in zip(preds, refs)) / len(preds) * 100, 2)


def _string_match_all(preds, refs):
    return round(sum(sum(1.0 if r.lower() in p.lower() else 0.0 for r in ref) / len(ref)
                     for p, ref in zip(preds, refs)) / len(preds) * 100, 2)


def ruler_scorer(df):
    np_pat = re.compile(r"[\x00-\x1f]")
    df = df.copy()
    df["predicted_answer"] = df["predicted_answer"].apply(
        lambda x: np_pat.sub("", x.strip()).strip()
    )
    scores = {}
    for task, df_task in df.groupby("task"):
        fn = _string_match_part if task.split("_")[0] == "qa" else _string_match_all
        scores[task] = {
            "string_match": fn(
                df_task["predicted_answer"].tolist(),
                df_task["answer"].tolist(),
            )
        }
    return scores


# ── Press construction ───────────────────────────────────────────────────

def _build_press(config: dict):
    """Build press from config dict. Returns None for no_press."""
    if config.get("threshold") is None:
        return None

    from kvpress import DMSPress, KVzapPress, MergingPress

    threshold = config["threshold"]
    merge_params = config.get("merge_params")
    dms = DMSPress(
        press=KVzapPress(model_type="mlp"),
        threshold=threshold,
        sliding_window_size=128,
    )

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
        position_sigma=merge_params.get("position_sigma", 0),
        key_merge_window=merge_params.get("key_merge_window", 0),
        diagnostics=merge_params.get("diagnostics", False),
    )


def _flatten(val) -> float:
    if isinstance(val, dict):
        return val.get("string_match", val.get("rouge1", val.get("f1", 0.0)))
    return float(val) if val is not None else 0.0


# ── Config generators ────────────────────────────────────────────────────

def _get_stage1_configs():
    """Stage 1: threshold sweep with diagnostics on merge variants."""
    configs = [{"name": "no_press", "threshold": None, "merge_params": None}]
    for t in [-3.0, -3.5, -4.0]:
        configs.append({
            "name": f"bare_dms_t{t}",
            "threshold": t,
            "merge_params": None,
        })
        configs.append({
            "name": f"merge_dms_t{t}",
            "threshold": t,
            "merge_params": {"diagnostics": True},
        })
    return configs


def _get_stage2_configs(threshold: float):
    """Stage 2: PSMR variants at selected threshold."""
    return [
        {
            "name": "bare_dms",
            "threshold": threshold,
            "merge_params": None,
        },
        {
            "name": "merge_val_only",
            "threshold": threshold,
            "merge_params": {
                "merge_keys": False,
                "position_sigma": 0,
                "key_merge_window": 0,
            },
        },
        {
            "name": "merge_val_pos",
            "threshold": threshold,
            "merge_params": {
                "merge_keys": False,
                "position_sigma": 8.0,
                "key_merge_window": 0,
            },
        },
        {
            "name": "merge_full_pos",
            "threshold": threshold,
            "merge_params": {
                "merge_keys": True,
                "position_sigma": 8.0,
                "key_merge_window": 16,
            },
        },
    ]


# ── Survivor density from diagnostics ────────────────────────────────────

def _summarize_survivors(diag_log, n_context_tokens):
    """Extract survivor density stats from MergingPress diagnostics."""
    if not diag_log:
        return {}

    per_head_survivors = []
    per_head_gaps = []

    for layer_entry in diag_log:
        for hd in layer_entry["per_head"]:
            keep_pos = hd["keep_positions"]
            n_kept = len(keep_pos)
            per_head_survivors.append(n_kept)

            if n_kept > 1:
                sorted_pos = keep_pos.sort()[0]
                gaps = (sorted_pos[1:] - sorted_pos[:-1]).float()
                per_head_gaps.append(gaps.mean().item())

    n_heads = len(per_head_survivors)
    mean_survivors = sum(per_head_survivors) / n_heads if n_heads else 0
    mean_gap = sum(per_head_gaps) / len(per_head_gaps) if per_head_gaps else 0
    eviction_rate = 1 - mean_survivors / n_context_tokens if n_context_tokens else 0

    return {
        "mean_survivors_per_head": round(mean_survivors, 1),
        "mean_gap_between_survivors": round(mean_gap, 1),
        "eviction_rate": round(eviction_rate, 4),
        "n_context_tokens": n_context_tokens,
    }


# ── Main evaluation ─────────────────────────────────────────────────────

@app.function(
    gpu="A100",
    timeout=14400,
    memory=65536,
    scaledown_window=2,
    secrets=_secrets,
    volumes={"/results": results_vol},
)
def run_sweep(stage: int = 1, threshold: float = -3.5, fraction: float = FRACTION):
    """Run PSMR evaluation stage."""
    import gc
    import random
    import time

    import numpy as np
    import pandas as pd
    import torch
    from datasets import load_dataset
    from tqdm import tqdm
    from transformers import pipeline as hf_pipeline

    import kvpress
    print(f"kvpress loaded from: {kvpress.__file__}")
    print(f"Stage {stage} | threshold={threshold} | fraction={fraction}")

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    print(f"GPU: {torch.cuda.get_device_name(0)}")

    configs = _get_stage1_configs() if stage == 1 else _get_stage2_configs(threshold)
    print(f"Configs: {[c['name'] for c in configs]}")

    # Load model once
    print(f"\nLoading {MODEL}...")
    t0 = time.monotonic()
    pipe = hf_pipeline(
        "kv-press-text-generation",
        model=MODEL,
        device_map="auto",
        trust_remote_code=True,
    )
    pipe.model.eval()
    model_load_s = round(time.monotonic() - t0, 1)
    print(f"Model loaded in {model_load_s}s")

    # Load dataset, filter to target tasks
    df_full = load_dataset("simonjegou/ruler", data_dir=DATA_DIR, split="test").to_pandas()
    df_tasks = df_full[df_full["task"].isin(TASKS)].copy()
    df_sample = df_tasks.sample(frac=fraction, random_state=42)
    print(f"Dataset: {len(df_sample)} samples (f={fraction})")
    for t in TASKS:
        print(f"  {t}: {(df_sample['task'] == t).sum()} samples")

    all_results = []
    survivor_stats = {}

    for i, config in enumerate(configs):
        name = config["name"]
        print(f"\n{'=' * 70}")
        print(f"[{i + 1}/{len(configs)}] {name}")
        print(f"{'=' * 70}")

        try:
            press = _build_press(config)
            df = df_sample.copy()
            df["predicted_answer"] = None

            df_grouped = df.groupby("context")
            n_contexts = df["context"].nunique()

            torch.cuda.empty_cache()
            t_infer = time.monotonic()

            diag_samples = []

            with torch.inference_mode():
                for ctx_idx, (context, df_group) in enumerate(
                    tqdm(df_grouped, total=n_contexts, desc=name)
                ):
                    questions = df_group["question"].to_list()
                    max_new_tokens = df_group["max_new_tokens"].iloc[0]
                    answer_prefix = df_group["answer_prefix"].iloc[0]

                    if press is not None and hasattr(press, "clear_diagnostics"):
                        press.clear_diagnostics()

                    output = pipe(
                        context,
                        questions=questions,
                        answer_prefix=answer_prefix,
                        press=press,
                        max_new_tokens=max_new_tokens,
                    )
                    df.loc[df_group.index, "predicted_answer"] = output["answers"]

                    # Collect survivor stats from first 5 contexts
                    if (
                        press is not None
                        and hasattr(press, "get_diagnostics")
                        and ctx_idx < 5
                    ):
                        diag = press.get_diagnostics()
                        if diag:
                            n_tok = len(
                                pipe.tokenizer.encode(context, add_special_tokens=False)
                            )
                            diag_samples.append(_summarize_survivors(diag, n_tok))

                    torch.cuda.empty_cache()

            infer_s = round(time.monotonic() - t_infer, 1)
            metrics = ruler_scorer(df)
            tasks = sorted(metrics.keys())
            scores = [_flatten(metrics[t]) for t in tasks]
            mean_score = sum(scores) / len(scores) if scores else 0.0

            result = {
                "name": name,
                "threshold": config.get("threshold"),
                "merge_params": config.get("merge_params"),
                "mean_score": round(mean_score, 2),
                "inference_seconds": infer_s,
                "per_task": {t: round(_flatten(metrics[t]), 2) for t in tasks},
                "n_samples": len(df),
            }

            if diag_samples:
                agg = {
                    "mean_survivors_per_head": round(
                        sum(d["mean_survivors_per_head"] for d in diag_samples)
                        / len(diag_samples),
                        1,
                    ),
                    "mean_gap_between_survivors": round(
                        sum(d["mean_gap_between_survivors"] for d in diag_samples)
                        / len(diag_samples),
                        1,
                    ),
                    "mean_eviction_rate": round(
                        sum(d["eviction_rate"] for d in diag_samples)
                        / len(diag_samples),
                        4,
                    ),
                    "n_probe_samples": len(diag_samples),
                }
                result["survivor_stats"] = agg
                t_key = config.get("threshold")
                if t_key is not None:
                    survivor_stats[str(t_key)] = agg
                print(
                    f"  Survivors/head: {agg['mean_survivors_per_head']:.0f} "
                    f"| Gap: {agg['mean_gap_between_survivors']:.0f} "
                    f"| Eviction: {agg['mean_eviction_rate']:.1%}"
                )

            all_results.append(result)

            print(f"  Mean: {mean_score:.2f} | Infer: {infer_s}s")
            for t in tasks:
                print(f"    {t}: {_flatten(metrics[t]):.2f}")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback

            traceback.print_exc()
            all_results.append({"name": name, "error": str(e)})

        gc.collect()
        torch.cuda.empty_cache()

    # ── Save results ─────────────────────────────────────────────────────
    output = {
        "stage": stage,
        "model": MODEL,
        "fraction": fraction,
        "tasks": TASKS,
        "configs": configs,
        "results": all_results,
        "model_load_seconds": model_load_s,
    }
    if survivor_stats:
        output["survivor_stats_by_threshold"] = survivor_stats

    fname = f"stage{stage}_results.json"
    with open(f"/results/{fname}", "w") as f:
        json.dump(output, f, indent=2)
    results_vol.commit()

    # ── Summary table ────────────────────────────────────────────────────
    print(f"\n{'=' * 90}")
    header = f"{'Name':<30} {'Mean':>6}"
    for t in TASKS:
        header += f" {t:>8}"
    header += f" {'Infer':>7}"
    print(header)
    print(f"{'-' * 90}")
    for r in all_results:
        if "error" in r:
            print(f"{r['name']:<30} ERROR: {r['error'][:40]}")
            continue
        line = f"{r['name']:<30} {r['mean_score']:>6.2f}"
        for t in TASKS:
            line += f" {r['per_task'].get(t, 0):>8.2f}"
        line += f" {r['inference_seconds']:>6.0f}s"
        if r.get("survivor_stats"):
            ss = r["survivor_stats"]
            line += (
                f"  surv={ss['mean_survivors_per_head']:.0f}/head"
                f" gap={ss['mean_gap_between_survivors']:.0f}"
            )
        print(line)

    # ── Stage 1: threshold recommendation ────────────────────────────────
    if stage == 1:
        print(f"\n--- Stage 1 Analysis ---")
        no_p = next(
            (r for r in all_results if r["name"] == "no_press" and "error" not in r),
            None,
        )
        for t in [-3.0, -3.5, -4.0]:
            bare = next(
                (
                    r
                    for r in all_results
                    if r["name"] == f"bare_dms_t{t}" and "error" not in r
                ),
                None,
            )
            merge = next(
                (
                    r
                    for r in all_results
                    if r["name"] == f"merge_dms_t{t}" and "error" not in r
                ),
                None,
            )
            if bare and merge and no_p:
                room = no_p["mean_score"] - bare["mean_score"]
                merge_delta = merge["mean_score"] - bare["mean_score"]
                ss = survivor_stats.get(str(t), {})
                surv = ss.get("mean_survivors_per_head", "?")
                gap = ss.get("mean_gap_between_survivors", "?")
                print(
                    f"  t={t}: room={room:+.2f}pp  merge_delta={merge_delta:+.2f}pp"
                    f"  survivors={surv}/head  gap={gap}"
                )
        print("\nPick threshold where room > 1pp AND gap < 30 for PSMR key merge.")

    print(f"\nModel load: {model_load_s}s")
    return output


@app.local_entrypoint()
def main(stage: int = 1, threshold: float = -3.5):
    results = run_sweep.remote(stage=stage, threshold=threshold)
    out = pathlib.Path("evaluation/results_psmr")
    out.mkdir(parents=True, exist_ok=True)
    fname = f"stage{stage}_results.json"
    with open(out / fname, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nLocal copy saved to {out}/{fname}")
