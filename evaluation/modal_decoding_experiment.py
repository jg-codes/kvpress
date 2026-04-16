"""
MergingDecodingPress vs CAMPress benchmark — Modal A100.

Apple-to-apple comparison of decoding-phase merge strategies:
  - CAMPress(KnormPress): sequential-neighbor merge via Bernoulli mask (upstream baseline)
  - CAMPress(SnapKVPress): same with SnapKV scorer
  - MergingDecodingPress(KnormPress): cosine-similarity merge (ours)
  - MergingDecodingPress(SnapKVPress): same with SnapKV scorer
  - DecodingPress(KnormPress): hard eviction baseline (no merging)
  - DecodingPress(SnapKVPress): hard eviction baseline (no merging)
"""

import json
import os
import glob
import pathlib

import modal

MODEL = "Qwen/Qwen3-8B"
DATASET = "ruler"
DATA_DIR = "4096"

# target_size controls how many tokens survive after compression
# With ruler-4096: ts=3072 ≈ CR 0.25, ts=2048 ≈ CR 0.50, ts=1024 ≈ CR 0.75
TARGET_SIZES = [3072, 2048, 1024]

PRESSES = [
    "cam_knorm",
    "cam_adakv_snapkv",
    "merging_decoding_knorm",
    "merging_decoding_adakv_snapkv",
    "decoding_knorm",
    "decoding_adakv_snapkv",
    "no_press",
]

app = modal.App("kvpress-decoding-experiment")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install("packaging", "setuptools", "wheel")
    .pip_install(
        "torch>=2.1", "transformers>=4.48", "datasets", "accelerate",
        "pyyaml", "omegaconf", "hydra-core",
    )
    .pip_install("kvpress @ git+https://github.com/jg-codes/kvpress.git@pr/merging-press")
    .pip_install("jieba", "bert_score", "fuzzywuzzy", "python-Levenshtein", "nltk", "rouge")
    .run_commands(
        "git clone --branch pr/merging-press --depth 1 https://github.com/jg-codes/kvpress.git /eval_repo",
    )
)

results_vol = modal.Volume.from_name("kvpress-decoding-results", create_if_missing=True)


def flatten_score(v):
    """Recursively extract a numeric score from nested dicts."""
    if isinstance(v, (int, float)):
        return v
    if isinstance(v, dict):
        if "accuracy" in v:
            return flatten_score(v["accuracy"])
        if "score" in v:
            return flatten_score(v["score"])
        vals = [flatten_score(x) for x in v.values() if isinstance(x, (int, float, dict))]
        return sum(vals) / len(vals) if vals else 0.0
    return 0.0


@app.function(
    image=image,
    gpu="A100",
    timeout=3600,
    volumes={"/results": results_vol},
)
def run_one(press_name: str, target_size: int, fraction: float = 0.1) -> dict:
    """Run a single decoding benchmark job."""
    import glob
    import os
    import sys

    sys.path.insert(0, "/eval_repo/evaluation")
    os.chdir("/eval_repo/evaluation")

    from evaluate import EvaluationConfig, EvaluationRunner

    output_tag = f"{press_name}__ts{target_size}__f{fraction:.3f}"

    config = EvaluationConfig(
        dataset=DATASET,
        data_dir=DATA_DIR,
        model=MODEL,
        device="cuda:0",
        press_name=press_name,
        compression_ratio=0.0,  # not used for decoding presses
        target_size=target_size if press_name != "no_press" else None,
        compression_interval=1 if press_name != "no_press" else None,  # trigger eviction every step
        fraction=fraction,
        seed=42,
        output_dir=f"/results/{output_tag}",
    )

    runner = EvaluationRunner(config)
    runner.run_evaluation()

    results_vol.commit()

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
        "target_size": target_size,
        "metrics": metrics,
        "files": result_files,
    }


@app.local_entrypoint()
def main():
    fraction = 0.1

    # Build job list
    jobs = []
    for press_name in PRESSES:
        if press_name == "no_press":
            jobs.append((press_name, 0, fraction))
        else:
            for ts in TARGET_SIZES:
                jobs.append((press_name, ts, fraction))

    print(f"\n{'='*80}")
    print("MergingDecodingPress vs CAMPress — Modal A100")
    print(f"Model: {MODEL} | Dataset: {DATASET}-{DATA_DIR} | fraction={fraction}")
    print(f"Presses: {len(PRESSES)} | Target sizes: {TARGET_SIZES} | Jobs: {len(jobs)}")
    print(f"{'='*80}\n")

    for i, (p, ts, f) in enumerate(jobs):
        tag = f"ts={ts}" if p != "no_press" else "baseline"
        print(f"  [{i+1:2d}] {p:<35s} {tag}")
    print()

    results = list(run_one.starmap(jobs, return_exceptions=True))

    # Save results locally
    output_dir = pathlib.Path("evaluation/results_decoding")
    output_dir.mkdir(parents=True, exist_ok=True)

    table = {}
    errors = []
    for i, r in enumerate(results):
        press_name, ts, _ = jobs[i]
        label = f"{press_name} (ts={ts})" if press_name != "no_press" else "no_press"

        if isinstance(r, Exception):
            errors.append({"variant": press_name, "target_size": ts, "error": str(r)})
            continue
        if "error" in r.get("metrics", {}):
            errors.append({"variant": press_name, "target_size": ts, "error": r["metrics"]["error"]})
            continue

        for rel_path, content in r.get("files", {}).items():
            file_path = output_dir / rel_path.split("/", 1)[-1] if "/" in rel_path else output_dir / rel_path
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(content)

        m = r["metrics"]
        tasks = sorted(m.keys())
        scores = [flatten_score(m[t]) for t in tasks]
        mean = sum(scores) / len(scores) if scores else 0.0
        table[label] = {"press_name": press_name, "target_size": ts, "mean": round(mean, 2), "n_tasks": len(tasks)}

    # Print summary table
    print(f"\n{'='*80}")
    print(f"{'Variant':<50} {'Mean':>6}  {'n':>3}")
    print(f"{'-'*80}")
    for label, r in sorted(table.items(), key=lambda x: (x[1]["target_size"], -x[1]["mean"])):
        print(f"{label:<50} {r['mean']:>6.1f}  {r['n_tasks']:>3}")

    # Head-to-head comparison: MergingDecodingPress vs CAMPress vs DecodingPress (hard evict)
    print(f"\n{'='*80}")
    print("Head-to-head: MergingDecoding vs CAM vs HardEvict (Knorm scorer)")
    print(f"{'-'*80}")
    print(f"{'TS':>5} {'~CR':>5} {'MergDec':>8} {'CAM':>8} {'HardEv':>8} {'Δ vs CAM':>9} {'Δ vs Hard':>10}")
    print(f"{'-'*80}")

    for ts in TARGET_SIZES:
        approx_cr = round(1.0 - ts / 4096, 2)
        md_label = f"merging_decoding_knorm (ts={ts})"
        cam_label = f"cam_knorm (ts={ts})"
        de_label = f"decoding_knorm (ts={ts})"

        md_val = table.get(md_label, {}).get("mean", "—")
        cam_val = table.get(cam_label, {}).get("mean", "—")
        de_val = table.get(de_label, {}).get("mean", "—")

        d_cam = f"{md_val - cam_val:+.1f}" if isinstance(md_val, float) and isinstance(cam_val, float) else "—"
        d_hard = f"{md_val - de_val:+.1f}" if isinstance(md_val, float) and isinstance(de_val, float) else "—"
        print(f"{ts:>5} {approx_cr:>5} {md_val:>8} {cam_val:>8} {de_val:>8} {d_cam:>9} {d_hard:>10}")

    # Same for SnapKV
    print(f"\n{'='*80}")
    print("Head-to-head: MergingDecoding vs CAM vs HardEvict (SnapKV/AdaKV scorer)")
    print(f"{'-'*80}")
    print(f"{'TS':>5} {'~CR':>5} {'MergDec':>8} {'CAM':>8} {'HardEv':>8} {'Δ vs CAM':>9} {'Δ vs Hard':>10}")
    print(f"{'-'*80}")

    for ts in TARGET_SIZES:
        approx_cr = round(1.0 - ts / 4096, 2)
        md_label = f"merging_decoding_adakv_snapkv (ts={ts})"
        cam_label = f"cam_adakv_snapkv (ts={ts})"
        de_label = f"decoding_adakv_snapkv (ts={ts})"

        md_val = table.get(md_label, {}).get("mean", "—")
        cam_val = table.get(cam_label, {}).get("mean", "—")
        de_val = table.get(de_label, {}).get("mean", "—")

        d_cam = f"{md_val - cam_val:+.1f}" if isinstance(md_val, float) and isinstance(cam_val, float) else "—"
        d_hard = f"{md_val - de_val:+.1f}" if isinstance(md_val, float) and isinstance(de_val, float) else "—"
        print(f"{ts:>5} {approx_cr:>5} {md_val:>8} {cam_val:>8} {de_val:>8} {d_cam:>9} {d_hard:>10}")

    if errors:
        print(f"\n--- Errors ({len(errors)}) ---")
        for e in errors:
            print(f"  {e['variant']} ts={e['target_size']}: {e['error']}")

    # Save summary JSON
    summary = {"table": table, "errors": errors, "fraction": fraction, "target_sizes": TARGET_SIZES}
    summary_path = output_dir / "decoding_experiment_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\nResults saved to {output_dir}/")
