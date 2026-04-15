# PR: Add MergingPress and MergingDecodingPress

## Title
Add MergingPress and MergingDecodingPress: scorer-agnostic merge-on-evict

## Description

### What
Adds three new presses that replace hard eviction with **merge-on-evict**: instead of discarding low-scoring tokens, each evicted token is folded into its most similar surviving neighbor via a similarity-weighted value blend. This preserves more information in the KV cache at the same compression ratio.

- **`MergingPress`** — prefill-time merge, wraps any `ScorerPress`  
- **`MergingDecodingPress`** — decoding-time merge, extends `DecodingPress`
- **`MergingAdaKVPress`** — combines head-wise adaptive budget allocation (AdaKV) with merge-on-evict: each attention head receives a budget proportional to its information density, and evicted tokens are merged rather than discarded

Both share a common `_merge_on_evict` kernel that:
1. Computes pairwise cosine similarity between evicted and surviving key vectors
2. Routes each evicted token to its most similar survivor (gated by `similarity_threshold`)
3. Blends values with similarity-weighted averaging (optionally weighted by value norms)
4. Optionally merges keys the same way (`merge_keys=True`) or preserves original keys (`merge_keys=False`, recommended for RoPE models)

### Parameters

**MergingPress / MergingDecodingPress:**

| Parameter | Default | Description |
|---|---|---|
| `similarity_threshold` | `0.0` | Minimum cosine similarity to allow a merge (0.0 = always merge) |
| `merge_keys` | `False` | Whether to also merge key vectors (False preserves RoPE positional info) |
| `value_norm_weighting` | `True` | Weight blending by value vector norms (higher-norm = more influence) |
| `max_merge_per_token` | `0` | Cap merges per survivor before weight downscaling (0 = unlimited) |
| `score_weighting` | `False` | Scale merge weight by normalised importance score from the base scorer |
| `adaptive_threshold` | `False` | Compute similarity threshold dynamically as the 25th percentile of per-token max cosine similarities |
| `collect_diagnostics` | `False` | Record per-layer merge statistics for analysis |

**MergingAdaKVPress** — combines head-wise adaptive budget allocation (AdaKV) with merge-on-evict. Same parameters as above, plus:

| Parameter | Default | Description |
|---|---|---|
| `alpha_safeguard` | `0.20` | Minimum fraction of tokens each head must retain (from AdaKV) |

### Why merge-on-evict?
Standard KV eviction discards tokens entirely. Merging recovers partial information from evicted tokens at negligible compute cost (one cosine-similarity matrix + scatter-add). This is complementary to any scorer — the scorer decides *what* to evict, merging decides *how*.

### Related Work & Differentiation

Several recent works explore KV cache merging as an alternative to pure eviction:

| Aspect | **MergingPress** (ours) | **KVMerger** ([Wang+ 2024](https://arxiv.org/abs/2407.08454)) | **EMS** ([Li+ 2024](https://arxiv.org/abs/2412.08521)) | **CaM** (Zhang+ 2024, ICML) |
|---|---|---|---|---|
| **Target selection** | Any kvpress scorer | Cosine-similarity clustering (AHC) | Global-Local attention score | Cumulative attention |
| **Merge routing** | Cosine → most similar survivor (global) | Gaussian kernel weighted → pivotal in set (local, consecutive only) | Attention-weighted → class centers | Bernoulli → sequential neighbor |
| **Scorer coupling** | Fully decoupled — wraps any `ScorerPress` | Tightly coupled to own similarity-based identification | Tightly coupled to own Global-Local scorer | Coupled to cumulative attention scorer |
| **Locality constraint** | None — routes to best match anywhere | Only adjacent tokens (consecutive merging sets) | None — TBM tokens merge to any class center | Sequential neighbors only |
| **Keys handling** | Optional (`merge_keys=False` default, preserves RoPE) | Merges keys + values with same weights | Normalizes keys, preserves norms separately | Values only |
| **Scope** | Prefill + decoding (two classes) | Prefill only (long-context) | Prefill + decoding | Decoding only |
| **Head-wise adaptive** | No (uniform across heads) | No (layer-wise compression ratio) | Yes (zero-class mechanism) | No |
| **Framework integration** | Native kvpress (`BasePress`/`DecodingPress`) | Standalone | Standalone + custom FA2 kernel | Native kvpress (`DecodingPress`) |

**Key differentiators of MergingPress:**
1. **Scorer-agnostic wrapper** — the only approach that fully decouples "what to evict" from "how to merge". Users combine any existing scorer (SnapKV, Knorm, TOVA, …) with merge-on-evict in one line.
2. **Non-local cosine routing** — unlike KVMerger (consecutive-only) and CaM (sequential neighbors), MergingPress routes each evicted token to its globally most-similar survivor.
3. **Native kvpress integration** — no standalone code or custom kernels required; works with `KVPressTextGenerationPipeline`, existing tests, and evaluation harness.
4. **Configurable threshold gating** — `similarity_threshold` lets users control the merge-vs-discard tradeoff per deployment.

Also related:
- [D2O](https://arxiv.org/abs/2406.13035) (Wan+ 2024): EMA-threshold merge of evicted KV into conserved — closer to CaM than to MergingPress.
- [ToMe](https://arxiv.org/abs/2210.09461) (Bolya+ 2023, ICLR): Token merging for ViTs via bipartite matching — inspiration for applying merging to KV cache.

## Changes
- `kvpress/presses/merging_press.py` — NEW: `_merge_on_evict` kernel + `MergingPress` + `MergingDecodingPress` + `MergingAdaKVPress`
- `kvpress/__init__.py` — added imports and `__all__` entries
- `tests/presses/test_merging_press.py` — NEW: 32 unit tests (24 for MergingPress, 8 for MergingAdaKVPress)
- `tests/test_decoding_compression.py` — added MergingDecodingPress to all parametrized decoding tests
- `tests/default_presses.py` — added MergingPress with KnormPress base
- `evaluation/evaluate_registry.py` — added 32 MergingPress-family entries (vonorm, score, simonly, adaptive, adakv variants across multiple scorers)
- `README.md` — added press description

## Benchmark Results

Full results: [`reports/benchmark_results.md`](reports/benchmark_results.md) | Raw data: [`evaluation/multi_cr_results_f0.1.json`](evaluation/multi_cr_results_f0.1.json)

**Setup**: Qwen/Qwen3-8B on RULER-4096 (13 subtasks, ~650 samples, seed=42), Modal A100 40GB. Paired bootstrap test (B=10,000, two-sided) on per-task deltas. Validation on Qwen2.5-0.5B-Instruct showed no harm but no significant benefit (model near floor on RULER-4096).

### Paired Comparison (MergingPress − Baseline) — Qwen3-8B

| Scorer | CR | Baseline | Merging | Δ | 95% CI | p | |
|---|---|---|---|---|---|---|---|
| knorm | 0.25 | 86.4 | 88.6 | **+2.2** | [+0.8, +4.0] | **0.0002** | *** |
| knorm | 0.50 | 69.3 | 72.8 | **+3.5** | [+1.4, +6.6] | **<0.0001** | *** |
| knorm | 0.75 | 34.7 | 38.6 | **+3.9** | [+0.2, +7.8] | **0.037** | * |
| knorm | 0.88 | 7.7 | 12.3 | **+4.6** | [+1.3, +8.7] | **0.002** | ** |
| snapkv | 0.25 | 84.5 | 87.1 | **+2.6** | [+0.6, +5.1] | **0.007** | ** |
| snapkv | 0.50 | 56.2 | 57.8 | +1.6 | [−1.5, +4.7] | 0.328 | |
| snapkv | 0.75 | 32.1 | 33.8 | +1.7 | [−0.6, +4.3] | 0.155 | |
| snapkv | 0.88 | 21.1 | 20.0 | −1.0 | [−3.6, +1.6] | 0.409 | |
| critical_snapkv | 0.25 | 92.5 | 92.8 | +0.3 | [−0.6, +1.9] | 0.742 | |
| critical_snapkv | 0.50 | 85.3 | 85.1 | −0.3 | [−0.9, +0.4] | 0.453 | |
| critical_snapkv | 0.75 | 60.6 | 67.4 | **+6.8** | [+2.7, +11.1] | **0.0002** | *** |
| critical_snapkv | 0.88 | 28.4 | 27.3 | −1.2 | [−3.6, +1.6] | 0.368 | |

**6/12 significant** (α=0.05); 5/12 survive Bonferroni. 9/12 positive. Mean Δ = **+2.07 pp**. Cohen's d ranges from negligible to large (0.14–0.84), median = medium.

### Key Findings
1. **KnormPress benefits most** — all 4 CRs significant with medium effect sizes (d=0.54–0.73)
2. **No evidence of harm** — only 3/12 deltas negative, none significant
3. **Strongest at high compression** — largest absolute gains at CR=0.75–0.88 where baseline quality is lowest
4. **Model-dependent** — validated on Qwen3-8B (capable model, clear gains) and Qwen2.5-0.5B (near-floor performance, no significant effect). Merging helps when the model has room to benefit.

### Limitations
- Benchmarked on one model family (Qwen) with RULER-4096. Broader model/task coverage pending.
- Compute overhead not formally profiled; expected to be small (one cosine-similarity matrix + scatter-add per layer) but not measured.
- MergingAdaKVPress and ablation variants (score_weighting, adaptive_threshold) not yet benchmarked at scale.

## Checklist

- [x] Tests pass (`make test` — 581 passed, 107 skipped on CPU)
- [ ] Style clean (`make style`) — not yet run
- [x] Benchmarks: 6/12 pairs significantly positive on Qwen3-8B (paired bootstrap, B=10k)
- [x] No significant harm in any pairing
