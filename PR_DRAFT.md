# PR: Add MergingPress: scorer-agnostic merge-on-evict for KV cache compression 🤖🤖🤖

## What

`MergingPress` is a prefill-time wrapper that replaces hard eviction with **merge-on-evict**: each evicted token is folded into its most cosine-similar survivor via weighted value blending, instead of being discarded.

It wraps any `ScorerPress` — scoring is delegated entirely; only the eviction step changes. This makes it composable with all existing scorers (KnormPress, SnapKVPress, etc.).

### How it works

1. Score tokens using the wrapped `ScorerPress`
2. Partition into keep/evict sets by score
3. Compute batched cosine similarity between evicted and surviving keys
4. Route each evicted token to its most similar survivor (gated by `similarity_threshold`)
5. Blend values via similarity-weighted scatter-add (float32 accumulation)
6. Keys are preserved unchanged by default (protects RoPE positional encoding)

### Perturbation bound

For evicted token *i* routed to survivor *j* with cosine similarity *w*:

> ‖ΔO_merge‖ ≤ 1/(1+w) · ‖ΔO_evict‖

At w ≥ 0.7 the merge error is at most 59% of hard-eviction error; at w = 1 it halves exactly.

## Parameters

| Parameter | Default | Description |
|---|---|---|
| `press` | — | Any `ScorerPress` whose scores determine which tokens survive |
| `similarity_threshold` | `0.0` | Minimum cosine similarity to merge (0.0 blocks only opposite-direction) |
| `merge_keys` | `False` | Merge key vectors too (`False` preserves Rotary Positional Encoding info) |
| `value_norm_weighting` | `True` | Scale merge weight by relative value-vector L2 norm |
| `max_merge_per_token` | `0` | Cap merges per survivor to prevent dilution (0 for unlimited) |

### Empirical defaults (RULER-4096, Qwen3-8B)

- `merge_keys=True` hurts quality (−2.5 pp at CR=0.75) — RoPE corruption (?)
- `value_norm_weighting=True` improves accuracy (~1.9 pp)
- `similarity_threshold=0.0` is sufficient — nearly no tokens have negative max similarity; empirical threshold unclear and in general cases may not be required
-  `max_merge_per_token=0` (unlimited) works well up to CR=0.75; at CR=0.88 the AdaKV results below show a broad −0.8 pp regression (1 win / 7 losses), suggesting too many evicted tokens pile onto the same survivors. Capping at 3–5 may help at extreme compression, but generalisation is unclear.

## Benchmark results

RULER-4096, Qwen3-8B, fraction=1.0 (all 13 subtasks), seed=42:

### Average scores

| CR | MergingPress(KnormPress) | KnormPress | Δ | % lift |
|----|--------------------------|------------|------|--------|
| 0.25 | **88.3** | 87.2 | +1.1 | +1.3% |
| 0.50 | **72.2** | 68.3 | +3.9 | +5.7% |
| 0.75 | **38.6** | 32.6 | +6.0 | +18.3% |
| 0.88 | **13.6** | 8.9 | +4.7 | +53.3% |

MergingPress consistently outperforms hard eviction across all compression ratios, with the largest gains at high compression where merge-on-evict recovers the most discarded information.

### Per-task breakdown

| Task | no_press | M+K 0.25 | K 0.25 | Δ | M+K 0.50 | K 0.50 | Δ | M+K 0.75 | K 0.75 | Δ | M+K 0.88 | K 0.88 | Δ |
|------|----------|----------|--------|-----|----------|--------|-----|----------|--------|------|----------|--------|------|
| cwe | 98.9 | **96.9** | 96.7 | +0.2 | **92.4** | 89.2 | +3.1 | **53.9** | 38.1 | +15.9 | **9.8** | 5.9 | +3.9 |
| fwe | 95.3 | **89.7** | 89.4 | +0.3 | **83.7** | 80.9 | +2.9 | **65.3** | 54.9 | +10.4 | **33.2** | 18.6 | +14.6 |
| niah_mk1 | 100.0 | **100.0** | 99.8 | +0.2 | **95.2** | 92.0 | +3.2 | **42.2** | 38.4 | +3.8 | **9.6** | 8.0 | +1.6 |
| niah_mk2 | 100.0 | **93.8** | 92.0 | +1.8 | **46.6** | 39.2 | +7.4 | 2.8 | **3.2** | −0.4 | 0.2 | 0.2 | 0.0 |
| niah_mk3 | 100.0 | **66.8** | 61.8 | +5.0 | **11.6** | 8.4 | +3.2 | 0.8 | **1.2** | −0.4 | 0.0 | 0.0 | 0.0 |
| niah_mq | 99.9 | **99.8** | 99.7 | +0.1 | **94.5** | 92.8 | +1.6 | **47.8** | 37.9 | +9.9 | **8.7** | 5.8 | +3.0 |
| niah_mv | 100.0 | **99.9** | 99.6 | +0.3 | **93.6** | 92.1 | +1.5 | **57.9** | 48.9 | +8.9 | **10.9** | 7.0 | +3.8 |
| niah_s1 | 100.0 | 100.0 | 100.0 | 0.0 | 100.0 | 100.0 | 0.0 | **93.6** | 75.0 | +18.6 | **40.6** | 19.6 | +21.0 |
| niah_s2 | 100.0 | 100.0 | 100.0 | 0.0 | **99.6** | 99.4 | +0.2 | **87.4** | 79.2 | +8.2 | **43.4** | 32.8 | +10.6 |
| niah_s3 | 100.0 | 97.2 | 97.2 | 0.0 | **89.8** | 87.0 | +2.8 | **19.6** | 17.6 | +2.0 | 0.0 | 0.0 | 0.0 |
| qa_1 | 81.6 | **60.0** | 58.4 | +1.6 | **31.2** | 29.4 | +1.8 | **13.8** | 11.8 | +2.0 | **10.8** | 8.6 | +2.2 |
| qa_2 | 63.4 | **47.4** | 46.2 | +1.2 | **26.0** | 24.6 | +1.4 | **11.8** | 11.0 | +0.8 | **10.2** | 9.2 | +1.0 |
| vt | 100.0 | **96.9** | 93.0 | +3.9 | **74.8** | 53.1 | +21.7 | 5.2 | **7.2** | −2.0 | 0.0 | 0.0 | 0.0 |
| **Average** | **95.3** | **88.3** | **87.2** | **+1.1** | **72.2** | **68.3** | **+3.9** | **38.6** | **32.6** | **+6.0** | **13.6** | **8.9** | **+4.7** |

M+K = MergingPress(KnormPress), K = KnormPress. Knorm and no_press baselines from the [kvpress leaderboard](https://huggingface.co/spaces/nvidia/kvpress-leaderboard).

**Key observations:**
- Largest per-task gains at CR=0.50: **vt +21.7**, niah_mk2 +7.4, niah_mk3 +3.2
- At CR=0.75: **niah_s1 +18.6**, cwe +15.9, fwe +10.4, niah_mq +9.9
- At CR=0.88: **niah_s1 +21.0**, fwe +14.6, niah_s2 +10.6
- A few minor regressions at CR=0.75–0.88 on near-zero tasks (niah_mk2/mk3, vt) where both methods could be near the noise floor?

### Scorer generality: AdaKVPress (f=0.1, ~650 samples)

Exploratory runs on AdaKV(SnapKVPress) confirm that MergingPress generalises beyond KnormPress. These used fraction=0.1 (~650 of ~6500 RULER samples), so treat as directional:

| CR | MergingPress(AdaKV) | AdaKV(SnapKV) | Δ | % lift |
|----|---------------------|---------------|------|--------|
| 0.25 | **93.0** | 92.2 | +0.8 | +0.9% |
| 0.50 | **66.6** | 64.0 | +2.6 | +4.1% |
| 0.75 | **39.0** | 37.4 | +1.6 | +4.2% |
| 0.88 | 23.8 | **24.6** | −0.8 | −3.3% |

Pattern matches KnormPress: positive gains at CR 0.25–0.75, with an inversion at CR=0.88 where the merge overhead may dilute the few surviving tokens. Per-task win/loss breakdown: CR=0.25 has 5 wins / 0 losses, CR=0.50 has 7/2, CR=0.75 has 5/6 (net positive due to larger wins on niah_s1 +10.6, vt +12.2), CR=0.88 has 1/7. The CR=0.88 regression (−0.8 pp) is small but broad — suggesting that `max_merge_per_token` capping or a higher `similarity_threshold` could help at extreme compression.

### Computational overhead

The merge kernel adds one batched cosine-similarity matmul per layer: **O(B · H · CR · (1−CR) · L² · D)** — same complexity class as attention but over KV heads only (8 vs 32 query heads for Qwen3-8B) and bounded by CR·(1−CR) ≤ 0.25. Runs **once at prefill**; decoding is unaffected.

Theoretical peak: **~6% of attention FLOPs** at CR=0.50, i.e. **~2–3% of total prefill FLOPs**. No extra forward passes, no learned parameters.

## Changes

| File | Lines | Description |
|------|-------|-------------|
| `kvpress/presses/merging_press.py` | +281 | `_merge_on_evict` kernel + `MergingPress` dataclass |
| `tests/presses/test_merging_press.py` | +322 | 17 tests (validation, correctness, precision, edge cases) |
| `kvpress/__init__.py` | +2 | Import + `__all__` entry |
| `evaluation/evaluate_registry.py` | +4 | `merging_knorm` and `merging_snapkv` configs |
| `tests/default_presses.py` | +8 | Parametrized test matrix entry |
| `README.md` | +1 | One-line description |

**Total: 6 files, +618 lines**

## Design choices vs. related work

| Aspect | MergingPress (this PR) | CAMPress ([#196](https://github.com/NVIDIA/kvpress/pull/196), merged) |
|--------|-------------|----------------------|
| Phase | Prefill | Decoding |
| Merge routing | Position-agnostic (max cosine similarity) | Sequential neighbors |
| Merge weight | Cosine similarity + optional value-norm weighting | Bernoulli sampling from cumulative attention ratio |
| Scorer | Any ScorerPress (composable) | Any ScorerPress via DecodingPress |
| Key handling | Keys preserved by default (RoPE-safe) | Keys not merged |

> **Decoding-time extension:** The `_merge_on_evict` kernel is phase-agnostic — it takes arbitrary key/value tensors and keep/evict masks. Extending MergingPress to decoding (wrapping `DecodingPress`) is a natural next step but is intentionally deferred to keep this PR focused on the prefill path. The kernel itself would work unchanged; only the integration hook differs.

**References:**
- Token Merging — Bolya et al., ICLR 2023 ([arXiv:2210.09461](https://arxiv.org/abs/2210.09461))
- D2O — Wan et al., 2024 ([arXiv:2406.13035](https://arxiv.org/abs/2406.13035))
- KeepKV — Huang et al., 2025 ([arXiv:2504.09936](https://arxiv.org/abs/2504.09936))
- CaM — Yao et al., ICML 2024 ([OpenReview](https://openreview.net/forum?id=LCTmppB165))

## Usage

```python
from kvpress import KnormPress, MergingPress, KVPressTextGenerationPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-8B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

press = MergingPress(KnormPress(compression_ratio=0.5))
pipe = KVPressTextGenerationPipeline(model=model, tokenizer=tokenizer, press=press)
output = pipe("Your long context here...", max_new_tokens=50)
```

## Tests

17 test methods in `tests/presses/test_merging_press.py` — 18 passed, 1 skipped (`test_quantized_cache_compatibility` requires `optimum-quanto`).

Coverage: parameter validation, compression-ratio delegation, identity at zero compression, model forward pass (KnormPress + SnapKVPress), merge-vs-hard-eviction difference, threshold gating, key preservation, fp16/bf16 numerical stability, repeated compression, value-norm weighting, information preservation, batching, `max_merge_per_token` validation + effect, short-sequence edge case, quantized cache compatibility.

## CI

Awaiting `/ok to test` from a collaborator. Local results:
- `ruff check` ✅ — no issues on all changed files
- `pytest tests/presses/test_merging_press.py` ✅ — 18 passed, 1 skipped (no GPU needed for unit tests)
- `make style` / `make test` — not run locally (full suite requires GPU for `default_presses` integration tests)

## AI disclosure

This PR was developed with AI assistance. Commits authored by AI are marked with 🤖🤖🤖. In fact, AI did most of the work. The API design, parameter selection, empirical tuning (...), and docstring proofreading are human contributions.

## Checklist

- [x] Code follows `AGENTS.md` guidelines (dataclass, BasePress, SPDX headers)
- [x] All commits signed off (DCO)
- [x] AI commits marked with 🤖🤖🤖
- [x] `ruff check` passes on all changed files
- [x] 18/19 tests pass locally (1 skipped — requires `optimum-quanto`)
- [x] Added to `kvpress/__init__.py`, `tests/default_presses.py`, `evaluation/evaluate_registry.py`, `README.md`
- [ ] `make style` / `make test` on CI (awaiting `/ok to test`)
