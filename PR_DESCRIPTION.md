# Add MergingPress: Scorer-Agnostic Merge-on-Evict Wrapper

## Summary

`MergingPress` is a composable wrapper that turns any `ScorerPress` hard eviction into **merge-on-evict**: each evicted token is folded into its nearest surviving neighbor rather than being permanently discarded. The wrapper delegates all scoring to the inner press and changes only the eviction step.

## Motivation

Every `ScorerPress` in kvpress shares the same eviction strategy: `score → topk → gather`. The dropped tokens are permanently lost. Meanwhile, published KV-cache merging methods (D2O, KVMerger, KeepKV, KVSlimmer) each bundle their own scorer, preventing reuse of the 16+ scorers already available in kvpress.

`MergingPress` bridges this gap — wrap any existing scorer, replace its hard eviction with merge-on-evict, and preserve information that hard eviction destroys.

## Algorithm

For each evicted token:
1. Find the kept token with highest **cosine similarity** in key-space
2. If similarity ≥ threshold τ (default 0.0 = merge all), merge:
   - **Keys**: score-weighted average — `k_merged = (w_kept * k_kept + w_evict * k_evict) / (w_kept + w_evict)`
   - **Values**: same weighted average, ensuring convex combination within the attention manifold
3. Multi-merge accumulation: when multiple evicted tokens target the same survivor, weights accumulate properly (no magnitude explosion)

The similarity threshold τ ∈ [0, 1] gates semantically distant merges. Use 0.0 to merge all (robust default) or 0.8 for conservative gating.

## Evaluation

### KL Divergence (12 samples × 3 compression ratios, Qwen2.5-7B-Instruct)

| Method | Mean KL ↓ | Δ vs hard eviction | p-value |
|--------|-----------|---------------------|---------|
| SnapKV (hard) | 14.63 ± 4.72 | — | — |
| **SnapKV + MergingPress** | **10.88 ± 1.96** | **−3.75 (−26%)** | **< 0.001** |
| H2O (hard) | 13.96 ± 4.06 | — | — |
| **H2O + MergingPress** | **9.69 ± 1.06** | **−4.26 (−31%)** | **< 0.001** |

### RULER-4096 (Qwen2.5-7B-Instruct)

<!-- TODO: Fill from VM benchmark results -->

| Method | CR=0.25 | CR=0.50 | CR=0.75 |
|--------|---------|---------|---------|
| No compression | 94.4 | — | — |
| SnapKV | 54.4 | 40.3 | 28.8 |
| **Merging+SnapKV** | *running* | *running* | *running* |
| **Merging+KNorm** | *running* | *running* | *running* |

## Changes

- `kvpress/presses/merging_press.py` — New press implementation (156 lines)
- `tests/presses/test_merging_press.py` — 9 unit tests covering:
  - Validation (requires ScorerPress, threshold bounds)
  - Compression ratio delegation
  - Zero-compression identity
  - Integration with KNormPress and SnapKVPress on a real model
  - Merge differs from hard eviction (statistical)
  - Threshold gating behavior
  - Key modification verification
- `kvpress/__init__.py` — Export MergingPress
- `tests/presses/test_presses.py` — Add MergingPress to parametrized test suite

All existing tests pass (496 passed, 98 skipped).

## Design Decisions

1. **Scorer-agnostic**: Wraps any `ScorerPress` subclass without modification
2. **Convex combination**: Keys and values use score-weighted average (not additive), preventing magnitude explosion when multiple tokens merge into one survivor
3. **Per-head processing**: Each attention head merges independently, respecting head-specific attention patterns
4. **Similarity threshold**: Optional gate prevents semantically distant merges from corrupting the cache

## Related

- D2O (Wan et al., 2024) — cosine-similarity target selection (similar matching, but bundles its own scorer)
- KVSlimmer (Wang et al., 2026) — asymmetric key/value merge (similar insight, bundles its own scorer)
- Token Merging / ToMe (Bolya et al., 2023) — bipartite matching in vision transformers

Signed-off-by: Johannes Gast <johannes@gast.dev>
