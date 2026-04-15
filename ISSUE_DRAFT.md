# Upstream Issue Draft

## Title
[Feature Request] MergingPress: scorer-agnostic merge-on-evict for KV cache compression

## Body

### Description
I'd like to propose **MergingPress**, a scorer-agnostic wrapper that replaces hard eviction with merge-on-evict: instead of discarding low-scoring tokens, each evicted token is folded into its most similar surviving neighbor via cosine-similarity-weighted value blending.

This is complementary to all existing scorers (SnapKV, Knorm, TOVA, etc.) — the scorer decides *what* to evict, MergingPress changes *how* evicted tokens are handled.

### Motivation
Standard KV eviction discards tokens entirely. For high compression ratios this causes significant information loss. Merging recovers partial information from evicted tokens at negligible compute cost (one cosine-similarity matrix + scatter-add per layer).

### Implementation
- **MergingPress** — prefill-time merge, wraps any `ScorerPress`
- **MergingDecodingPress** — decoding-time merge, extends `DecodingPress`
- Shared `_merge_on_evict` kernel with configurable `similarity_threshold`, `merge_keys`, `value_norm_weighting`

### Relationship to existing merge approaches

| Method | Scorer | Routing | Keys | Scope | Framework |
| --- | --- | --- | --- | --- | --- |
| **MergingPress** | Any kvpress scorer | Cosine → most-similar survivor (global) | Optional (default off, RoPE-safe) | Prefill + decoding | Native kvpress |
| [KVMerger](https://arxiv.org/abs/2407.08454) | Own similarity clustering | Gaussian kernel → pivotal (consecutive only) | Merged | Prefill | Standalone |
| [EMS](https://arxiv.org/abs/2412.08521) | Own Global-Local score | Attention-weighted → class centers | Normalized (norms preserved) | Prefill + decoding | Standalone + custom FA2 |
| CAMPress (#196) | Cumulative attention | Bernoulli → sequential neighbor | Values only | Decoding | Native kvpress |
| [D2O](https://arxiv.org/abs/2406.13035) | Own attention | EMA-threshold merge | Both | Decoding | Standalone |

MergingPress is the only approach that **fully decouples the scorer from the merge strategy**, letting users combine any existing `ScorerPress` with merge-on-evict in one line. Unlike KVMerger (consecutive-only) and CaM (sequential neighbors), it routes evicted tokens to the globally most-similar survivor.

### Benchmarks
[Will add benchmark results here before PR]

I have a working implementation with 18+ tests passing — happy to open a PR if there's interest.
