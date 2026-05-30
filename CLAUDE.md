# kvpress — Claude Code Project Guide

## Project Context
MergingPress: merge-on-evict wrapper for KV cache compression. Wraps any BasePress scorer, replaces hard eviction with cosine-similarity-weighted merge into nearest surviving token.

## Key Architecture
- **ScorerPress** subclasses (KnormPress, SnapKVPress, etc.): implement `score()` → called via `compress()`
- **Hook-based presses** (DMSPress): override `forward_hook` directly, use `module.masked_key_indices`
- **MergingPress**: wraps either kind via `_is_hook_based_press()` detection
- **Evaluation**: `evaluate.py` CLI + `evaluate_registry.py` for press/dataset/scorer lookup
- **Modal**: `evaluation/modal_*.py` scripts for GPU evaluation on Modal

## Pitfalls & Known Issues
- `import kvpress` MUST be called before using `kv-press-text-generation` pipeline (registers it)
- PyTorch multidimensional indexing: use `tensor[tuple(indices)]` not `tensor[indices]` to avoid deprecation warning
- DMSPress uses thresholds (not compression_ratio) — pass `--threshold` not `--compression_ratio`
- Modal runs MUST use `--detach` or client disconnect kills them
- `modal app logs <app-id>` to monitor; results also saved to Modal volume `/results/`

## MergingPress Research Findings

### Parameter Bounds (empirically validated, RULER-4096/Qwen3-8B, f=0.1)

- **merge_keys=False strictly dominates True** (-2.5pp empirically from earlier ScorerPress runs). RoPE corruption. PSMR (dev/psmr branch) explores tiered key merging restricted to RoPE-safe positions — not yet validated.
- **similarity_threshold=0.0 is optimal for DMSPress** — DMSPress evicts low-importance tokens where even low-w merges reduce error.
- **Ceiling effect at t=-4**: bare DMS is only 0.37pp below no_press → merging has nothing to recover. Delta ≈ 0.
- **t=-3 is the interesting operating point**: bare DMS is 1.47pp below no_press → merging recovers +0.40pp mean.

### Task-Specific Patterns (RULER-4096, Qwen3-8B, DMSPress t=-3)

| Task | Bare DMS | Merge default | Delta | Notes |
|------|----------|---------------|-------|-------|
| fwe | 85.33 | 89.33 | **+4.00** | Consistent benefit — redundant tokens merge well |
| niah_mk1 | 98.15 | 100.0 | **+1.85** | Recovers lost needle info |
| niah_mq | 99.56 | 100.0 | +0.44 | Small but consistent |
| qa_1 | 80.85 | 78.72 | -2.13 | See EXP-07b: isolated merging effect is NOT significant (p=0.10). Half is DMS eviction. |
| qa_2 | 54.55 | 59.4 (f=1.0) | 0.00 | Merging perfectly neutral on qa_2 (23 helps = 23 hurts). All regression is DMS eviction. |
| cwe | 95.58 | 94.42 | -1.16 | Common words share key similarity but different values |
| vt, niah_s*, niah_mk2/3, niah_mv | ~same | ~same | ~0 | Near-ceiling, no room |

**Key insight**: Merging helps retrieval (FWE +4pp, NIAH +1.85pp) and is neutral on QA when properly isolated from DMS eviction. The previously reported qa_1 regression (-2.13pp) was misattributed — half was DMS eviction, and the isolated merging effect is NOT significant (p=0.10). Paper framing: MergingPress is a strict improvement over hard eviction for retrieval tasks, neutral for QA.

**EXP-07b (mechanism isolation)**: Always compare M(DMS) vs bare_DMS to measure merging's effect, NOT M(DMS) vs no_press (which conflates eviction with merging).

### Dominant Strategies

- **MergingPress(DMSPress) at t=-3, default params**: +0.40pp mean over bare DMS. Best general-purpose.
- **merge_fraction=0.75 at t=-3**: +0.29pp mean (less than default). Helps CWE (+1.16) but loses FWE gain.
- **At t=-4**: All merge variants ≈ bare DMS. Not worth the overhead.
- **perturbation_gate=1.0 DISPROVEN** (EXP-06): Kills FWE/NIAH gains (-3.33pp FWE, -1.85pp NIAH) without fixing qa_1 (-2.13pp in both gated and ungated). The qa_1 regression is from cumulative small-error merges, not gatable catastrophic ones.

### Disproven Hypotheses

| Hypothesis | Tested in | Result |
|------------|-----------|--------|
| Compression-as-denoising (distractor removal) | EXP-07 (n=500) | NULL — eviction profiles identical between merge_better/worse |
| Perturbation gating fixes qa_1 | EXP-06 | DISPROVEN — kills gains without fixing regression |
| qa_2 improves with merging | EXP-07 (n=500) | DISPROVEN — f=0.1 signal was sampling noise |
| Merging causes qa_1 regression | EXP-07b isolation | MISATTRIBUTED — isolated effect p=0.10, half was DMS eviction |

## Evaluation Checklist
1. Always track wall-clock time (model_load, inference) alongside accuracy
2. Compare per-task, not just mean — merging may help retrieval but hurt reasoning
3. Use paired comparisons (same seeds, same data fraction) for valid deltas
4. Log compression ratios — DMSPress CR varies by content
5. Validate f=0.1 signals at full scale (n=500+) before claiming results — qa_2 false positive was a lesson
