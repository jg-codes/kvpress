# MergingPress — Project Wrap-Up

> Archival document ("einmotten"). Captures origins, process, results, pitfalls, and future directions so any continuation can pick up where we left off.

**Status**: PR open, awaiting NVIDIA CI  
**Date**: 2025-07-16  

---

## Links

| What | URL |
|---|---|
| **PR** | [NVIDIA/kvpress#219](https://github.com/NVIDIA/kvpress/pull/219) |
| **Issue** | [NVIDIA/kvpress#214](https://github.com/NVIDIA/kvpress/issues/214) |
| **HF leaderboard submission** | [kvpress-leaderboard/discussions/15](https://huggingface.co/spaces/nvidia/kvpress-leaderboard/discussions/15) |
| **Fork** | [jg-codes/kvpress](https://github.com/jg-codes/kvpress), branch `pr/merging-press` |

---

## 1. Origins: ContentAdaptivePress → KL Divergence → MergingPress

### PR #212: ContentAdaptivePress (closed prematurely)

The project started as [PR #212](https://github.com/NVIDIA/kvpress/pull/212) — "Add ContentAdaptivePress and ContentAdaptiveWrapper" — opened on 2026-04-10, closed the next day. It was submitted prematurely before benchmarks were ready, with 529 additions across 6 files. The core idea was sound (content-adaptive compression), but the execution was rushed: no benchmarks, no statistical validation, and a scope that tried to do too much at once (wrapper + standalone + adaptive). **Lesson: never open a PR before you have results.** The branch `feature/content-adaptive-press` still exists on the fork for reference.

This failure prompted a reset: strip the idea down to just the merge kernel, benchmark rigorously, and let the evidence drive the PR scope.

### KL divergence motivation

The initial motivation was measuring **how much information merge-on-evict recovers** compared to hard eviction, using KL divergence between compressed and uncompressed output distributions.

### Early KL divergence findings (Qwen2.5-7B-Instruct, now superseded)

These results were reported in the original issue #214 body (since rewritten) and motivated the entire project:

| Scorer | KL (hard evict) | KL (merge-on-evict) | Reduction | p-value |
|---|---|---|---|---|
| SnapKV | 14.70 ± 2.04 | 10.88 ± 1.96 | **−26%** | < 0.001 |
| H2O | 14.11 ± 1.44 | 9.69 ± 1.06 | **−31%** | < 0.001 |

MergingPress reduced the KL divergence from the uncompressed model by roughly a quarter to a third — meaning the compressed model's output distribution stayed closer to the original.

### Submodularity investigation

H2O's theoretical justification relies on attention being submodular. We tested this empirically:

- **Qwen2-0.5B**: Submodularity held in only **46%** of cases (p < 10⁻⁶ against the null that it always holds)
- **Qwen2.5-7B**: Submodularity held in only **57%** of cases

This was a secondary finding — it challenges H2O's theoretical motivation but didn't directly shape MergingPress's design. Worth revisiting in a follow-up paper if the empirical results on larger models confirm the pattern.

### Why KL was dropped from the PR

The PR pivoted to **task-level accuracy** (RULER benchmark) instead of KL divergence because:
1. KL divergence aggregates across the full vocabulary — it's hard to interpret practically
2. The NVIDIA leaderboard uses RULER accuracy, making it the natural comparison metric
3. Reviewers care about "does the model answer correctly?" not "how different is the distribution?"

**Future direction**: KL divergence remains a useful **mechanistic diagnostic**. A follow-up could report KL alongside accuracy to explain *why* MergingPress helps — it literally preserves more of the original distribution.

---

## 2. What We Built

### Implementation (6 files, +618 lines in PR)

| File | Lines | What |
|---|---|---|
| `kvpress/presses/merging_press.py` | +281 | `_merge_on_evict` kernel + `MergingPress` dataclass |
| `tests/presses/test_merging_press.py` | +322 | 17 test methods (18 pass, 1 skip) |
| `kvpress/__init__.py` | +2 | Import + `__all__` |
| `evaluation/evaluate_registry.py` | +4 | `merging_knorm` + `merging_snapkv` configs |
| `tests/default_presses.py` | +8 | Parametrized test matrix |
| `README.md` | +1 | One-line description |

### Branch history

9 atomic commits on `pr/merging-press` (7 AI 🤖🤖🤖, 2 human). Clean rebase on `upstream/main`.

### Development branches (local)

| Branch | Status | Notes |
|---|---|---|
| `pr/merging-press` | **active** — pushed to origin | The PR branch |
| `merging-press` | old dev branch | Pre-cleanup, has extra eval files. Stash `stash@{0}` |
| `feature/merging-press` | older dev branch | Earlier iteration. Stash `stash@{1}` |
| `experimental/archive` | unknown | May contain early experiments |
| `fix/criticalkv-head-dim` | unrelated | Separate upstream fix |

---

## 3. Key Results

### RULER-4096, Qwen3-8B, f=1.0 (full dataset)

| CR | MergingPress(KnormPress) | KnormPress | Δ | % lift |
|----|--------------------------|------------|------|--------|
| 0.25 | **88.3** | 87.2 | +1.1 | +1.3% |
| 0.50 | **72.2** | 68.3 | +3.9 | +5.7% |
| 0.75 | **38.6** | 32.6 | +6.0 | +18.3% |
| 0.88 | **13.6** | 8.9 | +4.7 | +53.3% |

### Best per-task gains: niah_s1 +21.0 (CR=0.88), vt +21.7 (CR=0.50), cwe +15.9 (CR=0.75)

### Paired bootstrap significance (f=0.1, 3 scorers × 4 CRs = 12 conditions)

- **6/12 significant** at α=0.05 (5 survive Bonferroni)
- **9/12 positive** deltas
- **0/12 significant negatives** — "do no worse" property
- KnormPress benefits most (all 4 CRs significant, d=0.54–0.73)
- CriticalSnapKV spike at CR=0.75: +6.8 pp (d=0.84, largest single effect)

### AdaKV generality (f=0.1, directional)

Positive at CR 0.25–0.75 (+0.8 to +2.6 pp), slight inversion at CR=0.88 (−0.8 pp). Win/loss: 5/0 → 7/2 → 5/6 → 1/7 across CRs.

---

## 4. Workspace Artifacts (not in PR)

These files are gitignored or untracked — the actual evidence backing the PR numbers:

### Untracked files (in working directory)
- `PR_DRAFT.md` — the source for PR #219's body (canonical reference)
- `PR_DESCRIPTION.md` — older, broader draft (superseded by PR_DRAFT.md)
- `evaluation/modal_leaderboard.py` — Modal A100 GPU script for running full-dataset evals

### Git-ignored evaluation data
- `evaluation/results_targeted/ada/` — 8 AdaKV experiment dirs (f=0.1)
- `evaluation/results_targeted/ada_summary.json` — AdaKV summary stats
- `evaluation/results_lb/` — 5 leaderboard result dirs (f=0.01)
- `reports/leaderboard_submission/` — 4 full-fraction result dirs + HF submission script
- `reports/benchmark_results.md` — full f=0.1 statistical analysis (tables, effect sizes, CIs)
- `reports/friends_pitch.md` — non-technical explanation
- `reports/*.log` — QA logs (all clean)

### To preserve before cleanup

If you want to git-clean or reset this workspace, **archive these first**:
1. `evaluation/results_targeted/` — irreplaceable benchmark data (hours of A100 compute)
2. `evaluation/results_lb/` — leaderboard baseline data
3. `reports/leaderboard_submission/` — HF submission package
4. `reports/benchmark_results.md` — full statistical writeup
5. `PR_DRAFT.md` — the canonical PR body

---

## 5. Pitfalls & Lessons Learned

### Process
1. **f=0.1 fast iteration saved 10× compute** — directional results at fraction=0.1 (~650 samples) matched full-dataset trends. Use this pattern for any future kvpress eval.
2. **Paired bootstrap is the right test** — don't eyeball means. B=10,000 resamples with per-task pairing gives rigorous p-values. Pre-built in `reports/benchmark_results.md`.
3. **Modal A100 parallel eval** — 5 jobs × ~1.7h = wall time ~1.7h not ~8.5h. ~$1.62/h per A100. Keep `modal_leaderboard.py` as template.
4. **Clean branch from the start** — the old `merging-press` and `feature/merging-press` branches had 30+ commits including evaluation scripts, JSON results, Modal configs. The final PR branch (`pr/merging-press`) was a clean 9-commit rebase. Start clean next time.
5. **PR_DESCRIPTION.md vs PR_DRAFT.md** — having two overlapping docs caused confusion. Use one canonical file next time.
6. **Issue before PR** — opening issue #214 first gave the PR a `Closes #214` link. Good workflow.

### Technical
1. **`merge_keys=True` hurts** — RoPE corruption costs ~2.5 pp at CR=0.75. Default to `False`.
2. **`value_norm_weighting=True` helps** — ~1.9 pp improvement. Scale merges by value norms.
3. **`similarity_threshold` doesn't matter much** — swept 0.0–0.7, max spread 0.6 pp. Default 0.0 is fine.
4. **`max_merge_per_token=0` (unlimited) is a blind spot** — at CR=0.88, too many evicted tokens pile onto few survivors. Capping merges per survivor (`max_merge_per_token=3–5`) might help. Sweep failed due to registry sync issue — **untested future direction**.
5. **CriticalSnapKV is already strong** — merging adds little at low CR because the baseline already retains 97.5% quality. Merging helps most when the base scorer leaves room for improvement.
6. **AdaKV inversion at CR=0.88** — broad but small (−0.8 pp). The few surviving tokens get diluted by too many merges. `max_merge_per_token` is the likely fix.

### AI workflow
- 7/9 commits AI-authored (🤖🤖🤖), 2 human — the human work was API design, parameter tuning, and proofreading
- AI disclosure in PR: honest about the ratio, well-received
- `replace_string_in_file` tool is unreliable for complex edits — verify with `read_file` or use Python scripts as fallback

---

## 6. Honest Retrospective

### What went well
- The final PR is clean: 6 files, +618 lines, 9 atomic commits. Reviewable.
- Statistical rigour (paired bootstrap, Bonferroni) gave us defensible claims.
- The f=0.1 fast iteration pattern saved real money and time.
- Composable design (wraps any ScorerPress) was the right call — proved it on 3 different scorers.

### What we burned money on
- **29 GPU jobs when 5 sufficed** — launched all 7 presses × 4 CRs + no_press at f=1.0 before confirming that only merging_knorm was needed for the leaderboard. The f=0.1 pairwise comparison covered the PR story. Cost: unnecessary A100 hours.
- **Empty speed_results.json / leaderboard_summary.json** — the runtime profiling pipeline produced empty outputs. We never diagnosed why; it wasn't blocking the PR, so we moved on. The theoretical overhead analysis (2–3% of prefill FLOPs) was used instead. Wall-clock numbers remain an open gap.
- **Lost results to Modal timeouts** — first full eval run: all containers hit the 1h default timeout, zero results saved. Had to re-run everything at 7h timeout.
- **Three overlapping markdown drafts** (ISSUE_DRAFT.md, PR_DESCRIPTION.md, PR_DRAFT.md) that drifted apart. Consolidation came too late.

### The arc
PR #212 (premature, closed day 1) → KL divergence experiments (compelling but niche metric) → RULER benchmarks (the right metric for the leaderboard) → clean rebase → PR #219 (open, awaiting CI). The biggest lesson: **start with the evaluation, not the code.** If we'd run RULER on a simple merge-on-evict prototype first, we'd have known the story (KnormPress benefits most, high CR matters most) before writing 690 lines.

---

## 7. Future Directions

### Near-term (if PR is accepted)
1. **Decoding-time MergingPress** — the `_merge_on_evict` kernel is phase-agnostic. Wrapping `DecodingPress` is straightforward. Deferred from this PR to keep scope focused.
2. **`max_merge_per_token` sweep** — test caps of 3, 5, 10 at CR=0.75–0.88 to fix the dilution problem.
3. **Wall-clock profiling** — `run_speed()` from kvpress's eval tools. Theoretical overhead is ~2–3% of prefill FLOPs but never measured empirically.
4. **More models** — only Qwen3-8B tested at full scale. Llama-3.1-8B and Mistral would strengthen generalization claims.

### Medium-term
5. **MergingPress(KVzapPress)** — wrapping kvzap's MLP-based scorer. If kvzap already leads the leaderboard, adding merge-on-evict should push it further.
6. **KL divergence revisited** — report KL alongside accuracy to explain the mechanism. The early data (−26% to −31% KL reduction) was compelling but dropped for scope. A paper could combine both metrics.
7. **Submodularity paper fragment** — the finding that attention submodularity holds in <60% of cases is independently interesting. Could become a short paper or appendix.

### Revisiting PR #212
- The ContentAdaptivePress idea (content-aware per-layer compression) is still valid. With MergingPress as a foundation, a second PR could add adaptive layer-wise compression ratios. Do it right this time: benchmark first, PR second.

### Long-term
8. **Adaptive merge routing** — instead of max cosine similarity, learn a merge policy (e.g., attention-weighted routing). More complexity, but could beat fixed cosine.
9. **Cross-layer merging** — tokens evicted in layer L could be merged with survivors in layer L+1 instead of within the same layer.

---

## 8. Deduplication Notes

### What to keep
- **`PR_DRAFT.md`** — canonical PR body (already submitted as PR #219 body). Keep for reference.
- **`reports/benchmark_results.md`** — full statistical analysis, the most detailed record of f=0.1 results.
- **This file (`WRAP_UP.md`)** — the archival record.

### What's superseded
- **`PR_DESCRIPTION.md`** — older, broader draft covering MergingDecodingPress and MergingAdaKVPress (which were dropped from the final PR). Content overlaps with PR_DRAFT.md. **Can be deleted** — all useful content preserved in this wrap-up and in PR_DRAFT.md.

### What's safely in git history
- Old branch (`merging-press`, `feature/merging-press`) — 30+ development commits, evaluation scripts, ablation results. Accessible via `git log merging-press` and stashes.
- Original issue #214 body (KL divergence data) — lost from GitHub when we rewrote the issue. **Preserved above in §1.**

---

*Last updated: 2026-04-15. Project lead: Johannes.*
