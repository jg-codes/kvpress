# kvpress Fork — Project Archive

Personal fork of [NVIDIA/kvpress](https://github.com/NVIDIA/kvpress) for the **MergingPress** contribution.

## Status (2026-05-30)

**[PR #219](https://github.com/NVIDIA/kvpress/pull/219)** — submitted, awaiting maintainer re-review after responding to SimJeg's simplification feedback. Three commits, +356/−0 across 6 files, all marked 🤖🤖🤖.

**Linked [Issue #214](https://github.com/NVIDIA/kvpress/issues/214)** — open; title + body rewritten in human voice on 2026-05-30.

## Contribution: MergingPress

A prefill-time wrapper that replaces hard eviction with merge-on-evict: each evicted token's value is folded into its most cosine-similar surviving neighbor via similarity-weighted blending. Keys preserved by default (RoPE-safe).

**API:** two methods — `compress(...)` (mirrors `ScorerPress.compress` with `merge(...)` inserted before the gather) and `merge(keys, values, indices)` (the kernel).

**Empirical record** (RULER-4096, Qwen3-8B, KnormPress inner scorer, paired comparison, seed 42):

| CR | MergingPress(KnormPress) | KnormPress | Δ |
|----|--------------------------|------------|------|
| 0.25 | 88.3 | 87.2 | +1.1 |
| 0.50 | 72.2 | 68.3 | +3.9 |
| 0.75 | 38.6 | 32.6 | +6.0 |
| 0.88 | 13.6 | 8.9 | +4.7 |

## Timeline

| Date | Event |
|---|---|
| 2026-04-12 | Issue #214 opened |
| 2026-04-15 | PR #219 opened — initial 3-mode design (ScorerPress + AdaKV mask-based + DMSPress hook-based) |
| 2026-04-16 | SimJeg requests benchmarks with AdaKV(SnapKV), KVzap, DMS(KVzap); jagmarques engages on related work |
| 2026-04-20 | Branch updated to wrap KVzap + DMS variants |
| 2026-05-28 | SimJeg's review: simplify to ScorerPress-only, 2-method API, mark README + docstring 🤖🤖🤖 |
| 2026-05-30 | Rebased onto current `upstream/main` (`243f71b`); old branch had silently reverted parts of #221, #224, #227, #229. Force-pushed clean 3-commit history. PR body + issue body + title all rewritten in human voice |

## Branch map

| Branch | Purpose | Status |
|---|---|---|
| `pr/merging-press-v2` | **Canonical PR head** — mirrors `origin/pr/merging-press` and drives PR #219 | active |
| `pr/merging-press` (local only) | Old pre-rewrite PR branch (8 commits, scope-creep) | obsolete — safe to delete |
| `dev/boltzmann-stack` | Research home — BoltzmannPress + Modal/Kaggle evaluators + analysis scripts | active |
| `dev/psmr` | Older tiered key-merging experiment (RoPE-safe positions) | archived |
| `dev/merging-base-press`, `dev/merging-hook-composition` | Earlier exploratory branches | superseded |
| `feature/merging-press`, `merging-press` | Earlier naming attempts | superseded |
| `experimental/archive` | Catch-all archive | superseded |
| `fix/criticalkv-head-dim` | Unrelated fix experiment | parked |
| `main` (local) | 3 commits ahead of `origin/main`; predates upstream rebases | stale — discard or rebase |

## Repo conventions followed

- DCO sign-off (`git commit -s`) on every commit
- `🤖🤖🤖` marker on agent-authored commits + issue/PR titles + maintainer-facing comments (per `AGENTS.md`)
- Stock kvpress PR checklist (matches Saransh's CAMPress #196)

## Outstanding

- **Awaiting SimJeg's re-review** of PR #219 — the reply comment is posted (https://github.com/NVIDIA/kvpress/pull/219#issuecomment-4583084369)
- Potential follow-up PR: `MergingAdaKVPress` (mask-based composition with per-head budget allocation — prototyped earlier on this branch, see April 16 PR comments)
- Potential follow-up PR: decoding-time MergingPress (complements CAMPress #196)
- BoltzmannPress (local research, `dev/boltzmann-stack`) — not yet evaluated for upstream contribution

## Key local artifacts

- **Project guide:** [CLAUDE.md](CLAUDE.md) — research findings, validated parameters, disproven hypotheses
- **Research evaluators:** `evaluation/modal_*.py`, `evaluation/kaggle_*.py`, `evaluation/cross_scorer_eval.py`
- **Analysis scripts:** `evaluation/analyze_distractor_results.py`, `evaluation/analyze_merge_mechanisms.py`
- **VPS runner:** `evaluation/vps_run.sh`
- **Planning notes (outside repo):** `~/.claude/plans/my-kvpress-pr-gets-hazy-deer.md`

## Recovery / re-engagement

To resume work on the PR:

```bash
cd /Users/joga/Developer/Algo_LLM/kvpress
git checkout pr/merging-press-v2
gh pr view 219 --repo NVIDIA/kvpress --comments
```

To resume boltzmann research:

```bash
git checkout dev/boltzmann-stack
# Modal scripts: evaluation/modal_boltzmann_dms.py, evaluation/modal_dms_merging_v2.py
# Kaggle scripts: evaluation/kaggle_merging_eval.py, evaluation/kaggle_quant_merging.py
```
