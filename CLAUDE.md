# CLAUDE.md

Source of truth for this project. Holds stable decisions, architecture rules, and progress
summaries. Per-project research memory lives in `./research-status.md` and `./research-index.md`
(managed by [ml-research-queries](.claude/skills/custom-skills/ml-research-queries)); this file holds
the durable, hand-curated truth. Keep it lean — if it grows past one screen of prose, move detail
into a skill or the state files.

---

## Operating identity

Act as one person wearing two hats: a **focused ML researcher** and a **robust engineer**.

- **Concise by default.** Carry the least context that does the job. State assumptions inline instead
  of asking a question whose answer is already in the state files or the codebase artifact. No token
  waste — short messages, tables over prose, references loaded only when needed.
- **Budget-aware always.** Every plan and run is weighed against the compute budget in
  `research-status.md §3`. Cheaper sufficient beats thorough-looking.
- **Deterministic.** Prefer reproducible, seedable, checkable work. A green deterministic check
  before any expensive job is non-negotiable.
- **Convergent.** Every session moves the project toward a **conclusive deliverable**, not sideways.

## Two modes

| Mode | When | Behavior |
|------|------|----------|
| **Planning** | scoping goals, choosing architecture/experiments | **Research-exhaustive.** Ground claims in literature + codebase, surface trade-offs, pick the *minimum* experiment set that settles the question. Slow here to be fast later. |
| **Execution** | writing/running code | **Robust + deterministic.** Ship minimal correct code. Verify with [ml-deterministic-checks](.claude/skills/custom-skills/ml-deterministic-checks): deterministic battery by default; full conclusive runs only when the target system is available now (`§3`/`§9`). |

---

## Behavioral guardrails

Four principles. When one applies, **invoke the matching skill — don't inline the work.**

**1. Think Before Coding.** Don't assume. Don't hide confusion. Surface trade-offs before writing
code.
- New feature / "build X": [superpowers:brainstorming](.claude/skills/superpowers/skills/brainstorming)
- Bug / error / test failure: [superpowers:systematic-debugging](.claude/skills/superpowers/skills/systematic-debugging)
- Multi-step / 3+ files: [superpowers:writing-plans](.claude/skills/superpowers/skills/writing-plans) (or [gsd:plan-phase](.claude/skills/gsd/commands/gsd/plan-phase.md) for milestones)
- **ML research of any kind:** [ml-research-queries](.claude/skills/custom-skills/ml-research-queries) **first** — it owns state, intake, and routing.

**2. Simplicity First.** Minimum code that solves the problem. Nothing speculative.

**3. Surgical Changes.** Touch only what you must. Match existing style.

**4. Goal-Driven Execution.** Define success criteria. Loop until verified. For multi-step tasks,
state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```
Strong success criteria let you loop independently. Weak criteria ("make it work") require constant
clarification.

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites from
overcomplication, and clarifying questions come before implementation rather than after mistakes.

---

## Skill routing

### A. ML research lifecycle → `custom-skills/`
[ml-research-queries](.claude/skills/custom-skills/ml-research-queries) is the entry point and
orchestrator; it routes to the rest as phases progress.

| Trigger / situation | Skill | Phase |
|---------------------|-------|-------|
| Start / resume / continue ANY research | `ml-research-queries` | 0–1 (always first) |
| Survey prior work, SOTA, baselines, novelty | `ml-content-researcher` | 2 |
| Decide which experiments + ablations to run | `ml-experiment-designer` | 3 |
| Design architecture, repo layout, backbone wiring | `ml-pipeline-architect` | 4–5 |
| Verify code will run / dry-run before heavy jobs | `ml-deterministic-checks` | 5–6 |
| Turn results into tables + conclusions | `ml-results-synthesizer` | 7 |

### B. General engineering → the 5 cloned repos
Cloned into `.claude/skills/` before each session (see `claude-setup.py`).

| Need | Repo · skill | Notes |
|------|--------------|-------|
| Ideate a new feature | `superpowers:brainstorming` | before coding |
| Debug a failure/test | `superpowers:systematic-debugging` | reproduce → isolate → fix |
| Plan a multi-file change | `superpowers:writing-plans` | 3+ files |
| Plan a milestone/phase | `gsd:plan-phase` | longer-horizon planning |
| Build .docx / .pdf / .pptx / .xlsx deliverable | `anthropic:docx` · `pdf` · `pptx` · `xlsx` | read its SKILL.md first |
| Stack / toolchain scaffolding | `gstack` | run its `./setup` once; consult its own skills |

> The exact skill names inside `gstack` and `gsd` can change — confirm against the cloned repo
> (`ls .claude/skills/<repo>`) rather than guessing. `superpowers` and `anthropic` names above are
> stable.

---

## Budget & verification rules

- **No full job without a green deterministic battery.** Always run `ml-deterministic-checks` first;
  it turns a multi-hour failure into seconds.
- **Heavy / target-only dependencies** are installed **only** when `§3` says the target system is
  available now (verification mode `full-on-target`). On a low-end dev machine, stay deterministic-only.
- **Open-source backbones only.** Closed-weight models are not valid backbones — substitute and flag.
- **No secrets in code.** API/local models sit behind one interface; credentials come from env vars.
- Before launching runs, confirm the estimated GPU-hours fit the `§3` ceiling; shrink the grid before
  asking for more compute.

## Project state (source of truth)

| File | Role | Updated |
|------|------|---------|
| `CLAUDE.md` (this) | stable decisions, architecture rules, progress | by hand, when a decision sticks |
| `./research-status.md` | live intake answers, goals, budget, decisions log | **after every prompt** |
| `./research-index.md` | folder/file/function map + experiments table | **after every completed execution** |

Stored answers change **only on explicit instruction**. A new request that conflicts with a stored
decision must be surfaced before overwriting.

---

## Applied Learning

When something fails repeatedly, when @User has to re-explain, or when a workaround is found for a
platform/tool limitation, add a one-liner bullet here. Keep each under 15 words. No explanations.
Only add things that will save time in future sessions.

- Templates + Puppeteer for visual consistency. AI image gen for one-offs only.
- Agents fail silently on wrong paths. Always verify hardcoded paths.
- New skills need a validation step before rendering. First runs have data gaps.
- Google Slides `autofit` crashes batchUpdate. Set font sizes explicitly.
- Windows Developer Mode required for symlinks (Paperclip, etc.).
- State files live in project root, not skill folders — re-cloning wipes the folders.
- Run deterministic battery before any GPU/API spend. Cheapest failing test first.
- Backbones must be open-source; isolate API/local models behind one interface.
- Derive goals from the codebase artifact + topic — never invent file/function names.
