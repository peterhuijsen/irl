# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Layout

This repo holds coursework for Leiden University's Reinforcement Learning course (Bachelor AI). Each `assignment_N/` directory is an independent Python project with its own `pyproject.toml` and `uv.lock` — there is no top-level project. Always `cd` into the relevant assignment directory before running commands.

- `assignment_1/` — Jupyter notebooks only (`assignment_1a.ipynb`, `assignment_1b.ipynb`).
- `assignment_2/` — Tabular RL on a Shortcut gridworld (Q-learning, SARSA, Expected SARSA, n-step SARSA, decaying-epsilon extension). Experiment driver: `ShortCutExperiment.py`. Agents: `ShortCutAgents.py`. Envs: `ShortCutEnvironment.py` (also `WindyShortcutEnvironment`).
- `assignment_3/` — Model-based RL on a Windy Gridworld (Dyna, Prioritized Sweeping). Skeleton from course staff (Thomas Moerland, Leiden); fill in `MBRLAgents.py` and `MBRLExperiment.py`. Helper plotting utilities in `Helper.py`.

The `main.py` files in assignments 2 and 3 are placeholder stubs — real entry points are `ShortCutExperiment.py` and `MBRLExperiment.py`.

## Commands

Dependency management is `uv` (Python ≥3.12). Run inside the relevant assignment directory:

```
uv sync                          # install deps from uv.lock
uv run python ShortCutExperiment.py    # assignment_2 experiments
uv run python MBRLExperiment.py        # assignment_3 experiments
uv run jupyter lab               # assignment_1 notebooks
```

There are no tests or linters configured.

## Architecture Notes

**Assignment 2** uses a shared `Agent` base class in `ShortCutAgents.py`; all agents implement `select_action(state)` and `update(...)`. `ShortCutExperiment.py` orchestrates runs via `AgentType` / `ExecutionType` enums and parallelizes repetitions with `joblib.Parallel`. Learning curves are smoothed with `scipy.signal.savgol_filter` and plotted via the local `LearningCurvePlot` helper. A `global_seed = 42` is set at module scope — keep seeding deterministic when modifying.

**Assignment 3** mirrors the pattern: `DynaAgent` / `PrioritizedSweepingAgent` in `MBRLAgents.py` interact with `WindyGridworld` in `MBRLEnvironment.py`. Plotting/smoothing helpers live in `Helper.py` (`LearningCurvePlot`, `smooth`). The experiment sweeps `wind_proportions` × `n_planning_updates`.
