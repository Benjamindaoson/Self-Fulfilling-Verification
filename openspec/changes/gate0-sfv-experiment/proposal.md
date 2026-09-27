## Why

Post-verification success can be causally miscredited if a physical verification action itself completes an insertion. A frozen, paired-clone Gate-0 experiment is needed to cheaply falsify that effect before any larger robotics system or scientific claim is authorized.

## What Changes

- Add one CPU-only MuJoCo connector-style insertion environment with randomized natural task rollouts.
- Freeze a privileged-state success predicate, continuous margins, verification strengths, naturalness bins, and GO/CONTINUE/KILL rules before the confirmatory run.
- Clone every task terminal state into no-action, passive, weak, medium, and strong verification branches while preserving full state and trajectories.
- Produce reproducible Parquet ledgers, bootstrap causal metrics, leakage-safe retrospective classifiers, figures, and fixed-format reports.
- Add executable tests and smoke/main commands that audit determinism, state identity, metric correctness, and group isolation.

## Capabilities

### New Capabilities

- `gate0-sfv-experiment`: Reproducible natural-rollout, clone-branch, analysis, and scientific-gate behavior for Gate-0.

### Modified Capabilities

None.

## Impact

- Adds a small Python package, MuJoCo model construction, YAML configurations, tests, and generated run artifacts.
- Runtime dependencies are limited to MuJoCo and CPU scientific Python packages; no GPU, ROS, external model, network API, or distributed service is introduced.
- Generated evidence is written under `runs/<run_id>/`; source-controlled configuration and OpenSpec artifacts remain project-local.
