---
pretty_name: Self-Fulfilling Verification Gate-0
language:
- en
tags:
- robotics
- embodied-ai
- mujoco
- causal-inference
- failure-analysis
- reproducible-research
size_categories:
- 1K<n<10K
---

# Self-Fulfilling Verification Gate-0

This dataset is the complete frozen main run for a CPU-only MuJoCo study of
verification-induced outcome changes in connector insertion.

## Research question

Can a physical action used to verify task success complete an insertion itself,
making post-verification success look attributable to the preceding task
action? If so, can the pre-verification outcome still be recovered from the
resulting state or trajectory?

## Dataset scope

- 500 natural parent terminal states.
- Five paired verification branches per parent.
- 2,500 verification branches.
- 3,000 trajectories including the parent task trajectories.
- 500 MuJoCo integration-state snapshots.
- Leakage-safe 350/150 parent-level train/test split.
- Frozen configuration, manifest, metrics, bootstrap results, reports, and
  figures.

All observations are generated in simulation. The dataset contains no human
participants, personal information, camera recordings, or real-robot data.

## Directory layout

```text
config.yaml
manifest.json
figures/
reports/
results/
snapshots/
trajectories/
```

The `results/` directory contains parent and verification ledgers, split
assignments, recoverability outputs, and aggregate metrics. `trajectories/`
contains compressed NumPy traces, and `snapshots/` contains frozen MuJoCo state
arrays used to create paired branches.

## Main result and scientific status

The strongest verification probe produced a verification-induced completion
rate of `0.137` with a 95% bootstrap interval of `[0.108, 0.169]`, while causal
soundness fell to `0.474`. However, 59 of 61 induced completions were in the
predeclared very-near-boundary region, and the strongest predeclared full-trace
model recovered the pre-verification outcome with `1.000` balanced accuracy.

The frozen terminal decision is therefore:

```text
SCIENTIFIC_STATUS = KILL
TERMINATION_GATE = GATE_0
```

The data supports a narrow outcome-contamination finding. It does not establish
history erasure, information-theoretic non-identifiability, widespread robot
failure, real-robot validity, or generalization beyond this connector model.

## Reproduction

Source code and exact setup instructions:

https://github.com/Benjamindaoson/Self-Fulfilling-Verification

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m gate0.analyze --run <downloaded-run-directory>
```

Re-analysis validates the frozen config, input ledgers, snapshots, trajectories,
dependency provenance, and source provenance before regenerating outputs.
