# Self-Fulfilling Verification in Robotic Manipulation — Gate-0

This repository runs one CPU-only MuJoCo experiment: whether a physical verification probe can complete a connector insertion and make post-verification success look attributable to the preceding task action.

## Local setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## Reproduce

```powershell
# Engineering smoke test (20 parents, separate seeds)
.\.venv\Scripts\python.exe -m gate0.run --config configs/gate0_smoke.yaml

# Frozen confirmatory Gate-0 (500 parents, 2,500 branches)
.\.venv\Scripts\python.exe -m gate0.run --config configs/gate0_main.yaml

# Re-analyze an existing immutable run
.\.venv\Scripts\python.exe -m gate0.analyze --run runs/<run_id>

# Verification suite
.\.venv\Scripts\python.exe -m pytest
```

Every run stores its frozen config and canonical hash in `runs/<run_id>/manifest.json`. Debug, smoke, and main seed ranges do not overlap. Generated claims are bounded by `reports/gate0_report.md` and `reports/gate0_decision.md`; a GO result does not authorize Gate-1.
