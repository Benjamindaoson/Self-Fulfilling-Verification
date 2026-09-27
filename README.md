# Self-Fulfilling Verification in Robotic Manipulation — Gate-0

This repository runs one CPU-only MuJoCo experiment: whether a physical verification probe can complete a connector insertion and make post-verification success look attributable to the preceding task action.

## Local setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.lock.txt
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
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

Every run requires a clean Git worktree and stores its frozen config hash, source commit, lock-file hash, installed dependency versions, and immutable input-ledger hashes in `runs/<run_id>/manifest.json`. Re-analysis verifies the config, ledgers, snapshots, and every trajectory before writing outputs. Debug, smoke, and main seed ranges do not overlap. Generated claims are bounded by `reports/gate0_report.md` and `reports/gate0_decision.md`; a GO result does not authorize Gate-1.
