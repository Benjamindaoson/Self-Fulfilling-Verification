# Self-Fulfilling Verification in Robotic Manipulation — Gate-0

This repository runs one CPU-only MuJoCo experiment: whether a physical verification probe can complete a connector insertion and make post-verification success look attributable to the preceding task action.

## Result

The frozen Gate-0 decision is **KILL**, not because the physical effect was
absent, but because the original strong claim did not survive:

- the strongest probe produced a verification-induced completion rate of
  `0.137` (95% bootstrap CI `[0.108, 0.169]`);
- causal soundness fell to `0.474`;
- 59 of 61 strong-probe completions were in the predeclared very-near-boundary
  region; and
- full verification traces recovered the pre-verification outcome with `1.000`
  balanced accuracy under the strongest predeclared model suite.

The evidence supports a narrow outcome-contamination result. It does not
support history erasure, information-theoretic non-identifiability, or
generalization beyond this simulated connector task.

## Evidence and data

- Compact evidence bundle: [`artifacts/gate0-main/`](artifacts/gate0-main/)
- Full frozen run: [Hugging Face Dataset](https://huggingface.co/datasets/jlai300/self-fulfilling-verification-gate0)
- Generated report: [`artifacts/gate0-main/reports/gate0_report.md`](artifacts/gate0-main/reports/gate0_report.md)
- Frozen decision: [`artifacts/gate0-main/reports/gate0_decision.md`](artifacts/gate0-main/reports/gate0_decision.md)

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
