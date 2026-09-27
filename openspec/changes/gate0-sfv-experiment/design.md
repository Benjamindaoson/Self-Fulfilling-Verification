## Context

The project starts empty and must deliver one CPU-only, falsifiable Gate-0 experiment. The confirmatory run must distinguish task success at task termination (`Y_pre`) from success after a physical verification branch (`Y_post`), preserve paired counterfactual evidence, and prevent threshold, seed, split, or branch-selection leakage.

## Goals / Non-Goals

**Goals:**

- Generate natural connector-insertion terminal states from pre-task randomization in MuJoCo.
- Snapshot the complete integration state once and replay five deterministic verification branches from identical bytes.
- Freeze scientific parameters in YAML and record their canonical hash in every manifest.
- Compute paired causal metrics, bootstrap intervals, naturalness audits, and group-safe CPU classifiers.
- Emit raw trajectories, Parquet ledgers, plots, fixed-format reports, and an automatic terminal decision.

**Non-Goals:**

- Multiple tasks, learned control, robot hardware, cameras, ROS, GPU use, VLA/VLM/LLM models, or Gate-1.
- Information-theoretic non-identifiability claims from finite classifier performance.
- Retuning the frozen main configuration after observing confirmatory outcomes.

## Decisions

1. **Minimal connector model.** Use one free rectangular peg and a four-wall socket built from MuJoCo primitives. A scripted force/torque controller runs before a fixed zero-force settling period. This is simpler to audit than importing a robot arm and still supplies contact, alignment, jamming, and insertion mechanics.
2. **Privileged success predicate.** Compute depth, lateral error, and tilt from simulator state. Success is their frozen conjunction; normalized margin is the minimum normalized component margin. Verification never defines success.
3. **Integration-state cloning.** Serialize `mjSTATE_INTEGRATION` as float64 bytes and hash those exact bytes. Each branch restores the same bytes, calls forward dynamics, and records the restored hash before applying its own action.
4. **Natural pre-task randomization.** Sample initial pose, tilt, friction, clearance, controller gains/noise, target noise, actuation noise, and task/settling durations from fixed ranges before task execution. No terminal-state-conditioned perturbation or selection is allowed.
5. **Verification ladder.** V0 and V1 apply zero force; V2-V4 apply frozen axial force levels for one horizon. A distinct debug seed range is used to choose physically distinguishable but non-trivial levels before the main run.
6. **One-row-per-branch ledgers plus NPZ traces.** Parquet stores searchable scalar evidence; compressed NPZ files store every timestep. This avoids a database and keeps provenance explicit.
7. **Leakage-safe retrospective evaluation.** A single deterministic group split by `parent_episode_id` is reused for R1 final-state, R2 summary, and R3 resampled full-trace features. Logistic regression, random forest, and MLP are all predeclared and attempted; the gate uses the maximum balanced accuracy across this fixed suite as its strongest lightweight baseline for each target/regime. No family is added, removed, or selected after observing the confirmatory results. Majority baselines, split assignments, class counts, and per-parent holdout predictions are persisted; undefined single-class cases are reported as `N/A`, not silently scored.
9. **Fail-closed provenance.** A scientific run requires a clean Git commit and records the dependency-lock hash, installed versions, and immutable simulator-ledger hashes. Re-analysis verifies the config hash, ledger hashes, snapshot state hashes, and every trajectory hash before writing derived evidence.
8. **Frozen decision function.** YAML thresholds map measured H1-H5, naturalness, V0 stability, physical distinctness, and probe plausibility to GO, CONTINUE, or KILL. Reports reproduce the exact inputs and forbidden claims.

## Risks / Trade-offs

- [A primitive connector is not a full manipulator] → Limit the claim to this Gate-0 mechanism and forbid prevalence/generalization claims.
- [Zero gravity and scripted forces simplify hardware physics] → Preserve contact dynamics, record the simplification, and require later mechanism replication only after GO.
- [A finite model suite cannot prove history erasure] → Phrase results as recoverability under tested baselines and explicitly prohibit impossibility claims.
- [Classifier metrics can be unstable with rare classes] → Report class counts, majority baselines, grouped holdout predictions, bootstrap intervals, and `N/A` where undefined.
- [A force probe can become obvious recovery] → Freeze an upper plausibility threshold and KILL if the strong probe trivially completes nearly all failures.
- [V0 drift invalidates causal credit] → Include settling plus an automatic invalidation/KILL check when control completion exceeds its threshold.
