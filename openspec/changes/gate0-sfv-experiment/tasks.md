## 1. Project and Frozen Contract

- [x] 1.1 Add isolated Python packaging, dependency lock inputs, ignore rules, and runnable documentation
- [x] 1.2 Add distinct smoke/debug/main YAML configurations with frozen predicate, randomization, verification, analysis, and gate values
- [x] 1.3 Initialize CodeGraph after the code skeleton exists and create the validated repository architecture artifact

## 2. Test-Driven Simulation Core

- [x] 2.1 Write and observe failing tests for the success predicate and canonical manifest hash
- [x] 2.2 Write and observe failing real-MuJoCo tests for integration-state identity, deterministic clone replay, immutable `Y_pre`, V0 behavior, and parent linkage
- [x] 2.3 Implement the minimal MuJoCo connector, natural task controller, settling, snapshots, five verification branches, full trajectories, and ledgers to pass the tests

## 3. Test-Driven Analysis

- [x] 3.1 Write and observe failing tests for VIC, VID, transition matrices, bootstrap intervals, and leakage-free group splitting
- [x] 3.2 Implement R1/R2/R3 feature extraction, three classifier families for both targets, metrics, plots, reports, and frozen gate decision to pass the tests

## 4. Experimental Execution

- [x] 4.1 Run the full test suite and a 20-parent smoke experiment; repair failures and audit all sanity checks
- [x] 4.2 Run a separate-seed debug calibration sweep and freeze physically distinct, non-trivial main probe strengths
- [x] 4.3 Run at least 500 natural parent rollouts with five paired branches and re-run analysis from saved ledgers

## 5. Independent Verification and Delivery

- [x] 5.1 Audit generated manifests, hashes, counts, transitions, split isolation, required files, prohibited claims, and exact reproduction commands
- [x] 5.2 Run final tests and smoke/local commands, validate OpenSpec, review the Git diff for secrets or unsupported claims, and create a reproducible Git commit
