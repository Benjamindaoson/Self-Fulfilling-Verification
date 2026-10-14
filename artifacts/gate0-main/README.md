# Frozen Gate-0 evidence bundle

This directory contains the compact, Git-friendly evidence bundle copied
byte-for-byte from the frozen main run:

`20260927T023524Z_gate0-main-frozen-v2_18b4a30e`

The source copies remain under the ignored local `runs/` directory. SHA-256
comparisons were performed before publication to confirm that every copied
artifact matches its source.

## Contents

- `manifest.json`: run, environment, source, dependency, and ledger provenance.
- `config.yaml`: frozen confirmatory configuration.
- `results/metrics.json`: complete computed metrics and gate outputs.
- `results/bootstrap_results.json`: bootstrap intervals.
- `reports/gate0_report.md`: detailed scientific report.
- `reports/gate0_decision.md`: automatic terminal decision.
- `figures/`: six generated result figures.

## Evidence boundary

The terminal scientific status is `KILL`. The physical probe can induce
completion and contaminate post-hoc success attribution, but the effect is
concentrated near the success boundary and the full trace retains enough
information to recover the pre-verification outcome. Do not interpret this
bundle as evidence for history erasure, information-theoretic
non-identifiability, real-robot performance, or broad task generalization.

The full 500-parent run, including 3,000 trajectories and 500 simulator
snapshots, is published at:

https://huggingface.co/datasets/jlai300/self-fulfilling-verification-gate0
