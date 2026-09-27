## ADDED Requirements

### Requirement: Frozen experiment contract
The system SHALL load all success, randomization, verification, analysis, and decision parameters from a YAML configuration and SHALL store a stable canonical configuration hash in the run manifest before executing any rollout.

#### Scenario: Reordered mappings have the same manifest hash
- **WHEN** two configuration mappings differ only in key order
- **THEN** the system records the same canonical configuration hash

### Requirement: Natural task terminal states
The system SHALL create every parent terminal state through a complete task rollout whose randomized initial conditions, environment parameters, controller parameters, and termination timing are sampled before task execution and recorded without terminal-state-conditioned selection.

#### Scenario: Parent evidence is persisted
- **WHEN** a task rollout terminates
- **THEN** its seed, sampled parameters, task trajectory, terminal state identifier, immutable `Y_pre`, component margins, normalized margin, snapshot bytes, and snapshot hash are stored

### Requirement: Privileged pre-verification success
The system SHALL compute `Y_pre` and `Y_post` from the same pre-frozen conjunction of minimum insertion depth, maximum lateral error, and maximum angular error, and SHALL expose each continuous component margin.

#### Scenario: Threshold conjunction is enforced
- **WHEN** any one component violates its configured threshold
- **THEN** the success predicate returns false even if the other components pass

### Requirement: Paired verification branches
The system SHALL restore V0, V1, V2, V3, and V4 from identical complete MuJoCo integration-state bytes for each parent, preserve the parent's `Y_pre`, and record every branch timestep and terminal `Y_post`.

#### Scenario: Branches share one causal parent
- **WHEN** five branches are created for a parent terminal state
- **THEN** all five rows have the same parent identifier, initial state hash, and `Y_pre`, while only the verification action differs

### Requirement: Deterministic control replay
The system SHALL reproduce identical V0 trajectories and outcomes when the same model and snapshot are restored twice under deterministic conditions.

#### Scenario: Repeated V0 replay
- **WHEN** V0 is executed twice from the same saved integration state
- **THEN** the serialized trajectories and outcomes are numerically identical

### Requirement: Causal and naturalness metrics
The system SHALL report per-branch transition counts, Verification-Induced Completion among `Y_pre=0` parents with `control_Y_post=0`, Verification-Induced Disruption among `Y_pre=1` parents, causal soundness, credit delta, fixed-bin margin audits, and 95% parent-level bootstrap confidence intervals.

#### Scenario: Paired VIC calculation
- **WHEN** a failed parent remains failed in V0 but succeeds in a probe branch
- **THEN** that probe row contributes one Verification-Induced Completion event

### Requirement: Leakage-safe retrospective recovery
The system SHALL predict both `Y_pre` and `Y_post` under R1 final-state, R2 trajectory-summary, and R3 resampled full-trace regimes using LogisticRegression, RandomForestClassifier, and MLPClassifier with a deterministic group split by `parent_episode_id`.

#### Scenario: No parent crosses split boundary
- **WHEN** train and test indices are generated
- **THEN** their parent identifier sets are disjoint

### Requirement: Evidence bundle
The system SHALL write manifest, parent, branch, and trajectory-index Parquet ledgers, metrics and bootstrap JSON, required PNG/PDF figures, and fixed-format Gate-0 report and decision files beneath one immutable run directory.

#### Scenario: Re-analysis from ledger
- **WHEN** `python -m gate0.analyze --run <run_dir>` is invoked
- **THEN** analysis regenerates metrics, figures, and reports without rerunning simulation or changing the manifest configuration

### Requirement: Automatic scientific gate
The system SHALL emit exactly GO, CONTINUE, or KILL by applying the frozen gate thresholds and SHALL list claims that are prohibited by the observed evidence.

#### Scenario: No natural completion effect
- **WHEN** the configured natural VIC criterion fails
- **THEN** the terminal decision is KILL at GATE_0 and no Gate-1 implementation is authorized

### Requirement: Executable verification
The project SHALL provide tests for the predicate, state identity, branch reproducibility, immutable `Y_pre`, V0 behavior, parent linkage, VIC, VID, group leakage, and stable manifest hashing, plus runnable smoke and main commands in an isolated project environment.

#### Scenario: Fresh local verification
- **WHEN** the documented test and smoke commands run in the project `.venv`
- **THEN** they exercise real MuJoCo state cloning and generate an auditable run bundle
