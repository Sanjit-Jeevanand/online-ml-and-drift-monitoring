# Online ML Monitoring, Governance & Autonomous Model Lifecycle

This repository implements a **production-oriented machine learning system** that goes far beyond model training.  
Its core goal is not peak offline accuracy, but **trustworthy, observable, and governable ML behavior over time**.

The system is designed to answer a hard real-world question:

> *How do you safely deploy, monitor, retrain, evaluate, and promote machine learning models in a non-stationary environment — without breaking production?*

This project demonstrates a **full ML lifecycle**, including drift detection, shadow deployment, promotion gates, rollback, and autonomous governance execution.

---

## High-Level Architecture

  → Data
  → Validation
  → Feature Contract
  → Model Training (Baseline / Candidate)
  → Versioned Model Registry
  → Online Inference API
  → Structured Logging
  → Snapshot Aggregation
  → Drift Detection
  → Retraining Decision
  → Candidate Evaluation
  → Shadow Deployment
  → Governance State Machine
  → Autonomous Promotion / Rollback

Everything is **artifact-driven**, immutable, and auditable.

---

## Problem Framing

- **Task**: Binary risk prediction (credit default–style)
- **Output**: Calibrated probability (0–1), not a decision
- **Labels**: Delayed (known only after a future horizon)
- **Constraints**:
  - Strict temporal realism
  - No training–serving skew
  - Explicit feature contracts
  - CPU-only, cost-aware serving

---

## Repository Structure

online-ml-and-drift-monitoring/
│
├── artifacts/            # Immutable system outputs (source of truth)
│   ├── features/         # Frozen feature matrices + metadata
│   ├── models/           # Versioned model registry
│   │   ├── baseline/
│   │   ├── xgboost/
│   │   └── lightgbm/
│   │       ├── v1.0.0/
│   │       └── v1.0.1/
│   ├── drift/            # Drift metrics & retraining decisions
│   ├── shadow/           # Shadow evaluation & promotion decisions
│   └── governance/       # Governance state & transition history
│
│
├── config/               # Declarative system configuration
│   ├── features.yaml
│   ├── model.yaml
│   ├── drift.yaml
│   ├── service.yaml
│   └── data.yaml
│
├── data/                 # Raw and validated datasets
│   ├── raw/
│   └── interim/
│       └── splits/
│
├── docker/               # Container build context (support files)
│
├── logs/                 # Structured inference logs (jsonl)
│
├── notebooks/            # Exploration (non-production)
│
├── scripts/              # Executable workflows (entry points)
│   ├── ingest_data.py
│   ├── build_features.py
│   ├── train_baseline.py
│   ├── train_candidate.py
│   ├── evaluate_candidate.py
│   ├── promote_from_shadow.py
│   ├── execute_promotion.py
│   └── auto_governance_runner.py
│
├── snapshots/            # Baseline & current monitoring snapshots
│
├── src/                  # Core system logic (library code)
│   ├── data/             # Loading, validation, splitting
│   ├── features/         # Feature contracts & preprocessing
│   ├── models/           # Training, evaluation, registry
│   ├── inference/        # Online prediction service
│   ├── monitoring/       # Logging, aggregation, drift detection
│   ├── retraining/       # Retraining policy & triggers
│   └── governance/       # State machine & transition rules
│
├── tests/                # Contract & pipeline tests
│
├── Dockerfile            # Production container
├── pyproject.toml        # Project & dependency definition
└── README.md

Artifacts are **never overwritten**. Promotion creates new versions.

---

## Key System Components

### Feature Contract & Preprocessing
- Explicit semantic feature definitions
- Shared preprocessing pipeline for training and inference
- Feature metadata extraction and versioning
- Structural prevention of training–serving skew

### Model Registry
- Immutable, versioned model artifacts
- Semantic versioning (`vX.Y.Z`)
- Models include:
  - trained estimator
  - preprocessing pipeline
  - metrics & calibration
  - lineage metadata
- Rollback is always possible

### Online Inference Service
- FastAPI-based synchronous API
- Strict request/response schemas
- Deterministic preprocessing
- Structured JSON logging
- Containerized, cloud-ready

### Observability & Monitoring
- Per-request structured inference logs
- Windowed aggregation
- Immutable snapshots of system behavior
- Feature and prediction distribution tracking

### Drift Detection
- Statistical drift metrics:
  - KS test (numeric features)
  - PSI (categorical features)
  - Prediction distribution drift
- Explicit threshold policies
- Deterministic drift decisions:
  - `NO_ACTION`
  - `MONITOR`
  - `RETRAIN_RECOMMENDED`
  - `RETRAIN_REQUIRED`

### Retraining & Evaluation
- Retraining only when policy allows
- Candidate models isolated from production
- Head-to-head evaluation against deployed model
- Promotion requires measurable improvement

### Shadow Deployment
- Candidate models run invisibly alongside production
- Real traffic, identical inputs
- Paired request-level comparison
- Promotion gated by:
  - sufficient traffic
  - prediction stability
  - latency safety
  - error parity

### Governance State Machine
- Explicit model states (e.g. `SHADOWING`, `PROMOTABLE`)
- Valid transitions only
- Full state history persisted
- Manual override supported (but audited)

### Autonomous Governance Execution
- Cron-based execution via AWS EventBridge
- Stateless ECS tasks
- Idempotent, safe to rerun
- No 24×7 processes
- Automation without loss of control

---

## Cloud Deployment (AWS)

- **ECS Fargate** — container execution without servers
- **EventBridge** — cron scheduling for governance
- **CloudWatch Logs** — centralized execution logs
- **IAM** — least-privilege task roles
- **ECR** — container registry
- **Scale-to-zero** where possible to minimize cost

This setup mirrors real production systems while remaining student-budget friendly.

---

## Status

✔ End-to-end lifecycle implemented  
✔ Autonomous governance operational  
✔ Promotion blocked correctly under uncertainty  
✔ Cloud deployment validated  
✔ Rollback tested  

The system is complete, extensible, and intentionally conservative.

---