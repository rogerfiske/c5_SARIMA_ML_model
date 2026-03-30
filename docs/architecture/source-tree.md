# Source Tree

## Recommended Layout

```text
c5_SARIMA_ML_model/
├── README.md
├── pyproject.toml
├── poetry.lock
├── .env.example
├── docs/
│   ├── prd.md
│   ├── architecture.md
│   ├── architecture/
│   │   ├── tech-stack.md
│   │   ├── source-tree.md
│   │   └── coding-standards.md
│   ├── stories/
│   └── qa/
├── configs/
│   ├── datasets/
│   ├── features/
│   ├── models/
│   └── runs/
├── data/
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   ├── features/
│   └── forecasts/
├── artifacts/
│   ├── manifests/
│   ├── runs/
│   ├── models/
│   ├── metrics/
│   ├── plots/
│   └── champion/
├── src/
│   └── c5_forecasting/
│       ├── cli/
│       ├── config/
│       ├── domain/
│       ├── data/
│       ├── features/
│       ├── models/
│       ├── ranking/
│       ├── evaluation/
│       ├── registry/
│       └── pipelines/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── regression/
├── docker/
│   ├── Dockerfile
│   └── entrypoint.sh
└── .github/
    └── workflows/
```

## Layout Rules

- `data/raw/` is immutable.
- `configs/` contains only human-edited configuration inputs.
- `artifacts/` contains generated run outputs and manifests.
- `src/c5_forecasting/domain/` owns domain constants and hard rules.
- `src/c5_forecasting/models/` must never bypass the common model interface.
- `tests/regression/` owns no-zero forecast and tie-break stability checks.
