# Tech Stack

## Decision Summary

This project is a Python 3.11, batch-first, no-UI forecasting platform with Dockerized
execution and GitHub Actions for CI. The stack is intentionally narrow to reduce friction
during early experimentation.

## Approved Stack

| Category | Technology | Version | Notes |
| --- | --- | --- | --- |
| Language | Python | 3.11.15 | Primary runtime |
| Dependency management | Poetry | 2.0.x | Lockfile-managed installs |
| Data processing | pandas | 3.0.1 | Dataframe and time-series operations |
| Numerical computing | NumPy | 2.2.x | Vectorized operations |
| Statistical forecasting | statsmodels | 0.14.6 | SARIMA / SARIMAX baselines |
| ML challengers | scikit-learn | 1.6.x | Structured-data baseline models |
| Gradient boosting | XGBoost | 2.1.x | Nonlinear ranking challenger |
| Serialization | joblib | 1.4.x | Lightweight artifact persistence |
| Config typing | pydantic | 2.11.x | Config validation |
| CLI | Typer | 0.16.x | Typed command-line interface |
| Structured logging | structlog | 25.2.x | Run-oriented logs |
| Columnar storage | PyArrow | 20.0.x | Parquet I/O |
| Plotting | Matplotlib | 3.10.x | Diagnostics and reports |
| Testing | pytest | 8.4.x | Unit and integration testing |
| Coverage | pytest-cov | 6.1.x | Coverage gate |
| Lint / format | Ruff | 0.11.x | Code quality tooling |
| Type checking | mypy | 1.15.x | Static typing |
| Containers | Docker | 28.x | Reproducible batch runtime |
| CI | GitHub Actions | N/A | Lint, test, smoke runs, schedules |

## Guardrails

- Do not add a web framework during MVP.
- Do not add a database during MVP.
- Do not add MLflow during MVP.
- Keep notebooks exploratory only.
- Keep CPU-first execution unless a later epic proves otherwise.
