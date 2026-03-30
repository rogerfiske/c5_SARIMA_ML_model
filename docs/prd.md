# c5_SARIMA_ML_model Product Requirements Document (PRD)

Version: 0.1-draft  
Status: Draft for owner review  
Prepared by: ChatGPT acting as Senior Project Representative  
Date: 2026-03-30

## Goals and Background Context

### Goals

- Deliver a rigorous, reproducible experimental forecasting platform for next-day part-demand ranking.
- Predict the **top 20 most likely part IDs** from the valid universe **1..39** for the next event/day.
- Treat `0` only as a historical count value meaning "part not used that day", never as a forecasted part ID.
- Establish a trustworthy dataset contract and anomaly policy grounded in confirmed production history.
- Evaluate candidate methods with rolling backtests and ranking-focused metrics rather than relying on a single modeling family.
- Use SARIMA/SARIMAX as a baseline family, not as a fixed architectural mandate.
- Produce BMAD-ready epics and stories that Claude Code / Windsurf can implement in gated phases.
- Keep the MVP strictly **no UI / no web app**: file-driven pipelines, reports, and artifacts only.

### Background Context

This project starts from a daily aggregated matrix dataset with columns `date, P_1..P_39`, where each `P_x` column is the count used for part `x` on that date. The business question is not merely to forecast counts; it is to rank the 20 part numbers most likely to be required on the next day/event. That means the primary product output is a **ranked next-event candidate set**, with count forecasts and probability estimates used as supporting signals rather than the only objective.

A repository audit shows the project is still at a planning and setup stage. It contains BMAD scaffolding, one production dataset, a senior representative playbook, and Perplexity-generated starter documents. It does **not** yet contain a real implementation, test suite, CI/CD pipeline, or deployable forecasting service. Accordingly, this PRD treats the repo as a **greenfield experimental service project with legacy reference inputs**, even though the BMAD materials can still be used for brownfield-style disciplined execution.

The prior skeleton documents are a helpful starting point, but they incorrectly hard-assume a perfect fixed total of 30 every day. Production review has now confirmed that exception rows are legitimate operating-condition rows, not corrupt data. The system therefore must preserve and model these exceptions explicitly.

## Data Reality Snapshot

### Dataset profile

- Source dataset: `data/raw/c5_aggregated_matrix.csv`
- Date range observed in audit: `2008-09-08` through `2026-03-29`
- Row count observed in audit: `6412`
- Missing calendar dates observed in audit: `0`
- Valid part-ID universe for prediction: `1..39`
- Historical encoding: `P_x = 0` means part `x` did not occur on that day
- Average count of non-zero part IDs per day observed in audit: approximately `21.9`

### Confirmed operating-condition exception dates

These rows are confirmed by production review and are **valid**.

#### Reduced output / manpower / holiday effects

- `2008-12-25` -> total 20
- `2009-12-25` -> total 20
- `2010-12-25` -> total 20
- `2011-07-03` -> total 25
- `2011-08-28` -> total 25
- `2011-12-25` -> total 25
- `2012-12-25` -> total 25

#### Additional requested output

- `2012-05-15` -> total 35
- `2012-11-29` -> total 35

### Data policy implications

- These rows remain in the **raw source of truth**.
- They must be annotated as operational events, not silently normalized away.
- A curated variant may still be generated for controlled experiments, but the raw dataset remains authoritative.
- Any model or evaluation logic that assumes a hard daily total of 30 must be revised.

## Change Log

| Date | Version | Description | Author |
|---|---:|---|---|
| 2026-03-30 | 0.1 | Initial PRD draft prepared from repo audit, owner guidance, and validated production exception rules | ChatGPT |

## Scope

### In Scope

- Dataset ingestion, validation, and event annotation
- Experimental forecasting pipelines for next-day prediction
- Ranked top-20 part-ID output for next day/event
- Baseline and candidate model comparison
- Rolling backtesting and evaluation framework
- Artifact generation: CSV, JSON, Markdown reports, plots, and serialized model outputs
- Minimal DevOps and reproducibility hardening needed to operate the experimental platform
- BMAD-ready epics, stories, acceptance criteria, and architect handoff

### Out of Scope for MVP

- Web UI, dashboards, or human-facing front-end application
- Multi-user authentication and authorization
- Real-time online inference API with SLA commitments
- Auto-remediation or autonomous model promotion without owner approval
- ERP integration, procurement automation, or downstream order execution
- Mobile or desktop client applications

## Product Outcomes

The MVP is successful when the project owner can run a controlled forecasting workflow that:

1. reads the approved dataset and event annotations,
2. trains and evaluates multiple candidate model families,
3. produces a ranked top-20 next-day part list from valid part IDs `1..39`,
4. never emits `0` as a candidate part ID,
5. compares results against robust baselines using rolling backtests, and
6. identifies a transparent champion configuration suitable for further hardening.

## Requirements

### Functional

1. **FR1:** The system shall ingest a daily historical dataset with schema `date, P_1..P_39`.
2. **FR2:** The system shall validate column presence, data types, duplicate dates, date continuity, and non-negative integer count values.
3. **FR3:** The system shall preserve an immutable raw dataset copy and produce a validated working dataset for downstream processing.
4. **FR4:** The system shall support an event annotation layer for confirmed operational exceptions such as holidays, reduced-output days, and additional-output days.
5. **FR5:** The system shall retain confirmed exception rows in the raw modeling corpus and shall not auto-correct those rows to a total of 30.
6. **FR6:** The system shall support an optional curated dataset variant for comparative experiments, with every transformation documented and reversible.
7. **FR7:** The system shall define the forecast candidate universe strictly as **part IDs 1 through 39**.
8. **FR8:** The system shall treat historical zero counts as absence indicators only and shall never represent `0` as a predicted next-event part ID.
9. **FR9:** The system shall produce a ranked **top 20** next-event list containing only valid part IDs from `1..39`.
10. **FR10:** The system shall assign a score to every part ID for next-day ranking, with configurable scoring formulas such as probability of non-zero occurrence, expected count, or ensemble score.
11. **FR11:** The system shall output per-part supporting diagnostics for the next-event forecast, including at minimum score, rank, and model family used; expected count and probability shall be included where available.
12. **FR12:** The system shall implement a baseline ladder that includes naïve and seasonal baselines before any advanced model is considered a champion.
13. **FR13:** The system shall implement SARIMA or SARIMAX as one baseline/candidate family, not as the only supported modeling approach.
14. **FR14:** The system shall support additional count-aware and ranking-aware candidate model families, including at minimum one generalized count model family and one machine-learning candidate family.
15. **FR15:** The system shall support rolling-origin backtesting with configurable train/validation/test windows.
16. **FR16:** The system shall evaluate models using ranking metrics and count/probability metrics, not count error alone.
17. **FR17:** The system shall produce experiment reports comparing all model runs against benchmark baselines on the same backtest windows.
18. **FR18:** The system shall support reproducible experiment execution through versioned configuration files and fixed random seeds where applicable.
19. **FR19:** The system shall expose a command-line or task-runner interface for ingest, feature build, train, backtest, forecast, and report steps.
20. **FR20:** The system shall serialize model artifacts, run metadata, dataset version metadata, and evaluation results to a documented artifact layout.
21. **FR21:** The system shall export next-event outputs in machine-readable formats at minimum CSV and JSON, and shall generate a concise human-readable markdown report for each forecast run.
22. **FR22:** The system shall support event and calendar features as optional exogenous inputs, including confirmed special-operation dates.
23. **FR23:** The system shall support champion-versus-challenger evaluation and shall require explicit owner approval before promoting a new champion configuration.
24. **FR24:** The system shall record sufficient provenance to reproduce any reported forecast, including code version, config version, dataset fingerprint, and model artifact reference.
25. **FR25:** The system shall support pairwise comparison of ranking outputs to determine whether a candidate materially outperforms recency, frequency, and seasonal baselines.
26. **FR26:** The system shall support explanation artifacts appropriate to the active model family, such as feature importance, lag contribution summaries, or component diagnostics.
27. **FR27:** The system shall fail fast on schema-breaking data issues and soft-flag domain exceptions that have been explicitly approved in the event annotation policy.
28. **FR28:** The system shall provide a documented tie-breaking policy for ranked outputs when two part IDs share the same score.
29. **FR29:** The system shall support generation of a complete daily score vector for all 39 part IDs, even though only the top 20 are operationally surfaced.
30. **FR30:** The system shall support BMAD-compatible document and story sharding after PRD and architecture are approved.

### Non Functional

1. **NFR1:** The implementation shall be Python 3.11 based unless a later approved architecture revision changes this.
2. **NFR2:** The MVP shall be repo-local and batch-oriented, with no web UI and no interactive front-end requirement.
3. **NFR3:** All pipeline steps shall be runnable in a clean environment using documented commands and configuration.
4. **NFR4:** The codebase shall favor clarity, modularity, and inspectability over premature complexity.
5. **NFR5:** The project shall maintain deterministic behavior where technically possible, including fixed seeds and stable sort/tie-break rules.
6. **NFR6:** All critical pipeline steps shall emit structured logs suitable for debugging failed runs.
7. **NFR7:** Unit and integration tests shall cover ingestion, validation, ranking logic, backtesting logic, and artifact generation.
8. **NFR8:** Every released experiment run shall be traceable to a dataset fingerprint and config fingerprint.
9. **NFR9:** The platform shall support local developer execution first and containerized execution second.
10. **NFR10:** The initial system shall run on commodity developer hardware, with optional heavier training support on RunPod or equivalent environments later.
11. **NFR11:** Model evaluation shall prioritize methodological rigor over raw throughput.
12. **NFR12:** The system shall not silently overwrite artifacts from prior experiment runs.
13. **NFR13:** Failures in one candidate model family shall not corrupt the run metadata or suppress baseline results.
14. **NFR14:** The repository shall be structured so BMAD-generated stories can be implemented incrementally without large-scale document rewrites.
15. **NFR15:** The MVP shall avoid unnecessary external service dependencies unless they clearly improve experiment rigor or reproducibility.

## User and Stakeholder Profiles

### Primary

- **Project Owner / Senior Representative:** defines goal state, approves requirements, reviews checkpoints, approves champion changes.
- **Implementation Agent Team (BMAD / Claude Code / Windsurf):** implements stories, runs tests, produces artifacts, and stops at gates for review.
- **Research Analyst / Data Scientist:** interprets backtests, compares models, tunes features, and proposes challengers.

### Secondary

- **Operations / Planning Consumer:** receives exported next-day ranked part lists and diagnostics, but is not a direct UI user in MVP.

## Core Decisions

### Prediction target

The primary target is the ranked set of the **20 most likely part IDs** to appear on the next day/event.

### Interpretation rules

- The forecast universe is `1..39`.
- `0` is **not** a forecastable part ID.
- A part is operationally "present" on a day if its observed count is greater than zero.
- Ranking score definitions are configurable, but each experiment must declare the exact score used.

### Primary evaluation philosophy

Because the observed average number of non-zero part IDs per day is already high, simple binary overlap metrics can overstate model value. Therefore, model acceptance must emphasize metrics that reward **correct prioritization**, **count-awareness**, and **calibrated probabilities**, not just broad inclusion.

## Success Metrics

### Primary ranking metrics

- **Weighted Recall@20:** actual next-day counts used as weights over the selected top-20 set.
- **nDCG@20:** ranking quality measured against actual next-day counts.
- **Mean score calibration / Brier-style occurrence metrics:** for models that output probabilities.
- **Champion lift over baseline:** paired rolling-window improvement versus naïve and seasonal baselines.

### Secondary metrics

- Precision@20 using `count > 0`
- Recall@20 using `count > 0`
- Jaccard overlap with actual next-day non-zero set
- Per-part MAE / RMSE on counts
- Log loss or proper scoring rules where model family permits
- Stability of top-20 membership across adjacent days
- Artifact completeness and reproducibility pass rate

### MVP promotion rule

A challenger may be recommended as champion only if it:

- beats the declared baseline ladder on the primary ranking metrics over the agreed holdout windows,
- does not materially regress reproducibility or artifact quality,
- passes test and runbook checks, and
- is explicitly approved by the owner after review.

## Constraints and Assumptions

### Business / domain assumptions

- The dataset reflects aggregated daily part usage.
- Confirmed exception dates are legitimate operating events.
- The owner may later provide additional domain signals such as holiday flags, manpower notes, or planned output changes.
- The immediate objective is research and rigorous implementation, not operational UI delivery.

### Technical assumptions

#### Repository Structure: Monorepo

A single Python repository is sufficient for the MVP.

#### Service Architecture: Modular batch forecasting service

The MVP should be implemented as a modular, file-driven, batch-oriented service with clear pipeline stages rather than microservices.

#### Testing Requirements: Unit + Integration with research-grade evaluation checks

The MVP requires strong unit coverage plus integration tests for end-to-end batch flows and evaluation correctness.

#### Additional technical assumptions and requests

- Preferred language: Python 3.11
- Configuration: YAML or TOML plus environment overrides
- Packaging: `src/` layout
- Quality tooling: Ruff, pytest, mypy or pyright, pre-commit
- Reproducibility: artifact folders keyed by timestamp and run ID
- Serialization: documented use of parquet/csv/json/pickle/joblib as appropriate
- Initial orchestration: Makefile, task runner, or CLI entry points before any heavier workflow engine
- Optional compute target for heavier experimentation: RunPod
- GitHub Actions or equivalent CI should be added once the basic project skeleton is stable

## Repository and Artifact Direction

Recommended high-level repository direction for implementation:

- `docs/`
- `config/`
- `data/raw`, `data/interim`, `data/processed`
- `src/<package_name>/ingestion`
- `src/<package_name>/features`
- `src/<package_name>/models`
- `src/<package_name>/ranking`
- `src/<package_name>/evaluation`
- `src/<package_name>/pipelines`
- `artifacts/`
- `tests/`

This remains a recommendation, not an implementation mandate, until architecture is approved.

## Epic List

### Epic 1 - Foundation, Dataset Contract, and Canary Forecast Flow

Establish the repository, dataset contract, event-annotation policy, and a fully runnable canary next-day ranking pipeline so the team has a trustworthy foundation and an executable vertical slice from raw data to top-20 output.

### Epic 2 - Evaluation Engine and Baseline Ladder

Build the rolling backtest framework, ranking metrics, and benchmark ladder so every future model is evaluated under the same rigorous rules and can be compared honestly.

### Epic 3 - Candidate Models, Feature Experiments, and Champion Selection

Implement SARIMA-family, count-aware, and machine-learning challenger families plus feature experiments and controlled champion/challenger comparison.

### Epic 4 - Reproducibility, Minimal DevOps, and Operational Batch Hardening

Add the minimum operational hardening needed to run the platform reliably: containers, CI, artifact/version controls, runbooks, and scheduled batch support.

## Epic Details

## Epic 1 - Foundation, Dataset Contract, and Canary Forecast Flow

**Goal:** Create a stable experimental base that can ingest the real dataset, preserve known operational exceptions correctly, and produce a first valid top-20 forecast artifact. This epic delivers immediate value because it turns the current planning repository into a runnable research platform with a trustworthy data contract.

### Story 1.1 - Create the executable project skeleton

As an implementation agent,  
I want a runnable project skeleton with documented commands and directories,  
so that all future work lands in a consistent structure.

#### Acceptance Criteria

1. A Python project skeleton exists with documented setup and run commands.
2. The repo contains dedicated directories for docs, config, data, source, tests, and artifacts.
3. A basic command or task runner can execute a no-op or canary pipeline end to end.
4. Linting and test scaffolding are present and runnable.
5. The project explicitly declares that MVP scope excludes any UI or web application.

### Story 1.2 - Implement raw ingestion and schema validation

As a research analyst,  
I want reliable ingestion and validation of the historical matrix,  
so that all downstream experiments begin from a verified dataset.

#### Acceptance Criteria

1. The ingestion pipeline reads the raw CSV and validates the required schema `date, P_1..P_39`.
2. Duplicate dates, missing columns, negative counts, and malformed dates cause a clear failure.
3. Date continuity is checked and reported.
4. A raw immutable copy or fingerprinted reference is persisted for provenance.
5. Validation results are emitted as a structured report.

### Story 1.3 - Implement event annotation and anomaly policy

As the project owner,  
I want confirmed exception dates represented as valid operational events,  
so that the platform models reality instead of rewriting history.

#### Acceptance Criteria

1. The system supports an event annotation source for special dates.
2. The confirmed reduced-output and additional-output dates are encoded in that source.
3. Those dates are preserved in the raw modeling corpus.
4. Validation soft-flags those rows as approved exceptions rather than errors.
5. The anomaly policy is documented in the repo and referenced by the pipeline.

### Story 1.4 - Build working datasets and a curated experiment variant

As a data scientist,  
I want both raw and curated working datasets,  
so that I can compare modeling behavior with and without normalization assumptions.

#### Acceptance Criteria

1. The pipeline produces a validated raw working dataset.
2. The pipeline can optionally generate a curated experiment dataset variant.
3. Every curated transformation is explicitly documented and reversible.
4. Dataset fingerprints are produced for both variants.
5. The default experiment path remains the raw validated dataset unless overridden by config.

### Story 1.5 - Deliver a canary next-day top-20 forecast flow

As the project owner,  
I want the first valid next-day ranking artifact,  
so that the project demonstrates a real vertical slice before advanced modeling begins.

#### Acceptance Criteria

1. A baseline canary model produces scores for all 39 part IDs for the next day.
2. The exported top-20 artifact contains only part IDs from `1..39`.
3. `0` never appears in the forecast output.
4. The run exports CSV, JSON, and Markdown summary artifacts.
5. The run captures provenance including dataset fingerprint, config fingerprint, and timestamp.

## Epic 2 - Evaluation Engine and Baseline Ladder

**Goal:** Build the evaluation system before pursuing complex model work so that every modeling claim is grounded in consistent rolling backtests and ranking-aware metrics. This epic is the methodological backbone of the project.

### Story 2.1 - Implement rolling-origin backtesting

As a research analyst,  
I want rolling-origin evaluation windows,  
so that candidate models are compared under realistic next-day forecasting conditions.

#### Acceptance Criteria

1. The backtest engine supports configurable rolling or expanding windows.
2. The engine evaluates the same forecast horizon and dates across all compared models.
3. The engine persists per-window outputs and aggregate summaries.
4. Failed model runs are isolated without suppressing successful baseline runs.
5. The backtest engine is covered by integration tests.

### Story 2.2 - Implement ranking and calibration metrics

As the project owner,  
I want ranking-aware metrics that reflect operational usefulness,  
so that model selection is not distorted by weak binary overlap measures.

#### Acceptance Criteria

1. The evaluation layer computes nDCG@20 and Weighted Recall@20.
2. The evaluation layer computes binary Precision@20 and Recall@20 as secondary metrics.
3. Probability-capable models expose at least one calibration/proper scoring metric.
4. Metric definitions are documented and tested.
5. A comparison report clearly distinguishes primary and secondary metrics.

### Story 2.3 - Implement the baseline ladder

As a data scientist,  
I want strong simple baselines,  
so that advanced models must prove they beat credible alternatives.

#### Acceptance Criteria

1. The system includes at minimum a frequency baseline, a recency baseline, and a seasonal baseline.
2. Each baseline produces a full 39-part score vector and a top-20 output.
3. Baseline outputs follow the same artifact schema as advanced models.
4. Baselines participate in the same backtest engine as challengers.
5. Baseline performance is summarized in a reusable benchmark report.

### Story 2.4 - Add experiment comparison and champion gating

As the project owner,  
I want a formal comparison and promotion process,  
so that model changes do not become champion by anecdote.

#### Acceptance Criteria

1. The system compares challengers against the current champion and baseline ladder on identical windows.
2. The comparison report highlights lifts, regressions, and statistical or practical tradeoffs.
3. Champion recommendation is separated from champion promotion.
4. Promotion requires explicit owner approval.
5. All comparison artifacts are versioned and reproducible.

## Epic 3 - Candidate Models, Feature Experiments, and Champion Selection

**Goal:** Add substantive modeling depth after the evaluation framework is in place. This epic turns the platform from a benchmark harness into a genuine model-development environment while keeping every candidate accountable to the same ranking-focused standards.

### Story 3.1 - Implement SARIMA/SARIMAX candidate family

As a research analyst,  
I want a properly evaluated SARIMA-family implementation,  
so that the project can test the original concept fairly without letting it dominate the design.

#### Acceptance Criteria

1. The codebase supports SARIMA or SARIMAX candidate runs for relevant per-part or transformed targets.
2. Calendar and approved event features can be supplied as exogenous inputs where applicable.
3. The implementation documents how counts or probabilities are derived from model outputs.
4. SARIMA-family outputs flow through the shared ranking and backtest interfaces.
5. Diagnostics and failure cases are logged clearly.

### Story 3.2 - Implement a count-aware candidate family

As a data scientist,  
I want a count-aware model family,  
so that the project can test methods better matched to discrete usage counts.

#### Acceptance Criteria

1. At least one generalized count-based family is implemented, such as Poisson, Negative Binomial, or equivalent count-aware state-space/regression approach.
2. The model can emit either expected counts, occurrence probabilities, or both.
3. The model participates in the common experiment interface.
4. Feature inputs and transformations are documented.
5. Results are compared directly against the SARIMA-family and baseline ladder.

### Story 3.3 - Implement a machine-learning ranking challenger

As the project owner,  
I want at least one non-classical challenger,  
so that the platform can test whether feature-driven ranking models outperform time-series-only baselines.

#### Acceptance Criteria

1. A machine-learning challenger family is implemented using lag, calendar, and optional event features.
2. The model can generate scores for all 39 part IDs for a given next day.
3. The output is converted into the common ranking artifact schema.
4. Feature importance or equivalent explanation artifacts are generated where supported.
5. The challenger is evaluated through the shared backtest engine.

### Story 3.4 - Build ensemble logic and champion selection reporting

As the project owner,  
I want to test whether combining models improves ranking quality,  
so that champion selection is based on best demonstrated performance rather than model preference.

#### Acceptance Criteria

1. The platform supports at least one ensemble strategy such as weighted score blending or stacked ranking.
2. Ensemble weights or logic are configuration-driven and reproducible.
3. Ensemble runs are evaluated under the same benchmark rules as single models.
4. The comparison report identifies whether the ensemble materially improves primary metrics.
5. Champion recommendation artifacts summarize the best-performing configuration and its tradeoffs.

## Epic 4 - Reproducibility, Minimal DevOps, and Operational Batch Hardening

**Goal:** Add the minimum operational discipline required to make the research platform maintainable, repeatable, and safe for continued BMAD-driven development. This epic is intentionally lean: enough hardening to support disciplined work, not a premature production platform.

### Story 4.1 - Containerize and standardize project execution

As an implementation agent,  
I want consistent local and containerized run paths,  
so that experiments can be reproduced across machines and compute environments.

#### Acceptance Criteria

1. The repo contains a documented local execution path and a containerized execution path.
2. Container builds are able to run baseline training/backtest jobs.
3. Environment assumptions are documented explicitly.
4. Dependency installation is reproducible.
5. A smoke test verifies the container path.

### Story 4.2 - Add CI quality gates and artifact checks

As the project owner,  
I want automated quality gates,  
so that story implementations do not erode reproducibility or correctness.

#### Acceptance Criteria

1. CI runs linting, unit tests, and key integration tests.
2. CI validates artifact schemas for at least one representative pipeline run.
3. Failures surface clearly in a machine-readable and human-readable form.
4. The CI workflow does not require a UI or external dashboard.
5. Required checks are documented for story completion.

### Story 4.3 - Add runbooks, release gates, and scheduled batch support

As the project owner,  
I want operational guidance and a controlled batch path,  
so that the platform can be run repeatedly without improvisation.

#### Acceptance Criteria

1. A runbook exists for ingest, train, backtest, and forecast workflows.
2. A release-gate checklist exists for promoting a new champion configuration.
3. A scheduled batch entry point exists for daily or on-demand next-event forecast generation.
4. Scheduled runs preserve artifact provenance and do not overwrite prior run outputs.
5. Operational docs clearly describe rollback and recovery steps for failed runs.

## Risks and Mitigations

### Risk 1 - Weak metrics could overstate value

Because many parts are non-zero on many days, simple overlap metrics can make mediocre models look useful.

**Mitigation:** make weighted ranking metrics and calibration primary, keep binary overlap secondary, and compare against strong baselines.

### Risk 2 - Overcommitting to SARIMA

A concept seed centered on SARIMA could bias the team toward a suboptimal architecture.

**Mitigation:** define SARIMA/SARIMAX as one candidate family only, and require head-to-head comparison against count-aware and ML challengers.

### Risk 3 - Silent data rewriting

Treating valid exception rows as errors would distort both model fit and evaluation.

**Mitigation:** preserve raw truth, annotate special dates, and keep curated variants explicitly separate.

### Risk 4 - Premature DevOps expansion

The repo could spend too much effort on infra before proving forecasting value.

**Mitigation:** stage DevOps only after foundation and evaluation layers are functioning.

## Open Questions for Architecture Phase

- What exact package layout and naming should be adopted for the implementation repo?
- Which candidate count-aware model family should be the first non-SARIMA challenger?
- Should the machine-learning challenger be framed as multi-label occurrence prediction, count regression, or direct ranking?
- What artifact registry format is simplest while remaining reproducible?
- What is the cleanest champion/challenger manifest structure for BMAD-driven implementation?
- Which daily score definition should be the default operational ranking score: probability of occurrence, expected count, or a hybrid?

## Checklist Readiness Summary

This PRD is intentionally ready for the next BMAD phases even though architecture has not yet been written:

- clear business goal: yes
- explicit target definition: yes
- explicit `0` exclusion rule: yes
- anomaly/data policy grounded in production validation: yes
- no-UI MVP scope: yes
- sequential epics and stories: yes
- architecture handoff topics identified: yes

## Next Steps

### Architect Prompt

Use this PRD to create a no-UI `architecture.md` for a **greenfield experimental forecasting service**. Preserve the raw-vs-curated dataset policy, the confirmed exception-date event annotation approach, the rule that forecast candidates are valid part IDs `1..39` only, and the requirement that `0` must never appear in the ranked top-20 next-event output. Design for Python 3.11, modular batch pipelines, rolling backtests, reproducible artifacts, and minimal but real DevOps hardening. Treat SARIMA/SARIMAX as one candidate family rather than the sole architectural center.

### Product Owner / Sharding Prompt

After PRD and architecture are approved, shard this PRD into BMAD-ready epic and story documents for the **team-no-ui / greenfield-service** workflow, preserving acceptance criteria and execution gating.
