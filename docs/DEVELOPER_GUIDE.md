# Developer guide

Read [ARCHITECTURE.md](ARCHITECTURE.md) first: it records the design decisions (D1 to D13) that
the code follows. The low-data engine replaced parts of D1 to D4: data-size gates, the
one-standard-error rule as the way to pick the final model, Platt-only calibration and TF-IDF as
the default embedding. [LOW_DATA_INFERENCE.md](LOW_DATA_INFERENCE.md) describes the current
inference engine and what it measured. Where the two documents disagree, follow
LOW_DATA_INFERENCE.md.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[ocr,dev]"
pytest
predictor --data-dir /tmp/pd serve --no-browser
```

`wordllama` is a core dependency. It carries the bundled pretrained embeddings, so the semantic
layer needs no download, and `predictor doctor` tells you whether it loaded. The optional
`neural` extra (sentence-transformers) still needs `predictor models download` once.

## Code map

```
predictor/
  config/          default.toml (all defaults), settings.py (load + validate overrides,
                   DEPRECATED_KEYS), taxonomy.json (question types), generic_words.txt (OOS evidence
                   stop list)
  database/        SQLAlchemy models, engine/session, FTS5 search table, migrations.py (adds new
                   columns to existing databases and records schema_version)
  ingestion/       file sniffing, hashing, PDF/DOCX/image/text extraction
  ocr/             Tesseract wrapper (line reconstruction, confidence), OpenCV preprocessing
  preprocessing/   text cleaning, math normalisation, stemming, quality flags
  parsing/         exam metadata, question-tree segmentation, question-type classifier
  syllabus/        syllabus parser, document merge, TopicTree (unit/topic/concept), alignment (A/B/C/D)
  embeddings/      backends.py (TF-IDF, sentence-transformers, backend choice), pretrained.py
                   (bundled WordLlama model and the hybrid WordLlama + TF-IDF backend),
                   embedding cache, vector index
  topic_modeling/  question families: exact, paraphrase, concept recurrence
  temporal/        Panel (exams x items) with until(t): appearances, marks, formats, semantic soft
                   counts, data-quality sums; hazard/Markov/gap/rotation statistics
  features/        grouped features computed at a cutoff, GENERIC_FEATURES (scale-free subset),
                   FeatureStore cache
  inference/       bayes.py (hierarchical Bayesian recurrence, prequential log scores),
                   evidence.py (evidence profile, component status, run-level message),
                   generic.py + generic_prior.json (general ranking model trained on simulated
                   courses), repository.py (cross-course feature rows, model registry),
                   uncertainty.py (rank intervals, evidence strength)
  models/          base.py (interface), statistical.py (reference baselines, Markov),
                   components.py (ensemble components: Bayesian recurrence, temporal, semantic,
                   coverage, question type, co-occurrence, general model, course logistic),
                   learned.py (random forest, gradient boosting, pooled HMM), meta.py (tuned
                   variants, evidence-aware ensemble), registry.py, calibration.py (Platt and
                   isotonic, validated by a nested check)
  evaluation/      metrics, rolling-origin backtest + final-ranking choice + validation summary +
                   leakage audit, staged and leave-one-component-out ablation
  prediction/      final ranking, categories, evidence lines, evidence strength, rank intervals,
                   why-not, format forecast, families
  generation/      grounded question formulations, verify.py (checks every formulation must pass),
                   predicted paper simulation
  analysis/        exam template discovery, coverage, co-occurrence tests
  visualization/   chart data with question ids behind every point
  search/          FTS5 + semantic search
  export/          CSV / XLSX / PDF / JSON
  services/        AppContext and the services the API and CLI call
  app/             FastAPI app, routes, background job runner
  ui/static/       vanilla JS single-page app (no build step)
  demo/            synthetic demo course generator (also the test fixture); the demo course is
                   flagged synthetic
  cli.py           command-line entry point
```

## Data flow

```
SourceFile -> DocumentPage (text per page)
           -> Exam / ExamSection / ExamQuestion tree            (IngestService)
           -> CourseTopic tree (merged documents)               (IngestService)
AnalysisService.run:
  load topics + included exams (order_index) + leaf questions
  get_backend, get_pretrained     -> alignment backend (hybrid by default), WordLlama for semantics
  SyllabusAligner.align           -> QuestionTopicMapping (manual mappings kept)
  QuestionTypeClassifier          -> ExamQuestion.question_types (manual types kept)
  find_recurrence                 -> exact/paraphrase/concept links
  build_panel (topic, concept, unfiltered topic) + semantic soft evidence per paper
  general_model_for               -> general ranking model (shipped prior + other real courses)
  BacktestEngine.run              -> ModelResult, BacktestFold, calibration, leakage audit
  build_profile, component_status, low_data_message -> evidence profile, statuses, run message
  topic_uncertainty               -> rank interval per topic
  run_ablation, syllabus_filter_check, run_type_forecast, backtest_families
  build_topic_predictions         -> Prediction (topic, concept, family layers)
  generate_formulations + FormulationVerifier -> PredictedQuestion (verified formulations only)
  discover_structure, coverage, charts, simulate_papers -> AnalysisArtifact (including "evidence")
  save_course_features, register_course_model -> course_feature_set, model_registry
```

`BacktestEngine.run` predicts every paper after the first from the papers before it (rolling
origin, `models.min_train_exams = 1`) and makes the real forecast with the same code at `t = T`.
Every enabled model runs on every fold. The evidence-aware ensemble (`models/meta.py`) is the final
ranking unless a single method beat it by more than `ensemble.replace_if_worse_by_se` standard
errors of the paired difference. Only the topic layer runs the leakage audit and calibration; the
concept and unfiltered panels skip both.

Papers of a synthetic course (the demo) get `Exam.source = "demo"`. Their feature rows are stored
flagged and never train the general model. Generated formulations and simulated papers are never
stored as exams or questions.

## The leakage rule

Every model sees history only through `Panel.until(t)` (a copy holding exams `0..t-1`) and
`FeatureStore.at(t)` / `FeatureStore.training_rows(t, ...)`, which are built on it. Never index
`ctx.panel.Y[t]` or later inside a model. Meta models (tuned variants, the ensemble) read other
models' predictions for target `t`, which were made from exams before `t`, and may read
`ctx.fold_metric[...]` only for targets `s < t`. If you cache a whole-panel computation in
`ctx.cache`, read only the entries for exams before `t`, as `TemporalModel` does with its
prequential log scores.

`evaluation.backtest.leakage_audit` scrambles every exam-level panel array from `t` onward:
appearances (which it also flips), marks, best similarity, formats, question counts, repeats,
semantic soft counts and data-quality sums. It then reruns every model on the folds up to `t`,
including the tuned variants and the ensemble, and checks that predictions for `t` do not change.
Meta models are rerun because they read earlier folds, so on the scrambled panel they must see the
same earlier predictions and metrics. The audit runs on the topic layer of every analysis that
has at least one held-out paper. It also runs in `tests/unit/test_temporal_and_models.py`, which
proves it catches a deliberately leaky model, and in `tests/unit/test_low_data.py` at 3, 5 and 12
papers.

If you add an exam-indexed array to `Panel`, add it to `until`, `drop_exam` and the list of arrays
`leakage_audit` scrambles. Otherwise the audit cannot see a leak through it. Arrays that describe
items only (`static`, `item_sim`, `format_prior`) come from the syllabus and hold no exam text.

Three subtler points (the first two are documented in ARCHITECTURE.md):

* Alignment TF-IDF is fitted on the syllabus only, and the pretrained part of the hybrid backend
  fits nothing. The feedback pass that expands topic vocabulary uses confidently mapped questions
  from all papers; this is treated as annotation (like a person labelling every paper), not as a
  predictive feature. The semantic soft evidence in the panel comes from the same alignment
  scores, so the same caveat applies to it.
* Question similarity for recurrence uses stateless character n-grams (or a downloaded
  sentence-transformer), so no statistics from later papers enter earlier comparisons. The bundled
  WordLlama model is not used here: on 26 reworded test questions it found the right topic for 11,
  against 14 for character n-grams.
* The general ranking model never learns from the course it predicts. `general_model_for` updates
  the shipped prior with the scale-free rows of other real courses only, and skips synthetic
  courses. Course A's rows can change the general model used for course B, never course B's panel,
  frequencies or posteriors (`test_other_courses_do_not_change_this_courses_history`).

## Adding a prediction model

1. Subclass `models.base.BaseModel`. Set `name`, `display`, `family`, a `description` and
   `complexity` (higher means more complex; it breaks ties, and when a single method replaces the
   ensemble, the simplest one within one standard error of the best is used). Then set the
   attributes the evidence-aware ensemble reads:
   * `role`: `"baseline"` (a reference method, scored on every fold but not part of the
     ensemble), `"component"` (an ensemble member) or `"ensemble"`.
   * `df`: the effective number of parameters the model estimates from this course. The
     ensemble's reliability prior is `(exams + pk) / (exams + pk + df)`, so a large `df` keeps the
     weight small until many papers exist. `df = 0` (the general model) gives reliability 1.
   * `prior_knowledge = True` when the information does not come from the course's own history
     (pretrained, syllabus, cross-course). Such a component starts with one pseudo-exam (`pk = 1`)
     and gets more weight, by up to a factor of `1 + ensemble.gate_strength`, on topics whose own
     history taught little.
   * `scope`: `"course"` (the default), `"global"` (pretrained or cross-course knowledge) or
     `"syllabus"` (used by the coverage component). It is stored in `model_result.scope`.
2. Implement `predict(t, ctx) -> ModelOutput(scores, info)`. Use `ctx.store.at(t)` features or
   `ctx.panel.until(t)`; learned models use `ctx.store.training_rows(t, min_history, ...)`. There
   is no minimum dataset size: return a forecast for every `t`, including `t = 0` for a course
   with no papers. Two exits exist, and neither is a size gate:
   * When an input the model needs does not exist (no earlier paper to compare with, no syllabus
     units or weights, a single question format), return `components.unavailable(K, reason)`. The
     ensemble leaves the model out for that target. If this happens on the final forecast, the
     status table shows UNAVAILABLE with your reason.
   * When fitting is mathematically impossible (every training label is the same), return
     `components.general_fallback(t, ctx, reason)`. The report counts those folds.
3. Leave `gate()` alone. It answers only whether a model can run at all, and no model in the
   registry overrides the default `(True, "")`. The old `[models.sufficiency]` thresholds are
   gone: `df` and the skill measured on earlier papers decide how much a model counts.
4. Register it in `models/registry.py` and add its name to `models.enabled` in `default.toml`. For
   an ensemble component, also add the name to `COMPONENT_ORDER`: the ensemble's members are the
   built models named there, and it reads each member's `df` and `prior_knowledge`. To tune a
   hyperparameter on earlier folds, register each variant as a hidden model and wrap them in
   `meta.TunedModel` with a default (see `recency` and `beta_binomial`). It keeps the default
   unless another variant is ahead by more than one standard error of the paired difference.
5. Add a component to a stage in `evaluation/ablation.STAGES` so the staged ablation measures it.
   The leave-one-component-out rows cover every ensemble member automatically.
6. If the model refits a classifier or another slow model on every call, add its name to
   `inference/uncertainty.SLOW`.
   The leave-one-paper-out rank intervals then keep its full-data scores instead of refitting it
   once per paper.
7. Add a unit test, and add a component's name to `COMPONENTS` in `tests/unit/test_low_data.py`
   so the no-cutoff test checks that it runs at 2, 3, 5, 9, 15 and 18 papers. The leakage audit
   covers it automatically.

The model then appears in the comparison table with its status, weight and reliability. The
statuses (ACTIVE, LIMITED, DOWNWEIGHTED, UNAVAILABLE, REFERENCE) and the weight formula are in
[LOW_DATA_INFERENCE.md](LOW_DATA_INFERENCE.md), section 5. Any visible single method, baseline or
component, becomes the final ranking only if it beat the ensemble on the backtest by more than
`ensemble.replace_if_worse_by_se` standard errors of the paired difference.

## Adding a feature

Add the computation in `features/builder.compute_features` and its name to one group in
`FEATURE_GROUPS` (a new group also needs a label in `GROUP_LABELS`). Groups label the signal
contributions in each topic's explanation, which come from the course-specific logistic model, so
choose the group that matches the evidence family. If zero is the wrong value for a course with no
papers, set the feature in the `T == 0` branch as well. The ablation study removes ensemble
components, not feature groups.

Then decide whether the feature is generic or course-only:

* **Generic** (`GENERIC_FEATURES`): scale-free, means the same thing in every course, and
  computable from the appearance matrix and syllabus units alone, because the simulated courses
  the general model trains on have nothing else. Recency rates, gaps, streaks and Bayesian
  posterior summaries are examples. The general ranking model uses these features, the course
  logistic pulls its weights for them toward the general model, and `course_feature_set` stores
  them for other courses. After changing the list, rebuild the shipped model with
  `python -m predictor.inference.generic` (deterministic for a given seed) and commit the new
  `inference/generic_prior.json`. Until you do, `load_prior` finds a feature mismatch and trains a
  smaller prior (120 simulated courses) once per process. Stored rows of other courses with the old
  feature list are ignored until those courses are analysed again. Expect the regression snapshot
  to change too, because the general model is an ensemble component.
* **Course-only**: anything that depends on question text, marks, syllabus weights or data
  quality. Keep it out of `GENERIC_FEATURES`, and add it to `COURSE_ONLY` in
  `models/components.py` if the course logistic should use it; its weight then has a prior
  centred on zero. The random forest and gradient boosting models use every feature without
  further changes.

## Adding a question type

Add an entry to `config/taxonomy.json` with `id`, `label`, `format` (one of definition, theory,
derivation, numerical, diagram, objective), `weight` and regex `patterns`. Users can do the same
without touching the code by placing a `taxonomy.json` in their data folder.

## Adding a file format

Add an extractor returning `ingestion.types.ExtractionResult` and dispatch to it from
`ingestion/extract.py` (`sniff_type` and `extract_document`). Add the extension to
`ingestion.supported_extensions`.

## Changing the database schema

`Base.metadata.create_all` creates missing tables but never changes existing ones, so a new table
needs no migration. To add a column, add it to `database/models.py` with a default, then add
`(table, column, SQL type with default)` to `MIGRATIONS` in `database/migrations.py` under a new
version and raise `SCHEMA_VERSION`. `Database.create_all` runs `migrate` every time a database is
opened: it adds missing columns with `ALTER TABLE ... ADD COLUMN` and records the version in
`schema_version`. Migrations only add. Nothing is dropped or rewritten.

The low-data upgrade (schema version 2) added `course.is_synthetic`, `exam.source`,
`analysis_run.engine_version`, the component columns of `model_result` (`role`, `scope`,
`status`, `weight`, `reliability`, `evidence`), `prediction.evidence_strength` and
`prediction.uncertainty`. It also added three tables: `model_registry` (scope "global" for the
general models used, "course" for a run's course logistic), `course_feature_set` (one course's
scale-free feature rows, flagged when synthetic) and `schema_version`.
`tests/fixtures/schema_v1.sql` holds the version 1 schema, and
`test_migration_from_version_1_keeps_data` checks that a database built from it keeps its rows.

## Configuration

`config/default.toml` is the only place defaults live. `Settings` is read-only;
`settings.merged({...})` derives a new object. Overrides from `<data_dir>/settings.json` and from a
course's `settings` column are validated against the defaults: unknown keys and wrong types raise
`SettingsError`. The exception is `DEPRECATED_KEYS` in `config/settings.py`. Settings removed in
later versions (`models.sufficiency`, `calibration.min_rows`, `calibration.min_positives`) still
load from old files and are ignored. When you remove a setting, add it there so existing
installations keep working.

The low-data engine adds `embeddings.pretrained`, `embeddings.hybrid_pretrained_weight`,
`alignment.semantic_temperature`, `temporal.bayes_half_lives`, `models.course_prior_precision`,
`calibration.methods` and the `[ensemble]` section, and sets `models.min_train_exams` to 1.
[LOW_DATA_INFERENCE.md](LOW_DATA_INFERENCE.md), section 12, explains each one.

## Logging

`utils.logging.log_event(logger, "event_name", key=value)` writes structured fields. The console
shows warnings; `<data_dir>/logs/predictor.log` keeps INFO and above as JSON lines. Loggers:
`predictor.ingestion`, `predictor.ocr`, `predictor.embeddings`, `predictor.evaluation`,
`predictor.analysis`, `predictor.jobs`.

## Tests

| Folder | What it covers |
|---|---|
| `tests/unit` | parsing formats, metadata, syllabus parsing and merge, text normalisation, quality flags, question types, metrics (hand-computed and against simulation), temporal statistics, hazard, rotation test, features, leakage audit, every component running on a 5-paper panel, final-ranking choice and the one-standard-error helper, calibration, generation grounding, structure, co-occurrence, format forecast, settings validation |
| `tests/unit/test_low_data.py` | low-data behaviour on planted panels with 0 to 25 papers: every component runs at 2, 3, 5, 9, 15 and 18 papers, semantic, Bayesian and general components keep weight at 2 to 5 papers, course-learned models are downweighted at 3 papers and gain weight at 25, Bayesian intervals for 1 of 2 against 8 of 16, question counts do not change exam-level evidence, the leakage audit including meta models, the low-data message, reproducibility, ranking with zero papers |
| `tests/unit/test_repository_and_migrations.py` | the general model learns only from other real courses, other courses never change this course's history, a version 1 database (`tests/fixtures/schema_v1.sql`) migrates and keeps its data |
| `tests/integration` | real PDF, scanned PDF (OCR), DOCX with Word numbering, images, duplicates; the full demo pipeline (alignment accuracy, out-of-syllabus handling, planted patterns, component statuses, exports, manual edits surviving re-analysis, generated material never counted as history, the evidence artifact, rank intervals and ablation metrics); the HTTP API |
| `tests/regression` | a snapshot of the seeded demo run: parsing counts, mapping statuses, the final model (the ensemble) and whether it was calibrated, NDCG of every model, the top-10 ranking and categories |

OCR tests skip automatically when the `tesseract` binary is missing.

After an intended behaviour change, refresh the regression snapshot and review the diff:

```bash
python tests/regression/snapshot_tools.py
git diff tests/regression/snapshots/
```

The integration tests score models against the app's own mapped labels. When one of them shows the
final ranking ahead of frequency, it says nothing about the true topics (see Conventions).

## HTTP API

The UI uses the same JSON API you can call yourself. With the server running, the interactive
reference is at <http://127.0.0.1:8765/api/docs>. Main routes:

| Route | Purpose |
|---|---|
| `GET/POST /api/courses`, `GET/PATCH/DELETE /api/courses/{id}` | courses (PATCH accepts per-course `settings`) |
| `POST /api/courses/{id}/files` (multipart `files`, `kind`, `syllabus_version`) | upload |
| `GET /api/courses/{id}/files`, `GET /api/files/{id}/pages`, `POST /api/files/{id}/reprocess` | files |
| `GET /api/courses/{id}/exams`, `PATCH /api/exams/{id}` | paper metadata |
| `GET /api/exams/{id}/questions`, `PATCH/DELETE /api/questions/{id}`, `POST .../split`, `POST .../merge-next` | question review |
| `GET /api/courses/{id}/syllabus`, `POST /api/courses/{id}/topics`, `PATCH/DELETE /api/topics/{id}` | syllabus |
| `GET /api/courses/{id}/mappings`, `PUT/DELETE /api/questions/{id}/mapping` | mapping overrides |
| `POST /api/courses/{id}/analyze`, `GET /api/runs/{id}` | run analysis, poll progress |
| `GET /api/courses/{id}/results`, `GET /api/runs/{id}/predictions?layer=topic|concept|family` | latest run with its predictions and the `excluded`, `sufficiency` and `evidence` artifacts; each topic prediction carries `evidence_strength` and `uncertainty` (rank interval and level) |
| `GET /api/runs/{id}/topics/{topic_id}` | evidence for one topic |
| `GET /api/runs/{id}/models` | backtest table: role, scope, status, weight, reliability and metrics of every visible model, plus per-fold metrics |
| `GET /api/runs/{id}/artifacts/{key}` | stored artifacts: `evidence`, `ablation`, `calibration`, `charts`, `structure`, `coverage`, `papers` ... |
| `GET /api/runs/{id}/artifacts/evidence` | run-level mode and message, evidence profile, component status table, validation summary, uncertainty summary, general model source, generation checks |
| `POST /api/runs/{id}/papers` | new predicted-paper variants |
| `GET /api/courses/{id}/search?q=...` | search |
| `GET /api/runs/{id}/export?format=pdf|xlsx|csv|json` | export (the JSON export includes the `evidence` artifact) |
| `GET /api/health` | app version, OCR status, embeddings (including whether the bundled pretrained model loaded) |

The `sufficiency` artifact keeps its old key for older clients. It now holds the same component
statuses as `evidence`, and it marks a component as not enabled only when the component is
UNAVAILABLE.

## Conventions

* Keep data processing, ML, business logic and UI in their packages; services glue them together.
* No magic numbers in code: add a setting with a comment in `default.toml`.
* User-visible text is plain and specific. Never claim certainty about future exams, and never
  tell the user that inference is disabled or switched off.
* Do not add dataset-size cutoffs. Express thin evidence through a small reliability prior, a wide
  rank interval and a LIMITED status, and keep every model running.
* Count exams, not questions, for anything temporal. A paper with forty questions is one paper.
* Synthetic material (the demo course, generated formulations, simulated papers, the simulated
  courses behind the general model) never counts as historical evidence.
* Every new ML component needs a backtest-visible reason to exist (see the ablation study).
* Report what the backtest measured. The ensemble's gains over frequency are shown on simulated
  courses. On the 12-paper demo course it scores above plain frequency against the app's own
  mapped labels (NDCG@11 0.754 against 0.709), but against the generator's true topics it scores
  slightly below frequency, within one standard error. Section 8 of
  [LOW_DATA_INFERENCE.md](LOW_DATA_INFERENCE.md) reports both measurements.
