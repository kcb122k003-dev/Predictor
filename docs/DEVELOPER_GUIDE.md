# Developer guide

Read [ARCHITECTURE.md](ARCHITECTURE.md) first: it records the design decisions (D1 to D13) that
the code follows.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[ocr,dev]"
pytest
predictor --data-dir /tmp/pd serve --no-browser
```

## Code map

```
predictor/
  config/          default.toml (all defaults), settings.py (load + validate overrides),
                   taxonomy.json (question types), generic_words.txt (OOS evidence stop list)
  database/        SQLAlchemy models, engine/session, FTS5 search table
  ingestion/       file sniffing, hashing, PDF/DOCX/image/text extraction
  ocr/             Tesseract wrapper (line reconstruction, confidence), OpenCV preprocessing
  preprocessing/   text cleaning, math normalisation, stemming, quality flags
  parsing/         exam metadata, question-tree segmentation, question-type classifier
  syllabus/        syllabus parser, document merge, TopicTree (unit/topic/concept), alignment (A/B/C/D)
  embeddings/      TF-IDF and sentence-transformers backends, embedding cache, vector index
  topic_modeling/  question families: exact, paraphrase, concept recurrence
  temporal/        Panel (exams x items) with until(t), hazard/Markov/gap/rotation statistics
  features/        grouped features computed at a cutoff, FeatureStore cache
  models/          baselines, Bayesian, Markov, hazard, semantic, logistic, RF, GBM, HMM,
                   tuned variants, ensemble, registry, calibration
  evaluation/      metrics, expanding-window backtest + selection + leakage audit, ablation
  prediction/      final ranking, categories, evidence, why-not, format forecast, families
  generation/      grounded question formulations, predicted paper simulation
  analysis/        exam template discovery, coverage, co-occurrence tests
  visualization/   chart data with question ids behind every point
  search/          FTS5 + semantic search
  export/          CSV / XLSX / PDF / JSON
  services/        AppContext and the services the API and CLI call
  app/             FastAPI app, routes, background job runner
  ui/static/       vanilla JS single-page app (no build step)
  demo/            synthetic demo course generator (also the test fixture)
  cli.py           command-line entry point
```

## Data flow

```
SourceFile -> DocumentPage (text per page)
           -> Exam / ExamSection / ExamQuestion tree            (IngestService)
           -> CourseTopic tree (merged documents)               (IngestService)
AnalysisService.run:
  load topics + included exams (order_index) + leaf questions
  SyllabusAligner.align           -> QuestionTopicMapping (manual mappings kept)
  QuestionTypeClassifier          -> ExamQuestion.question_types (manual types kept)
  find_recurrence                 -> exact/paraphrase/concept links
  build_panel (topic, concept, unfiltered topic)
  BacktestEngine.run              -> ModelResult, BacktestFold, calibration, leakage audit
  run_ablation, syllabus_filter_check, run_type_forecast, backtest_families
  build_topic_predictions         -> Prediction (topic, concept, family layers)
  generate_formulations           -> PredictedQuestion
  discover_structure, coverage, charts, simulate_papers -> AnalysisArtifact
```

## The leakage rule

Every model sees history only through `Panel.until(t)` (a copy holding exams `0..t-1`) and
`FeatureStore.at(t)` / `FeatureStore.training_rows(t, ...)`, which are built on it. Never index
`ctx.panel.Y[t]` or later inside a model. Meta models (tuned variants, the ensemble) may read
`ctx.fold_metric[...]` only for targets `s < t`.

`evaluation.backtest.leakage_audit` scrambles all exams from `t` onward and checks predictions
for `t` do not change. It runs on every analysis and in `tests/unit/test_temporal_and_models.py`,
which also proves it catches a deliberately leaky model.

Two subtler points, both documented in ARCHITECTURE.md:

* Alignment TF-IDF is fitted on the syllabus only. The feedback pass that expands topic
  vocabulary uses confidently mapped questions from all papers; this is treated as annotation
  (like a person labelling every paper), not as a predictive feature.
* Question similarity for recurrence uses stateless character n-grams (or a pretrained model),
  so no statistics from later papers enter earlier comparisons.

## Adding a prediction model

1. Subclass `models.base.BaseModel`; set `name`, `display`, `family`, `complexity` (higher means
   more complex; the one-SE rule prefers lower) and a `description`.
2. Implement `predict(t, ctx) -> ModelOutput(scores)`. Use `ctx.store.at(t)` features or
   `ctx.panel.until(t)`; for learned models use `ctx.store.training_rows(t, min_history, ...)` and
   fall back with `learned._fallback` when the rows are too few.
3. Implement `gate(panel, ctx)` returning `(False, reason)` when data is insufficient. Put
   thresholds in `[models.sufficiency]` in `default.toml`.
4. Register it in `models/registry.py` and add its name to `models.enabled`.
5. Add a unit test; the leakage audit covers it automatically.

The model then appears in the comparison table, the ensemble's candidate list and selection.

## Adding a feature

Add the computation in `features/builder.compute_features` and its name to one group in
`FEATURE_GROUPS`. Groups drive the ablation study and the contribution breakdown, so choose the
group that matches the evidence family.

## Adding a question type

Add an entry to `config/taxonomy.json` with `id`, `label`, `format` (one of definition, theory,
derivation, numerical, diagram, objective), `weight` and regex `patterns`. Users can do the same
without touching the code by placing a `taxonomy.json` in their data folder.

## Adding a file format

Add an extractor returning `ingestion.types.ExtractionResult` and dispatch to it from
`ingestion/extract.py` (`sniff_type` and `extract_document`). Add the extension to
`ingestion.supported_extensions`.

## Configuration

`config/default.toml` is the only place defaults live. `Settings` is read-only;
`settings.merged({...})` derives a new object. Overrides from `<data_dir>/settings.json` and from a
course's `settings` column are validated against the defaults: unknown keys and wrong types raise
`SettingsError`.

## Logging

`utils.logging.log_event(logger, "event_name", key=value)` writes structured fields. The console
shows warnings; `<data_dir>/logs/predictor.log` keeps INFO and above as JSON lines. Loggers:
`predictor.ingestion`, `predictor.ocr`, `predictor.embeddings`, `predictor.evaluation`,
`predictor.analysis`, `predictor.jobs`.

## Tests

| Folder | What it covers |
|---|---|
| `tests/unit` | parsing formats, metadata, syllabus parsing and merge, text normalisation, quality flags, question types, metrics (hand-computed and against simulation), temporal statistics, hazard, rotation test, features, leakage audit, model gating, selection, calibration, generation grounding, structure, co-occurrence, format forecast, settings validation |
| `tests/integration` | real PDF, scanned PDF (OCR), DOCX with Word numbering, images, duplicates; the full demo pipeline (alignment accuracy, out-of-syllabus handling, planted patterns, exports, manual edits surviving re-analysis); the HTTP API |
| `tests/regression` | a snapshot of the seeded demo run: parsing counts, mapping statuses, selected model, NDCG of every model, the top-10 ranking and categories |

OCR tests skip automatically when the `tesseract` binary is missing.

After an intended behaviour change, refresh the regression snapshot and review the diff:

```bash
python tests/regression/snapshot_tools.py
git diff tests/regression/snapshots/
```

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
| `GET /api/courses/{id}/results`, `GET /api/runs/{id}/predictions?layer=topic|concept|family` | results |
| `GET /api/runs/{id}/topics/{topic_id}` | evidence for one topic |
| `GET /api/runs/{id}/models`, `GET /api/runs/{id}/artifacts/{key}` | backtest, charts, structure, ablation ... |
| `POST /api/runs/{id}/papers` | new predicted-paper variants |
| `GET /api/courses/{id}/search?q=...` | search |
| `GET /api/runs/{id}/export?format=pdf|xlsx|csv|json` | export |

## Conventions

* Keep data processing, ML, business logic and UI in their packages; services glue them together.
* No magic numbers in code: add a setting with a comment in `default.toml`.
* User-visible text is plain and specific. Never claim certainty about future exams.
* Every new ML component needs a backtest-visible reason to exist (see the ablation study).
