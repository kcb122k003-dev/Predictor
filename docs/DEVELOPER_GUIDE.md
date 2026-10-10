# Developer guide

Read [ARCHITECTURE.md](ARCHITECTURE.md) first: it records the design decisions (D1 to D18) that
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
  syllabus/        syllabus parser, document merge, TopicTree (unit/topic/concept), alignment
                   (A/B/C/D, feedback from earlier papers only)
  embeddings/      backends.py (TF-IDF, sentence-transformers, backend choice), pretrained.py
                   (bundled WordLlama model and the hybrid WordLlama + TF-IDF backend),
                   embedding cache, vector index
  topic_modeling/  question families: exact, paraphrase, concept recurrence
  temporal/        Panel (exams x items) with until(t): appearances, marks, formats, semantic soft
                   counts, data-quality sums; hazard/Markov/gap/rotation statistics
  features/        grouped features computed at a cutoff, GENERIC_FEATURES (scale-free subset),
                   FeatureStore cache
  inference/       bayes.py (hierarchical Bayesian recurrence, prequential log scores),
                   evidence.py (evidence profile, input quality, component status, run-level
                   message),
                   generic.py + generic_prior.json (general ranking model trained on simulated
                   courses), repository.py (cross-course feature rows, model registry),
                   uncertainty.py (rank intervals, evidence strength)
  models/          base.py (interface), statistical.py (reference baselines, Markov),
                   components.py (ensemble components: Bayesian recurrence, temporal, semantic,
                   coverage, question type, co-occurrence, general model, course logistic),
                   learned.py (random forest, gradient boosting, pooled HMM), meta.py (tuned
                   variants, best single method chosen on earlier papers, evidence-aware
                   ensemble), registry.py, calibration.py (Platt and isotonic, validated by a
                   nested check)
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

The Syllabus Explorer and the clearer Predictions tab added these modules:

| Module | What it does |
|---|---|
| `analysis/topic_history.py` | builds the run's `topic_history` artifact (papers, questions, per-topic counts, timeline rows) from the same counted mappings as the topic panel |
| `parsing/format_labels.py` | student-facing format labels (Definition, Explain, Derive, Numerical calculation, ...) and the format families they group into; display only, never a model input |
| `prediction/format_guide.py` | the question format guide of one topic: suggested family, description, placeholder template, reason, alternatives and reliability |
| `prediction/priority_reason.py` | one sentence saying why a topic has its priority, built only from recorded counts; `priority_word` shows old "Extremely High" runs as "Very high" |
| `services/fingerprint.py` | the per-run data fingerprint, the freshness check and `ENGINE_VERSION` |
| `services/explorer_service.py` | explorer course list, the tree with per-node stats, node detail, syllabus and question sources, PDF page images |
| `app/routes_explorer.py` | HTTP routes for the explorer, freshness, sources and page images |
| `ui/static/js/topic_panel.js` | the node panel shared by the explorer and the Predictions tab drawer: history, past questions, format guide, source, mapping correction, kind tags |
| `ui/static/js/view_explorer.js` | the Syllabus Explorer view: tree, search, filters, sort, and the routes `#/explorer`, `#/course/{id}/explorer` and `#/course/{id}/explorer/{node}` |

## Data flow

```
SourceFile -> DocumentPage (text per page)
           -> Exam / ExamSection / ExamQuestion tree            (IngestService)
           -> CourseTopic tree (merged documents)               (IngestService)
AnalysisService.run:
  load topics + included exams (order_index) + leaf questions
  data_fingerprint                -> AnalysisRun.config["fingerprint_parts"] and data_fingerprint
  get_backend, get_pretrained     -> alignment backend (hybrid by default), WordLlama for semantics
  SyllabusAligner.align           -> QuestionTopicMapping (manual mappings kept; causal feedback)
  QuestionTypeClassifier          -> ExamQuestion.question_types (manual types kept)
  format_labels, families_of      -> student-facing labels and families per question (display only)
  find_recurrence                 -> exact/paraphrase/concept links
  build_panel (topic, concept, unfiltered topic) + semantic soft evidence per paper
  build_history, _check_history   -> topic_history (same counted mappings as the topic panel)
  general_model_for               -> general ranking model (shipped prior unless
                                     models.use_simulated_prior = false, + other real courses)
  BacktestEngine.run              -> ModelResult, BacktestFold, calibration, leakage audit
  build_profile, component_status, low_data_message -> evidence profile, statuses, run message
  topic_uncertainty               -> rank interval per topic
  run_ablation, syllabus_filter_check, run_type_forecast, backtest_families
  build_topic_predictions         -> Prediction (topic, concept, family layers)
  _format_guides (build_format_guide) -> format guide per topic
  generate_formulations + FormulationVerifier -> PredictedQuestion (verified formulations only,
                                     in the order of the topic's format guide)
  priority_reason                 -> one sentence per topic
  discover_structure, coverage, charts, simulate_papers -> AnalysisArtifact (including "evidence"
                                     and "topic_history")
  _persist_results                -> Prediction.features gets "history", "format_guide",
                                     "priority_reason"; AnalysisRun.engine_version = ENGINE_VERSION
  save_course_features, register_course_model -> course_feature_set, model_registry
```

`BacktestEngine.run` predicts every paper after the first from the papers before it (rolling
origin, `models.min_train_exams = 1`) and makes the real forecast with the same code at `t = T`.
Every enabled model runs on every fold. The evidence-aware ensemble (`models/meta.py`) is the final
ranking unless `best_single` beat it. `best_single` (`meta.BestSingleModel`) is an out-of-sample
selector: for each held-out paper it uses the method that did best on the papers before it.
`evaluation.backtest.select_final` compares the two on the same held-out papers and replaces the
ensemble only when `best_single` leads by more than a one-sided t-bound of the paired differences
at `ensemble.replace_confidence` (0.95). With 5 held-out papers that means more than 2.13 standard
errors; with 11, more than 1.81. Only the topic layer runs the leakage audit and calibration; the
concept and unfiltered panels skip both.

A course with a syllabus and no papers still runs (`T = 0`). No fold exists, so the ensemble uses
its prior weights (syllabus structure and the general model), every topic gets the rank range
`[1, K]` with level High, and no probability is shown.

Papers of a synthetic course (the demo) get `Exam.source = "demo"`. Their feature rows are stored
flagged and never train the general model. Generated formulations and simulated papers are never
stored as exams or questions.

## The leakage rule

Every model sees history only through `Panel.until(t)` (a copy holding exams `0..t-1`) and
`FeatureStore.at(t)` / `FeatureStore.training_rows(t, ...)`, which are built on it. Never index
`ctx.panel.Y[t]` or later inside a model. Meta models (tuned variants, `best_single`, the ensemble)
read other models' predictions for target `t`, which were made from exams before `t`, and may read
`ctx.fold_metric[...]` only for targets `s < t`. Those fold scores use a causal K: the backtest
scores fold `s` with the K known once paper `s` is in (`k_known(s)`, the median topics per paper
over papers `0..s`), not the K of the whole history. Otherwise a later paper with many topics would
move K and change the earlier scores a meta model chooses from.
`test_later_papers_never_change_earlier_predictions` puts every topic into papers 6 onward, which
changes the run's K, and checks that no model's prediction for papers 1 to 6 changes. If you cache
a whole-panel computation in `ctx.cache`, read only the entries for exams before `t`, as
`TemporalModel` does with its prequential log scores.

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
  fits nothing. The feedback pass that expands topic vectors and vocabulary is causal
  (`SyllabusAligner._align_causal` in `syllabus/alignment.py`). A first pass maps every question
  without feedback. Then, paper by paper in exam order, the topic vectors are reset to the syllabus,
  expanded only with confident (status A) first-pass matches from earlier papers, and used to map
  that paper. A later paper's wording cannot change how an earlier paper is mapped, and the semantic
  soft evidence in the panel, which comes from the same scores, is causal too. The causal path needs
  a time order on every question (`QuestionItem.order`, which `AnalysisService` sets from the exam
  index); without it, `align` falls back to feedback from all questions. On the demo this change
  moved the status counts from A 109 / B 27 / C 4 / D 4 to A 104 / B 30 / C 6 / D 4. With the rest
  of the code unchanged, it moved the backtest scores against mapped labels by -0.072
  (co-occurrence) to +0.016 (coverage): the ensemble fell by 0.023, frequency by 0.031 and semantic
  evidence by 0.033, while coverage, the course logistic (+0.002) and the random forest (+0.003)
  went up. Most old app-label scores were optimistic because the held-out paper's own wording fed
  the alignment.
* Question similarity for recurrence uses stateless character n-grams (or a downloaded
  sentence-transformer), so no statistics from later papers enter earlier comparisons. The bundled
  WordLlama model is not used here: for 26 reworded test questions, the nearest past question it
  found in the demo bank was on the same topic for 11, against 14 with character n-grams.
* The general ranking model never learns from the course it predicts. `general_model_for` updates
  the shipped prior with the scale-free rows of other real courses only, and skips synthetic
  courses. Course A's rows can change the general model used for course B, never course B's panel,
  frequencies or posteriors (`test_other_courses_do_not_change_this_courses_history`). With
  `models.use_simulated_prior = false` there is no shipped prior: the general model is learned from
  other real courses alone, and with none of them the general component is UNAVAILABLE.

Mapping corrections are labels, and they do not leak into earlier folds either: the causal
alignment feedback uses earlier papers' first-pass automatic matches, never your corrections.
`test_mapping_correction_marks_results_stale_and_updates_counts_without_leakage` corrects a
question in the last paper, re-runs the analysis and checks that every backtest metric for earlier
papers is unchanged to 1e-9.

## Topic history, format guides and freshness

The Syllabus Explorer and the Predictions tab show the same numbers because they read the same
stored run. Every count comes from the run's `topic_history` artifact. Rank, priority, likelihood,
the format guide and the priority reason come from the topic `Prediction` rows, which both views
serialise with `results_service.prediction_dict`. Neither view recomputes results from the live data
when a request arrives; unit and sub-topic counts are summed from the stored artifact. If you add a number to either view, store it with the run and read it from there.

### The topic_history artifact

`build_history` (`analysis/topic_history.py`) runs right after the topic panel is built, from
exactly the question records that fill it. A question counts for a topic when one of its counted
mappings (status A, and B at the default strictness, as `counts_for_prediction` decides) rolls up
to the topic. Rank 1 is the primary topic and rank 2 the secondary one.

| Key | Contents |
|---|---|
| `version` | artifact version (1) |
| `usable_papers` | T, the number of papers in the analysis |
| `papers` | one entry per usable paper in panel order: `index`, `exam_id`, `label`, `year` (null when undated), `session`, `file`, `file_id`, `provisional` (reasons its counts may be unreliable: unknown or low-confidence year, low-OCR pages, questions flagged for review) |
| `excluded_papers` | papers left out of the analysis: `exam_id`, `label`, `year`, `kind` (`duplicate`, `undated` or `excluded`), `reason`, `duplicate_of` |
| `missing_years` | calendar years between the first and last usable paper with no usable paper |
| `undated_papers` | panel indexes of usable papers with no year |
| `questions` | map from question id (as a string) to a compact record: `e` (paper index), `nodes` (counted mappings as `[node, rank, weight]`), `uncounted` (matches that do not count, as `[node, status, score]`), `status`, `manual`, `conf`, `labels`, `families`, `fmt` (the model's coarse format), `marks`, `rel` (`exact`, `paraphrase`, `concept` or `new`), `prev` (the earlier questions it relates to), `fam` (recurrence family key), `pc` (parse confidence), `review` |
| `topics` | map from topic id (as a string) to `node_stats` for that topic, described below |
| `course_families` | format families across every counted question of the course (`family`, `display`, `questions`, `papers`, `of_questions`); used only to infer a format for topics with no past question |

Each `topics` entry holds `usable_papers`; `papers` (indexes where the topic appeared) and
`exam_frequency` (how many); `question_frequency`, `primary_questions`, `secondary_questions` and
`secondary_only_papers`; `per_paper` (question ids per paper) and `roles`; `first_index`,
`last_index`, `last_label` and `last_year`; `recent_window` and `recent_hits` (the last 3 papers);
`marks` (`known`, `min`, `max`, `median`, from primary questions); `labels` and `families`, each
with `questions` and `papers`; `multi_label_questions`; `repetition` (counts of `exact`,
`paraphrase`, `concept` and `new`); `uncounted_questions`; `probable_questions`; and `provisional`
with up to 12 `provisional_reasons`.

Rules the artifact follows:

* Exam frequency counts a paper once however many of its questions hit the topic. Question
  frequency counts every counted question.
* `relation_of` compares a question with earlier papers only, so same-paper pairs never count as
  repeats. A repeat or paraphrase link to a question of a different format family counts as the
  same concept. A shared syllabus node is a shared concept only when it is finer than the topic
  (`topic_modeling/families.py`).
* A missing year or an excluded paper is missing data, never "not examined". `timeline` returns
  them as rows of kind `missing` or `excluded`.
* Counts are provisional when a contributing paper has reasons, a question has parse confidence
  below 0.7 or is flagged, or at least half of the counted questions are probable (status B,
  not manual) matches.
* Units and sub-topics are not stored. The explorer computes their stats on request with
  `node_stats`, counting each question once, and `node_index` limits the scan to questions mapped
  below the node. On the demo the overview takes about 0.02 s and a node about 0.03 s; building the
  history for 400 topics and 6,000 questions takes 0.09 s.

`_check_history` in `services/analysis_service.py` must pass on every run. It compares each
topic's `exam_frequency` with the column sum of `panel.Y` and its `question_frequency` with the
column sum of `panel.n_questions`, and logs `history_mismatch` at ERROR level on any difference.
A mismatch is a bug: the views would show numbers the model never used. If you change how the
panel counts (strictness, roll-up, which papers are usable), change `build_history` the same way.
`test_counts_match_the_model_panel_and_the_predictions_tab` checks the result end to end.

Each topic's `Prediction.features["history"]` holds a compact copy (`_history_summary`) for the
prediction cards, with the most common family as `top_format` and up to 6 labels.

### The question format guide

`AnalysisService._format_guides` calls `build_format_guide` for every topic during the run, using
the topic's counted questions from `topic_history`, and stores the result in
`Prediction.features["format_guide"]`. The guide is the only place the app states a format; the old
evidence lines "Most common question format" and "Predicted format if it appears" were removed
because they contradicted topics' own history.

| Field | Meaning |
|---|---|
| `family`, `display` | suggested family: the one asked in the most papers containing the topic; ties go to more questions, then the more recent family, then `FAMILY_ORDER` |
| `basis` | `history`, `weak_history`, `syllabus`, `course_pattern` or `none` |
| `evidence` | `established` (basis `history`: the family in 2 or more papers, and certain matches, status A or manual, in at least 2 of them), `weak` (one paper, or resting on probable matches) or `inferred` (no past question: from the syllabus entry's kinds, else the course-wide mix) |
| `papers`, `questions`, `last_label` | the family's papers and questions for this topic, and the paper it last appeared in |
| `topic_papers`, `usable_papers` | papers containing the topic, and T |
| `description` | one specific sentence built from the topic's representative question in that family (primary role first, then certain matches, then the most recent), or from the syllabus wording when there is none |
| `template` | numerical and design families only: `text` with every value replaced by a named `[placeholder]`, `placeholders` (`name`, `unit`, `example`), `source` and `note`; null when no value was found |
| `illustrative` | the practice question. For numerical and design families it is the real past question word for word (`basis: "past_values"`, with `source`, and `probable_match` when the link is status B). Otherwise `_attach_illustrative` attaches a verified generated formulation (`basis: "generated"`), or leaves null and sets `illustrative_note` |
| `marks_note` | the marks of the topic's primary past questions in that family |
| `why` | the reason with its counts; states ties as ties and says when counts overlap |
| `alternatives` | up to two more families from the same order, each with `description`, `support` and, where it applies, `template` and `illustrative` |
| `reliability` | `topic_history`, `course_mix`, `folds` and `text` from the format forecast backtest (on the demo, over 11 earlier papers, the topic's most frequent past format matched a format it was asked in 37% of the time, and the course-wide mix 40%) |

A past numerical question shown with its values must pass the same topic check as a generated
question (`FormulationVerifier.check` through the `accept` callback). The app never invents new
values, because it cannot check that they give a solvable problem.

`priority_reason` (`prediction/priority_reason.py`) builds the "why this priority" sentence from
recorded facts only: appearances, the last 3 papers, last appearance, syllabus match strength and
secondary-only appearances. For a topic that never appeared it says there is no historical
evidence in the T usable papers and that missing from past papers is not evidence that the topic
will not be examined. Never describe such a topic as unlikely or rarely tested.

### Freshness and ENGINE_VERSION

At the start of every run, `fingerprint.data_fingerprint` hashes five parts of the course data. The
run stores them in `AnalysisRun.config["fingerprint_parts"]` and their combined hash in
`AnalysisRun.data_fingerprint`.

| Part | What it covers |
|---|---|
| `papers` | every paper except generated and simulated ones: id, included or not, order index, year, source, label, full marks |
| `questions` | every question of those papers, included or not: text, marks, parent, leaf flag, optional flag, OR group, and question types only when you set them |
| `mappings` | your manual mappings (question, topic, status, rank) |
| `syllabus` | the topics of the current syllabus version: number, title, parent, concepts, excluded flag, kinds, hours, marks weight, order |
| `settings` | `analysis_settings(settings)`: every section except `NON_ANALYSIS_SECTIONS` (`app`, `logging`, `server`, `ui`, `ocr`, `ingestion`, `quality`, `parsing`, `search`, `export`), with `data_dir`, `model_dir` and keys ending in `_dir` or `_path` removed |

Automatic mappings and automatic question types are outputs of the analysis, so they are left out
and re-running never makes a run look stale. `fingerprint.freshness(session, run, settings)`
recomputes the parts, lists the ones that differ in `changed`, and adds `engine` when
`run.engine_version` differs from `ENGINE_VERSION` (now `"2.1"`). It returns `run_id`, `checked`,
`stale`, `changed` and `message`. A run made before change tracking has no stored parts, so
`checked` is false and the explorer shows the message asking for a re-run. When `stale` is true,
both tabs show "Results out of date" with the reasons and a "Re-analyse now" button. Nothing is recomputed in the background.

Bump `ENGINE_VERSION` when a change alters what a run stores for the same data: counts in
`topic_history`, format guides, priority reasons, practice questions, stored wording, or model
scores. A change to the UI alone, which only reads stored fields, needs no bump. When you add a
setting that cannot change an analysis's results, put it in a section listed in
`NON_ANALYSIS_SECTIONS`, or add its section there; otherwise changing it marks every run out of
date. When you add a stored input that does change results, add it to the matching part in
`data_fingerprint`.

### Mapping corrections

`PUT /api/questions/{id}/mapping` calls `ReviewService.set_mapping(qid, topic_ids, status)`. The
status is A, B or D; D with no topics marks the question outside the syllabus and is stored as one
manual row with a null `topic_id`. Before it deletes the question's rows, `set_mapping` snapshots
every automatic mapping (`_auto_snapshot`, the fields in `AUTO_FIELDS`) into the rank-1 manual
row's `evidence["replaced_automatic"]`. A second correction carries the first snapshot over, so the
original automatic mapping is never lost. `DELETE /api/questions/{id}/mapping`
(`clear_manual_mapping`, the panel's "Use automatic mapping") removes the manual rows and, when no
automatic row exists, re-creates the automatic rows from the snapshot. It skips snapshots of
topics deleted since; the next analysis maps those questions again.

The explorer shows a saved correction at once as "your correction, not yet counted": `_live_manual`
and `_mapping_now` in `explorer_service.py` compare the live manual rows with what the run used.
Counts and predictions change only after re-analysis, and the `mappings` part of the fingerprint
marks the run out of date until then. The analysis now honours a manual "outside the syllabus"
mark; earlier versions ignored it. `test_undoing_a_correction_restores_the_automatic_mapping`
covers the undo.

## Adding a prediction model

1. Subclass `models.base.BaseModel`. Set `name`, `display`, `family`, a `description` and
   `complexity` (higher means more complex; it breaks ties between methods with the same mean, and
   the compatibility helper `select_model` uses it to prefer the simplest method within one
   standard error of the best). Then set the attributes the evidence-aware ensemble reads:
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
   * `meta` and `meta_order`, only for a model that reads other models' predictions.
     `ordered_models` runs every base model first, then the meta models by `meta_order`: tuned
     variants 0, `BestSingleModel` 1, the ensemble 2. Give a new meta model an order after
     everything it reads.
2. Implement `predict(t, ctx) -> ModelOutput(scores, info)`. Use `ctx.store.at(t)` features or
   `ctx.panel.until(t)`; learned models use `ctx.store.training_rows(t, min_history, ...)`. There
   is no minimum dataset size: return a forecast for every `t`, including `t = 0` for a course
   with no papers. Two exits exist, and neither is a size gate:
   * When an input the model needs does not exist (no earlier paper to compare with, no syllabus
     units or weights, a single question format), return `components.unavailable(K, reason)`. The
     ensemble leaves the model out for that target. If this happens on the final forecast, the
     status table shows UNAVAILABLE with your reason. This is the only way to UNAVAILABLE: a
     component with little data stays in as LIMITED, and one that ranked earlier papers badly
     stays in as DOWNWEIGHTED.
   * When fitting is mathematically impossible (every training label is the same), return
     `components.general_fallback(t, ctx, reason)`. The report counts those folds.
3. Leave `gate()` alone. It answers only whether a model can run at all, and no model in the
   registry overrides the default `(True, "")`. The old `[models.sufficiency]` thresholds are
   gone: `df` and the skill measured on earlier papers decide how much a model counts.
4. Register it in `models/registry.py` and add its name to `models.enabled` in `default.toml`. For
   an ensemble component, also add the name to `COMPONENT_ORDER`: the ensemble's members are the
   built models named there, and it reads each member's `df` and `prior_knowledge`. Every visible
   model except `random` is also a candidate for `BestSingleModel`, so it can become the final
   ranking through `best_single`. To tune a hyperparameter on earlier folds, register each variant
   as a hidden model and wrap them in `meta.TunedModel` with a default (see `recency` and
   `beta_binomial`). It keeps the default unless another variant is ahead on earlier folds beyond a
   one-sided 95% t-bound of the paired differences.
5. If the quality of the component's inputs varies between courses, add an entry to
   `input_quality` in `inference/evidence.py`. It is computed from papers before the cutoff only,
   and the ensemble multiplies the component's reliability prior by it; a component without an
   entry gets 1. A course-learned model also belongs in `COURSE_LEARNED` there (input quality
   0.5 + 0.25 * marks known + 0.25 * parse confidence, and it drives the run-level message), and a
   temporal one in `TEMPORAL_COMPONENTS` (1 - 0.5 * missing years / span).
6. Add a component to a stage in `evaluation/ablation.STAGES` so the staged ablation measures it.
   The leave-one-component-out rows cover every ensemble member automatically.
7. If the model refits a classifier or another slow model on every call, add its name to
   `inference/uncertainty.SLOW`.
   The leave-one-paper-out rank intervals then keep its full-data scores instead of refitting it
   once per paper.
8. Add a unit test, and add a component's name to `COMPONENTS` in `tests/unit/test_low_data.py`
   so the no-cutoff test checks that it runs at 2, 3, 5, 9, 15 and 18 papers. The leakage audit
   covers it automatically.

The model then appears in the comparison table with its status, weight and reliability. The
statuses (ACTIVE, LIMITED, DOWNWEIGHTED, UNAVAILABLE, REFERENCE) and the weight formula are in
[LOW_DATA_INFERENCE.md](LOW_DATA_INFERENCE.md), section 5. A single method becomes the final
ranking only through `best_single`, and only when `best_single` beat the ensemble on the same
held-out papers by more than the one-sided 95% t-bound (`ensemble.replace_confidence`). On the demo
the ensemble scores 0.710 and `best_single` 0.707 (difference +0.004, standard error 0.019), so the
ensemble stays. In 112 simulated backtests `best_single` never replaced it, and
`test_selection_does_not_replace_the_ensemble_on_noise` checks that two equally good rankings lead
to a replacement in fewer than 8% of 400 trials.

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

The student-facing format label comes from `TYPE_LABELS` in `parsing/format_labels.py`. Map the new
type there (next section); otherwise its questions show only the labels their instruction verbs
produce, or "Unclassified".

## Adding a format label or family

Format labels and families are display only. The models read the classifier's coarse `format`,
never the labels, so adding one changes no backtest number. It does change the format guides a run
stores, so bump `ENGINE_VERSION` (see "Freshness and ENGINE_VERSION").

To add a label:

1. Add its id and display name to `LABELS` in `parsing/format_labels.py`. The dict order is the
   display order.
2. Make it reachable: map a classifier type to it in `TYPE_LABELS`, or add a pattern to
   `VERB_LABELS`, which reads the instruction verbs in the question's own text. Items under "Write
   short notes on" stay short notes (`format_labels` drops other labels for a short item with no
   instruction verb), so check that your pattern does not fire on a bare item name.
3. Add the id to the `labels` of one family in `FAMILIES`. The explorer's format table, timeline,
   question items and format filter, and the format guide, all work in families. A question whose
   labels fall in no family shows there as "Not classified".
4. Add a case to `tests/unit/test_format_guide.py`
   (`test_labels_read_the_instruction_and_allow_several` covers labels).

To add a family:

1. Add it to `FAMILIES` with `display` and `labels`, and place it in `FAMILY_ORDER`. That order
   breaks ties between families asked in as many papers and questions (more specific first), and
   `families_of` returns families in it. `families_of` drops `short_note` when a more specific
   family applies.
2. Set `FAMILY_GENERATOR_FORMAT[family]` to the generator format that illustrates it, or `None`
   when no generator template fits. With `None` the guide shows no generated practice question and
   says why.
3. If a syllabus kind should imply the family for topics without history, add it to `KIND_FAMILY`.
4. In `prediction/format_guide.py`, add a `FRAMES` sentence, an `OPENERS` pattern (strips the
   instruction from a past question to find its object), a `FAMILY_CUES` pattern (picks the
   sentence that asks for this format) and a `DEFAULT_VERBS` entry. Without them the guide falls
   back to the verb "explain" and a bare "{verb} {obj}." sentence.
5. A family whose questions carry values, like `numerical` and `design`, also needs its id where
   `build_format_guide` and `_alternative` call `_numerical_parts`, and in the family set that
   `_representative` sends through the topic check. Its practice question is then a real past
   question that passed that check.
6. Add tests to `tests/unit/test_format_guide.py`. The regression snapshot stores no guides, so it
   does not change.

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
`prediction.uncertainty`. It also added three tables: `model_registry` (scope "global": one row
per distinct general model used, keyed by a fingerprint of the prior source and the other courses'
feature sets, which each run records in its config; scope "course": a run's course logistic),
`course_feature_set` (one course's scale-free feature rows, flagged when synthetic) and
`schema_version`.
`tests/fixtures/schema_v1.sql` holds the version 1 schema, and
`test_migration_from_version_1_keeps_data` checks that a database built from it keeps its rows.

The Syllabus Explorer needed no migration. The per-topic history summary, format guide and
priority reason live in `prediction.features` (JSON), the full history in an `analysis_artifact`
row with key `topic_history`, the fingerprint parts in `analysis_run.config`, and the replaced
automatic mapping in the manual row's `question_topic_mapping.evidence`. New runs write
`analysis_run.engine_version = "2.1"`.

## Configuration

`config/default.toml` is the only place defaults live. `Settings` is read-only;
`settings.merged({...})` derives a new object. Overrides from `<data_dir>/settings.json` and from a
course's `settings` column are validated against the defaults: unknown keys and wrong types raise
`SettingsError`. The exception is `DEPRECATED_KEYS` in `config/settings.py`. Settings removed in
later versions (`models.sufficiency`, `calibration.min_rows`, `calibration.min_positives`,
`ensemble.replace_if_worse_by_se`, `models.selection_rule`, `models.logistic_C`,
`models.ensemble_max_members`) still load from old files and are ignored. When you remove a
setting, add it there so existing installations keep working.

The low-data engine adds `embeddings.pretrained`, `embeddings.hybrid_pretrained_weight`,
`alignment.semantic_temperature`, `temporal.bayes_half_lives`, `models.course_prior_precision`,
`models.use_simulated_prior`, `calibration.methods` and the `[ensemble]` section (including
`ensemble.replace_confidence`), and sets `models.min_train_exams` to 1.
[LOW_DATA_INFERENCE.md](LOW_DATA_INFERENCE.md), section 12, explains each one.

Settings are part of each run's fingerprint, apart from the sections in
`services/fingerprint.NON_ANALYSIS_SECTIONS` and any folder or file path. Put a setting that cannot
change an analysis's results in one of those sections, so changing it does not mark runs out of
date.

## Logging

`utils.logging.log_event(logger, "event_name", key=value)` writes structured fields. The console
shows warnings; `<data_dir>/logs/predictor.log` keeps INFO and above as JSON lines. Loggers:
`predictor.ingestion`, `predictor.ocr`, `predictor.embeddings`, `predictor.evaluation`,
`predictor.analysis`, `predictor.jobs`. A `history_mismatch` event on `predictor.analysis` means
the topic history and the model's panel disagree (see `_check_history`); treat it as a bug.

## Tests

| Folder | What it covers |
|---|---|
| `tests/unit` | parsing formats, metadata, syllabus parsing and merge, text normalisation, quality flags, question types, metrics (hand-computed and against simulation), temporal statistics, hazard, rotation test, features, leakage audit, every component running on a 5-paper panel, the one-standard-error helper `select_model` kept for compatibility, calibration, generation grounding, structure, co-occurrence, format forecast, settings validation |
| `tests/unit/test_low_data.py` | low-data behaviour on planted panels with 0 to 25 papers: every component runs at 2, 3, 5, 9, 15 and 18 papers, semantic, Bayesian and general components keep weight at 2 to 5 papers, course-learned models are downweighted at 3 papers and gain weight at 25, Bayesian intervals for 1 of 2 against 8 of 16, question counts do not change exam-level evidence, the leakage audit including meta models, later papers never changing earlier predictions (causal K, `test_later_papers_never_change_earlier_predictions`), semantic evidence capped per paper (`test_semantic_evidence_is_capped_per_paper`), input quality changing reliability (`test_input_quality_changes_component_reliability`), the ensemble not replaced on noise (`test_selection_does_not_replace_the_ensemble_on_noise`, under 8% of 400 trials), calibration rejecting uninformative scores (`test_calibration_rejects_scores_unrelated_to_outcomes`, at most 5% of 60 trials), the general component reported UNAVAILABLE without the simulated prior (`test_without_simulated_prior_the_general_model_is_reported_unavailable`), the low-data message, reproducibility, ranking with zero papers |
| `tests/unit/test_repository_and_migrations.py` | the general model learns only from other real courses, other courses never change this course's history, the simulated prior can be turned off and each distinct general model gets one registry row (`test_simulated_prior_can_be_turned_off`), a version 1 database (`tests/fixtures/schema_v1.sql`) migrates and keeps its data |
| `tests/unit/test_topic_history.py` | paper counts kept apart from question counts, secondary roles, sub-topic roll-up (a group counts each question once), the candidate index never changing a result, uncounted (status C) matches, missing years as missing data, excluded and duplicate papers, undated papers, multi-label formats, repetition classes (same-paper pairs ignored), provisional reasons, zero papers |
| `tests/unit/test_format_guide.py` | format labels and families, numerical templates and placeholder naming (a value never takes another value's name or a quantity word that contradicts its unit), object phrases, a guide built from history, the topic check for past numerical questions, weak and inferred formats, priority wording for never-appeared topics, short-note items |
| `tests/integration/test_explorer.py` | the tree shows every syllabus node, counts equal the model panel and the Predictions tab, identical prediction dicts, original wording and sources, a duplicate paper shown as excluded, group nodes without a prediction of their own, format counts, marginal probabilities shown only when calibrated, illustrative questions labelled and grounded and never history, never-appeared wording, source passages and page images, a mapping correction marking results stale and leaving earlier folds unchanged, excluded topics and papers, undo restoring the automatic mapping, a course with only a syllabus |
| `tests/integration` | real PDF, scanned PDF (OCR), DOCX with Word numbering, images, duplicates; the full demo pipeline (alignment accuracy, out-of-syllabus handling, planted patterns, component statuses, exports, manual edits surviving re-analysis, generated material never counted as history, the evidence artifact, rank intervals and ablation metrics); a course with a syllabus and no papers (`test_course_with_a_syllabus_and_no_papers_still_gets_a_ranking`: full rank range, level High, no probability); the HTTP API |
| `tests/regression` | a snapshot of the seeded demo run: parsing counts, mapping status counts, the final model (the ensemble) and whether it was calibrated, NDCG of every model, the top-10 ranking and categories |

The full suite has 133 tests. OCR tests skip automatically when the `tesseract` binary is missing.

After an intended behaviour change, refresh the regression snapshot and review the diff:

```bash
python tests/regression/snapshot_tools.py
git diff tests/regression/snapshots/
```

The snapshot last changed with the Syllabus Explorer work. Paraphrase detection had chained
short-note titles into long questions, because `token_set_ratio` scores a title contained in a
longer question as 100. `topic_modeling/families.py` now caps the ratio with `token_sort_ratio`
when one question has fewer than half the distinct content terms of the other. That changes a model feature: the
ensemble's NDCG moved from 0.717 to 0.710 and the course logistic's from 0.682 to 0.710, several
categories changed, and "Extremely High Priority" became "Very High Priority". The status counts
stayed at A 104 / B 30 / C 6 / D 4. In the ablation table, exact-repeat recall for the frequency
stage fell from about 0.43 to about 0.17 because repeat families are no longer chained, so the
target is finer; that is a change of definition, not lost accuracy.

History: the refresh before that one came with causal alignment feedback, which moved the status
counts to A 104 / B 30 / C 6 / D 4 and the ensemble's NDCG to 0.717 at the time. It also took in
the review fixes made since the previous snapshot (causal K, capped semantic evidence and others),
so its diff is not the effect of causal feedback alone. "The leakage rule" above gives that effect.

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
| `GET /api/courses/{id}/mappings`, `PUT/DELETE /api/questions/{id}/mapping` | mapping overrides (PUT keeps the automatic mapping it replaces and DELETE restores it; see "Mapping corrections") |
| `POST /api/courses/{id}/analyze`, `GET /api/runs/{id}` | run analysis, poll progress (the course needs syllabus topics; a run with zero papers is allowed and returns a ranking with full rank ranges and no probabilities) |
| `GET /api/courses/{id}/results`, `GET /api/runs/{id}/predictions?layer=topic|concept|family` | latest run with its predictions and the `excluded`, `sufficiency` and `evidence` artifacts (`evidence` is null for runs from the earlier engine, and the UI then asks you to rerun); `results` also returns `freshness`; each topic prediction carries `evidence_strength`, `uncertainty` (rank interval and level), `history` (compact counts), `format_guide`, `priority_reason` and `priority` (Very high, High, Moderate or Low) |
| `GET /api/explorer/courses` | courses with node count, papers included, latest run and its freshness |
| `GET /api/courses/{id}/explorer` | the tree: every syllabus node with compact stats and a compact prediction, plus `run`, `freshness`, `papers`, `usable_papers`, `excluded_papers`, `missing_years`, `roots` and `families` |
| `GET /api/courses/{id}/explorer/nodes/{node}` | node detail: `node` (with path and source refs), `stats`, `timeline`, `questions`, `uncounted`, `children`, `prediction` (the same `prediction_dict` as the Predictions tab), `child_predictions` or `parent_prediction` for groups and sub-topics, `formulations`, `related`, `freshness`, `calibrated`; `history_available` is false, with a `message`, when the run has no `topic_history` |
| `GET /api/courses/{id}/freshness` | whether the latest run still matches the course data, and what changed |
| `GET /api/topics/{id}/source`, `GET /api/questions/{id}/source` | a syllabus entry's source passages (file, page, line, passage), or a question's original and extracted wording with file, page, line and page text |
| `GET /api/files/{id}/pages/{n}/image?highlight=...` | a PDF page rendered as PNG with PyMuPDF (a core dependency) and the passage marked; the `X-Highlight-Top` and `X-Highlight-Bottom` headers give its position as fractions of the page height |
| `GET /api/runs/{id}/topics/{topic_id}` | evidence for one topic |
| `GET /api/runs/{id}/models` | backtest table: role, scope, status, weight, reliability and metrics of every visible model, plus per-fold metrics |
| `GET /api/runs/{id}/artifacts/{key}` | stored artifacts: `evidence`, `topic_history`, `ablation`, `calibration`, `charts`, `structure`, `coverage`, `papers` ... |
| `GET /api/runs/{id}/artifacts/evidence` | `mode` and `message` (run level), `profile` (evidence profile), `components` (status table with reason, weight, reliability, skill, df), `validation` (held-out papers, fold spread, 95% t-interval of the final ranking's mean metric, baselines), `general_model` (source, real courses and rows used, fingerprint), `uncertainty` (jackknife replicates, posterior draws, components held fixed, level counts, median rank range, overall level), `evidence_quality`, `ensemble` (member weights, reliability, skill and folds, excluded components with reasons, whether it is final), `generation_checks`, `synthetic_course`, `notes` |
| `POST /api/runs/{id}/papers` | new predicted-paper variants |
| `GET /api/courses/{id}/search?q=...` | search |
| `GET /api/runs/{id}/export?format=pdf|xlsx|csv|json` | export (the JSON export includes the `evidence` artifact; the predictions table has `suggested_format`, `format_basis`, `format_description`, `why_this_format` and `priority_reason` columns; practice questions carry the label ILLUSTRATIVE PRACTICE QUESTION) |
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
* Report what the backtest measured, including results that go against the ensemble. On
  simulated courses the in-sample backtest puts the ensemble 0.03-0.08 NDCG above frequency
  (planted generator), but on the next paper alone the gain lies between -0.008 ± 0.012 and
  +0.032 ± 0.012 and is clearly positive only at 5 papers. On the 12-paper demo course the ensemble
  scores NDCG@11 0.710 ± 0.026 against the app's own mapped labels, above plain frequency
  (0.678 ± 0.024; paired difference +0.033 ± 0.019, better on 8 of the 11 held-out papers and
  worse on 3) and below the general model alone (0.730 ± 0.022). The course logistic scores 0.710
  and the best single method chosen on earlier papers 0.707. Against the generator's true topics
  the ensemble scores 0.731 and frequency 0.734 (difference -0.003 ± 0.016), so the demo shows no
  gain over plain frequency there. Calibration is valid on the demo (Platt on this run): Brier
  0.193 against 0.248 for the base rate. The leakage audit passes for all 34 models. Section 8 of
  [LOW_DATA_INFERENCE.md](LOW_DATA_INFERENCE.md) reports both measurements.
* The Syllabus Explorer, the prediction cards, format labels, format guides and priority reasons
  are display only and did not change accuracy. The demo numbers above moved from the previous
  version (ensemble 0.717) because of an engine change, the paraphrase fix described under Tests,
  not because of the interface. Do not describe interface work as an accuracy gain.
* Both tabs read the stored run. Never compute a count in a view that the topic panel did not use,
  and keep `_check_history` passing.
* A topic missing from every past paper has no historical evidence. Never describe it as unlikely
  or rarely tested.
* Tag every value the panel shows with what it is (`kindTag` in `topic_panel.js`: fact, estimate,
  model, input, illustrative, inferred, weak). The Bayesian recurrence rate is a model input,
  never the likelihood. Show a probability only when the run's calibration is valid, and say that
  topic probabilities are marginal and do not add up to 100%.
* Label every generated practice question as illustrative. For numerical problems show a real past
  question word for word, never invented values.
