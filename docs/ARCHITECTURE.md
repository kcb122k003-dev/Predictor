# Architecture and Design Analysis

This document answers the "first task" of the specification (section 73) before any
substantial code was written. It records the analysis, the decisions that came out of it,
and the plan the code follows. Later sections of the code reference these decisions by
number (for example `D3`).

The low-data upgrade (engine version 2.0) changed several of these decisions. A changed
decision keeps its original text so you can see what was decided first, followed by a note
marked **Revised in the low-data upgrade**. Sections that described the old behaviour
(mainly 3, 5, 6, 11 and 15) now describe the engine as built, and shorter notes mark the
changes in sections 4, 7, 9, 10, 12 and 13. For how the ranking engine
works with two, five or fifteen papers, and for the full measurements, read
[Low-data inference](LOW_DATA_INFERENCE.md).

---

## 1. Reading of the specification

The application predicts which **syllabus topics** (and, as a secondary layer, which
question formulations) are likely to appear in the next exam of one course. It has three
inputs: past papers, course contents, and the order in which the papers were set. It must
run offline, treat the syllabus as a hard boundary, validate every model against the past
with time-ordered backtests, and say so plainly when the data cannot support a claim.

The specification is long, but most of it reduces to five engineering problems:

1. **Turning messy documents into structured question records** (OCR, segmentation,
   metadata, quality flags, manual correction).
2. **Mapping each question onto the syllabus** with an auditable confidence and an
   in/out-of-syllabus verdict.
3. **Building a time-indexed panel**: for every exam `t` and every topic `k`, did `k`
   appear in `t`, with how many marks, in which question type, and was it a repeat?
4. **Predicting row `T+1` of that panel** with models that are selected by time-aware
   backtesting rather than by preference.
5. **Explaining and presenting** the result with honest uncertainty.

---

## 2. Contradictions and unrealistic expectations

| # | Tension in the spec | Decision |
|---|---|---|
| D1 | "Works fully offline" vs "modern neural embeddings". Pretrained models must be downloaded once, and the download host may be unreachable. | The default embedding backend is a TF-IDF model over word stems plus character n-grams, fitted locally. It needs no download. A neural backend (sentence-transformers) is optional and is enabled only after an explicit `predictor models download` command. The app never downloads anything on its own.<br><br>**Revised in the low-data upgrade:** TF-IDF is no longer the only model that works without a download. The default backend is `auto`: a sentence-transformer if you downloaded one, otherwise a hybrid of the bundled WordLlama model and syllabus-fitted TF-IDF, otherwise TF-IDF alone. WordLlama (256-dimensional static embeddings, MIT licence) ships inside the `wordllama` pip package, so it needs no download and runs at every course size. The hybrid cosine is 0.3 × pretrained + 0.7 × TF-IDF (`embeddings.hybrid_pretrained_weight`). On the demo course the hybrid, with the causal feedback pass described in section 9, puts 131 of 140 in-syllabus questions on the right topic and keeps all 4 old-syllabus questions outside. Repeat detection still uses character n-grams unless a sentence-transformer is installed, because character n-grams found the right topic for more reworded questions (14 of 26, against 11 for WordLlama). The app still downloads nothing on its own. |
| D2 | "Use Markov, HMM, survival, gradient boosting, calibration" vs "you will often have 5 to 15 exams". A single topic with 10 exams has 10 binary observations. No per-topic model can be fitted on that. | Every learned model is **pooled across topics**: one row per (exam, topic). Ten exams and 40 topics give about 280 training rows after warm-up. Per-topic parameters use empirical-Bayes shrinkage toward the pooled estimate. Complex models are gated by data-sufficiency rules and must also win the backtest.<br><br>**Revised in the low-data upgrade:** the data-sufficiency gates (`models.sufficiency`) were removed and no other threshold replaced them. Every component runs on every backtest fold at every course size. Instead of a gate, each component gets a reliability prior that grows with the number of papers and shrinks with the number of parameters it estimates from this course (section 3). That prior is multiplied by the quality of the component's inputs (mapping confidence, parse confidence, share of marks known, missing calendar years, formats seen, syllabus weight source), and the ensemble weight also depends on how well the component ranked earlier papers. A component reports "unavailable" only when an input it needs does not exist. Pooling stays: the hierarchical Bayesian recurrence pools at course, unit and topic level, and the course logistic model is pulled toward a general ranking model trained on simulated courses and other real courses in your library. Set `models.use_simulated_prior = false` and the simulated part is dropped. A component no longer has to win the backtest to contribute. Old settings files that still contain `models.sufficiency` load normally and the key is ignored. See section 3 and [Low-data inference](LOW_DATA_INFERENCE.md#2-what-replaced-the-cutoff). |
| D3 | "Calibrated probabilities" vs tiny validation sets. Isotonic regression on 200 points overfits. | Platt scaling (two parameters) on the percentile rank of each topic, fitted only on out-of-sample backtest predictions. Probabilities are shown only when nested calibration beats the base-rate Brier score. Otherwise the UI shows "relative score" and says why.<br><br>**Revised in the low-data upgrade:** Platt scaling and isotonic regression now compete. For each held-out paper, each calibrator is fitted on earlier held-out papers only, and the one with the lower nested log loss is used. The fixed row and positive-count minimums (`calibration.min_rows`, `calibration.min_positives`) were removed. Probabilities are shown only when the nested Brier gain over the base rate clears a one-sided 95% t-bound: the gain must exceed the t quantile for the number of papers times the standard error of the per-paper gain. With few papers both the quantile and the standard error are large, so you see relative scores with a rank interval and an evidence strength instead. |
| D4 | Backtests on very few exams. With 5 exams and 3 warm-up exams you get 2 folds; a metric averaged over 2 folds is close to noise. | The backtest reports the standard error for every metric and the number of folds. Model selection uses the one-standard-error rule, which prefers the simpler model when the gap is inside the noise. Below 4 exams, no backtest runs and the app uses transparent statistics only.<br><br>**Revised in the low-data upgrade:** there is no minimum number of exams for backtesting. The backtest is rolling-origin over every fold: with T papers, papers 2 to T are each predicted from the papers before them, which gives T − 1 folds (`models.min_train_exams` defaults to 1). One paper gives no fold, and the app says accuracy cannot be measured yet. Each report gives the number of folds, the fold-to-fold standard deviation, a 95% t-interval for the mean metric, and paired comparisons with random, frequency, recency-frequency and Bayesian recurrence. The one-standard-error rule no longer chooses the final ranking; it survives only in the question-type forecast (layer 2) and in `select_model`, a compatibility function the analysis no longer calls. The evidence-aware ensemble is the final ranking unless `best_single` beat it on the same held-out papers beyond a one-sided 95% t-bound of the paired differences (`ensemble.replace_confidence`, default 0.95). `best_single` is an out-of-sample selector: for each held-out paper it uses the method that did best on the papers before it, so its score is not inflated by picking the winner after seeing every fold. Tuned variants (decay, window, half-life, Bayesian recency discount) keep their default unless another variant is ahead on earlier folds beyond a one-sided 95% t-bound. |
| D5 | Using the **current** syllabus to backtest **old** exams leaks a small amount of future knowledge (the candidate set is defined with today's syllabus). | Unavoidable unless historical syllabi are supplied. The data model supports syllabus versions; when a historical version exists it is used for that period. The backtest report states which syllabus each fold used. |
| D6 | "Detect rotation cycles" and "examiner tendencies". Ten observations rarely support a periodicity claim, and examiner identity is almost never printed. | Rotation is tested against a permutation null (shuffle the appearance positions, compare gap regularity). The app reports a rotation only if `p < 0.10` with at least three gaps. Examiner is an optional metadata field used for descriptive tables only, never as a predictive feature. |
| D7 | "Generate plausible question formulations" vs "never introduce unsupported material". Free-form text generation cannot guarantee grounding. | Generation is template plus retrieval: templates come from the course's own historical question openers, slots are filled only with syllabus phrases, and a grounding checker rejects any output that contains a content word absent from the syllabus and the topic's historical questions. |
| D8 | "Numerical data must be physically consistent". Checking units and physics for arbitrary courses is not possible without formula knowledge. | The app does not invent numbers. A predicted numerical question reuses a historical numerical question's structure verbatim and says so. |
| D9 | OCR of equations and handwriting. Tesseract reads printed text well, equations poorly, and handwriting badly. | Equation-like and low-confidence regions are flagged for review, and every question text is editable. Dedicated math OCR models are out of scope for this version because of their size. |
| D10 | "Exam year" assumes one exam per year on the Gregorian calendar. Real data includes Bikram Sambat years (2076, 2079), regular and back exams in one year, and spring/fall sessions. | Time is an **exam sequence index**, not a calendar year. Each exam stores year, session, exam type and an editable `order_index`. Recency is measured in exam cycles. |
| D11 | XGBoost, LightGBM and CatBoost are listed as candidates. They add large native dependencies and solve the same problem as scikit-learn's histogram gradient boosting at this data size. | Gradient boosting uses `HistGradientBoostingClassifier`. The model registry accepts more models later, but none of these is required. |
| D12 | "High cosine similarity does not mean identical". | Exact-repeat detection requires both high embedding similarity and a high token-set string ratio. Paraphrase requires high similarity and the same syllabus topic. Thresholds live in configuration. |
| D13 | OR alternatives and "attempt any five". Does an optional question count as an appearance? | Yes for topic prediction: the examiner chose to put the topic on the paper. OR groups and optional instructions are stored, used by the structure model, and visible in the review screen. |

---

## 3. What needs a lot of data and what does not

*Revised in the low-data upgrade.* The original table listed, for each component, the
course size below which it was switched off. That cutoff no longer exists. Every component
runs at every course size unless an input it needs does not exist, and what changes with data
is how reliable it is and how much weight it gets in the final ranking.

Each ensemble component has a reliability prior:

```
reliability = (papers + pk) / (papers + pk + df)
```

`df` is the number of parameters the component estimates from this course, and `pk` is 1 for
components that bring outside knowledge (pretrained semantics, syllabus structure, the
cross-course model), otherwise 0. Semantic evidence gets it only when the pretrained model is
present. A paper without a year is excluded at ingestion; if you
include it anyway in Review, it counts as half a paper. Questions never count: a paper with
forty questions is one paper. The table gives the reliability prior for courses where every
paper has a year, before the input-quality factor described below.

| Component | Knowledge comes from | df | Reliability prior at 2 / 5 / 15 papers | Behaviour with few papers |
|---|---|---:|---|---|
| General ranking model | simulated courses (unless `models.use_simulated_prior` is false), plus other real courses in your library; UNAVAILABLE when neither exists | 0 | 1.00 / 1.00 / 1.00 | needs no course data; carries much of the weight before any fold exists |
| Semantic evidence (pretrained) | bundled WordLlama model + syllabus TF-IDF | 0.5 | 0.86 / 0.92 / 0.97 | runs from the first paper; topics with little history of their own lean on it more |
| Syllabus coverage | syllabus units, hours, marks or breadth | 0.5 | 0.86 / 0.92 / 0.97 | works with zero papers; unavailable only when the syllabus has a single unit and no hours, marks or sub-topics |
| Recency-frequency | this course | 1 | 0.67 / 0.83 / 0.94 | keeps its default decay until another is reliably better on earlier papers |
| Hierarchical Bayesian recurrence | this course, pooled at course, unit and topic level | 1 | 0.67 / 0.83 / 0.94 | mostly prior with one or two papers; credible intervals narrow as papers accumulate |
| Question-type fit | this course + syllabus format tags | 1 | 0.67 / 0.83 / 0.94 | unavailable only when past papers use one format and the syllabus has no format tags |
| Two-state Markov | this course | 2 | 0.50 / 0.71 / 0.88 | runs on every fold |
| Temporal recurrence | this course | 3 | 0.40 / 0.63 / 0.83 | uses the gap hazard only when its prequential log score beats the simple Bayesian rate; otherwise the simple rate |
| Co-occurrence | this course | 3 | 0.40 / 0.63 / 0.83 | unavailable until two consecutive earlier papers exist |
| Pooled hot/cold HMM | this course | 5 | 0.29 / 0.50 / 0.75 | smoothed EM, runs on short histories |
| Course-specific logistic | this course, pulled toward the general model | 22.5 | 0.08 / 0.18 / 0.40 | with zero training rows it equals the general model; with no general model its prior is centred on zero |
| Random forest, gradient boosting | this course | 40 | 0.05 / 0.11 / 0.27 | on a fold where every training label is the same, the general model's score is used and the fold is reported |

The weight of component m is `quality × reliability × exp(2 × skill) × g(x)`, normalised over
the components. Quality (0 to 1) is computed from papers before the cutoff: mapping confidence
for semantic evidence, formats seen for question-type fit, the syllabus weight source for
coverage, missing calendar years for the temporal components, and parse confidence and the
share of marks known for the course-learned models. Skill is how much better than the
average component it ranked earlier papers, in units of the fold spread, shrunk toward zero
by folds / (folds + 3). The per-topic factor g(x) gives outside-knowledge components more
weight on topics whose own history says little. A component is shown as
LIMITED while its reliability is below 0.5, DOWNWEIGHTED when it ranked earlier papers
clearly worse than the average component (skill below −0.25 and weight below half an equal
share), and UNAVAILABLE only when an input it needs does not exist.
On simulated courses the course logistic model gets 2 to 3% of the ensemble weight at 2 to 3
papers and 11% at 20.

Parts that are not ensemble components:

| Part | With few papers | With more papers |
|---|---|---|
| Ingestion, OCR, segmentation, syllabus parsing, alignment | work from the first file | unchanged |
| Time-aware backtest | T − 1 folds; with one fold accuracy cannot be measured, and with fewer than about five the confidence interval is wider than most differences between methods | more folds give a narrower interval, and skill on earlier papers starts to separate the components |
| Probability calibration | relative scores, rank intervals and evidence strength, because the standard error of the Brier gain is large | probabilities once the nested Brier gain over the base rate clears a one-sided 95% t-bound |
| Question-type forecast (layer 2) | all three forecast methods compete on every fold; the simplest within one standard error of the best is used | same rule, with more folds behind it |
| Rotation detection (descriptive) | a topic is tested only after four appearances (three gaps) | reported only when a permutation test gives p < 0.10 (D6) |

With a syllabus and no papers the analysis still runs. The ranking then comes from the
general model and the syllabus structure, every topic gets the rank range 1 to K with High
uncertainty, and no probability is shown.

There are no size thresholds to configure. The run-level label reads "Low-data advanced
inference" while the most reliable course-learned model (logistic, gradient boosting or
random forest) has reliability below 0.5, and "Advanced inference" after that. On a course
like the demo the logistic model estimates about 22.5 parameters, so the label changes at
about 25 papers. The label is wording only: it switches nothing on or off, and it never says
that inference is disabled. The status of every component and the reason for it are
written into the analysis report. Details: [Low-data inference](LOW_DATA_INFERENCE.md), sections 3, 5 and 10.

---

## 4. System architecture

Everything runs on the user's machine. The browser talks only to `127.0.0.1`.

```
 ┌────────────────────────── Browser (localhost UI, no CDN) ───────────────────────────┐
 │  Courses · Files · Review · Syllabus · Mapping · Analyze · Predictions · Analytics   │
 │  Models · Paper simulation · Search · Export · Settings          (vanilla JS + Plotly)│
 └───────────────────────────────────────┬──────────────────────────────────────────────┘
                                         │ JSON over HTTP (127.0.0.1 only)
 ┌───────────────────────────────────────┴──────────────────────────────────────────────┐
 │ app/        FastAPI routes, request schemas, background job runner                    │
 ├───────────────────────────────────────────────────────────────────────────────────────┤
 │ services    course · ingestion · analysis orchestration · search · export             │
 ├───────────────┬──────────────────┬───────────────────────┬────────────────────────────┤
 │ ingestion/    │ parsing/         │ syllabus/             │ embeddings/                │
 │ ocr/          │  metadata        │  parser, merge        │  backends (tfidf, hybrid,  │
 │ preprocessing/│  exam_parser     │  alignment (A/B/C/D)  │   pretrained, neural)      │
 │               │  question_types  │                       │  cache, vector index       │
 ├───────────────┴──────────────────┴───────────────────────┴────────────────────────────┤
 │ topic_modeling/ (question families, recurrence)   temporal/ (panel, hazard, rotation) │
 │ features/ (feature groups at a cutoff)            analysis/ (structure, coverage, co-occ)│
 ├───────────────────────────────────────────────────────────────────────────────────────┤
 │ inference/ (hierarchical Bayesian recurrence, evidence profile, general ranking       │
 │             model, model repository, rank intervals)                                  │
 │ models/ (baselines, ensemble components, course logistic, GBM, RF, HMM,               │
 │          evidence-aware ensemble, calibration)                                        │
 │ evaluation/ (metrics, rolling-origin backtest, final choice, ablation)                │
 │ prediction/ (final ranking, categories, explanations, why-not, type prediction)       │
 │ generation/ (grounded formulations, verification, paper simulation)                   │
 ├───────────────────────────────────────────────────────────────────────────────────────┤
 │ database/ SQLite (SQLAlchemy) + FTS5 search index + embedding cache                   │
 │ export/  CSV · XLSX · PDF        config/  TOML defaults + JSON user overrides         │
 └───────────────────────────────────────────────────────────────────────────────────────┘
```

Design rules:

* **Generic vs course-specific.** OCR, embeddings (including the bundled pretrained
  model), the question-type taxonomy and the parsers are generic and shared. The general
  ranking model is generic too: it is trained on simulated courses and updated with the
  scale-free feature rows of other real courses in your library, never with the course
  being predicted and never with synthetic courses such as the demo. With
  `models.use_simulated_prior = false` (a checkbox on the Settings page) it learns from other
  real courses alone, and with none in the library it is UNAVAILABLE. Everything else learned
  from history (Bayesian rates, hazards, course logistic weights, calibration) is fitted per
  course, per analysis run, and stored with that run. *Revised in the low-data upgrade:* the
  original rule said one course's history never enters another course's model. Now course A
  can change the general model used for course B, but course A's papers never enter course
  B's panel, frequencies, Bayesian posteriors or features (tested in
  `test_other_courses_do_not_change_this_courses_history`).
* **Exams, not questions, are the unit of temporal evidence.** Frequencies, recurrence,
  backtest folds and reliability count papers. Questions feed the language side
  (similarity, alignment, semantic evidence). A paper with forty questions is one paper;
  `test_questions_do_not_inflate_temporal_evidence` checks this.
* **Leakage is prevented by construction.** Feature functions receive a panel already
  truncated at the cutoff (`panel.until(t)`), so they cannot read exam `t` or later.
  A test perturbs future exams and asserts that earlier predictions do not change.
* **Every number has a source.** Questions keep file, page and line; topics keep the
  source document and page; predictions keep the questions and features behind them.
* **Manual edits win.** Any field a user edits is marked `user_edited` and is never
  overwritten by re-analysis.

---

## 5. ML pipeline

```
files ──► page text (native or OCR, with confidence)
      ──► exam metadata + question tree (section → question → sub-question, marks, OR groups)
      ──► question types (rule taxonomy, extensible JSON)
syllabus files ──► topic tree per document ──► merged canonical tree (with source refs)

questions × topics ──► alignment scores (semantic similarity, by default the pretrained + TF-IDF
                       hybrid, plus IDF keyword coverage; the feedback pass for a paper uses
                       only questions from earlier papers)
                   ──► status A clearly in / B probably in / C uncertain / D outside
                   ──► (A and validated B only) incidence panel Y[t, k] with one row per exam,
                       marks, types, semantic soft evidence (question level)

panel ──► features at each cutoff t (frequency, recency, gaps, hazard, Bayesian posterior,
          semantic, marks, types, co-occurrence, syllabus weight, repeat behaviour, data quality)
      ──► baselines and ensemble components, every one run on every fold (no data-size gate)
      ──► rolling-origin backtest: predictions for every target t from the second paper on,
          using exams < t only
      ──► nested tuning, evidence-aware ensemble weights and calibration (only earlier targets)
      ──► final choice: the ensemble, unless best_single (the method that did best on earlier
          papers, chosen afresh for each paper) beat it beyond a one-sided 95% t-bound
      ──► final ranking for exam T+1 with rank intervals, evidence strength and, where
          validated, calibrated probability bands
      ──► explanations (component contributions, Bayesian posterior, semantic evidence, why-not)
      ──► layer 2: question-type forecast, recurring question families, verified grounded formulations
      ──► optional predicted papers built from the discovered exam template
```

### Prediction units

* **Unit** (syllabus depth 1): used for coverage analysis and as a feature.
* **Topic** (depth 2, or depth 1 when the syllabus is flat): the primary prediction unit.
* **Concept** (deepest syllabus nodes): secondary layer and concept-level recall.
* **Question family** (cluster of near-identical questions across years): exact and
  paraphrased recurrence.

Questions are aligned to the finest syllabus node with good evidence and rolled up.

### Candidate models

*Revised in the low-data upgrade.* Baselines are kept for comparison (status REFERENCE).
Components are the members of the evidence-aware ensemble, which produces the final
ranking. No model has a minimum number of papers. `df` is the number of parameters a
component estimates from this course; it sets the reliability prior in section 3.

| Model | Role | Family | Learns | df | Complexity rank |
|---|---|---|---|---:|---|
| Random | baseline | baseline | nothing (expected value computed exactly) | | 0 |
| Frequency | baseline | baseline | count / exams | | 1 |
| Last exam | baseline | baseline | appeared in previous paper | | 1 |
| Window-N frequency | baseline | baseline | N tuned on earlier folds | | 2 |
| Recency-weighted (EWMA) | baseline | baseline | half-life tuned on earlier folds | | 2 |
| Linear decay | baseline | baseline | none | | 2 |
| Frequency + recency | baseline | baseline | rank average | | 2 |
| Best single method (`best_single`) | baseline, rival to the ensemble | selection | for each held-out paper, the method with the best mean score on the papers before it | | 4 |
| Recency-frequency | component | recency | decay strategy (none, linear, last-N window or half-life) chosen on earlier folds | 1 | 2 |
| Hierarchical Bayesian recurrence | component | Bayesian | beta-binomial pooled at course, unit and topic level; prior strength by empirical Bayes; recency discount chosen on earlier folds | 1 | 3 |
| Semantic evidence (pretrained) | component | semantic | similarity-weighted soft counts from past in-syllabus questions | 0.5 | 3 |
| Syllabus coverage | component | structure | unit-level Bayesian rate × square root of the topic's relative share of its unit (syllabus hours, marks or breadth) | 0.5 | 2 |
| Question-type fit | component | structure | topic's usual formats against the recent paper mix | 1 | 2 |
| Topic co-occurrence | component | structure | smoothed lift of topics following the previous paper's topics | 3 | 3 |
| Two-state Markov (empirical Bayes) | component | temporal | transition rates shrunk to pooled | 2 | 3 |
| Temporal recurrence | component | survival | pooled gap hazard, used only when its prequential log score beats the simple Bayesian rate | 3 | 3 |
| Pooled hot/cold HMM | component | temporal | two shared hidden states, smoothed EM | 5 | 6 |
| General ranking model | component | transfer | logistic on 22 scale-free features, trained on 400 simulated courses (optional, `models.use_simulated_prior`) and updated with other real courses | 0 | 2 |
| Course-specific logistic | component | ML | 22 scale-free and 23 course-only features (including optional rate, semantic density, syllabus centrality, mapping and parse confidence, marks known); Gaussian prior centred on the general model | 22.5 | 4 |
| Random forest | component | ML | bagged trees on all features | 40 | 6 |
| Gradient boosting | component | ML | histogram gradient boosting on all features | 40 | 6 |
| Evidence-aware ensemble | final ranking | ensemble | weights from input quality × reliability × exp(2 × skill on earlier folds) × per-topic gate | 0 | 5 |

The original list had a "Semantic soft recurrence" model, a plain "Pooled hazard", an L2
logistic regression fitted on the course's rows alone, data-size gates on the hazard, the
logistic model, the trees and the HMM, and an ensemble that averaged the percentile ranks
of the best earlier members. The rows above replace them. Formulas and settings are in
[Low-data inference](LOW_DATA_INFERENCE.md#4-components).

---

## 6. Database schema (SQLite)

*Revised in the low-data upgrade.* The original table used a `_json` suffix for JSON
columns and a few planned names that the code never used. The names below match
`predictor/database/models.py`. Columns and tables in **bold** were added in schema
version 2.

| Table | Key columns |
|---|---|
| `course` | id, name, code, description, created_at, settings, **is_synthetic** (synthetic courses, such as the demo, never train cross-course models) |
| `source_file` | id, course_id, kind (`exam` / `syllabus`), filename, sha256, mime, size_bytes, stored_path, status, error, page_count, extraction_summary, syllabus_version_id, uploaded_at |
| `document_page` | id, file_id, page_no, text, method (`native` / `ocr` / `docx` / `text`), ocr_confidence, quality_flags, details |
| `syllabus_version` | id, course_id, label, is_current, effective_from_order, created_at |
| `course_topic` | id, course_id, syllabus_version_id, parent_id, depth, number, title, description, concepts, objectives, hours, marks_weight, kinds, aliases, source_refs, order_no, excluded, user_edited |
| `exam` | id, course_id, source_file_id, title, subject, year, calendar, session, exam_type, exam_date, order_index, full_marks, pass_marks, duration, examiner, instructions, structure, metadata_confidence, include_in_analysis, exclusion_reason, duplicate_of_id, user_edited_fields, **source** (`upload` / `demo` / `manual`), created_at |
| `exam_section` | id, exam_id, label, title, instructions, attempt_count, order_no |
| `exam_question` | id, exam_id, section_id, parent_id, label, path_label, depth, text, raw_text, normalized_text, context_text, marks, marks_source, or_group, is_optional, is_leaf, order_no, page_no, line_no, question_types, type_scores, type_user_edited, options, equations, quality_flags, parse_confidence, needs_review, user_edited |
| `question_topic_mapping` | id, question_id, topic_id, rank, confidence, status (A/B/C/D), semantic_similarity, keyword_overlap, unknown_term_ratio, matched_terms, evidence_text, evidence, method (`auto` / `manual`) |
| `embedding_cache` | key (sha256 of backend name + text), backend, dim, vector blob |
| `analysis_run` | id, course_id, status, progress, message, started_at, finished_at, config, data_fingerprint, summary, **engine_version** (`2.0` for the low-data engine) |
| `analysis_artifact` | id, run_id, key, data (charts, structure, coverage, ablation, calibration, evidence profile, model table, and more) |
| `model_result` | id, run_id, layer, model_name, display_name, family, complexity, enabled, gate_reason, selected, metrics, metric_se, notes, **role** (`baseline` / `component` / `ensemble`), **scope** (`course` / `syllabus` / `global`), **status** (ACTIVE / LIMITED / DOWNWEIGHTED / UNAVAILABLE / REFERENCE), **weight**, **reliability**, **evidence** (status reason, df, skill, fold spread, 95% interval) |
| `backtest_fold` | id, run_id, layer, model_name, target_exam_id, target_index, target_label, n_train_exams, metrics |
| `prediction` | id, run_id, layer, topic_id, item_key, label, rank, score, probability, prob_low, prob_high, calibrated, category, confidence, features, contributions, evidence, why_not, **evidence_strength** (Strong / Moderate / Limited / Minimal), **uncertainty** (rank interval and level) |
| `predicted_question` | id, run_id, topic_id, text, question_type, marks_low, marks_high, basis, rank, evidence_question_ids, grounding |
| **`model_registry`** | id, scope (`global` / `course`), course_id, run_id, name, version, source (`simulation` / `simulation+repository` / `repository` / `none` for scope `global`, `course` for scope `course`), fingerprint, params, training, created_at |
| **`course_feature_set`** | id, course_id, run_id, layer, is_synthetic, feature_names, n_rows, n_exams, X, y, created_at; one row per course and layer, replaced on each analysis; scale-free features only, no topic names or raw counts |
| **`schema_version`** | id, version, applied_at, note |
| `search_index` (FTS5) | kind, ref_id, course_id, text |

`model_result.enabled` and `model_result.gate_reason` are kept from version 1, because
migrations never drop columns. No model is gated any more, so new runs store
`enabled = true` and an empty reason; the component status and its reason take their place.

`model_registry` separates the two kinds of learned knowledge. Scope `global` holds one row
per distinct general ranking model ever used, keyed by a fingerprint of the prior source and
the other courses' feature sets it learned from; each analysis run records that fingerprint
in its `config`, so you can tell which general model a run used. Scope `course` holds the
course-specific logistic model's coefficients for a run. `course_feature_set` is the
only thing other courses learn from. Rows of synthetic courses are stored flagged and never
used for cross-course training.

An existing database is upgraded in place when it is opened: SQLAlchemy's `create_all`
creates the missing tables, then `database/migrations.py` adds the new columns with
`ALTER TABLE ... ADD COLUMN` and a default and records the version in `schema_version`.
Nothing is dropped or rewritten. `tests/fixtures/schema_v1.sql` is the version 1 schema
used by the migration test.

Mapping to the names in the specification: `CourseContentDocument` = `source_file` with
`kind='syllabus'`; `Topic` and `CourseTopic` = `course_topic`; `QuestionSegment` = the
`exam_question` tree; `Model` = `model_result` (per run) and `model_registry` (fitted
parameters); `BacktestRun` = `analysis_run`; `BacktestResult` = `backtest_fold`;
`PredictionEvidence` = `prediction.evidence`.

---

## 7. UI architecture

A single-page app served by the same local FastAPI process. No build step and no CDN:
plain ES modules, one CSS file, and Plotly served from the installed `plotly` Python
package. Hash routes:

* `#/` course list and creation.
* `#/course/:id/files` drag-and-drop upload for papers and course content, with duplicate
  and processing status.
* `#/course/:id/review` exam metadata table (year, session, type, order, include) and a
  question tree editor (edit text, marks, type, split, merge, delete; filter "needs review").
* `#/course/:id/syllabus` topic tree editor (rename, move, add aliases, hours, exclude).
* `#/course/:id/mapping` question to topic mappings with A/B/C/D status, evidence, override.
* `#/course/:id/predict` the **Analyze & Predict** button, progress, an inference status
  panel (mode such as "Low-data advanced inference", evidence quality, prediction
  uncertainty, and a table of components with status, weight and reason), and the ranked
  dashboard grouped by priority. Each topic shows its historical coverage (papers and
  questions), rank range and evidence strength. Clicking a topic opens the evidence panel:
  why it ranked, model contributions, the Bayesian posterior, the temporal pattern used,
  semantically related past questions, source questions, syllabus location, verified
  formulations, and why-not notes for low-ranked topics. (Revised in the low-data upgrade;
  the original page showed data-sufficiency notes.)
* `#/course/:id/analytics` charts; clicking a point lists the underlying questions.
* `#/course/:id/models` backtest table, per-fold chart, ablation, calibration, selection
  rationale, the evidence profile, validation on held-out papers (folds, spread, comparison
  with baselines) and the ensemble components with their reliability.
* `#/course/:id/paper` predicted paper simulation.
* `#/course/:id/search` keyword and semantic search.
* `#/settings` thresholds, decay, strictness, OCR, embedding backend, and the checkbox "Use
  the simulated cross-course prior" (`models.use_simulated_prior`).

---

## 8. Development phases

Each phase ends with passing tests and a runnable command.

| Phase | Deliverable | Runnable check |
|---|---|---|
| 1 | Package layout, config, logging, database | `predictor doctor` |
| 2 | File ingestion, hashing, duplicate files | ingest a PDF/DOCX/TXT into SQLite |
| 3 | Native text + OCR with preprocessing and confidence | OCR an image page |
| 4 | Exam metadata and question segmentation | parsed question tree for fixtures |
| 5 | Syllabus parsing and merging | topic tree with source refs |
| 6 | Embeddings and syllabus alignment (A/B/C/D) | mapping table with evidence |
| 7 | Question families and recurrence | exact/paraphrase/concept/topic counts |
| 8 | Temporal panel, gaps, hazard, rotation test | per-topic timelines |
| 9 | Feature builder with leakage guard | leakage test passes |
| 10 | Baseline models | ranked list from frequency/recency |
| 11 | Backtest engine and metrics | model comparison table |
| 12 | Bayesian, temporal, ML models, ensemble, calibration, ablation | selection report |
| 13 | Type forecast and grounded question generation, paper simulation | formulations with evidence |
| 14 | API, UI, charts, search, export | full workflow in the browser |
| 15 | Tests: unit, integration, regression; performance caching | `pytest` green |
| 16 | Install scripts, user and developer docs, demo course | one-command install |

---

## 9. Local models and hardware

Target machine: 4 CPU cores, 8 GB RAM, no GPU.

| Need | Default | Optional | Cost |
|---|---|---|---|
| OCR | Tesseract 5 (LSTM engine) via `pytesseract`, OpenCV preprocessing | extra language packs | about 1 to 3 s per page at 300 dpi |
| Embeddings | Hybrid (revised in the low-data upgrade): WordLlama static embeddings (256-d, about 16 MB of weights inside the `wordllama` pip package, no download) at weight 0.3, plus TF-IDF (word stems 1-2 grams + char 3-5 grams) fitted on the syllabus at weight 0.7. TF-IDF alone if the pretrained model is unavailable | `BAAI/bge-small-en-v1.5` (33M params, 384-d, about 130 MB) or `all-MiniLM-L6-v2` (22M, about 90 MB) through sentence-transformers; with the default `auto` setting a downloaded model is used | TF-IDF: milliseconds. WordLlama: a token lookup and an average per text. MiniLM: about 600 questions in 15 to 30 s on CPU, cached afterwards |
| Vector search | NumPy cosine over a dense matrix | FAISS if installed | a course has a few thousand vectors; brute force takes under 10 ms |
| Learning | scikit-learn, SciPy | none | seconds |

Why the TF-IDF default was defensible: syllabus alignment is mostly a matter of shared
technical vocabulary ("Bernoulli", "Rankine cycle", "fugacity"), which stems and
character n-grams capture, and character n-grams also absorb OCR errors. A neural model
helps most with paraphrases that share no words. The backtest and the alignment review
screen let you compare both on your own course.

*Revised in the low-data upgrade:* TF-IDF stays inside the default, because the bundled
pretrained model does not beat it for alignment. On the demo course WordLlama alone put 118
of 140 questions on the right topic and marked only 1 of 4 old-syllabus questions as
outside (measured before the feedback pass became causal). The hybrid keeps TF-IDF's
separation (131 of 140 with causal feedback, all 4 old-syllabus questions outside) and adds
general language knowledge for semantic evidence, similarity between topics, and checks on
generated questions. See [Low-data inference](LOW_DATA_INFERENCE.md#alignment-backends).

Fitting the alignment TF-IDF on the **syllabus only** keeps exam text out of the
representation, which removes a subtle leakage path (document frequencies computed from
future exams). The pretrained model fits nothing on exam text either. The alignment
feedback pass, which moves topic vectors toward confidently mapped questions, is causal:
when it aligns a paper it uses only questions from earlier papers (by exam order), so a
later paper's wording cannot change how an earlier paper is mapped. Before this fix the
held-out paper's own wording fed its alignment, and the demo's backtest scores against
mapped labels were optimistic by about 0.035 (each fell by 0.02 to 0.05 once feedback became
causal; status counts moved from A 109 / B 27 / C 4 / D 4 to A 104 / B 30 / C 6 / D 4).

---

## 10. Deterministic and statistical vs learned

| Part | Approach | Reason |
|---|---|---|
| File hashing, duplicates, page extraction | deterministic | exact by nature |
| OCR | pretrained model (Tesseract) | generic, no course data needed |
| Question segmentation, marks, OR groups, metadata | deterministic rules + confidence + manual correction | no labelled data; rules are inspectable and fixable |
| Question type | rule taxonomy (JSON, extensible) | no labelled data; user corrections stored for a future learned classifier |
| Syllabus parsing and merge | deterministic + fuzzy matching | inspectable |
| Alignment | hybrid of the bundled pretrained model and TF-IDF (or a downloaded sentence-transformer) + IDF keyword coverage + thresholds; feedback from earlier papers only | similarity is learned elsewhere; thresholds are explicit |
| Families (exact / paraphrase) | similarity + string ratio thresholds | explicit and testable |
| Frequency, recency, gaps, co-occurrence, structure | statistics | small data |
| Bayesian recurrence, Markov, hazard | hierarchical beta-binomial and survival statistics with shrinkage; the gap hazard is used only when its prequential log score beats the simple rate | small data, pooled |
| General ranking model | logistic regression trained on simulated courses, updated with other real courses (other real courses alone when `models.use_simulated_prior` is false) | needs no data from the course being predicted |
| Course logistic, RF, GBM, HMM | learned on every fold; weight = reliability prior × measured skill on earlier papers (revised in the low-data upgrade; originally gated and required to win the backtest) | present at every size, influence grows with evidence |
| Calibration | Platt scaling or isotonic regression on out-of-sample predictions, chosen by nested log loss | probabilities shown only when the Brier gain over the base rate clears a one-sided 95% t-bound |
| Ensemble weights | input quality × reliability × exp(2 × skill on earlier folds) × per-topic gate (revised; originally a rank average of members chosen on earlier folds) | no weights fitted by optimisation; skill is shrunk toward zero when there are few folds |
| Question formulations | templates + retrieval + grounding check, then topic, semantic and question-type verification | guarantees no unsupported content |

---

## 11. Backtesting framework

Exams are sorted by `order_index`: `e_0, e_1, ..., e_{T-1}`. The next, unseen exam is
`e_T`.

*Revised in the low-data upgrade:* the framework is rolling-origin over every fold, there is
no minimum number of exams, and the evidence-aware ensemble replaced one-SE selection as the
main rule (see D4).

1. **Targets.** For every `t` from `min_train_exams` (default 1) to `T-1`, every model sees
   exams `e_0..e_{t-1}` and predicts the topic set of `e_t`. With T papers that is T − 1
   folds; one paper gives none, and with no paper the analysis still runs on prior
   knowledge (section 3). The final prediction uses the same code with `t = T`.
2. **Feature cutoff.** Features for target `t` are computed from `panel.until(t)`, which
   physically removes rows `t` and later.
3. **Training rows for learned models.** For target `t`, training rows are
   `(features(panel.until(s)), Y[s])` for every `s < t` with at least `min_history`
   (default 1) exams before `s`. Labels from `t` or later are never used. The general
   ranking model never trains on the course being predicted.
4. **Nested tuning.** Hyperparameters (decay strategy, window length, half-life, Bayesian
   recency discount), ensemble weights and calibration for target `t` are chosen using only
   the backtest results of targets `s < t`. A tuned model keeps its default unless another
   variant is ahead beyond a one-sided 95% t-bound of the paired differences. Meta models
   (tuned variants, ensemble skill, `best_single`) score earlier folds with the K known at
   that time (the median topics per paper over that fold's paper and the papers before it,
   all before the cutoff), so later papers cannot shift K
   for earlier folds (`test_later_papers_never_change_earlier_predictions`). Because each
   stored prediction at `s` was itself computed from exams before `s`, reusing it is safe.
5. **Metrics per fold**, then mean, standard error, fold-to-fold standard deviation and a
   95% t-interval across folds.
6. **Final choice.** The evidence-aware ensemble is the final ranking unless `best_single`
   beat it on the same held-out papers beyond a one-sided 95% t-bound of the paired
   differences (`ensemble.replace_confidence`, default 0.95). `best_single` uses, for each
   held-out paper, the method that did best on the papers before it, so its scores are out of
   sample; picking the winner after seeing every fold would favour whichever method got lucky.
   If it does replace the ensemble, the report says so. With no fold, the ensemble runs on its
   reliability priors alone.
7. **Reported baselines.** Random (exact expectation), most frequent, most recent,
   recency-weighted, and the final choice, always side by side. The validation summary adds
   paired differences against random, frequency, recency-frequency and Bayesian recurrence,
   with the number of papers on which the final ranking was better and worse.
8. **Leakage audit.** Every exam from the audit fold onward is scrambled and its outcomes
   flipped, every model (including tuned variants and the ensemble) is rerun, and the
   predictions for the audit fold must not change. The current syllabus is the candidate
   set for every fold (D5).
9. **Ablation.** On the same folds, staged ablation (frequency, + recency, + Bayesian
   smoothing, + semantic, + topic structure, + temporal dynamics, full ensemble) and
   leave-one-component-out. Each stage is the evidence-aware ensemble restricted to the
   listed components plus the earlier stages' members; baselines such as frequency stay in.
   Each change carries its paired standard error and is called reliable only when it falls
   outside a two-sided 95% t-interval of the paired per-paper differences. The note lists
   reliable gains and reliable losses separately.

Primary metric: NDCG@K, where K is configurable and defaults to the median number of
distinct topics per past exam (bounded to 3..15). Also reported: Precision@K, Recall@K,
Hit Rate@K, MRR, Top-1/3/5/10 hit, concept-level recall, exact-question (family) recall,
Brier score and expected calibration error.

---

## 12. Success metrics

| Area | Target |
|---|---|
| Segmentation on digital PDFs and DOCX in the test fixtures | 95% of question boundaries and marks correct |
| Year detection on fixtures | 95% correct, and the rest flagged for review |
| Syllabus alignment on labelled fixtures | 85% top-1 topic accuracy; out-of-syllabus questions never mapped to status A |
| Prediction | the selected model's NDCG@K is at least the best simple baseline's; if no learned model beats the baselines by more than one standard error, a baseline is selected and the report says so. *Revised in the low-data upgrade:* the final ranking is reported next to random, frequency, recency-frequency and Bayesian recurrence on the same folds, with paired differences, standard errors and a 95% interval; no gain inside that interval is claimed, and `best_single` replaces the ensemble only if it beat the ensemble beyond a one-sided 95% t-bound of the paired differences |
| Calibration | probabilities shown only when nested Brier score beats the base-rate Brier (revised: the per-paper Brier gain must clear a one-sided 95% t-bound) |
| Leakage | the perturbation test passes for every model |
| Syllabus boundary | zero predicted topics or formulations outside the current syllabus |
| Performance | a 15-paper course analyses in under 60 s on a 4-core laptop with the default backend; unchanged documents are never reprocessed |

---

## 13. Minimum viable version vs advanced version

**Minimum viable version** (the first fully working release, this repository):

* Ingestion of PDF (digital and scanned), DOCX, images and text; OCR with preprocessing.
* Metadata, segmentation, OR groups, marks, question types, quality flags, manual edits.
* Syllabus tree with merge, source references, versions; alignment with A/B/C/D status.
* Families and recurrence, temporal panel, features with leakage guard.
* Baselines, Bayesian, Markov, hazard, semantic, logistic regression, gated RF/GBM/HMM,
  ensemble, calibration, ablation, one-SE selection. (Revised in the low-data upgrade: the
  gates and one-SE selection as the main rule were replaced by the evidence-aware ensemble,
  see D2 and D4.)
* Ranked predictions with categories, evidence, contributions and why-not.
* Type forecast, recurring families, grounded formulations, paper simulation.
* Local web UI, charts, search, CSV/XLSX/PDF export, tests, install scripts, docs.

**Advanced version** (future work, designed for but not required):

* Neural embedding backend switched on by default after download (supported now, opt-in).
  Done in the low-data upgrade: with the default `auto` setting a downloaded
  sentence-transformer is used, and a bundled pretrained model runs without any download.
* A learned question-type classifier trained on user corrections across courses.
* Math OCR for equations; layout analysis models for two-column papers.
* Optional local LLM (for example through Ollama) for paraphrasing formulations, always
  passed through the grounding checker.
* Cross-course transfer for generic components only. Partly done in the low-data upgrade:
  the general ranking model learns from the scale-free feature rows of other real courses,
  while each course's own statistics stay isolated.
* Packaged desktop installer (PyInstaller) and a native window wrapper.

---

## 14. Implementation plan

The phases in section 8 are implemented in order. Each phase lands with its unit tests.
Integration and regression tests use a synthetic course generated from a known random
process (`predictor/demo/generator.py`), so tests can check that models recover planted
patterns (for example a topic that rotates every two exams) and that an out-of-syllabus
topic from an old syllabus never reaches the predictions.

The low-data upgrade added `tests/unit/test_low_data.py`. It runs courses of 0 to 25
papers and checks that every component runs at every size, that semantic and
Bayesian components are active with 2 to 5 papers, that course-learned models are
downweighted when uncertain and gain weight with data, that questions do not inflate
temporal evidence, that no model (including meta models) leaks future papers, that later
papers never change earlier predictions, that zero papers still give a ranking, that the
general model is reported unavailable without the simulated prior, that the message never
says inference is disabled, and that results are reproducible.
`tests/unit/test_repository_and_migrations.py` checks the schema migration, that the
simulated prior can be turned off, and that other courses never change a course's own
history. `tests/integration/test_end_to_end.py` runs a course with a syllabus and no papers
end to end.

---

## 15. Implementation status (as built)

| Spec area | Where | Status |
|---|---|---|
| Ingestion: PDF (digital, scanned, mixed), DOCX, images, text; hashing; duplicates (file and paper level) | `ingestion/`, `services/ingest_service.py` | done |
| OCR with deskew, denoise, contrast, binarisation, rotation detection, confidence | `ocr/` | done |
| Quality flags (low confidence, garbled symbols, implausible words, broken equations) | `preprocessing/quality.py` | done |
| Equations kept raw and normalised | `preprocessing/textnorm.py` | done (text equations; no math OCR) |
| Question tree: sections, numbering styles, marks, OR groups, optional questions, MCQ, inline parts | `parsing/exam_parser.py` | done |
| Metadata incl. Bikram Sambat years, sessions, exam types | `parsing/metadata.py` | done |
| Extensible question-type taxonomy | `parsing/question_types.py`, `config/taxonomy.json` | done (rule-based) |
| Syllabus tree, merge, source references, versions | `syllabus/` | done |
| Alignment with A/B/C/D and strictness | `syllabus/alignment.py` | done |
| Exact / paraphrase / concept / topic recurrence | `topic_modeling/families.py`, panel features | done |
| Temporal analysis, hazard (used only when justified by prequential log score), Markov, rotation test, co-occurrence, marks, format transitions | `temporal/`, `features/`, `models/components.py`, `prediction/type_forecast.py` | done |
| Pretrained semantic layer (bundled WordLlama, hybrid alignment, semantic soft evidence) | `embeddings/pretrained.py`, `embeddings/backends.py` | done |
| Hierarchical Bayesian recurrence (posterior mean, median, credible interval, effective sample size, prior contribution) | `inference/bayes.py` | done |
| Evidence profile at the exam level, component status, low-data message | `inference/evidence.py` | done |
| General ranking model and local model repository | `inference/generic.py`, `inference/generic_prior.json`, `inference/repository.py` | done |
| Baselines, ensemble components, tuned variants, evidence-aware ensemble (no data-size gates) | `models/` | done |
| Rolling-origin backtest over every fold, validation summary, final choice, leakage audit | `evaluation/backtest.py` | done |
| Platt and isotonic calibration with a nested significance check, probability bands | `models/calibration.py` | done |
| Rank intervals, uncertainty level, evidence strength | `inference/uncertainty.py`, `prediction/ranking.py` | done |
| Ablation (leave-one-out, staged) and syllabus-filter check | `evaluation/ablation.py` | done |
| Ranking, categories, evidence, contributions, why-not, excluded | `prediction/ranking.py` | done |
| Grounded formulations and paper simulation | `generation/` | done |
| Verification of generated formulations (syllabus, topic, semantic, question type) | `generation/verify.py` | done |
| Schema migrations for existing databases, synthetic-data flags | `database/migrations.py`, `database/models.py` | done |
| UI, charts with click-through, search, export, settings | `app/`, `ui/`, `search/`, `export/` | done |
| Install scripts, CLI, docs, tests | `scripts/`, `cli.py`, `docs/`, `tests/` | done |
| Examiner tendencies | `exam.examiner` field | stored and editable through the API only; no analysis, by design (D6) |
| Historical syllabus per backtest fold | `SyllabusVersion` | partial: historical versions explain excluded questions; the current syllabus is the candidate set for every fold (D5) |
| Learned question-type classifier from user corrections | none | future work |
| Cross-course transfer | `inference/repository.py` | partial: the general ranking model learns from scale-free feature rows of other real courses; each course's panel, frequencies, posteriors and features stay isolated |
| Math OCR, handwriting recognition | none | future work (D9) |

### Measured against the success metrics (synthetic demo, 12 papers)

*Revised in the low-data upgrade.* These numbers come from the upgraded engine with default
settings (hybrid alignment with causal feedback, evidence-aware ensemble). The full results,
including simulated courses with known true topics, the staged ablation and the alignment
backends, are in
[Low-data inference, section 8](LOW_DATA_INFERENCE.md#8-results).

| Target (section 12) | Result |
|---|---|
| Segmentation on digital PDFs and DOCX | 144 of 144 demo questions found with correct marks across PDF, DOCX and text papers; fixture tests cover TU, KU and generic formats |
| Year detection | 12 of 12 demo papers; fixture tests cover AD, BS and file-name years |
| Alignment top-1 accuracy at least 85%, out-of-syllabus never A | 93.6% (131 of 140) with the default hybrid backend and causal feedback; all 9 misses have status B; 4 of 4 old-syllabus questions marked D; 0 in-syllabus questions marked D; status counts A 104 / B 30 / C 6 / D 4 |
| Final ranking compared with the simple baselines | against the app's own mapped labels on 11 held-out papers: evidence-aware ensemble NDCG@11 0.717 ± 0.027, recent-window frequency 0.687, most frequent topics 0.678, random 0.456; ensemble minus frequency +0.039 ± 0.019, better on 9 of 11 papers, which is inside a two-sided 95% t-interval (±0.042), so it is not a reliable gain. The ensemble was kept: `best_single` scored 0.707 (difference +0.010 ± 0.018). Against the generator's planted true topics, which a real course cannot measure, the ensemble does **not** beat plain frequency: 0.738 against 0.734, difference +0.004 ± 0.017 |
| Probabilities shown only when they beat the base rate | isotonic chosen over Platt by nested log loss; nested Brier 0.191 vs 0.248 base rate on 10 later papers (gain 0.057 ± 0.005, skill 23%), beyond the one-sided 95% bound, so probabilities are shown; expected calibration error 0.042 |
| Leakage audit passes | passes for 34 models, including tuned variants and the ensemble |
| No prediction or formulation outside the syllabus | asserted in `tests/integration/test_end_to_end.py`; 53 generated formulations checked, 2 rejected by verification (both reused past numericals filed under the wrong topic) |
| A 15-paper course in under 60 s | 12 papers analysed in about 9 s on 4 cores (about 7 s before the upgrade), mostly tree models refitted on every fold; ingestion under 1 s for digital files |

Part of the gain against mapped labels comes from the semantic component, which uses the
same alignment scores that produce the labels. Gains over frequency are shown mainly on
simulated courses in in-sample rolling-origin backtests: with planted patterns, the ensemble
beat frequency by 0.03 to 0.08 NDCG (2 to 20 papers, 8 courses per size), and the general
model alone was as good as or slightly better than the ensemble on those generators. On the
next paper only (fit on T papers, score paper T + 1, 40 courses per size) the gain is small:
between -0.008 ± 0.012 and +0.032 ± 0.012, and clearly positive only at T = 5. Simulated courses
are only as realistic as their generators.
