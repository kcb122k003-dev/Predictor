# Architecture and Design Analysis

This document answers the "first task" of the specification (section 73) before any
substantial code was written. It records the analysis, the decisions that came out of it,
and the plan the code follows. Later sections of the code reference these decisions by
number (for example `D3`).

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
| D1 | "Works fully offline" vs "modern neural embeddings". Pretrained models must be downloaded once, and the download host may be unreachable. | The default embedding backend is a TF-IDF model over word stems plus character n-grams, fitted locally. It needs no download. A neural backend (sentence-transformers) is optional and is enabled only after an explicit `predictor models download` command. The app never downloads anything on its own. |
| D2 | "Use Markov, HMM, survival, gradient boosting, calibration" vs "you will often have 5 to 15 exams". A single topic with 10 exams has 10 binary observations. No per-topic model can be fitted on that. | Every learned model is **pooled across topics**: one row per (exam, topic). Ten exams and 40 topics give about 280 training rows after warm-up. Per-topic parameters use empirical-Bayes shrinkage toward the pooled estimate. Complex models are gated by data-sufficiency rules and must also win the backtest. |
| D3 | "Calibrated probabilities" vs tiny validation sets. Isotonic regression on 200 points overfits. | Platt scaling (two parameters) on the percentile rank of each topic, fitted only on out-of-sample backtest predictions. Probabilities are shown only when nested calibration beats the base-rate Brier score. Otherwise the UI shows "relative score" and says why. |
| D4 | Backtests on very few exams. With 5 exams and 3 warm-up exams you get 2 folds; a metric averaged over 2 folds is close to noise. | The backtest reports the standard error for every metric and the number of folds. Model selection uses the one-standard-error rule, which prefers the simpler model when the gap is inside the noise. Below 4 exams, no backtest runs and the app uses transparent statistics only. |
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

| Component | Works with 3 to 5 exams | 6 to 9 exams | 10 to 19 exams | 20+ exams |
|---|---|---|---|---|
| Ingestion, OCR, segmentation, syllabus parsing | yes | yes | yes | yes |
| Syllabus alignment (pretrained or TF-IDF similarity) | yes | yes | yes | yes |
| Frequency, recency, windowed counts | yes, descriptive only | yes | yes | yes |
| Time-aware backtest | no (needs at least 4) | 2 to 5 folds, wide error bars | usable | good |
| Empirical-Bayes rate and Markov models | descriptive | yes | yes | yes |
| Pooled hazard (time since last appearance) | no | weak | yes | yes |
| Regularised logistic regression | no | if at least 5 training exams and 30 positive rows | yes | yes |
| Random forest, gradient boosting | no | no | only with about 400+ rows and 10+ exams | yes |
| Pooled two-state HMM | no | no | no (below 12 exams) | candidate |
| Probability calibration | no | Platt, if it beats base rate | Platt | Platt, maybe isotonic |
| Rotation detection | no | rarely significant | sometimes | yes |
| Question-type transition model | no | weak | yes | yes |

The thresholds are configuration values (`models.sufficiency`), and every rejection is
written into the analysis report in plain language.

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
 │ ocr/          │  metadata        │  parser, merge        │  backends (tfidf, neural)  │
 │ preprocessing/│  exam_parser     │  alignment (A/B/C/D)  │  cache, vector index       │
 │               │  question_types  │                       │                            │
 ├───────────────┴──────────────────┴───────────────────────┴────────────────────────────┤
 │ topic_modeling/ (question families, recurrence)   temporal/ (panel, hazard, rotation) │
 │ features/ (feature groups at a cutoff)            analysis/ (structure, coverage, co-occ)│
 ├───────────────────────────────────────────────────────────────────────────────────────┤
 │ models/ (baselines, Bayesian, Markov, hazard, logistic, GBM, RF, HMM, ensemble,       │
 │          sufficiency gates, calibration)                                              │
 │ evaluation/ (metrics, expanding-window backtest, nested selection, ablation)          │
 │ prediction/ (final ranking, categories, explanations, why-not, type prediction)       │
 │ generation/ (grounded question formulations, paper simulation)                        │
 ├───────────────────────────────────────────────────────────────────────────────────────┤
 │ database/ SQLite (SQLAlchemy) + FTS5 search index + embedding cache                   │
 │ export/  CSV · XLSX · PDF        config/  TOML defaults + JSON user overrides         │
 └───────────────────────────────────────────────────────────────────────────────────────┘
```

Design rules:

* **Generic vs course-specific.** OCR, embeddings, the question-type taxonomy and the
  parsers are generic and shared. Everything learned from history (rates, hazards,
  logistic weights, calibration) is fitted per course, per analysis run, and stored with
  that run. One course's history never enters another course's model.
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

questions × topics ──► alignment scores (semantic + IDF keyword coverage)
                   ──► status A clearly in / B probably in / C uncertain / D outside
                   ──► (A and validated B only) incidence panel Y[t, k], marks, types, soft similarity

panel ──► features at each cutoff t (frequency, recency, gaps, hazard, semantic, marks,
          types, co-occurrence, syllabus weight, repeat behaviour)
      ──► candidate models (gated by data sufficiency)
      ──► expanding-window backtest: predictions for every target t, using exams < t only
      ──► nested tuning, ensemble construction and calibration (only earlier targets)
      ──► one-standard-error model selection
      ──► final ranking for exam T+1, calibrated probability bands where valid
      ──► explanations (additive contributions, evidence bullets, why-not)
      ──► layer 2: question-type forecast, recurring question families, grounded formulations
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

| Model | Family | Learns | Complexity rank |
|---|---|---|---|
| Random | baseline | nothing (expected value computed exactly) | 0 |
| Frequency | baseline | count / exams | 1 |
| Last exam | baseline | appeared in previous paper | 1 |
| Window-N frequency | baseline | N tuned on earlier folds | 2 |
| Recency-weighted (EWMA) | baseline | half-life tuned on earlier folds | 2 |
| Linear decay | baseline | none | 2 |
| Frequency + recency | baseline | rank average | 2 |
| Beta-binomial (empirical Bayes) | Bayesian | pooled prior + recency-weighted counts | 3 |
| Two-state Markov (empirical Bayes) | temporal | transition rates shrunk to pooled | 3 |
| Pooled hazard | survival | P(appear \| exams since last appearance) | 3 |
| Semantic soft recurrence | semantic | similarity-weighted recency | 3 |
| Logistic regression (L2) | ML | 20+ engineered features, pooled | 4 |
| Random forest | ML | gated | 6 |
| Gradient boosting | ML | gated | 6 |
| Pooled hot/cold HMM | temporal | gated, EM | 6 |
| Ensemble | ensemble | rank average of the best earlier performers | 5 |

---

## 6. Database schema (SQLite)

| Table | Key columns |
|---|---|
| `course` | id, name, code, description, created_at, settings_json |
| `source_file` | id, course_id, kind (`exam` / `syllabus`), filename, sha256, mime, stored_path, status, error, syllabus_version_id, duplicate_of_id, page_count |
| `document_page` | id, file_id, page_no, text, method (`native` / `ocr` / `docx` / `text`), ocr_confidence, quality_flags |
| `syllabus_version` | id, course_id, label, is_current, effective_from_order |
| `course_topic` | id, course_id, syllabus_version_id, parent_id, depth, number, title, description, concepts_json, hours, kinds_json, aliases_json, source_refs_json, order_no, excluded, user_edited |
| `exam` | id, course_id, source_file_id, title, year, session, exam_type, exam_date, order_index, full_marks, duration, examiner, instructions_json, structure_json, metadata_confidence_json, include_in_analysis, exclusion_reason, duplicate_of_id, user_edited |
| `exam_section` | id, exam_id, label, title, instructions, order_no |
| `exam_question` | id, exam_id, section_id, parent_id, label, path_label, text, raw_text, normalized_text, context_text, marks, marks_source, or_group, is_optional, is_leaf, order_no, page_no, line_no, question_types_json, type_confidence, type_user_edited, equations_json, quality_flags_json, needs_review, user_edited |
| `question_topic_mapping` | id, question_id, topic_id, rank, confidence, status (A/B/C/D), semantic_similarity, keyword_overlap, matched_terms_json, evidence_text, method (`auto` / `manual`) |
| `embedding_cache` | key (sha of backend + model + text), backend, dim, vector blob |
| `analysis_run` | id, course_id, status, progress, message, started_at, finished_at, config_json, data_fingerprint, summary_json |
| `analysis_artifact` | id, run_id, key, json (charts, structure, co-occurrence, ablation, calibration) |
| `model_result` | id, run_id, layer, model_name, enabled, gate_reason, selected, complexity, metrics_json, fold_metrics_json, notes |
| `backtest_fold` | id, run_id, model_name, layer, target_exam_id, target_order, n_train_exams, metrics_json |
| `prediction` | id, run_id, layer, item_kind, topic_id, family_key, rank, score, probability, prob_low, prob_high, calibrated, category, confidence, features_json, contributions_json, evidence_json, why_not_json |
| `predicted_question` | id, run_id, topic_id, text, question_type, marks_low, marks_high, basis, evidence_question_ids_json, grounding_json |
| `search_index` (FTS5) | kind, ref_id, course_id, text |

Mapping to the names in the specification: `CourseContentDocument` = `source_file` with
`kind='syllabus'`; `Topic` and `CourseTopic` = `course_topic`; `QuestionSegment` = the
`exam_question` tree; `Model` = `model_result`; `BacktestRun` = `analysis_run`;
`BacktestResult` = `backtest_fold`; `PredictionEvidence` = `prediction.evidence_json`.

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
* `#/course/:id/predict` the **Analyze & Predict** button, progress, data-sufficiency notes,
  and the ranked dashboard grouped by priority. Clicking a topic opens the evidence panel:
  why it ranked, feature contributions, source questions, syllabus location, predicted
  formulations, and why-not notes for low-ranked topics.
* `#/course/:id/analytics` charts; clicking a point lists the underlying questions.
* `#/course/:id/models` backtest table, per-fold chart, ablation, calibration, selection
  rationale.
* `#/course/:id/paper` predicted paper simulation.
* `#/course/:id/search` keyword and semantic search.
* `#/settings` thresholds, decay, strictness, OCR, embedding backend.

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
| Embeddings | TF-IDF (word stems 1-2 grams + char 3-5 grams), fitted on the syllabus | `BAAI/bge-small-en-v1.5` (33M params, 384-d, about 130 MB) or `all-MiniLM-L6-v2` (22M, about 90 MB) through sentence-transformers | TF-IDF: milliseconds. MiniLM: about 600 questions in 15 to 30 s on CPU, cached afterwards |
| Vector search | NumPy cosine over a dense matrix | FAISS if installed | a course has a few thousand vectors; brute force takes under 10 ms |
| Learning | scikit-learn, SciPy | none | seconds |

Why the TF-IDF default is defensible: syllabus alignment is mostly a matter of shared
technical vocabulary ("Bernoulli", "Rankine cycle", "fugacity"), which stems and
character n-grams capture, and character n-grams also absorb OCR errors. A neural model
helps most with paraphrases that share no words. The backtest and the alignment review
screen let you compare both on your own course.

Fitting the alignment TF-IDF on the **syllabus only** keeps exam text out of the
representation, which removes a subtle leakage path (document frequencies computed from
future exams).

---

## 10. Deterministic and statistical vs learned

| Part | Approach | Reason |
|---|---|---|
| File hashing, duplicates, page extraction | deterministic | exact by nature |
| OCR | pretrained model (Tesseract) | generic, no course data needed |
| Question segmentation, marks, OR groups, metadata | deterministic rules + confidence + manual correction | no labelled data; rules are inspectable and fixable |
| Question type | rule taxonomy (JSON, extensible) | no labelled data; user corrections stored for a future learned classifier |
| Syllabus parsing and merge | deterministic + fuzzy matching | inspectable |
| Alignment | pretrained or TF-IDF similarity + IDF keyword coverage + thresholds | similarity is learned elsewhere; thresholds are explicit |
| Families (exact / paraphrase) | similarity + string ratio thresholds | explicit and testable |
| Frequency, recency, gaps, co-occurrence, structure | statistics | small data |
| Rate, Markov, hazard | Bayesian / survival statistics with shrinkage | small data, pooled |
| Logistic regression, RF, GBM, HMM | learned, gated, must win the backtest | used only when justified |
| Calibration | Platt scaling on out-of-sample predictions | two parameters |
| Ensemble weights | rank average of members chosen on earlier folds | no free weights to overfit |
| Question formulations | templates + retrieval + grounding check | guarantees no unsupported content |

---

## 11. Backtesting framework

Exams are sorted by `order_index`: `e_0, e_1, ..., e_{T-1}`. The next, unseen exam is
`e_T`.

1. **Targets.** For every `t` from `min_train_exams` to `T-1`, the model sees exams
   `e_0..e_{t-1}` and predicts the topic set of `e_t`. The final prediction uses the same
   code with `t = T`.
2. **Feature cutoff.** Features for target `t` are computed from `panel.until(t)`, which
   physically removes rows `t` and later.
3. **Training rows for learned models.** For target `t`, training rows are
   `(features(panel.until(s)), Y[s])` for every `s < t` with at least `min_history` exams
   before `s`. Labels from `t` or later are never used.
4. **Nested tuning.** Hyperparameters (half-life, window length, ensemble members,
   calibration) for target `t` are chosen using only the backtest results of targets
   `s < t`. Because each stored prediction at `s` was itself computed from exams before `s`,
   reusing it is safe.
5. **Metrics per fold**, then mean and standard error across folds.
6. **Selection.** The model with the best mean primary metric is found; then the simplest
   model whose paired difference to it is within one standard error is chosen.
7. **Reported baselines.** Random (exact expectation), most frequent, most recent,
   recency-weighted, and the final choice, always side by side.
8. **Leakage audit.** An automated test perturbs exams `>= t` and checks the predictions
   for `t` are unchanged; the report states the syllabus version used per fold.

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
| Prediction | the selected model's NDCG@K is at least the best simple baseline's; if no learned model beats the baselines by more than one standard error, a baseline is selected and the report says so |
| Calibration | probabilities shown only when nested Brier score beats the base-rate Brier |
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
  ensemble, calibration, ablation, one-SE selection.
* Ranked predictions with categories, evidence, contributions and why-not.
* Type forecast, recurring families, grounded formulations, paper simulation.
* Local web UI, charts, search, CSV/XLSX/PDF export, tests, install scripts, docs.

**Advanced version** (future work, designed for but not required):

* Neural embedding backend switched on by default after download (supported now, opt-in).
* A learned question-type classifier trained on user corrections across courses.
* Math OCR for equations; layout analysis models for two-column papers.
* Optional local LLM (for example through Ollama) for paraphrasing formulations, always
  passed through the grounding checker.
* Cross-course transfer for generic components only.
* Packaged desktop installer (PyInstaller) and a native window wrapper.

---

## 14. Implementation plan

The phases in section 8 are implemented in order. Each phase lands with its unit tests.
Integration and regression tests use a synthetic course generated from a known random
process (`examples/demo_generator.py`), so tests can check that models recover planted
patterns (for example a topic that rotates every two exams) and that an out-of-syllabus
topic from an old syllabus never reaches the predictions.

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
| Temporal analysis, hazard, Markov, rotation test, co-occurrence, marks, format transitions | `temporal/`, `features/`, `prediction/type_forecast.py` | done |
| Candidate models, gates, tuned variants, ensemble | `models/` | done |
| Expanding-window backtest, metrics, one-SE selection, leakage audit | `evaluation/backtest.py` | done |
| Calibration with validity check and bands | `models/calibration.py` | done |
| Ablation (leave-one-out, staged) and syllabus-filter check | `evaluation/ablation.py` | done |
| Ranking, categories, evidence, contributions, why-not, excluded | `prediction/ranking.py` | done |
| Grounded formulations and paper simulation | `generation/` | done |
| UI, charts with click-through, search, export, settings | `app/`, `ui/`, `search/`, `export/` | done |
| Install scripts, CLI, docs, tests | `scripts/`, `cli.py`, `docs/`, `tests/` | done |
| Examiner tendencies | `exam.examiner` field | stored and editable through the API only; no analysis, by design (D6) |
| Historical syllabus per backtest fold | `SyllabusVersion` | partial: historical versions explain excluded questions; the current syllabus is the candidate set for every fold (D5) |
| Learned question-type classifier from user corrections | none | future work |
| Cross-course transfer | none | future work; courses are fully isolated |
| Math OCR, handwriting recognition | none | future work (D9) |

### Measured against the success metrics (synthetic demo, 12 papers)

| Target (section 12) | Result |
|---|---|
| Segmentation on digital PDFs and DOCX | 144 of 144 demo questions found with correct marks across PDF, DOCX and text papers; fixture tests cover TU, KU and generic formats |
| Year detection | 12 of 12 demo papers; fixture tests cover AD, BS and file-name years |
| Alignment top-1 accuracy at least 85%, out-of-syllabus never A | 91.4% (128 of 140); 4 of 4 old-syllabus questions marked D; 0 in-syllabus questions marked D |
| Selected model at least as good as the best simple baseline | selected Recent-window frequency (NDCG@11 0.712), best simple baselines 0.698 to 0.712, random 0.461 |
| Probabilities shown only when they beat the base rate | nested Brier 0.199 vs 0.251 base rate, so probabilities are shown |
| Leakage audit passes | passes for all 19 model variants |
| No prediction or formulation outside the syllabus | asserted in `tests/integration/test_end_to_end.py` |
| A 15-paper course in under 60 s | 12 papers analysed in about 7 s (4-core container, TF-IDF backend); ingestion under 1 s for digital files |
