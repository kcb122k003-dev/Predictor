# Low-data inference

How Exam Predictor ranks topics when a course has two, five or fifteen past papers, and what
the measurements say about it. The short version: no part of the engine switches off because a course is
small. Every component runs at every size. Its influence depends on how much evidence supports it, and
uncertainty is shown instead of hidden.

## 1. Pipeline

```
syllabus + past papers
  -> alignment (pretrained + TF-IDF hybrid), strict syllabus filter (statuses A/B/C/D)
  -> exam x topic panel (exam-level outcomes) + semantic soft evidence (question-level)
  -> components, each predicting paper t from papers before t:
       pretrained/general knowledge   general ranking model, semantic evidence, syllabus coverage
       course evidence                 recency-frequency, hierarchical Bayesian recurrence, Markov,
                                       temporal (hazard only if justified), co-occurrence, question type
       course-learned models          logistic pulled toward the general model, gradient boosting,
                                       random forest, pooled HMM
  -> evidence-aware ensemble (weights from reliability x measured skill, per-topic gating)
  -> calibration (only if it beats the base rate on later papers by more than one standard error)
  -> rank intervals, credible intervals, evidence strength, explanations
  -> topics first, then verified question formulations
```

## 2. What replaced the cutoff

The old engine had `[models.sufficiency]`: below 4 papers there was no backtest, logistic regression needed 7,
trees 10, the HMM 12 and the hazard model 6. Below the cutoff a model produced nothing. All of that is gone,
and no other threshold replaced it.

* The backtest is rolling-origin over every fold: with T papers, papers 2..T are each predicted from the papers
  before them (T - 1 folds). One paper gives no fold, which is a fact about the data, not a setting.
* Every component runs on every fold. A component reports "unavailable" only when an input it needs does not
  exist. Two cases: co-occurrence before two consecutive papers exist, and syllabus coverage or question-type
  fit when the syllabus has no units, weights or format tags.
* Old `settings.json` files and course overrides that still contain `models.sufficiency`,
  `calibration.min_rows` or `calibration.min_positives` load normally; those keys are ignored.

## 3. Evidence, measured at the right level

`predictor/inference/evidence.py` builds an evidence profile for each run:

| Evidence | Counted in | Used for |
|---|---|---|
| Papers, papers without a year (count half), calendar gaps | exams | temporal statistics, reliability |
| Topics observed, topics per paper, coverage spread | exams | Bayesian pooling, diagnostics |
| In-syllabus questions, questions with marks, formats | questions | language and semantic evidence only |
| Held-out folds, fold spread, confidence interval | folds | validation report, ensemble skill |
| Calibration result | folds | whether percentages are probabilities |
| Pretrained model, syllabus weights, other real courses | availability | component status |

Question counts never enter temporal reliability. A paper with forty questions is one paper. A unit test
(`test_questions_do_not_inflate_temporal_evidence`) builds the same course with one and with ten questions per
topic per paper and checks that the Bayesian posteriors, temporal scores and ensemble reliabilities are identical.

## 4. Components

### Hierarchical Bayesian recurrence (`inference/bayes.py`)

Course -> unit -> topic beta-binomial at the exam level. For topic k in unit u, with recency weights
w_s = 0.5^(age/h):

```
n_eff = sum w_s                      h_k = sum w_s * Y[s, k]
mu_course = (sum_k h_k + 2*K*r0) / (n_eff*K + 2*K)          r0 = 0.4 generic prior rate
mu_unit   = (sum_{k in u} h_k + 2*kappa0*mu_course) / (|u|*n_eff + 2*kappa0)
p_k ~ Beta(mu_unit*kappa, (1-mu_unit)*kappa)
```

kappa is chosen by empirical Bayes: it maximises the beta-binomial marginal likelihood of all topics' counts
with a log-normal hyperprior centred on 2 (standard deviation 1 on the log scale), which dominates when there
are one or two papers. Each topic gets a posterior mean, median, 80% credible interval, effective sample size
(kappa + n_eff), prior contribution kappa / (kappa + n_eff) and its raw evidence (appearances, papers).

One appearance in two papers and eight in sixteen have similar means and very different intervals (tested:
the 80% interval for 1/2 is more than 1.5 times wider). The recency discount (none, 6, 3 or 1.5 papers) is chosen
on earlier folds. The default before evidence is a half-life of 6 papers: on 300 simulated courses it beat no
decay by 0.003 NDCG (standard error 0.001). The default is kept unless another discount is ahead by more than
one standard error. On the demo course the backtest chose no decay (+0.016 ± 0.009 over the default).

### Temporal recurrence

A pooled gap hazard ("how often do topics reappear k papers after their last appearance") and the simpler
gap-independent Bayesian rate are compared by their prequential log score on earlier papers. Each earlier paper
is predicted from the papers before it, so the score is a sequential marginal likelihood. The hazard's extra
parameters cost it on early papers, and it is used only when its cumulative log score is higher (a positive log
Bayes factor). On the demo it was not justified (log Bayes factor -0.88), so the simplified recurrence was used.

### Semantic evidence

The bundled WordLlama model (256-dimensional static embeddings, MIT licence, inside the `wordllama` pip package,
no download) runs at every course size. It is used in three places:

* Alignment: hybrid cosine = 0.3 x pretrained + 0.7 x syllabus-fitted TF-IDF.
* Semantic soft evidence: each in-syllabus question spreads one unit of evidence over topics with weights
  softmax(alignment score / 0.05), so near-miss and multi-topic questions count a little.
* Syllabus-only similarity between topics, the most similar past questions per topic (explanations), and
  verification of generated questions.

Questions with status D (outside the syllabus) never contribute, and C questions contribute only when the
strictness setting counts them.

### Other components

* **Recency-frequency**: frequency with no decay, linear decay, last-N windows or exponential half-lives, chosen
  on earlier papers. The default is kept unless another variant is reliably better.
* **Syllabus coverage**: Bayesian unit-level appearance rate times the topic's share of its unit from the
  syllabus (hours, then marks weights, then breadth). Works with zero papers.
* **Question-type fit**: a topic's usual formats (smoothed toward syllabus tags) against the recent paper mix.
* **Co-occurrence**: smoothed lift of topics following the previous paper's topics.
* **Markov** (empirical Bayes) and a pooled hot/cold **HMM** (smoothed EM, runs on short histories).
* **General ranking model** (`inference/generic.py`): logistic regression on scale-free features (recency rates,
  gaps, streaks, hazard, Markov persistence, Bayesian posterior, history length). It is trained on 400 simulated
  courses (104,420 rows) drawn from broad families of examiner behaviour: stable rates, persistence and
  avoidance, rotations, trends, cool-downs and hot/cold phases, with random course sizes and history lengths. It
  ships as `generic_prior.json` and is updated (MAP around the prior) with the rows of other real courses in your
  library. It never uses the course being predicted or courses marked synthetic.
* **Course-specific logistic**: the same scale-free features plus course-only features (semantic, marks, type,
  syllabus, repeats). Its weights have a Gaussian prior centred on the general model (course-only features on 0).
  With zero training rows it equals the general model. With many rows the course data dominate.
* **Gradient boosting, random forest**: run on every fold. Their reliability prior is tiny with few papers
  (about 40 effective parameters each).

## 5. Evidence-aware ensemble (`models/meta.py`)

```
S(x)   = sum_m w_m(x) S_m(x) / sum_m w_m(x)      S_m(x): component m's percentile rank of topic x
w_m(x) = rho_m * exp(tau * skill_m) * g_m(x)
rho_m  = (exams + pk) / (exams + pk + df_m)      df_m: parameters estimated from this course;
                                                 pk = 1 for outside-knowledge components, else 0
skill_m = mean over earlier folds of (metric_m - mean of members) / within-paper spread * folds/(folds + 3)
g_m(x) = 1 + u(x) for outside-knowledge components (semantic, coverage, general), else 1
u(x)   = posterior variance / prior variance of topic x in the Bayesian component
```

`tau` is 2 (`[ensemble] skill_temperature`); on the simulated courses below, tau = 4 changed mean NDCG by at
most 0.007 in either direction, so the less aggressive value was kept. Weights for paper t use only folds before t. With no folds, weights
are the reliability priors: the general model (df 0) and syllabus or pretrained components dominate, and Bayesian
recurrence follows. As papers accumulate, course-learned components gain reliability and must earn weight by
skill.

The ensemble is the final ranking unless a single method beat it on the backtest by more than one standard error
of the paired difference (`replace_if_worse_by_se`). Then that method is used, the simplest one within one
standard error of the best, and the report says so.

Component status shown in the app:

| Status | Meaning |
|---|---|
| ACTIVE | running, inputs available, reliability prior at least 0.5 |
| LIMITED | running and contributing, with high uncertainty (little course evidence for its parameters) |
| DOWNWEIGHTED | running, but earlier papers showed it ranks worse than the average component |
| UNAVAILABLE | an input does not exist for this course; the reason is shown |
| REFERENCE | baseline kept for comparison |

The run-level message is "Low-data advanced inference" while the course-learned ranking models (logistic, trees)
have reliability below 0.5. That holds below about 20 papers, because the logistic model estimates about 20
effective parameters. The message reads, for example: "Only 5 historical examinations are available. Advanced
semantic and Bayesian inference remains active, but course-specific learned ranking parameters have high
uncertainty." It never says that inference is disabled.

## 6. Calibration and uncertainty

* Platt scaling and isotonic regression on percentile ranks compete by nested log loss. For each held-out
  paper, the calibrator is fitted on earlier held-out papers only.
* Percentages are shown as probabilities only when the nested Brier score beats the base rate by more than one
  standard error of the per-paper difference. There is no row-count rule. With few papers the standard error is
  large, and the app shows relative scores with a rank interval and evidence strength instead.
* Rank interval per topic: the wider of (a) the range over leave-one-paper-out rankings, with ensemble weights
  fixed and classifier components held at their full-data scores, and (b) the 10th-90th percentile of ranks when
  the Bayesian component is replaced by draws from its posterior. Uncertainty is Low below a fifth of the topics,
  Medium below two fifths, High otherwise.
* Evidence strength per topic, from the Bayesian prior contribution: Strong (< 30%), Moderate (< 50%),
  Limited (< 75%), Minimal.

## 7. Validation and ablation

* Rolling-origin backtest on every fold. Each report gives folds, papers, the fold-to-fold standard deviation, a
  95% t-interval for the mean metric, and paired comparisons with random, frequency, recency and Bayesian
  recurrence, including papers better and worse.
* Staged ablation, each stage being the evidence-aware ensemble restricted to its components: frequency, then +
  recency, + Bayesian smoothing, + semantic, + topic structure (coverage, type, co-occurrence), + temporal (hazard,
  Markov, HMM), then the full ensemble (general, course logistic, trees). Leave-one-component-out runs as well.
  Metrics: Hit@1/3/5, Recall@K, Precision@K, NDCG@K, concept recall@K and exact-repeat recall. Each change has
  its paired standard error and is labelled "not reliable" when it is within one standard error.
* Leakage audit: every exam from the audit fold onward is scrambled and flipped, every model (including tuned
  variants and the ensemble) is rerun, and predictions for the audit fold must not change.

## 8. Results

### Simulated courses (true outcomes known)

Mean NDCG over all folds, 8 simulated courses per row, 24 topics each. "Planted" uses a generator independent of
the general model (core, regular, rare, alternating, every-third, cool-down, emerging and fading topics, shuffled).
"Generic" uses the simulator family the general model was trained on (different seeds), so the general model's
numbers there are optimistic.

| Papers | Planted: ensemble | frequency | Bayesian | general | Generic: ensemble | frequency | general |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 0.504 | 0.438 | 0.498 | 0.549 | 0.575 | 0.513 | 0.576 |
| 5 | 0.597 | 0.545 | 0.582 | 0.605 | 0.647 | 0.614 | 0.644 |
| 8 | 0.655 | 0.589 | 0.616 | 0.657 | 0.655 | 0.640 | 0.656 |
| 12 | 0.681 | 0.612 | 0.629 | 0.690 | 0.670 | 0.645 | 0.668 |
| 15 | 0.697 | 0.621 | 0.640 | 0.704 | 0.691 | 0.671 | 0.693 |
| 20 | 0.711 | 0.633 | 0.657 | 0.725 | 0.715 | 0.685 | 0.717 |

* At every size from 3 papers, the ensemble beats frequency by 0.05-0.08 NDCG on planted courses and 0.015-0.06
  on generic ones.
* The general model alone is as good as or slightly better than the ensemble on these generators, which share
  behaviour families with its training simulator. The ensemble is within 0.01-0.05 of the best single component
  chosen with hindsight (an upper bound that no method can reach without seeing the answers).
* With 2 papers (one fold, one paper of history) on the planted courses, every method scores below random
  (ensemble 0.349, random 0.437): the planted mix contains alternating and cool-down topics, so "it appeared last
  time" is anti-predictive at that point. One fold says little.
* Course-learned weight grows with data: logistic regression gets 2% of the ensemble at 2-3 papers and 11% at 20;
  random forest and gradient boosting grow from about 1% to 4-6%.

### Synthetic demo course (12 papers, 25 topics, 11 held-out papers)

Scored by the app against its own mapped labels (what a real course can measure):

| Method | NDCG@11 |
|---|---:|
| General ranking model (cross-course) | 0.758 ± 0.028 |
| **Evidence-aware ensemble (final)** | **0.753 ± 0.033** |
| Semantic evidence | 0.749 ± 0.024 |
| Hierarchical Bayesian recurrence | 0.724 ± 0.025 |
| Recent-window frequency (selected by the old engine) | 0.720 ± 0.020 |
| Most frequent topics | 0.709 ± 0.029 |
| Random selection | 0.460 |

Ensemble vs frequency: +0.044 ± 0.022, better on 8 of 11 papers and worse on 2. 95% interval of the ensemble's
mean: 0.679-0.827.

Scored against the generator's true topics, the measure the app cannot see for a real course, the picture is
different. Ensemble 0.734 ± 0.028, frequency 0.745, window 0.738, general model 0.740: all within one standard
error, so **the demo does not show an improvement over plain frequency.** Part of the gain against mapped labels
comes from the semantic component, which uses the same alignment scores that produce the labels. It cannot see
the held-out paper, but it shares the alignment's systematic choices. We report both measurements rather than
the favourable one.

Staged ablation on the demo (NDCG@11, paired change vs previous stage):

| Stage | NDCG | Change | Hit@1 | Recall@11 | Concept recall | Exact repeats |
|---|---:|---:|---:|---:|---:|---:|
| Frequency only | 0.709 | | 0.82 | 0.651 | 0.651 | 0.425 |
| + recency | 0.709 | +0.001 ± 0.009 | 0.82 | 0.653 | 0.653 | 0.461 |
| + Bayesian smoothing | 0.734 | +0.025 ± 0.019 | 1.00 | 0.662 | 0.662 | 0.451 |
| + semantic | 0.727 | -0.007 ± 0.012 | 0.91 | 0.649 | 0.649 | 0.440 |
| + topic structure | 0.729 | +0.002 ± 0.010 | 0.91 | 0.666 | 0.666 | 0.440 |
| + temporal dynamics | 0.731 | +0.003 ± 0.008 | 0.91 | 0.665 | 0.665 | 0.451 |
| Full ensemble | 0.753 | +0.021 ± 0.016 | 0.91 | 0.696 | 0.696 | 0.470 |

Leave-one-out on the demo: removing semantic evidence costs 0.015 ± 0.009 and removing the general model costs
0.014 ± 0.015. Removing recency-frequency *improves* NDCG by 0.007 ± 0.006, and removing coverage, co-occurrence,
temporal, Markov, HMM or the trees changes NDCG by +0.002 to +0.003 (standard error 0.0025). With 11 held-out
papers these are small effects, and several components add noise on this course. The demo syllabus has no
sub-topics below topics, so concept recall equals topic recall here.

Other demo measurements: alignment puts 127 of 140 in-syllabus questions on the right topic (TF-IDF alone: 128),
all 4 old-syllabus questions are "outside" and no in-syllabus question is. Calibration (Platt, chosen over
isotonic by nested log loss) validated on 10 later papers: Brier 0.182 vs 0.249 for the base rate (gain
0.067 ± 0.006). Generated formulations: 56 checked, 4 rejected (2 for topic, 2 for semantic fit, all
reused past numericals whose history was mapped to the wrong topic). The leakage audit passes for 33 models,
including the tuned and ensemble models.

### Alignment backends

| Backend | Demo questions on the right topic | Reworded questions (26) | Old-syllabus questions marked outside |
|---|---:|---:|---:|
| TF-IDF | 128/140 (app) | 10 | 4/4 |
| Hybrid (default) | 127/140 (app) | 11 | 4/4 |
| WordLlama alone | 118/140 (offline check) | 10 | 1/4 |

The bundled static embeddings do not beat syllabus-fitted TF-IDF for alignment, and alone they separate
in-syllabus from out-of-syllabus questions poorly. The hybrid keeps TF-IDF's separation and adds pretrained
knowledge. For detecting reworded repeats, character n-grams found the right topic for 14 of 26 reworded
questions and WordLlama for 11. Repeat detection therefore keeps character n-grams unless a larger
sentence-transformer is installed (`predictor models download`).

## 9. Global and course knowledge in the database

| Table | Holds |
|---|---|
| `model_registry` scope "global" | each general model used: source (simulation, simulation+repository), courses it learned from, parameters |
| `model_registry` scope "course" | the course-specific logistic model's coefficients for a run |
| `course_feature_set` | one row per course: scale-free feature rows and outcomes, flagged synthetic or not |
| `prediction`, `model_result` | course predictions, uncertainty, evidence strength; component status, weight and reliability |

Course A's rows can change the general model used for course B. They never enter course B's panel, frequencies,
Bayesian posteriors or features (tested in `test_other_courses_do_not_change_this_courses_history`). The demo
course is marked synthetic. Its papers have source "demo", and its rows are never used for cross-course training.
Generated formulations and simulated practice papers are never stored as exams or questions (tested).

`database/migrations.py` adds the new columns to existing databases with `ALTER TABLE ... ADD COLUMN` and records
the schema version. Nothing is dropped. `tests/fixtures/schema_v1.sql` is the version 1 schema used by the
migration test.

## 10. Behaviour by course size

| Papers | What happens |
|---|---|
| 0 | Ranking from syllabus coverage and the general model; every topic has a wide rank range |
| 1 | No held-out fold. Bayesian recurrence (mostly prior), semantic, general and syllabus components; no calibration |
| 2-5 | 1-4 folds. Weights mostly from reliability priors; course-learned models run with 1-5% weight (LIMITED); rank intervals wide; scores shown as relative unless calibration is validated |
| 6-15 | Skill on earlier papers starts to separate components; components that rank worse than average become DOWNWEIGHTED; calibration validated when it beats the base rate reliably |
| 15+ | Course-learned models reach reliability 0.4-0.6 and can earn substantial weight; the message changes to "Advanced inference" from about 20 papers |

## 11. Limitations

* With one held-out paper, accuracy cannot be measured, and with fewer than about five the confidence interval is
  wider than most differences between methods. The app says so.
* On the demo course the ensemble does not beat frequency against the true topics. Gains are demonstrated on
  simulated courses, which are only as realistic as their generators.
* The general model is trained on simulations. Until other real courses are in your library, it encodes generic
  examiner behaviour, not your institution's.
* The bundled pretrained model is a static word-embedding model. It knows general English but not technical
  paraphrase well (11 of 26 reworded questions). A sentence-transformer improves this and must be downloaded once.
* Alignment's feedback pass uses questions from all papers, so the mapping of an older paper can be influenced by
  wording in later papers. Outcomes (which topics a held-out paper contains) never are.
* Rank intervals hold classifier components fixed during the leave-one-paper-out step.
* An analysis takes about 15-20 seconds for a 12-paper course on 4 cores (about 7 seconds before this upgrade),
  mostly tree models refitted on every fold.

## 12. Settings

| Setting | Default | Meaning |
|---|---|---|
| `embeddings.backend` | auto | sentence-transformer if downloaded, else hybrid, else TF-IDF |
| `embeddings.pretrained` | wordllama | "none" turns the pretrained layer off |
| `embeddings.hybrid_pretrained_weight` | 0.3 | pretrained share of the hybrid alignment cosine |
| `alignment.semantic_temperature` | 0.05 | sharpness of semantic soft evidence |
| `temporal.bayes_half_lives` | [6, 3, 1.5] | recency discounts tried by Bayesian recurrence (plus none) |
| `models.min_train_exams` | 1 | first held-out paper (1 = every paper after the first) |
| `models.course_prior_precision` | 2.0 | pull of the course logistic toward the general model |
| `ensemble.skill_temperature` | 2.0 | how strongly measured skill moves weights |
| `ensemble.skill_prior_folds` | 3 | folds of evidence before skill counts fully |
| `ensemble.gate_strength` | 1.0 | per-topic shift toward outside knowledge for topics with little history |
| `ensemble.replace_if_worse_by_se` | 1.0 | a single method replaces the ensemble only if better by this many SEs |
| `calibration.methods` | platt, isotonic | calibrators compared by nested log loss |

Rebuild the shipped general model with `python -m predictor.inference.generic` (deterministic, about 15 seconds).
