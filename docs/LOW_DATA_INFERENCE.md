# Low-data inference

How Exam Predictor ranks topics when a course has two, five or fifteen past papers, and what
the measurements say about it. The short version: no part of the engine switches off because a course is
small. Every component runs at every size unless an input it needs does not exist (section 2). Its influence
depends on how much evidence supports it, and uncertainty is shown instead of hidden.

## 1. Pipeline

```
syllabus + past papers (papers without a year are excluded at ingestion)
  -> alignment (pretrained + TF-IDF hybrid; the feedback pass for a paper uses only earlier papers),
     strict syllabus filter (statuses A/B/C/D)
  -> exam x topic panel (exam-level outcomes, per-question input quality)
     + semantic soft evidence (question-level, capped at 1 per topic per paper)
  -> components, each predicting paper t from papers before t:
       pretrained/general knowledge   general ranking model, semantic evidence, syllabus coverage
       course evidence                 recency-frequency, hierarchical Bayesian recurrence, Markov,
                                       temporal (hazard only if justified), co-occurrence, question type
       course-learned models          logistic pulled toward the general model, gradient boosting,
                                       random forest, pooled HMM
  -> evidence-aware ensemble (weights from input quality x reliability x measured skill; per-topic gate
     from the posterior of the Bayesian recurrence variant actually chosen)
  -> final model: the ensemble, unless best_single (the method chosen on earlier papers) beats it
     beyond a one-sided 95% t-bound
  -> calibration (only if the Brier gain over the base rate on later papers clears a one-sided 95% t-bound)
  -> rank intervals (jackknife, united with Dirichlet-weight and posterior draws), credible intervals,
     per-topic evidence strength (posterior/prior variance ratio), explanations
  -> topics first, then verified question formulations

With no papers: general model + syllabus structure, every topic's rank range is [1, K], uncertainty High,
no probabilities.
```

## 2. What replaced the cutoff

The old engine had `[models.sufficiency]`: below 4 papers there was no backtest, logistic regression needed 7,
trees 10, the HMM 12 and the hazard model 6. Below the cutoff a model produced nothing. All of that is gone,
and no other threshold replaced it.

* The backtest is rolling-origin over every fold: with T papers, papers 2..T are each predicted from the papers
  before them (T - 1 folds). One paper gives no fold, which is a fact about the data, not a setting.
* Every component runs on every fold. A component reports "unavailable" only when an input it needs does not
  exist. Examples: semantic evidence before any paper exists, co-occurrence before two consecutive papers exist,
  syllabus coverage when the syllabus has a single unit and no hours, marks or sub-topics, question-type fit
  when past papers use one format and the syllabus has no format tags, and the general component when no
  general model exists (simulated prior switched off and no other real course in the library).
* Old `settings.json` files and course overrides that still contain `models.sufficiency`,
  `calibration.min_rows` or `calibration.min_positives` load normally; those keys are ignored (section 12 lists
  every ignored key).

## 3. Evidence, measured at the right level

`predictor/inference/evidence.py` builds an evidence profile for each run:

| Evidence | Counted in | Used for |
|---|---|---|
| Papers, calendar gaps (ingestion excludes a paper without a year; if you include it in Review, it counts as half a paper for reliability) | exams | temporal statistics, reliability |
| Topics observed, topics per paper, coverage spread | exams | Bayesian pooling, diagnostics |
| In-syllabus questions, questions with marks, formats, mapping and parse confidence | questions | language and semantic evidence, input quality |
| Held-out folds, fold spread, confidence interval | folds | validation report, ensemble skill |
| Calibration result | folds | whether percentages are probabilities |
| Pretrained model, syllabus weights, other real courses | availability | component status, input quality |

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
decay by 0.003 NDCG (standard error 0.001). The default is kept unless another discount is ahead on earlier folds
beyond a one-sided 95% t-bound of the paired differences. On the demo course the backtest chose no decay: it was
ahead of the default by 0.019 ± 0.010 on 11 earlier papers, beyond the bound.

### Temporal recurrence

A pooled gap hazard ("how often do topics reappear k papers after their last appearance") and the simpler
gap-independent Bayesian rate are compared by their prequential log score on earlier papers. Each earlier paper
is predicted from the papers before it, so the score is a sequential marginal likelihood. The hazard's extra
parameters cost it on early papers, and it is used only when its cumulative log score is higher (a positive log
Bayes factor). On the demo it was not justified (log Bayes factor -3.35 over 11 papers), so the simplified
recurrence was used.

### Semantic evidence

The bundled WordLlama model (256-dimensional static embeddings, MIT licence, inside the `wordllama` pip package,
no download) runs at every course size. It is used in three places:

* Alignment: hybrid cosine = 0.3 x pretrained + 0.7 x syllabus-fitted TF-IDF. The feedback pass, which moves
  topic vectors toward confidently mapped questions, uses only questions from papers earlier in exam order than
  the paper being aligned. A later paper's wording cannot change how an earlier paper is mapped.
* Semantic soft evidence: each in-syllabus question spreads one unit of evidence over topics with weights
  softmax(alignment score / 0.05), so near-miss and multi-topic questions count a little. Within one paper the
  shares for a topic combine as 1 - prod(1 - share), so a topic gets at most 1 per paper: six questions on one
  topic in one paper cannot outweigh one question on it in each of four papers.
* Syllabus-only similarity between topics, the most similar past questions per topic (explanations), and
  verification of generated questions.

Questions with status D (outside the syllabus) never contribute, and C questions contribute only when the
strictness setting counts them.

### Other components

* **Recency-frequency**: frequency with no decay, linear decay, last-N windows or exponential half-lives, chosen
  on earlier papers. The default is kept unless another variant is ahead beyond a one-sided 95% t-bound.
* **Syllabus coverage**: Bayesian unit-level appearance rate times the square root of the topic's share of its
  unit, relative to the unit's average topic, from the syllabus (hours, then marks weights, then breadth).
  Works with zero papers.
* **Question-type fit**: a topic's usual formats (smoothed toward syllabus tags) against the recent paper mix.
* **Co-occurrence**: smoothed lift of topics following the previous paper's topics.
* **Markov** (empirical Bayes) and a pooled hot/cold **HMM** (smoothed EM, runs on short histories).
* **General ranking model** (`inference/generic.py`): logistic regression on scale-free features (recency rates,
  gaps, streaks, hazard, Markov persistence, Bayesian posterior, history length). It is trained on 400 simulated
  courses (104,420 rows) drawn from broad families of examiner behaviour: stable rates, persistence and
  avoidance, rotations, trends, cool-downs and hot/cold phases, with random course sizes and history lengths. It
  ships as `generic_prior.json` and is updated (MAP around the prior) with the rows of other real courses in your
  library. It never uses the course being predicted or courses marked synthetic. With
  `models.use_simulated_prior = false` no simulated knowledge is used: the general model is learned from other
  real courses alone, and with none in the library it reports UNAVAILABLE and the ensemble reweights the rest.
* **Course-specific logistic**: the same scale-free features plus course-only features (semantic, marks, type,
  syllabus, repeats, the share of the topic's questions that were optional, semantic evidence density, syllabus
  centrality, mapping confidence, parse confidence and the share of questions with marks). Its weights have a
  Gaussian prior centred on the general model (course-only features on 0; centred on 0 throughout when no
  general model exists). With zero training rows it equals the general model. With many rows the course data
  dominate.
* **Gradient boosting, random forest**: run on every fold. Their reliability prior is tiny with few papers
  (about 40 effective parameters each).

## 5. Evidence-aware ensemble (`models/meta.py`)

```
S(x)    = sum_m w_m(x) S_m(x) / sum_m w_m(x)     S_m(x): component m's percentile rank of topic x
w_m(x)  = q_m * rho_m * exp(tau * skill_m) * g_m(x)
q_m     = input quality of component m (0-1), from papers before the cutoff only (table below)
rho_m   = (exams + pk) / (exams + pk + df_m)     df_m: parameters estimated from this course
                                                 (df_m = 0 gives rho_m = 1);
                                                 pk = 1 for outside-knowledge components, else 0
skill_m = mean over earlier folds of (metric_m - mean of members) / sd * folds / (folds + 3)
sd      = spread of members around each earlier fold's mean, at least 0.05 (ensemble.min_metric_sd)
g_m(x)  = 1 + gate_strength * u(x) for outside-knowledge components (semantic, coverage, general), else 1
u(x)    = posterior variance / prior variance of topic x under the Bayesian recurrence variant
          chosen for this paper
```

Weights are normalised over the components available for that paper. Semantic evidence gets the pseudo-exam
pk only when the pretrained model is present. `exams` counts papers before the cutoff at the exam level.

Input quality q_m multiplies the reliability prior, so the evidence profile changes the weights as well as the
report:

| Component | q_m |
|---|---|
| Semantic evidence | (0.5 + 0.5 x mean mapping confidence), times 0.8 without the pretrained model |
| Question-type fit | min(1, question formats observed / 3), at least 1/3 |
| Syllabus coverage | 1.0 for teaching-hour or marks weights, 0.7 for breadth, 0.5 for uniform weights |
| Recency, Bayesian recurrence, hazard, Markov, HMM, co-occurrence | 1 - 0.5 x missing calendar years / year span |
| Course logistic, gradient boosting, random forest | 0.5 + 0.25 x share of marks known + 0.25 x parse confidence |
| General ranking model | 1 |

`tau` is 2 (`[ensemble] skill_temperature`). On the planted and generic simulated courses of section 8 at 3, 5
and 12 papers, measured on an earlier build of the engine, tau = 4 changed the ensemble's mean NDCG by at most
0.007 in either direction, so the less aggressive value was kept. Weights for paper t use only folds
before t, and those folds are scored with the K known at the time (median topics per paper over that fold's
paper and the papers before it). With no folds, no skill is measured, so weights come from input quality, the
reliability priors and the per-topic gate: the general model (df 0)
and syllabus or pretrained components dominate, and the one-parameter components (Bayesian recurrence,
recency-frequency, question-type fit) come next. As papers accumulate, course-learned components
gain reliability and must earn weight by skill. A component is skipped only when an input it needs does not
exist (UNAVAILABLE); the ensemble then reweights the others.

The ensemble is the final ranking unless `best_single` beats it. `best_single` is an out-of-sample selector: for
each held-out paper it uses the method that scored best on the papers before it, so its fold scores are honest.
The ensemble is replaced only when `best_single` is ahead on the same held-out papers by more than a one-sided
95% t-bound of the paired differences (`ensemble.replace_confidence`, default 0.95). The report gives both means
and the difference with its standard error. Picking whichever method had the best overall backtest score would
favour the method that got lucky; with about 20 candidates and a handful of papers that happens often. The
question-type forecast (layer 2) keeps its own rule: the simplest method within one standard error of the best.

Component status shown in the app:

| Status | Meaning |
|---|---|
| ACTIVE | running, inputs available, reliability prior at least 0.5 |
| LIMITED | running and contributing, with high uncertainty (little course evidence for its parameters) |
| DOWNWEIGHTED | running, but earlier papers showed it ranks worse than the average component |
| UNAVAILABLE | an input does not exist for this course; the reason is shown |
| REFERENCE | baseline kept for comparison |

The run-level message is "Low-data advanced inference" while the most reliable course-learned model (logistic,
gradient boosting, random forest) has reliability below 0.5, and "Advanced inference" from there. On a course
like the demo the logistic model estimates about 22.5 parameters, so the wording changes at about 25 papers. The
change is wording only: it switches nothing on or off. The message reads, for example: "Only 5 historical
examinations are available. Advanced semantic and Bayesian inference remains active, but course-specific learned
ranking parameters have high uncertainty." With zero papers it reads: "No historical examinations are available
yet. The ranking uses the syllabus structure and the general cross-course model; every topic has high
uncertainty." It never says that inference is disabled.

## 6. Calibration and uncertainty

* Platt scaling (fitted by Newton's method) and isotonic regression on percentile ranks compete by nested log
  loss. For each held-out paper, the method is chosen and the calibrator fitted on earlier held-out papers only.
* Percentages are shown as probabilities only when the nested Brier gain over the base rate exceeds a one-sided
  95% t-bound times its standard error (`t_bound(n) * se` over n per-paper gains). There is no row count and no
  minimum number of positives. With few papers the bound is large, and the app shows relative scores with a band
  and evidence strength instead. When probabilities are shown, their band is the 10th-90th percentile of
  bootstrap calibrators.
* Rank interval per topic: the union of (a) the range over leave-one-paper-out rankings, with ensemble weights
  fixed and the slow classifier components (logistic, trees, HMM) held at their full-data scores, and (b) the
  10th-90th percentile of ranks over 200 draws, each combining ensemble weights resampled from a Dirichlet
  distribution (concentration 2 + 4 x held-out papers the weights were learned from) with a draw from the Bayesian
  posterior. The interval always includes the topic's current rank. Uncertainty is Low when the interval spans
  less than a fifth of the topics, Medium below two fifths, High otherwise. A topic whose posterior variance is
  still at least half of its prior variance is at least Medium, and at least three quarters makes it High.
  With zero papers every topic gets the full range [1, K] and level High.
* Evidence strength per topic, from the posterior/prior variance ratio of the Bayesian recurrence: Strong
  (< 0.3), Moderate (< 0.5), Limited (< 0.75), Minimal otherwise. A topic absent from many papers can be Strong:
  its low rate is well measured.

## 7. Validation and ablation

* Rolling-origin backtest on every fold. Each report gives folds, papers, the fold-to-fold standard deviation, a
  95% t-interval for the final ranking's mean metric (clipped to [0, 1]), and paired comparisons with random,
  frequency, recency and Bayesian recurrence, including papers better and worse. With fewer than 2 held-out
  papers the report says accuracy cannot be measured; with fewer than 4 it says the interval is too wide to
  separate methods.
* Final-model check: the ensemble against `best_single` on the same held-out papers, with the one-sided 95%
  rule from section 5.
* Staged ablation: each stage is the evidence-aware ensemble restricted to the listed components plus the earlier
  stages' members, and baselines such as frequency stay in. Stages: frequency, then + recency, + Bayesian
  smoothing, + semantic, + topic structure (coverage, type, co-occurrence), + temporal (hazard, Markov, HMM), then
  the full ensemble (general, course logistic, trees). Leave-one-component-out runs as well. Metrics: Hit@1/3/5,
  Recall@K, Precision@K, NDCG@K, concept recall@K and exact-repeat recall. Each change has its paired standard
  error. A change is reliable only when it lies outside a two-sided 95% t-interval of the paired per-paper
  differences. The note lists reliable gains and reliable losses separately, and says so when there are none.
* Leakage audit: every exam from the audit fold onward is scrambled and flipped, every model (including tuned
  variants and the ensemble) is rerun, and predictions for the audit fold must not change. Separately,
  `test_later_papers_never_change_earlier_predictions` changes the later papers of a course (including how many
  topics they contain, which moves the automatic K) and checks that every model's predictions for the earlier
  papers stay the same (causal K). It builds the panel directly, so it does not exercise the alignment feedback
  pass. No dedicated test covers the causal feedback. The regression snapshot pins the demo's status counts
  (A 104 / B 30 / C 6 / D 4); they were A 109 / B 27 / C 4 / D 4 before the feedback became causal, so reverting
  it would fail `test_data_layer_unchanged`.

## 8. Results

### Simulated courses (true outcomes known)

Mean NDCG over all folds of an in-sample rolling-origin backtest, 8 simulated courses per row, 24 topics each.
"Planted" uses a generator independent of the general model (core, regular, rare, alternating, every-third,
cool-down, emerging and fading topics, shuffled). "Generic" uses the simulator family the general model was
trained on (different seeds), so the general model's numbers there are optimistic.

| Papers | Planted: ensemble | frequency | Bayesian | general | Generic: ensemble | frequency | general |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.349 | 0.317 | 0.349 | 0.349 | 0.520 | 0.468 | 0.520 |
| 3 | 0.504 | 0.438 | 0.498 | 0.549 | 0.575 | 0.513 | 0.576 |
| 5 | 0.593 | 0.545 | 0.581 | 0.605 | 0.643 | 0.614 | 0.644 |
| 8 | 0.655 | 0.589 | 0.617 | 0.657 | 0.656 | 0.640 | 0.656 |
| 12 | 0.679 | 0.612 | 0.625 | 0.690 | 0.669 | 0.645 | 0.668 |
| 15 | 0.695 | 0.621 | 0.637 | 0.704 | 0.694 | 0.671 | 0.693 |
| 20 | 0.711 | 0.633 | 0.651 | 0.725 | 0.717 | 0.685 | 0.717 |

* Ensemble minus frequency: +0.03 to +0.08 NDCG on planted courses and +0.016 to +0.06 on generic ones.
* The general model alone is as good as or slightly better than the ensemble on both generators.
* With 2 papers (one fold, one paper of history) on the planted courses, every method scores below random
  (random 0.437): a third of the planted topics alternate or cool down, so "it appeared last time" is
  anti-predictive at that point. One fold says little.
* The ensemble was kept as the final model in all 112 simulated runs; `best_single` never replaced it.
* Course-learned weight grows with data: logistic regression gets 2-3% of the ensemble at 2-3 papers and 11% at
  20 (planted); random forest and gradient boosting grow from about 1% to 4-6%.

Next-paper check, strictly out of sample: fit on T papers, score paper T + 1. 40 courses per size, 12-topic
generator from `tests/unit/test_low_data.py`.

| T | Ensemble minus frequency (NDCG) |
|---:|---:|
| 3 | +0.025 ± 0.017 |
| 4 | -0.008 ± 0.012 |
| 5 | +0.032 ± 0.012 |
| 6 | -0.001 ± 0.012 |
| 8 | +0.011 ± 0.008 |

Selection replaced the ensemble 0 of 40 times at every size. On the next paper the ensemble's gain over frequency
is small and clearly positive only at T = 5.

### Synthetic demo course (12 papers, 25 topics, 144 questions, 11 held-out papers, K = 11)

Scored by the app against its own mapped labels (what a real course can measure), NDCG@11, mean ± standard error
over 11 papers:

| Method | NDCG@11 | Status, weight |
|---|---:|---|
| General ranking model (cross-course) | 0.730 ± 0.022 | ACTIVE, 21% |
| **Evidence-aware ensemble (final)** | **0.717 ± 0.027** | final |
| Semantic evidence (pretrained) | 0.712 ± 0.021 | ACTIVE, 21% |
| Best single method chosen on earlier papers | 0.707 ± 0.022 | reference |
| Random forest | 0.696 ± 0.042 | LIMITED, 3% (reliability 0.23) |
| Recent-window frequency | 0.687 ± 0.017 | reference |
| Course logistic | 0.682 ± 0.025 | LIMITED, 4% (reliability 0.34) |
| Two-state Markov | 0.682 ± 0.024 | ACTIVE, 12% |
| Most frequent topics | 0.678 ± 0.024 | reference |
| Temporal recurrence (hazard if justified) | 0.678 ± 0.024 | ACTIVE, 9% |
| Hierarchical Bayesian recurrence | 0.673 ± 0.026 | ACTIVE, 9% |
| Gradient boosting | 0.667 ± 0.048 | LIMITED, 2% |
| Recency-frequency | 0.666 ± 0.023 | ACTIVE, 10% |
| Syllabus coverage | 0.614 ± 0.043 | DOWNWEIGHTED, 4% |
| Pooled HMM | 0.604 ± 0.030 | DOWNWEIGHTED, 2% |
| Question-type fit | 0.553 ± 0.035 | DOWNWEIGHTED, 2% |
| Topic co-occurrence | 0.525 ± 0.028 | DOWNWEIGHTED, 1% |
| Random selection | 0.456 | reference |

95% interval of the ensemble's mean: 0.657-0.777 (width 0.12); fold standard deviation 0.090. Paired
differences, ensemble minus baseline: random +0.261 ± 0.024 (better on 11 papers, worse on 0), frequency
+0.039 ± 0.019 (9 better, 2 worse), recency-frequency +0.051 ± 0.019 (8 better, 2 worse), Bayesian recurrence
+0.043 ± 0.020 (7 better, 4 worse). Final-model check: ensemble 0.717 against `best_single` 0.707, difference
+0.010 ± 0.018, so the ensemble was kept. Before alignment feedback became causal, the held-out paper's own
wording fed the alignment, and most app-label scores were optimistic. Measured in isolation (current code with
only the feedback pass switched back to all papers), making the feedback causal changed the scores by -0.072
(co-occurrence) to +0.016 (coverage): ensemble -0.023, frequency -0.031, semantic evidence -0.033, while
coverage, course logistic (+0.002) and random forest (+0.003) went up. The larger drop in the regression
snapshot (ensemble 0.754 to 0.717) also includes other review fixes (causal K, capped semantic evidence, gate
changes), so causal feedback accounts for only part of it.

Scored against the generator's planted true topics, the measure the app cannot see for a real course (label
agreement 92.3%), the picture is different. Ensemble 0.738 ± 0.025, frequency 0.734 ± 0.020, window 0.745,
general model 0.747, semantic 0.747, random forest 0.749, Bayesian recurrence 0.685, course logistic 0.689.
Ensemble minus frequency is +0.004 ± 0.017, so **the demo does not show an improvement over plain frequency.**
Part of the gain against mapped labels comes from components such as semantic evidence, which use the same
alignment scores that produce the labels. They cannot see the held-out paper, but they share the alignment's
systematic choices. We report both measurements rather than the favourable one.

Staged ablation on the demo (NDCG@11, paired change vs previous stage ± standard error; reliable means outside the
two-sided 95% t-interval, t(10) = 2.23):

| Stage | NDCG | Change | Hit@1 | Recall@11 | Concept recall | Exact repeats |
|---|---:|---:|---:|---:|---:|---:|
| Frequency only | 0.678 | | 0.82 | 0.616 | 0.616 | 0.431 |
| + recency | 0.663 | -0.015 ± 0.008 | 0.82 | 0.593 | 0.593 | 0.429 |
| + Bayesian smoothing | 0.669 | +0.006 ± 0.017 | 0.82 | 0.601 | 0.601 | 0.460 |
| + semantic | 0.720 | +0.052 ± 0.019 (reliable) | 1.00 | 0.634 | 0.634 | 0.450 |
| + topic structure | 0.699 | -0.021 ± 0.016 | 0.91 | 0.633 | 0.633 | 0.440 |
| + temporal dynamics | 0.694 | -0.005 ± 0.003 | 0.91 | 0.633 | 0.633 | 0.450 |
| Full ensemble | 0.716 | +0.022 ± 0.010 | 0.91 | 0.667 | 0.667 | 0.460 |

Only "+ semantic" is a reliable change; no stage is a reliable loss. Leave-one-out from the full ensemble (0.717;
the staged "Full ensemble" row scores 0.716 because it also keeps the frequency baseline), change in NDCG when
the component is removed: general model -0.031 ± 0.014, semantic evidence -0.020 ± 0.017,
course logistic -0.007 ± 0.006, Markov -0.004 ± 0.006, question-type fit +0.016 ± 0.009, coverage
+0.009 ± 0.008, recency-frequency +0.005 ± 0.005, Bayesian recurrence +0.004 ± 0.010; the others within ±0.001.
On this course question-type fit and coverage add noise. The ensemble downweights them (2% and 4%) but does not
drop them. The demo syllabus has no sub-topics below topics, so concept recall equals topic recall here.

Other demo measurements:

* Alignment (hybrid, causal feedback) puts 131 of 140 in-syllabus questions on the right topic; all 9 misses have
  status B. All 4 old-syllabus questions are marked D (outside), and no other question is. Status counts: A 104,
  B 30, C 6, D 4; 134 of 144 questions are counted.
* Syllabus filter: with the filter 0.717 ± 0.027, without it 0.690 ± 0.026 (change -0.027 ± 0.019).
* Calibration is valid. Isotonic was chosen over Platt by nested log loss. Validated on 10 later papers: Brier
  0.191 vs 0.248 for the base rate (skill 23%, gain 0.057 ± 0.005); expected calibration error 0.042.
* Uncertainty: 15 topics Low, 10 Medium; median rank-range width 4; overall prediction uncertainty "low",
  evidence quality "moderate".
* Generated formulations: 53 checked, 2 rejected, 0 for question type. Each rejected one failed both the topic
  and the semantic check: a pipe Bernoulli numerical filed under "Pressure and its measurement" and a jet-impact
  numerical under "Laminar flow". Both are reused past numericals.
* The leakage audit passes for 34 models (target index 6), including the tuned and ensemble models.

### Alignment backends

| Backend | Demo questions on the right topic | Reworded questions aligned to the right topic (of 26; offline check before causal feedback) | Old-syllabus questions marked outside |
|---|---:|---:|---:|
| TF-IDF | 131/140 (app, causal feedback) | 10 | 4/4 |
| Hybrid (default) | 131/140 (app, causal feedback) | 11 | 4/4 |
| WordLlama alone | 118/140 (offline check before causal feedback) | 10 | 1/4 |

Status counts on the demo with causal feedback: TF-IDF A 102 / B 29 / C 9 / D 4, hybrid A 104 / B 30 / C 6 / D 4.
On the demo the hybrid is not shown to map more questions correctly than TF-IDF (131 of 140 each); it leaves
fewer questions at status C (6 against 9). The bundled static embeddings do not beat syllabus-fitted TF-IDF for
alignment, and alone they separate in-syllabus from out-of-syllabus questions poorly (1 of 4 old-syllabus
questions marked outside). The hybrid keeps TF-IDF's separation.

Repeat detection is a different measurement from the reworded-questions column above: for each of the 26
reworded questions, it checks whether the nearest past question in the demo bank is on the same topic. That was
true for 14 of 26 with character n-grams and 11 of 26 with WordLlama. Repeat detection therefore keeps character
n-grams unless a larger sentence-transformer is installed (`predictor models download`).

## 9. Global and course knowledge in the database

| Table | Holds |
|---|---|
| `model_registry` scope "global" | one row per distinct general model ever used, keyed by a fingerprint of the prior source and the other courses' feature sets: source (simulation, simulation+repository, repository, none), courses it learned from, parameters. Each run records the fingerprint in its config |
| `model_registry` scope "course" | the course-specific logistic model's coefficients for a run |
| `course_feature_set` | one row per course: scale-free feature rows and outcomes, flagged synthetic or not |
| `prediction`, `model_result` | course predictions, uncertainty, evidence strength; component status, weight and reliability |

Source "simulation" is the shipped prior alone; "simulation+repository" is that prior updated with other real
courses; "repository" means other real courses only (`models.use_simulated_prior = false`); "none" means the
simulated prior is off and no other real course exists, so the general component is UNAVAILABLE.

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
| 0 | The analysis runs with a syllabus and no papers. Ranking from the general model and syllabus structure; every topic has rank range 1..K and uncertainty High; no probabilities; the message says no historical examinations are available yet |
| 1 | No held-out fold. Bayesian recurrence (mostly prior), semantic, general and syllabus components; no calibration |
| 2-5 | 1-4 folds. Weights mostly from reliability priors; course-learned models run with 1-5% weight (LIMITED); rank intervals wide; scores shown as relative unless calibration is validated |
| 6-15 | Skill on earlier papers starts to separate components; components that rank worse than average become DOWNWEIGHTED; calibration validated when its Brier gain clears the one-sided 95% bound |
| 15+ | The course logistic (about 22.5 parameters) reaches reliability about 0.4 at 15 papers and 0.5 at about 25; gradient boosting and random forest (about 40 each) stay below 0.4 until at least 27 papers. Their weight still depends on measured skill (logistic about 11% at 20 papers on the planted simulations). On a demo-like course the message wording changes to "Advanced inference" at about 25 papers (wording only) |

## 11. Limitations

* With one held-out paper, accuracy cannot be measured, and with fewer than about five the confidence interval is
  wider than most differences between methods. The app says so.
* On the demo course the ensemble does not beat frequency against the true topics (+0.004 ± 0.017), and on
  simulated courses its next-paper gain over frequency is small (-0.008 to +0.032). The gains shown above come
  mostly from in-sample backtests on simulated courses, which are only as realistic as their generators.
* The general model is trained on simulated sequences (`inference/generic_prior.json`, 400 simulated courses,
  104,420 rows). Until other real courses are in your library, it encodes generic examiner behaviour, not your
  institution's. `models.use_simulated_prior = false` (the "Use the simulated cross-course prior" checkbox on the
  Settings page) turns it off. Keeping it on by default was a judgment call: the requirements said synthetic
  questions must never count as history. The simulated sequences never enter any course's history, frequencies
  or posteriors; they only shape the cross-course prior.
* Papers without a year are excluded at ingestion, because they cannot be placed in time. You can include one in
  Review; it then enters the panel at its order index, counts as half a paper in the components' reliability
  priors, and the evidence profile reports it under `exams_without_year`.
* The bundled pretrained model is a static word-embedding model. It knows general English but not technical
  paraphrase well: for 26 reworded questions, its nearest past question was on the same topic for 11, against 14
  for character n-grams, so repeat detection keeps character n-grams. A full sentence-transformer replaces them
  for repeat detection; it must be downloaded once, and this document has no measurement of it.
* Rank intervals hold classifier components fixed during the leave-one-paper-out step.
* Some components add noise on some courses (question-type fit and coverage on the demo). The ensemble
  downweights them but does not drop them.
* An analysis takes about 9 seconds for the 12-paper demo course on 4 cores (single-threaded BLAS).

## 12. Settings

| Setting | Default | Meaning |
|---|---|---|
| `embeddings.backend` | auto | sentence-transformer if downloaded, else hybrid (WordLlama + TF-IDF), else TF-IDF |
| `embeddings.pretrained` | wordllama | "none" turns the pretrained layer off |
| `embeddings.hybrid_pretrained_weight` | 0.3 | pretrained share of the hybrid alignment cosine |
| `alignment.semantic_temperature` | 0.05 | sharpness of semantic soft evidence |
| `temporal.bayes_half_lives` | [6, 3, 1.5] | recency discounts tried by Bayesian recurrence (plus none) |
| `models.min_train_exams` | 1 | first held-out paper (1 = every paper after the first) |
| `models.course_prior_precision` | 2.0 | pull of the course logistic toward the general model |
| `models.use_simulated_prior` | true | false = no simulated knowledge; general model only from other real courses |
| `ensemble.skill_temperature` | 2.0 | how strongly measured skill moves weights |
| `ensemble.skill_prior_folds` | 3 | folds of evidence before skill counts fully |
| `ensemble.min_metric_sd` | 0.05 | floor on the fold spread used to scale skill |
| `ensemble.gate_strength` | 1.0 | per-topic shift toward outside knowledge for topics with little history |
| `ensemble.replace_confidence` | 0.95 | `best_single` replaces the ensemble only beyond this one-sided t-bound |
| `calibration.methods` | platt, isotonic | calibrators compared by nested log loss |
| `calibration.band_low` / `band_high` | 0.10 / 0.90 | percentiles of bootstrap calibrators for the probability band |

Old settings files that contain `models.sufficiency`, `calibration.min_rows`, `calibration.min_positives`,
`ensemble.replace_if_worse_by_se`, `models.selection_rule`, `models.logistic_C` or
`models.ensemble_max_members` load normally; those keys are ignored. `ensemble.replace_if_worse_by_se` held the
old one-standard-error replacement rule, which `ensemble.replace_confidence` replaced.

Rebuild the shipped general model with `python -m predictor.inference.generic` (deterministic, about 12 seconds; it reproduces the shipped file exactly).
