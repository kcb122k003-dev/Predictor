# User guide

This guide walks you through installing Exam Predictor, adding a course, and reading what it
tells you. You do not need any machine-learning background.

> Exam prediction is probabilistic. Use the rankings to decide what to revise first, not to
> skip parts of the syllabus.

---

## 1. Installation

You need Python 3.10 or newer. Check with `python3 --version` (macOS/Linux) or `py --version`
(Windows).

**Linux and macOS**

```bash
cd Predictor
scripts/install.sh
```

**Windows**: open the `Predictor` folder and double-click `scripts\install.bat`.

The installer creates a private Python environment in `.venv`, installs the dependencies and runs
`predictor doctor`, which lists what is available.

**OCR for scanned papers.** Digital PDFs, Word files and text files work without OCR. For scans
and photos, install Tesseract:

| System | Command |
|---|---|
| Ubuntu / Debian | `sudo apt install tesseract-ocr` |
| Fedora | `sudo dnf install tesseract` |
| macOS (Homebrew) | `brew install tesseract` |
| Windows | installer from <https://github.com/UB-Mannheim/tesseract/wiki> |

For papers in another language, also install that Tesseract language pack and set
**Settings > OCR language(s)**, for example `eng+nep`.

**Pretrained language model (included).** The installer also installs `wordllama`, a normal
dependency like the others. It contains a small pretrained language model whose weights ship inside
the package, so nothing is downloaded when the app runs and it works without an internet
connection. The model knows general English: it can tell that two differently worded questions are
about the same thing, even when a course has only one past paper. It is used for every course,
whatever its size. `predictor doctor` reports it as "Pretrained semantic model: ready".

**Optional larger model.** If you want a larger sentence-transformer model, run once:

```bash
scripts/install.sh --neural
.venv/bin/predictor models download
```

Apart from installing the packages, this download is the only step that uses the internet.
Afterwards the app picks the model up automatically. You do not need it: the bundled model covers
general English, but it recognises technical rewording less well. With the larger model installed,
the app also uses it to detect reworded repeat questions. Without it, repeat detection uses
character matching: for 26 reworded test questions, the nearest past question it found was on the
same topic for 14, against 11 with the bundled model.

## 2. First launch

Run `scripts/start.sh` (or double-click `scripts\start.bat`). Your browser opens
`http://127.0.0.1:8765`. The address starts with 127.0.0.1, which means the app talks only to
your own computer.

Click **Load the synthetic demo course** to see a finished example. Its 12 papers are generated
(not real exams) and contain planted patterns: topics that alternate, topics that appear every
third paper, topics that became popular recently, and questions from an old syllabus. The demo
course is marked as synthetic, so its papers are never used to train the cross-course model
described in section 3.

The demo also shows the limits of what a 12-paper course can prove. Scored against the app's own
topic labels, the final ranking reaches 0.717 on the backtest, against 0.678 for simply ranking
topics by how often they appeared (the score is NDCG, where 1 is a perfect ranking). That is a
difference of +0.039 ± 0.019 over 11 held-out papers. Scored against the topics the generator
really planted, which you can never know for a real course, the final ranking scores 0.738 and
plain frequency 0.734, a difference of +0.004 ± 0.017. So the demo shows no improvement over
frequency on the planted topics. Part of the gain against the app's labels comes from semantic
evidence, which shares the mapping's systematic choices. Gains over frequency show up mainly in
backtests on simulated courses, and they are small when you score only the next unseen paper.
[docs/LOW_DATA_INFERENCE.md](LOW_DATA_INFERENCE.md) has the full measurements.

Your data lives in `~/ExamPredictorData` (change it with `predictor --data-dir PATH serve`).
Back up that folder to keep your courses.

If you used an earlier version, the app upgrades your database when it starts. It only adds new
columns and tables; nothing is deleted. Run the analysis again to get the inference status, rank
ranges and evidence strength, which older runs do not have. Until you do, the predictions page of
an older run shows a banner: "These results were produced by an earlier version of the engine.
Press Analyze & Predict ..."

## 3. Creating a course

On the first screen, enter a course name (for example *Chemical Engineering Thermodynamics*) and
click **Create course**. Each course keeps its own papers, statistics and predictions. One course's
papers never enter another course's topic counts, Bayesian estimates or features.

The one shared part is the general ranking model (section 9). After each analysis of a real course,
the app stores that course's pattern rows: scale-free numbers such as recency rates and gaps, with
no topic names or raw counts. The general model used for any other course is the shipped model
updated with these rows. A course never learns from its own rows this way, and courses marked as
synthetic, such as the demo, are never used. If you switch off the simulated prior (section 14),
the general model is learned from these rows alone.

## 4. Uploading past papers

Open **1. Upload files** and drag your past papers onto **Past exam papers**, or click it to choose
files. You can add PDFs (digital or scanned), Word `.docx` files, images (`.jpg`, `.png`, `.tif`)
and text files, as many as you have.

The file list shows, for each file:

* how it was read (native text, or OCR with a confidence percentage),
* pages that look suspicious (low OCR confidence, garbled symbols, broken equations),
* the paper it became and how many questions were found,
* duplicates: an identical file is refused, and a paper that matches an already uploaded paper
  (for example the same paper as PDF and Word) is kept but excluded from analysis.

Old `.doc` files are not supported. Save them as `.docx` or PDF first.

## 5. Uploading course contents

Drag the syllabus, course outline or topic lists onto **Course contents**. Several documents are
merged into one topic tree; each topic remembers which file and page it came from.

If your course changed its syllabus, upload the old syllabus as **A historical syllabus** and give
it a name. Past questions that match only the old syllabus are then explained as such and kept out
of the predictions.

## 6. Reviewing papers (step 2)

Open **2. Review papers**.

**The paper table.** Check the year of each paper. The app reads Gregorian years (2019) and
Bikram Sambat years (2076, with months such as Chaitra or Baishakh), sessions (Spring, Fall, month
names) and exam types (regular, back, mid-term and so on).

* *Order* decides the time sequence. It is computed from year and session. When two papers share a
  year (a regular and a back exam), make sure the order matches the order they were set.
* Untick *Include* to leave a paper out (for example an internal test that differs from the final).
* A paper without a detected year is excluded until you enter one.

**The question tree.** Click a paper to see its questions: main questions, sub-parts, marks, OR
alternatives and question types. Items with a yellow border need a look. Use:

* **Edit** to fix text, marks or the question type,
* **Split** when two questions were read as one (put each part on its own line),
* **Merge with next** when one question was split in two,
* **Add sub-part** or **Add question** for anything the parser missed,
* **Looks right** to clear a review flag,
* **Show extracted text** to compare with the original page text.

Your edits are never overwritten when the analysis runs again.

## 7. The syllabus (step 3)

**3. Syllabus** shows the topic tree. Levels:

* **Unit**: a top-level chapter or module.
* **Topic**: the main prediction unit (usually the second level, such as "4.1 Bernoulli's equation").
* **Concept**: the finest items.

You can rename items, change their parent, add teaching hours, list **aliases** (other names past
papers use for the same thing, which improves matching), add missing items, or **Exclude** items
that are not examinable.

## 8. Topic mapping (step 4)

After an analysis, **4. Topic mapping** lists every question with its syllabus status:

| Status | Meaning | Counts for predictions? |
|---|---|---|
| A clearly in | strong match in meaning and shared terms | yes |
| B probably in | good match with some ambiguity | yes, unless strictness is high |
| C uncertain | weak evidence either way | only with strictness 0.25 or lower |
| D outside | no syllabus topic matches, or most technical terms are absent from the syllabus | never |

Click a row to see why: the matched syllabus text, shared terms, unknown terms and the similarity
scores. Click **Change** to set the topic and status yourself. Manual choices always win.

The **syllabus strictness** setting (Settings page) controls which statuses count: 0 is lenient,
0.5 (default) counts A and validated B, 1 counts only A.

## 9. Running the analysis (step 5)

Open **5. Analyze & Predict** and press **Analyze & Predict**. A progress bar shows each step. The
12-paper demo course takes about 9 seconds on a 4-core computer, mostly spent refitting the tree
models for every past paper.

What happens:

1. Every question is mapped to the syllabus and classified by type. By default the mapping combines
   the bundled pretrained model with word matching fitted on your syllabus (section 14).
2. Repeated and reworded questions across years are linked.
3. The app builds a table of which topics appeared in which paper, with marks and formats. A paper
   counts once for timing statistics, however many questions it has. The question text is used
   separately, as language evidence.
4. **Components.** Thirteen components score every topic, each from a different kind of evidence:
   * knowledge from outside your papers: a general ranking model trained on simulated courses (and
     updated with your other real courses, section 3), semantic evidence from the pretrained model,
     and the coverage of each syllabus unit;
   * your course's history: recency-frequency, hierarchical Bayesian recurrence, a two-state Markov
     model, a temporal model (it uses the gaps between appearances only when earlier papers show
     that this predicts better), topic co-occurrence and question-type fit;
   * models learned from your course: a course-specific logistic model pulled toward the general
     model, gradient boosting, a random forest and a pooled hot/cold model.
5. **Backtesting.** Every component predicts every past paper using only the papers before it. For
   12 papers, papers 2 to 12 are predicted this way (11 held-out papers). This mirrors your real
   situation: predicting a paper you have not seen. A course with one paper has nothing to test
   against yet.
6. **Evidence-aware ensemble.** The components are combined into one ranking. A component's weight
   depends on its reliability (the number of papers compared with the number of parameters it
   estimates from your course) and on how well it ranked your earlier papers. Topics with little
   history of their own lean more on the components that bring outside knowledge (the general
   model, semantic evidence and syllabus coverage). The app also backtests a
   simpler rival: for each held-out paper it uses whichever single method ranked the papers before
   it best. The ensemble is the final ranking unless that rival beat it on the same papers by a
   clear margin, beyond a one-sided 95% bound on the paired differences. Only then is a single
   method used (the one that ranked your past papers best), and the page says so.
7. **Calibration.** Two calibration methods are tested on the held-out papers. Scores become
   probabilities only if the calibrated scores beat the base rate on later papers beyond a
   one-sided 95% bound (section 10).
8. **Uncertainty.** Each topic gets a rank range and an evidence strength.
9. Explanations, question formulations (each checked before it is shown), structure analysis and
   practice papers are produced.

Nothing switches off because a course is small. There is no minimum number of papers for
backtesting or for any model: every component runs on every held-out paper, and the evidence
behind it sets its weight. A component is marked UNAVAILABLE only when an input it needs does not
exist, for example co-occurrence before two consecutive papers exist. You can even analyse a course
that has a syllabus and no past papers yet (section 15). Until you have more papers than the models
learned from your course have parameters to estimate (about 25 papers for a course like the demo),
those parameters stay uncertain, and the predictions page tells you so, for example:

> Only 5 historical examinations are available. Advanced semantic and Bayesian inference remains
> active, but course-specific learned ranking parameters have high uncertainty.

That is the intended behaviour. [docs/LOW_DATA_INFERENCE.md](LOW_DATA_INFERENCE.md) describes each
component, how the weights are computed and what the measurements show.

## 10. Understanding the predictions

**Inference status.** The card at the top names the mode: *Low-data advanced inference* while the
most reliable model learned from your course has a reliability below 0.5 (about 25 papers for a
course like the demo), and *Advanced inference* after that. The label is wording only: it switches
nothing on or off. Two badges show the overall **evidence quality** (the typical evidence
strength of the topics in the top half of the ranking) and the **prediction uncertainty** (the most
common rank uncertainty level among the topics). Under the message you find the validation summary
and notes about your data, such as papers without a year or missing calendar years. Open
**Components** to see every component with its kind (outside knowledge, syllabus structure or
estimated from this course), its status, its weight in the ensemble and the reason.

| Status | Meaning |
|---|---|
| ACTIVE | running, its inputs are available and its reliability is at least 0.5 |
| LIMITED | running and contributing, with high uncertainty because your course has little evidence for its parameters |
| DOWNWEIGHTED | running, but it ranked your earlier papers worse than the average component, so its weight is small |
| UNAVAILABLE | an input it needs does not exist for this course; the reason is shown |
| REFERENCE | a baseline kept for comparison on the Model performance page; not part of the final ranking |

**Summary cards** show how many papers were analysed, how many questions fell inside the syllabus,
which method produced the final ranking (normally the evidence-aware ensemble) and whether
percentages are calibrated. **Why this ranking** compares the ensemble on your held-out papers
with the best single method chosen on earlier papers, gives the standard error of the difference
and states the calibration result.

**Priority categories**

| Category | Meaning |
|---|---|
| Extremely High Priority | high probability (70% or more when calibrated; with relative scores, a place in the top third of the typical number of topics per paper) and at least three independent signals agree (frequency, recency, recurrence timing, semantic recurrence, marks) |
| High Priority | strong evidence with some uncertainty |
| Moderate Priority | some supporting evidence |
| Low Priority | weak evidence |
| Excluded | past questions outside the current syllabus |

**Topic cards.** Each card shows the topic's appearances in the most recent papers (up to six), its
**historical coverage** (papers that contain it out of all papers, and the number of questions
behind it, for example "5/12 papers, 7 questions"), when it last appeared, its likely format, marks
range and syllabus match, its rank range, its evidence strength and a confidence label.

**Rank range.** The positions the topic could plausibly take with the papers you have. The app
ranks the topics again with each past paper left out in turn, and again 200 times with the
Bayesian estimates and the ensemble weights redrawn from their uncertainty (the middle 80% of those
draws is used). The rank range covers both results. Its uncertainty level is Low when it spans less
than a fifth of the topics, Medium below two fifths and High otherwise. A topic whose estimate
still rests mostly on the prior is at least Medium, or High when its papers have narrowed the
estimate very little. A topic at rank 2 with a range of 1 to 9 could plausibly end up anywhere in
the top nine. With no past papers, every topic shows the full range and High uncertainty.

**Evidence strength** shows how much your papers have narrowed the Bayesian estimate for this
topic. The app compares the uncertainty left after seeing your papers (posterior variance) with the
uncertainty before (prior variance, from the rates of the topic's unit and of the whole course):

| Evidence strength | Uncertainty left, as a share of the prior uncertainty |
|---|---|
| Strong | under 30% |
| Moderate | 30% to 50% |
| Limited | 50% to 75% |
| Minimal | 75% or more |

The label differs by topic. A topic that was absent from many papers can have strong evidence: you
then know well that it is rarely tested. The one-line evidence summary in each topic's details says
in how many papers, and with how many questions, the topic itself appeared.

A high score does not mean high certainty. A topic can rank near the top and still have a wide rank
range. Check the rank range and evidence strength before you skip anything.

**Percentages.** Percentages are probabilities only when the card says *Calibrated*. The app tests
two calibration methods (Platt scaling and isotonic regression): each is fitted on earlier held-out
papers and checked on later ones, and the method used for each later paper is the one that did
better on the papers before it. Calibration is accepted only if it predicts the later papers better
than the base rate (the overall share of topics that appear) beyond a one-sided 95% bound: the
average gain in Brier score must exceed its standard error times the t-value for that many papers.
With three papers or fewer it cannot pass, because it needs at least two later papers to check
against. On the demo it passes: Brier 0.191 against 0.248 for the base rate over 10 later papers
(gain 0.057 ± 0.005). When the card says
*Calibrated*, a topic shown at 70% appeared in about 70% of similar cases in held-out papers, and
the range (for example 66% to 74%) shows how precisely that is known. When it says *Relative
scores*, the number is the topic's position in the ranking on a 0 to 1 scale. Use it to order
topics only: 0.80 does not mean an 80% chance.

**Confidence** (High, Medium, Low) combines the topic's rank uncertainty, how well its questions
match the syllabus and, when calibrated, how wide the probability range is. A weak syllabus match or
a High rank uncertainty gives Low confidence. There is no rule based on the number of papers: few
papers show up as wide rank ranges, and those lower the confidence.

**Click a topic** to open its evidence:

* The header repeats the category, confidence, score, rank, rank range and evidence strength,
  followed by a one-line summary such as "Sparse historical evidence (2 of 4 papers, 3 questions)
  + strong syllabus match".
* *Why this topic ranked here*: appearances in recent and all papers, when it last appeared, the
  gaps between appearances, how often topics in this course come back after the same gap, typical
  marks, common formats, repeated questions and where it sits in the syllabus. It ends with the
  Bayesian estimate, the temporal model used, the most similar past question, the largest
  component contributions and the rank range.
* *Why it is not ranked higher*: for example "Recently repeated: topics reappear in the very next
  exam only 30% of the time in this course", or a note that the topic's rank is highly uncertain.
  These notes are learned from your papers, not from a fixed rule.
* *Model contributions*: each component's share of the final score (its ensemble weight times the
  topic's position in that component's ranking, on a 0 to 1 scale). The shares add up to the
  topic's score. This section appears when the
  ensemble produced the final ranking.
* *Bayesian recurrence*: the chance of appearing (posterior mean and median), its 80% credible
  interval, how often the topic was observed, how much of the estimate comes from the prior, the
  effective sample size and the recency discount chosen on earlier papers. One appearance in two
  papers and eight in sixteen give similar chances but very different intervals. This chance is the
  Bayesian component's own estimate. It is shown even when the ranking uses relative scores, and the
  calibration test does not check it.
* *Temporal pattern*: whether the gap-based model or the simpler Bayesian recurrence rate was
  used, with the log Bayes factor from earlier papers (a positive value favours the gap-based
  model).
* *Semantic evidence*: the past in-syllabus questions closest in meaning to the topic according to
  the pretrained model, with their similarity and whether they were mapped to this topic.
* *Signal contributions* (collapsed; click to open): how much each family of signals pushed the topic up
  (green) or down (red) in the course-specific logistic model. Related signals can carry large
  opposite values, so read their sum. The model contributions above are the main explanation.
* *Predicted question formulations*: examples of how the topic could be asked, labelled
  **PREDICTED QUESTION FORMULATION**. They reuse your course's own phrasing and syllabus terms.
  Numerical questions are shown as past numerical patterns; the app never invents numbers. Each
  formulation passed four checks before it is shown: its words come from the topic's syllabus text
  or past questions, it maps back to the intended topic, the pretrained model places it close to
  that topic, and its question type matches the intended format. Formulations that fail are
  dropped.
* *Past questions on this topic*, with their papers, marks, types and mapping status.

**Concept level and exact questions.** Topic predictions are the main output. Predicting the exact
wording is much harder; the Model performance page shows how well recurring question families were
predicted in backtests so you can judge that layer separately.

## 11. Analytics, models and practice papers

* **Analytics**: timeline of topics by paper (click a cell to see its questions), frequency by unit,
  recency-weighted importance, exams since last appearance, gaps between appearances with a
  rotation test (a rotation is reported only when gaps are more regular than chance), co-occurring
  topics with a significance test, question formats over time, marks share by unit, and the
  historical paper structure.
* **Model performance**: the final ranking and why it was chosen, the leakage audit (later papers
  are scrambled to confirm they never change earlier predictions), the backtest scores of every
  component and baseline with standard errors, each one's status and ensemble weight, per-paper
  results, calibration quality (the held-out Brier score against the base rate, and a reliability
  chart), the ablation study, the effect of the syllabus filter, format-forecast accuracy,
  exact-question recurrence results and the evidence and diagnostics card.
* **Predicted papers**: hypothetical papers following the historical structure (number of
  questions, sub-parts, short-notes question, unit balance). Paper A uses the top topics; B and C
  are other plausible combinations. Click **Generate other combinations** for more, or **Print**.

**Evidence and diagnostics.** This card on the Model performance page shows what the run is based
on:

* the evidence profile: papers (a paper without a year counts as half a paper of timing evidence),
  the years covered, in-syllabus questions, questions with marks, topics observed, topics per paper,
  question formats seen, where the syllabus weights come from, whether the pretrained model is
  available, how many of your other real courses the general model learned from, the overall
  evidence quality and the prediction uncertainty;
* validation on held-out papers: the number of folds (held-out papers), the mean score with its
  standard error, a 95% interval (it needs two or more folds), the fold-to-fold spread, and
  comparisons with random selection, plain frequency, recency-frequency and Bayesian recurrence,
  including on how many papers the final ranking did better or worse;
* how the rank ranges were computed, and how many generated formulations were checked and rejected;
* every ensemble component with its status, weight, reliability, measured skill and the reason.

**Ablation table.** The ablation study rebuilds the ensemble in stages: frequency only, then
recency, Bayesian smoothing, semantic evidence, topic structure (coverage, question type,
co-occurrence), temporal dynamics and finally the full ensemble. Each row gives the ranking score
(NDCG), Hit@1, Hit@3, Hit@5, recall, precision, concept recall and recall of exactly repeated
questions, plus the paired change against the previous stage with its standard error. A change is
reliable only when it falls outside a two-sided 95% interval of the per-paper differences; with 11
held-out papers that means more than about 2.2 standard errors away from zero. Every other change
is labelled *not reliable*. The note above the table lists reliable gains and reliable losses
separately. With fewer than five held-out papers, it says the differences show direction only. A
second table removes one component at a time from the full ensemble. With a single paper there is
no held-out paper, so no ablation is shown. On the demo course, only adding semantic evidence is a
reliable gain (+0.052 ± 0.019). No stage is a reliable loss, and every other change is within the
noise.

## 12. Search

**Search** finds past questions and syllabus items by keyword and by meaning. Filter by year, unit
or question type. Searching "Bernoulli" lists every related past question and the syllabus items
that mention it.

## 13. Exporting results

On the predictions page:

* **PDF report**: summary with the inference status and validation result, ranked topics with
  probabilities or relative scores, evidence strength and rank range, evidence for the top topics,
  predicted formulations, why-not notes, component statuses and weights with the model comparison,
  ablation and excluded questions.
* **Excel workbook**: an About sheet, then sheets for predictions (including evidence strength, rank
  range and uncertainty level), predicted questions, models (with status, weight and reliability),
  backtest folds, evidence (including each component's share of the score) and excluded questions.
* **CSV**: the ranking or the predicted questions as a single table.
* **JSON**: everything, including the evidence and diagnostics, for your own analysis.

From the command line: `predictor export RUN_ID --format pdf --out report.pdf`.

## 14. Settings

| Setting | What it changes |
|---|---|
| Syllabus strictness | which mapping statuses count (see section 8) |
| Clearly-in / probably-in / outside thresholds | the match scores behind A, B and D for the default hybrid backend. The `tfidf` and `sentence-transformers` backends read their own thresholds (`alignment.thresholds.tfidf` and `alignment.thresholds.neural`), which are not on this page |
| Top-K for evaluation | how many topics the backtest scores; `auto` uses the typical number per paper |
| First held-out paper | how many papers come before the first held-out paper. The default, 1, tests every paper after the first; a larger value only removes held-out papers. No model needs a minimum |
| Default recency half-life (papers) | the default decay of the recency-frequency component (kept unless another decay is ahead on earlier papers beyond a one-sided 95% bound), the recency-weighted importance chart and the question-format mix. The Bayesian recurrence chooses its own discount on earlier papers |
| Ensemble skill temperature | how strongly a component's skill on earlier papers moves its weight (default 2) |
| Course model pull toward the general model | how closely the course-specific logistic model stays to the general model (`models.course_prior_precision`, default 2) |
| Use the simulated cross-course prior | on by default: the general ranking model starts from simulated examiner behaviour and is updated with your other real courses. Turn it off and no simulated knowledge is used. Until another real (not synthetic) course is in your library, the general model then shows UNAVAILABLE, the other components share its weight and the course-specific logistic model is no longer pulled toward it. Once other real courses exist, the general model is learned from them alone |
| OCR language, resolution, page segmentation, deskew | text extraction from scans |
| Embedding backend, neural model name | how questions are matched to the syllabus (see below) |

Settings files from older versions still load. Keys that no longer do anything, such as the old
model selection rule and logistic regularisation C, are ignored.

**Embedding backend**

| Option | What it does |
|---|---|
| `auto` (default) | uses a downloaded sentence-transformer if one is on disk, otherwise `hybrid`, otherwise `tfidf` |
| `hybrid` | combines the bundled pretrained model with word matching (TF-IDF) fitted on your syllabus: similarity = 0.3 times the pretrained similarity + 0.7 times the TF-IDF similarity. If the `wordllama` package is missing, the app uses `tfidf` and says so in the run |
| `tfidf` | word matching only, for mapping questions to the syllabus. The pretrained model still provides semantic evidence and checks generated questions |
| `sentence-transformers` | the larger model from section 1. If it is not downloaded, the app uses `hybrid` (or `tfidf`) and says so in the run |

On the demo course, `hybrid` and `tfidf` each put 131 of 140 in-syllabus questions on the right
topic, and both mark all four old-syllabus questions as outside (both measured with feedback from
earlier papers only). `tfidf` leaves 9 questions at status C (uncertain), against 6 for `hybrid`.
The hybrid is not shown to map more demo questions correctly. It adds general language knowledge
while keeping the separation between in-syllabus and out-of-syllabus questions that the
pretrained model alone does poorly.

**Settings that are not on the page.** Add them to `settings.json` in your data folder, using the
same names and nesting as `predictor/config/default.toml`, then restart the app. For example:

```json
{"ensemble": {"gate_strength": 1.0, "replace_confidence": 0.95}}
```

| Setting | Default | What it changes |
|---|---|---|
| `ensemble.skill_prior_folds` | 3 | how many held-out papers it takes before measured skill counts fully |
| `ensemble.min_metric_sd` | 0.05 | the smallest fold-to-fold spread used to scale measured skill |
| `ensemble.gate_strength` | 1.0 | how much more a topic with little history of its own leans on the pretrained, syllabus and cross-course components |
| `ensemble.replace_confidence` | 0.95 | the best single method chosen on earlier papers replaces the ensemble only if it beat it beyond this one-sided bound |
| `embeddings.pretrained` | `wordllama` | `"none"` turns the pretrained model off everywhere |
| `embeddings.hybrid_pretrained_weight` | 0.3 | the pretrained share of the hybrid similarity |
| `temporal.bayes_half_lives` | [6, 3, 1.5] | recency discounts the Bayesian recurrence tries (plus no discount) |
| `calibration.methods` | platt, isotonic | calibration methods compared on held-out papers |

Settings apply to the next analysis. All defaults and their explanations are in
`predictor/config/default.toml`, and [docs/LOW_DATA_INFERENCE.md](LOW_DATA_INFERENCE.md) explains
the ensemble settings in detail.

## 15. Troubleshooting

| Problem | What to do |
|---|---|
| "OCR unavailable" on a scanned file | install Tesseract (section 1) and click **Reprocess** |
| Many "check pages" warnings | open the file's **Text** view; if the scan is poor, rescan at 300 dpi or type the questions in Review |
| A paper has no year | enter it in Review; the paper is then included |
| Questions merged or split wrongly | use Split and Merge in Review |
| A question is mapped to the wrong topic | click **Change** in Topic mapping; add an alias to the topic so similar wording maps correctly next time |
| Everything is "Relative scores" | percentages become probabilities only when calibration beats the base rate on later held-out papers beyond a one-sided 95% bound (section 10). With three papers or fewer that cannot happen; with more, it depends on your papers. Relative scores still order the topics: read them with the rank range and evidence strength, and add older papers when you can |
| You have no past papers yet | you can still analyse the course with only a syllabus. The ranking comes from the general ranking model and the syllabus structure. Every topic shows the full rank range (1 to the number of topics) and High uncertainty, scores are relative, and the status card says: "No historical examinations are available yet. The ranking uses the syllabus structure and the general cross-course model; every topic has high uncertainty." If you switched off the simulated prior and no other real course is in your library, only the syllabus structure is left to rank the topics |
| You have only one past paper | the analysis still runs. There is no earlier paper to test a prediction against, so accuracy, calibration and the ablation study cannot be measured, and scores are relative. Every component except co-occurrence runs. No skill can be measured yet, so the weights come from reliability alone: most of the weight goes to the general ranking model, semantic evidence, recency-frequency, Bayesian recurrence (which with one paper leans mostly on its prior) and syllabus coverage, and the models learned from your course get very little. Co-occurrence is UNAVAILABLE until two consecutive papers exist |
| You have two to five papers | every component runs and is tested on one to four held-out papers. Weights come mostly from each component's reliability, the models learned from your course run with small weights (usually LIMITED), rank ranges are wide and the status card says "Low-data advanced inference". With one held-out paper the accuracy figure is anecdotal, and the 95% interval needs at least two. This is the expected behaviour |
| A component shows UNAVAILABLE | an input it needs does not exist for this course, and the *Why* column names it. For example, co-occurrence needs two consecutive papers, and question-type fit needs more than one question format or format tags in the syllabus |
| "The pretrained semantic model is unavailable" | the `wordllama` package is missing or damaged, or `embeddings.pretrained` is set to `"none"`. Run the installer again and check `predictor doctor`. Until then, mapping uses TF-IDF only |
| A paper you uploaded is not in the analysis | give it a year in Review and leave *Include* ticked |
| The browser did not open | go to <http://127.0.0.1:8765> yourself |
| Port already in use | `predictor serve --port 8800` |

Logs are written to `~/ExamPredictorData/logs/predictor.log` (one JSON object per line).
