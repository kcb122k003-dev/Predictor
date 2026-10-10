# User guide

This guide walks you through installing Exam Predictor, adding a course, and reading what it
tells you. You do not need any machine-learning background. Two screens matter most once a course
is analysed: the **Predictions** tab (section 10), which ranks the topics, and the **Syllabus
Explorer** (section 11), where you can open any unit or topic and see every past question on it.

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
topic labels, the final ranking reaches 0.710 ± 0.026 on the backtest, against 0.678 for simply
ranking topics by how often they appeared (the score is NDCG, where 1 is a perfect ranking). That
is a difference of +0.033 ± 0.019 over 11 held-out papers: the final ranking did better on 8 of
them and worse on 3. Scored against the topics the generator really planted, which you can never
know for a real course, the final ranking scores 0.731 and plain frequency 0.734, a difference of
-0.003 ± 0.016. So the demo shows no improvement over frequency on the planted topics. Part of the
gain against the app's labels comes from semantic evidence, which shares the mapping's systematic
choices. Gains over frequency show up mainly in backtests on simulated courses, and they are small
when you score only the next unseen paper.
[docs/LOW_DATA_INFERENCE.md](LOW_DATA_INFERENCE.md) has the full measurements.

These numbers come from the current engine (version 2.1). Earlier versions of this guide quoted
0.717 for the final ranking and +0.004 against the planted topics. The scores moved because the
app no longer links a short-note title to a longer question just because the title's words
appear inside it, and that link feeds one of the model's inputs. The Syllabus Explorer and the
redesigned Predictions tab change how results are shown. They do not change the ranking or its
accuracy.

Your data lives in `~/ExamPredictorData` (change it with `predictor --data-dir PATH serve`).
Back up that folder to keep your courses.

If you used an earlier version, the app upgrades your database when it starts. It only adds new
columns and tables; nothing is deleted. Run the analysis again after an upgrade, because the topic
history, format guides and priority reasons are written when the analysis runs. A run made before
change tracking existed is not marked out of date: the Syllabus Explorer shows a note asking you to
re-run it and has no counts until you do. From this version on, a run made by an older engine
version is shown as out of date ("Results out of date ... the app was updated (new counts, format
guides or wording)") with a **Re-analyse now** button. A very old run, made before rank ranges and
evidence strength existed, shows a different banner on the Predictions tab: "These results were
produced by an earlier version of the engine. Press Analyze & Predict ..."

## 3. Creating a course

On the first screen, enter a course name (for example *Chemical Engineering Thermodynamics*) and
click **Create course**. Each course keeps its own papers, statistics and predictions. One course's
papers never enter another course's topic counts, Bayesian estimates or features.

The one shared part is the general ranking model (section 9). After each analysis of a real course,
the app stores that course's pattern rows: scale-free numbers such as recency rates and gaps, with
no topic names or raw counts. The general model used for any other course is the shipped model
updated with these rows. A course never learns from its own rows this way, and courses marked as
synthetic, such as the demo, are never used. If you switch off the simulated prior (section 15),
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

To see the same tree with each item's exam history, past questions and prediction, open the
**Syllabus Explorer** (section 11).

## 8. Topic mapping (step 4)

After an analysis, **4. Topic mapping** lists every question with its syllabus status:

| Status | Meaning | Counts for predictions? |
|---|---|---|
| A clearly in | strong match in meaning and shared terms | yes |
| B probably in | good match with some ambiguity | yes, unless strictness is high |
| C uncertain | weak evidence either way | only with strictness 0.25 or lower |
| D outside | no syllabus topic matches, or most technical terms are absent from the syllabus | never |

Click a row to see why: the matched syllabus text, shared terms, unknown terms and the similarity
scores. Click **Change** to set the topic and status yourself. Manual choices always win. You can
also correct a mapping from any past question in the Syllabus Explorer or in a topic's panel on
the Predictions tab, with **Correct mapping** (section 11). If you mark a question as outside the
syllabus, the next analysis keeps it out of every topic's counts. Older versions ignored that mark.

The **syllabus strictness** setting (Settings page) controls which statuses count: 0 is lenient,
0.5 (default) counts A and validated B, 1 counts only A.

## 9. Running the analysis (step 5)

Open **5. Analyze & Predict** and press **Analyze & Predict**. A progress bar shows each step. The
12-paper demo course takes about 10 seconds on a 4-core computer, mostly spent refitting the tree
models for every past paper.

What happens:

1. Every question is mapped to the syllabus and classified by type. By default the mapping combines
   the bundled pretrained model with word matching fitted on your syllabus (section 15).
2. Repeated and reworded questions across years are linked. A short-note title is not linked to a
   longer question just because its few words appear inside that question.
3. The app builds a table of which topics appeared in which paper, with marks and formats. A paper
   counts once for timing statistics, however many questions it has. The question text is used
   separately, as language evidence. From the same counted questions it records each topic's
   history (papers, questions, formats, repeats). The Predictions tab and the Syllabus Explorer
   both read this record, so they always show the same numbers.
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
9. Explanations, a one-line priority reason and a question format guide for every topic,
   practice questions (each checked before it is shown), structure analysis and practice papers
   are produced.

Nothing switches off because a course is small. There is no minimum number of papers for
backtesting or for any model: every component runs on every held-out paper, and the evidence
behind it sets its weight. A component is marked UNAVAILABLE only when an input it needs does not
exist, for example co-occurrence before two consecutive papers exist. You can even analyse a course
that has a syllabus and no past papers yet (section 16). Until you have more papers than the models
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
strength of the topics in the top half of the ranking) and the **rank uncertainty** (the most
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

If your data changed after the analysis ran, a **Results out of date** banner sits at the top of
the page. Section 11 explains it.

**Priority categories**

| Category | Meaning |
|---|---|
| Very High Priority | high probability (70% or more when calibrated; with relative scores, a place in the top third of the typical number of topics per paper) and at least three independent signals agree (frequency, recency, recurrence timing, semantic recurrence, marks) |
| High Priority | strong evidence with some uncertainty |
| Moderate Priority | some supporting evidence |
| Low Priority | weak evidence |
| Excluded | past questions outside the current syllabus |

Very High Priority used to be called *Extremely High Priority*. The rule did not change, only the
name. Results from runs made before the rename are listed under Very High.

### Topic cards

Topics are listed under the four priority headings, best rank first. The number on the left of a
card is the topic's rank. Each card has five fields. Most values carry a small tag that says what
kind of value you are reading, such as *Historical fact* or *Model ranking* (the full list is in
section 11).

**Predicted topic.** The topic's name in the app. When your syllabus words it differently, the
card adds *Syllabus wording:* with the original text. Under it you see the unit or chapter the
topic belongs to. A unit whose name the app supplied, because the document gave none, is marked
"(name inferred)".

**Priority.** Very high, High, Moderate or Low, followed by the rank, for example "High rank 4 of
25". Both are a *Model ranking*: the topic's position among this course's topics.

**Likelihood.** This field shows one of two different things. Read the wording before you read the
number.

* When calibration passed (see *Percentages* below), you see a probability such as "34% chance of
  appearing (range 28-41%)", tagged *Statistical estimate*. The page then also says: "Each topic's
  probability is estimated separately; several topics appear in one paper, so they do not add up
  to 100%." On the demo the 25 topics' probabilities add up to about 11, close to the number of
  topics a demo paper covers.
* Otherwise you see a relative score such as "Relative score 0.83 (not a probability) ·
  Confidence Medium", tagged *Model ranking*. The score is the topic's position in this course's
  ranking on a 0 to 1 scale: 1 is the top topic, 0 the last. Use it only to order topics. A score
  of 0.83 does not mean an 83% chance.

Under the number you find the rank range, for example "Rank range: 2-9 of 25 (medium
uncertainty)", and a bar.

**Historical support.** Counted directly from your papers and tagged *Historical fact*, for
example "Appeared in 5 of 12 usable papers; 7 questions; last appeared 2024 Fall Regular." Usable
papers are the papers the analysis used: duplicates and papers you excluded are left out. A paper
with several questions on the topic counts once for papers and once per question for questions.
The line below names the most common past format, for example "Most common format: Numerical
problem, in 4 of 12 usable papers (4 of the 5 papers containing this topic)". When two formats
are tied, it says "Equally common formats" and names both. If the topic appeared only as the
second topic of questions that mainly tested something else, the line says so. A *Counts
provisional* badge means something behind the counts is uncertain, such as a low OCR confidence
page or mostly probable (status B) matches; the first reason is shown next to it.

A topic with no past question reads "No historical evidence in the 12 usable papers", followed by
"Not appearing before is not evidence that it will not be examined." Its priority comes from the
syllabus structure, related questions and patterns learned across courses.

**General question format.** The way the topic is most likely to be asked, as one of nine format
families: Numerical problem, Derivation or proof, Conceptual explanation, Definition or statement,
Diagram-based explanation, Compare and contrast, Design or analytical problem, Short note and
Objective question. Under the name is a description built from the topic's own most recent
question in that format, for example "Calculate the head loss due to friction in a pipe from the
given diameter (mm), length (m), velocity (m/s) and friction factor." The suggested format is the
one asked in the most papers containing the topic. A tie goes to the format with more questions,
then to the one asked more recently. The tag next to the name tells you how solid the suggestion
is:

| Tag | When it is used |
|---|---|
| Historical fact | the topic was asked in this format in at least 2 papers through clear (status A) matches or mappings you set |
| Weak evidence | only one paper asked it this way, or the papers rest on probable (status B) matches |
| Inferred | the topic has no past question, so the format comes from words in its syllabus entry (such as "derivation" or "numerical") or, failing that, from the most common format across the course |

**Question format details.** Click this line under the fields to open it. Its summary already
shows the first line of the practice question. Inside you find:

* **The practice question.** One of three things:
  * *Illustrative practice question - not a real exam question*, tagged *Illustrative*. The app
    wrote it from your syllabus wording and your course's own phrasing. It is shown only after it
    passes four checks: its words come from the topic's syllabus text or past questions, it maps
    back to the intended topic, the pretrained model places it close to that topic, and its
    question type matches the format. On the demo these checks ran on 79 candidates, generated
    questions and past numericals offered for reuse: 4 failed the topic check and 3 the semantic
    check. The 3 that failed both were past numericals offered for a topic they do not fit, so 1
    generated question was rejected. It is an example of the format, never a prediction of the
    exact wording, and some read a little stiffly because they are built from templates.
  * *Past exam question to practise (paper, question)*, tagged *Historical fact*. This is what you
    get for numerical problems (and design or analytical problems); the next paragraph explains
    why.
  * A note saying why there is none, for example "No practice question in this format passed the
    grounding checks; use the description and the past questions instead."
* **General template** (numerical and design problems). The question's structure with every
  value replaced by a named placeholder in square brackets, for example "Calculate the head loss
  due to friction in a pipe of [diameter in mm] and [length in m] carrying water at [velocity in
  m/s]. Take the friction factor as [friction factor]." A *Fill in* line lists each placeholder
  with its unit and the value the past paper used.
* **Why this format.** The counts behind the suggestion: in how many of the papers containing the
  topic it was asked this way, when it was last asked this way, and how often the next format
  appeared. When questions combine formats (a derivation with a sketch, say), it tells you the
  counts overlap.
* **Marks.** What this topic's past questions in this format carried.
* **Reliability.** How often this kind of suggestion was right on your own earlier papers. On the
  11 earlier demo papers, the format a topic was asked in most often before matched a format it
  was actually asked in 37% of the time. The course-wide mix of formats did slightly better, at
  40%. Treat the suggested format as a guide, not a certainty.
* **Alternative formats.** Up to two other formats, each with its support, for example "asked in
  2 of the 5 papers containing this topic". For a topic with no history, an alternative can come
  from the syllabus entry and is marked as inferred.

**Why numerical topics show a past question.** For a numerical problem the app does not make up
new values. Invented numbers can give a problem that has no solution or contradicts itself, and the
app has no way to check that the numbers it picked are consistent. So it shows a real past question
on the topic, word for word, labelled with its paper and question number, for example "Past exam
question to practise (2025 Fall Regular 4(b))". You know that one can be solved. A new paper will
ask something similar with other values, and the general template shows which values to expect. The
past question must pass the same topic check as a generated one. If its link to the topic is only a
probable (status B) match, a *Weak evidence* note says so: check it under Past questions and
correct the mapping if it belongs elsewhere. On the demo, 15 topics have a generated practice
question, 6 have a past numerical question and 4 have none (each says why). Six topics show a
numerical template.

**Open a topic.** Click the topic name, or anywhere on the card outside the format details block,
to open the topic's panel in a drawer. It is the same panel as in the Syllabus Explorer: the
history in your papers, every past question in its original wording, the formats, the format guide,
how the model ranked it and the syllabus source (section 11 describes each part). **Open in
Syllabus Explorer** takes you to the same topic there. If you correct a mapping in the drawer, the
Predictions page shows the out-of-date banner as well.

### Reading the numbers

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

The label differs by topic. A topic that was absent from many papers can have strong evidence: its
past rate is then well known to be low. That describes your past papers only. It does not mean the
topic will not be examined. You meet this measure in three places: the Syllabus Explorer's *Strong
historical evidence* filter, the exports, and the *Why it ranked here* lines of a topic's panel,
which call it the precision of the recurrence rate (high, moderate, low or very low). The
*Evidence:* line there says in how many papers, and with how many questions, the topic itself
appeared, for example "Sparse historical evidence (2 of 4 papers, 3 questions) + strong syllabus
match". A topic seen in one paper is always called very sparse, and one seen in two papers at most
sparse, however precise the estimate. A topic with no past question reads "No historical evidence
(not found in any of the 12 usable papers)". Earlier versions called such topics "rarely tested".

A high score does not mean high certainty. A topic can rank near the top and still have a wide rank
range. Check the rank range and evidence strength before you skip anything.

**Percentages.** Percentages are probabilities only when the summary card says *Calibrated*. The
app tests two calibration methods (Platt scaling and isotonic regression): each is fitted on
earlier held-out papers and checked on later ones, and the method used for each later paper is the
one that did better on the papers before it. Calibration is accepted only if it predicts the later
papers better than the base rate (the overall share of topics that appear) beyond a one-sided 95%
bound: the average gain in Brier score must exceed its standard error times the t-value for that
many papers. With three papers or fewer it cannot pass, because it needs at least two later papers
to check against. On the demo it passes, with Platt scaling chosen: Brier 0.193 against 0.248 for
the base rate over 10 later papers (lower is better). When the card says *Calibrated*, a topic
shown at 70% appeared in about 70% of similar cases in held-out papers, and the range (for example
66% to 74%) shows how precisely that is known. Each topic's probability is estimated on its own,
and one paper covers several topics, so the percentages of all topics do not add up to 100%. When
the card says *Relative scores*, the number is the topic's position in the ranking on a
0 to 1 scale. Use it to order topics only: 0.80 does not mean an 80% chance.

**Confidence** (High, Medium, Low) appears next to a relative score. It combines the topic's rank
uncertainty, how well its questions match the syllabus and, when calibrated, how wide the
probability range is. A weak syllabus match or a High rank uncertainty gives Low confidence. There
is no rule based on the number of papers: few papers show up as wide rank ranges, and those lower
the confidence.

**How the model ranked it.** In a topic's panel, under the likelihood and the one-line *Why this
priority* sentence, a collapsed block tagged *Model ranking* holds the model's own explanation.
Click it to open:

* *Why it ranked here*: appearances in recent and all papers, when it last appeared, the gaps
  between appearances, how often topics in this course come back after the same gap, marks on the
  topic per paper, repeated questions and where it sits in the syllabus. Then the *Evidence:*
  summary, the recurrence rate, the temporal model used, the most similar past question, the
  largest component contributions and the rank range. Question formats are not repeated here: the
  format guide is the only place that states them. Older versions listed a "Most common question
  format" line and a "Predicted format if it appears" line that could contradict a topic's own
  history; both are gone.
* *Why it is not ranked higher*: for example "Recently repeated: it was in the last paper, and in
  this course topics reappear in the very next paper only 30% of the time", followed by the course
  average, or a note that the topic's rank is highly uncertain.
  These notes are learned from your papers, not from a fixed rule.
* *Model contributions*: each component's share of the final score (its ensemble weight times the
  topic's position in that component's ranking, on a 0 to 1 scale). The shares add up to the
  topic's score. This section appears when the ensemble produced the final ranking.
* *Historical recurrence rate*, tagged *Model input*: how often the topic came up in your papers,
  smoothed by the Bayesian model (posterior mean and median), its 80% credible interval, how often
  the topic was observed, how much of the rate comes from the prior, the effective sample size and
  the recency discount chosen on earlier papers. One appearance in two papers and eight in sixteen
  give similar rates but very different intervals. This rate is one input to the ranking. It is
  not the topic's chance of appearing (that is the Likelihood), and the calibration test does not
  check it.
* *Temporal pattern*: whether the gap-based model or the simpler Bayesian recurrence rate was
  used, with the log Bayes factor from earlier papers (a positive value favours the gap-based
  model).
* *Semantic evidence*: the past in-syllabus questions closest in meaning to the topic according to
  the pretrained model, with their similarity and whether they were mapped to this topic.
* *Signal contributions* (collapsed; click to open): how much each family of signals pushed the
  topic up (green) or down (red) in the course-specific logistic model. Related signals can carry
  large opposite values, so read their sum. The model contributions above are the main
  explanation.

**Concept level and exact questions.** Topic predictions are the main output. Predicting the exact
wording is much harder; the Model performance page shows how well recurring question families were
predicted in backtests so you can judge that layer separately.

## 11. Syllabus Explorer

The Syllabus Explorer lets you browse your course contents and see, for any unit or topic, how
often it was examined, every past question on it in its original wording, the formats it was asked
in, a suggested question format and where it sits in the syllabus. It reads the syllabus and papers
you already uploaded, so there is nothing new to upload. Its counts come from the same record as
the Predictions tab, so the two screens always agree.

### Opening it

Click **Syllabus Explorer** in the top bar (next to Courses, Settings and Help) and choose a course.
Each course card shows how many syllabus items and included papers it has, when it was last
analysed, and a badge: *analysed*, *results out of date* or *no counts yet*. You can also open it
from a course's sidebar, where it is the first entry below the five numbered steps.

Every item has its own address, so you can bookmark a topic. For example
`http://127.0.0.1:8765/#/course/1/explorer/21` opens item 21 of course 1.

If the course has not been analysed yet, you can still browse the syllabus. The counts, past
questions and predictions appear after you run **Analyze & Predict**.

The line under the heading summarises the papers: how many are usable, how many were excluded and
which calendar years have no paper.

### The syllabus tree

The left pane shows the syllabus as the app read it from your documents: headings and the entries
under them, at any depth. A heading is called a Unit or a Chapter only when your document calls it
that. When the app had to supply a name, the row is marked *inferred name*. On the demo, the
heading that reads "Practical:" in the document appears as "Laboratory / practical work", marked
inferred.

Each row shows:

* a coloured dot for the topic's priority (no dot when the item has no prediction),
* the number and title,
* the exam count, for example "5/12 papers",
* labels where they apply: *no past questions*, *inferred name*, *excluded* or *lab*.

Hover over a row for its priority, rank and paper count. Click the arrow to expand or collapse a
group, or use **Expand all** and **Collapse all**. A group's rows are built only when you open it,
so the tree stays fast: on the demo the overview loads in about 0.02 seconds and an item in about
0.03 seconds.

**Search.** Type in the search box to match titles, numbers, the original syllabus wording and the
concepts listed in an entry. Matching items stay visible together with the units above them,
which are dimmed. If a unit itself matches your search, all its topics are shown. The line
above the tree counts the matches, for example "8 of 25 topics match".

**Filters.** Open **Filters** to narrow the list. The heading shows how many are on, for example
"Filters (2 on)". **Clear filters** resets them.

| Filter | Options |
|---|---|
| Unit | one unit or group, or all units |
| History | *Appeared in each of the last 3 papers*; *Appeared in any of the last 3 papers*; *Never appeared in the supplied papers*; *Has past questions* |
| Format | topics asked in that format family in at least 2 papers, for example *Format: Numerical problem* |
| Priority | Very high, High, Moderate or Low |
| Evidence | *Strong historical evidence* (appeared in at least 3 papers, with strong or moderate evidence strength); *High model uncertainty* (rank uncertainty High) |

Search matches units and topics. The other filters describe topics, so they select topics only; a
unit stays on screen, dimmed, when one of its topics matches.

**Sort.** *Syllabus order* (the default) shows the tree. The other three show a flat list of
topics, each with its unit path underneath:

* *Historical frequency*: most papers first, then most questions.
* *Likelihood (rank)*: the model's ranking, rank 1 first.
* *Last appearance*: the most recently examined first; topics never examined show "never in the
  supplied papers".

### The topic panel

Click a row to open its panel on the right. The same panel opens in the drawer on the Predictions
tab. It has these parts, from top to bottom.

**Overview.** A breadcrumb shows where the item sits; click any level to go there. The title is
the original syllabus wording. When the app uses a shorter name, it is shown as "Short name in
this app", or "Name supplied by this app" when the name was inferred. Tags show the level (Unit,
Chapter, topic and so on), *lab or practical*, *excluded from predictions*, formats the syllabus
wording suggests (for example derivation or numerical) and teaching hours.

Then come the tiles:

| Tile | What it shows | Tag |
|---|---|---|
| Papers | "5 of 12 usable", with how many of the last 3 papers asked it | Historical fact |
| Questions | the number of counted questions, with how many treat it as a secondary topic | Historical fact |
| Last appeared | the most recent paper that asked it, or "Not in the supplied papers" | Historical fact |
| Typical marks | the marks range and median of questions with this as the main topic | Historical fact |
| Priority | Very high, High, Moderate or Low, with the rank | Model ranking |

Typical marks count only questions whose main topic this is. A question that touches the topic as
a secondary topic carries marks for its main topic too, so those marks are shown separately and
never as this topic's typical marks.

Below the tiles you find a *Counts provisional* badge when something behind the counts is
uncertain (click it for the reasons), the **Likelihood** block (the same probability or relative
score as the card, with the rank range), and **Why this priority**: one sentence built only from
recorded facts, for example "High priority (rank 4 of 25) because it appeared in 5 of 12 usable
papers, including 2 of the last 3 (most recently 2024 Fall Regular) and its past questions match
the syllabus entry strongly." For a topic that never appeared, the sentence says so and adds:
"Missing from past papers is not evidence that it will not be examined." Last is the collapsed
**How the model ranked it** block (section 10).

A sub-topic has its own counts but no prediction of its own: its panel names the priority and rank
of the topic it belongs to. A unit has no prediction either. Its panel lists its topics by model
ranking and has a table of its items with papers, questions, last appearance and priority. Its
counts combine its topics' questions, each question counted once.

**History in past papers.** A timeline with one row per paper, oldest first (papers without a
year at the top), tagged *Historical fact*:

| Column | Meaning |
|---|---|
| Year / paper | the paper's label; click it to list only that paper's questions below |
| Appeared? | Yes or No |
| Questions | the question numbers on this item, with "(secondary)" where it was the second topic |
| Formats | the format families of those questions |

Some rows are not ordinary papers. They are never shown as "No":

* **Missing years** (greyed out): a calendar year with no usable paper reads "No paper in the
  supplied data". The topic was not absent that year; you simply have no paper for it.
* **Excluded papers** (greyed out): "Paper excluded:" with the reason, for example a duplicate of
  another paper or one you excluded. An excluded paper without a year has no place in the
  timeline; the line under the table lists it.
* **Undated papers** that you included are ordinary rows marked "Undated: placed by upload order".

A row can also say *Provisional* with reasons, for example low parse confidence, a question flagged
for review, a page with low OCR confidence or an uncertain year. The line under the table repeats
the number of usable papers, the missing years and the excluded papers.

**How often it was asked.** Three counts, kept apart because they answer different questions:

* **Exam frequency**: papers with at least one counted question on the item, out of the usable
  papers. A paper with three questions on it counts once.
* **Question frequency**: counted questions, split into primary (the item is the question's main
  topic) and secondary (the question also covers it).
* **Repetition**: each question compared with the questions of earlier papers, never with the same
  paper. *Exact or near-exact repeat*: almost the same words. *Paraphrase*: the same question in
  other words. *Same concept*: tests the same idea in a different way; this needs a shared syllabus
  item finer than the topic, or similar wording, and a link between questions of different formats
  always counts here. *Different or new*: no close earlier question. The headline, for example "2
  of 7 questions repeated", counts exact repeats and paraphrases.

Read them together. "5 of 12 papers, 7 questions, 2 repeated" means the topic was asked in 5
papers, some papers asked it more than once, and 2 of the 7 questions closely repeated an earlier
one. A high exam frequency with many repeats suggests a topic whose standard questions are worth
practising. A high question frequency in few papers means it tends to come as several parts of one
paper.

A question counts for an item when its mapping status counts at your strictness (A, and B at the
default 0.5; section 8) and its topic is this item or lies inside it. The question's first topic
is its primary topic and the second its secondary topic.

**Question formats.** A table of format families with the number of questions and of papers
containing each, tagged *Historical fact*. One question can combine formats, so the counts can
overlap, and the page says so when they do. A sentence names the most common format, or the tied
formats when two are equal. A format seen in one paper only is called "seen once", never "most
common". The note "The most frequent past format is not certain to appear next time" is there for
a reason: see *Reliability* in section 10.

**Past questions.** Every counted question on the item, newest first, in its original wording.
Each shows:

* the paper, question number and marks,
* *primary topic* or *secondary topic*, the mapping status (A or B) and the match percentage,
* *choice question* when it was optional (an OR alternative),
* its format families, and the wording labels the app read from the question (for example
  Explain, Derive, Explain with an example),
* *As extracted* (click to open) when the stored text differs from the raw text read from the file,
* how it relates to earlier questions, for example "Paraphrase of an earlier question" with links
  to those questions,
* other topics it is also mapped to, the source file and page, and any provisional reasons,
* **View original** and **Correct mapping** buttons.

Two collapsed lists follow. *Possible matches not counted, uncertain mapping* holds questions that
might belong here but matched too weakly to count (status C), with their possible topics and match
scores. *Related questions mapped to other topics, not counted* shows the past questions closest in
meaning that count for another topic.

**Question format guide.** The suggested format, its description, why it was chosen, marks,
reliability, the template and practice question, and the other formats seen for the item. It reads
the same way as the format details on a prediction card (section 10).

**Syllabus source.** The file, page and line the item came from, with the extracted passage (the
heading line and the lines under it). Click **View source passage** to see it in place. The
section also lists the item's description, the concepts named in its entry, its sub-topics, the
average mapping confidence of its questions, and any ambiguity, such as uncertain matches. Concepts
listed inside one syllabus line are not separate items, so they have no counts of their own.

### Kind tags

Tags next to values tell you what kind of value you are reading. Help in the top bar has the same
list.

| Tag | Meaning |
|---|---|
| Historical fact | counted directly from the past papers you supplied, such as how many papers asked about a topic |
| Statistical estimate | estimated by a statistical model, with a range: the calibrated probability that the topic appears in the next paper |
| Model ranking | the topic's position among this course's topics (rank, priority, relative score); not a probability |
| Model input | a value the model uses to rank topics, such as the Bayesian recurrence rate; not the topic's chance of appearing |
| Illustrative | a generated practice question; it shows the form of a question, not the real exam wording |
| Inferred | not observed in your papers or documents: a format or a name the app inferred from the syllabus or course patterns |
| Weak evidence | observed, but too few times to establish a pattern |

### Viewing the original

**View original** on a past question opens the page it came from. For a PDF you see the rendered
page with the question marked on it, scrolled to the question; **Open the page full size** opens
the image in a new tab so you can zoom in. For a Word or text file you see the extracted page text
with the question's line marked. *Question text as stored* shows the text the app works with.

**View source passage** in the Syllabus source section does the same for the syllabus: the PDF page
with the passage marked, or the extracted page text with the line marked for Word and text files.

### Correcting a mapping

If a past question sits under the wrong topic, fix it from the question itself:

1. Click **Correct mapping** on the question (in the Past questions list or in the uncounted list).
2. Search for a topic by number or name. Your current topics are listed first.
3. Choose one **Primary** topic, and tick **Secondary** for any other topic the question covers.
   Or tick **Outside the syllabus (no topic)** if it does not belong to the course.
4. Pick a status: *A: clearly in the syllabus* or *B: probably in the syllabus*.
5. Click **Save**.

What happens next:

* The question shows your choice at once, marked *your correction, not yet counted*, with a line
  such as "Your correction: primary topic 4.1 Bernoulli's equation; status A. It counts from the
  next analysis; the counts on this page are still from the last one."
* The **Results out of date** banner appears. Counts, predictions and format guides change only
  after you re-analyse.
* The automatic mapping is kept. To undo your correction, open **Correct mapping** again and click
  **Use automatic mapping**. The question then shows *correction removed, not yet counted* until
  the next analysis.
* A question you mark as outside the syllabus is left out of every topic's counts from the next
  analysis on.

Corrections do not leak into the backtest. A correction is a label, and the matching that the
backtest uses for an earlier paper only learns from automatic matches in the papers before it. On
the demo, correcting a question in the last paper left every backtest result for earlier papers
unchanged (to 9 decimal places).

### The out-of-date banner

Each analysis stores a fingerprint of what it used: the papers (included or not, years, order), the
question text, marks, optional flags and types you set, your manual mappings, the syllabus and the
analysis settings. When any of these changes, or when the app was updated to a newer engine
version (now 2.1), both the Syllabus Explorer and the Predictions tab show:

> **Results out of date.** Since this analysis ran, topic mappings were corrected. The counts and
> predictions shown are from the earlier analysis; re-analyse to update them.

The sentence names every kind of change, for example "papers were added, removed, included or
excluded, or their year changed" or "the syllabus changed". Settings that do not affect the
analysis, such as folders, do not count. The app never recomputes behind your back: press
**Re-analyse now** when you are ready. In the explorer the analysis runs with a progress bar and
the page reloads when it finishes; on the Predictions tab it starts **Analyze & Predict**. Text
written during the analysis, such as format guides and priority reasons, also updates only then.

For a run made before change tracking existed, the explorer shows a note instead: "This analysis
was made before change tracking existed; re-run it to be sure the results match your current
data."

### Limits to keep in mind

* Counts depend on the automatic mapping. Probable (status B) matches count at the default
  strictness and are flagged. Check them, and correct the ones that are wrong.
* Past numerical questions are reused word for word for practice. The app does not invent values.
* Generated practice questions are built from templates filled with syllabus and past-question
  phrases. Some read stiffly, and on the demo 4 topics have none in their suggested format (a lab
  experiment, and three derivation topics for which no derivation built from their phrases passed
  the checks).
* Format suggestions are only as reliable as format history: on the demo the topic's most frequent
  past format matched the next appearance 37% of the time.
* Page images exist only for PDFs. Word and text files show the extracted text.

## 12. Analytics, models and practice papers

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
  Items marked "past numerical pattern" reuse a real past numerical question, for the reason given
  in section 10.

**Evidence and diagnostics.** This card on the Model performance page shows what the run is based
on:

* the evidence profile: papers (a paper without a year counts as half a paper of timing evidence),
  the years covered, in-syllabus questions, questions with marks, topics observed, topics per paper,
  question formats seen, where the syllabus weights come from, whether the pretrained model is
  available, how many of your other real courses the general model learned from, the overall
  evidence quality and the rank uncertainty;
* validation on held-out papers: the number of folds (held-out papers), the mean score with its
  standard error, a 95% interval (it needs two or more folds), the fold-to-fold spread, and
  comparisons with random selection, plain frequency, recency-frequency and Bayesian recurrence,
  including on how many papers the final ranking did better or worse;
* how the rank ranges were computed, and how many practice questions (generated ones and past numerical questions before reuse) were checked and how many failed each check;
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

The recall of exactly repeated questions is lower than in earlier versions (about 0.17 instead of
about 0.43 for the frequency-only stage on the demo). That is a change of definition, not lost
accuracy: repeat families are no longer chained through short-note titles, so each family is
smaller and harder to hit.

## 13. Search

**Search** finds past questions and syllabus items by keyword and by meaning. Filter by year, unit
or question type. Searching "Bernoulli" lists every related past question and the syllabus items
that mention it.

## 14. Exporting results

On the predictions page:

* **PDF report**: summary with the inference status and validation result, ranked topics with
  probabilities or relative scores, evidence strength and rank range, evidence for the top topics,
  practice questions (each labelled ILLUSTRATIVE PRACTICE QUESTION), why-not notes, component
  statuses and weights with the model comparison, ablation and excluded questions.
* **Excel workbook**: an About sheet, then sheets for predictions (including evidence strength, rank
  range, uncertainty level, the suggested format, its basis, its description, why this format and
  the priority reason), predicted questions (each labelled ILLUSTRATIVE PRACTICE QUESTION), models
  (with status, weight and reliability), backtest folds, evidence (including each component's share
  of the score) and excluded questions.
* **CSV**: the ranking or the predicted questions as a single table. The ranking has the same
  format columns (`suggested_format`, `format_basis`, `format_description`, `why_this_format`) and
  `priority_reason`.
* **JSON**: everything, including the evidence and diagnostics, for your own analysis.

From the command line: `predictor export RUN_ID --format pdf --out report.pdf`.

## 15. Settings

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

## 16. Troubleshooting

| Problem | What to do |
|---|---|
| "OCR unavailable" on a scanned file | install Tesseract (section 1) and click **Reprocess** |
| Many "check pages" warnings | open the file's **Text** view; if the scan is poor, rescan at 300 dpi or type the questions in Review |
| A paper has no year | enter it in Review; the paper is then included |
| Questions merged or split wrongly | use Split and Merge in Review |
| A question is mapped to the wrong topic | click **Change** in Topic mapping, or **Correct mapping** on the question in the Syllabus Explorer (section 11), then re-analyse; add an alias to the topic so similar wording maps correctly next time |
| "Results out of date" banner | your papers, questions, mappings, syllabus or analysis settings changed after the last analysis, or the app was updated. Press **Re-analyse now**; until you do, the counts and predictions are those of the last analysis (section 11) |
| You corrected a mapping but the counts did not change | corrections count from the next analysis. The question shows *your correction, not yet counted* until you re-analyse |
| A numerical topic shows a past question instead of a new practice question | this is deliberate: the app does not invent values it cannot check, so it shows a real past question word for word (section 10). Use the general template to see which values a new paper could change |
| A topic has no practice question | no generated question in its format passed the four checks; the card says so. Use the format description and the topic's past questions in the Syllabus Explorer instead |
| A topic shows "No historical evidence" | it did not appear in any usable paper. Its rank comes from the syllabus structure, related questions and patterns learned across courses. Not appearing before is not evidence that it will not be examined |
| A row in the explorer timeline says "No paper in the supplied data" | you have no usable paper for that year. Upload it if you have it; the topic was not counted as absent that year |
| "View original" shows text instead of a page image | page images exist only for PDF files; Word and text files show the extracted text with the question's line marked |
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
