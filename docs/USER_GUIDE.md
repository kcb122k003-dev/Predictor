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

**Optional neural model.** The app ships with an offline text-similarity model that needs no
download. If you want a pretrained neural model for matching reworded questions, run once:

```bash
scripts/install.sh --neural
.venv/bin/predictor models download
```

This is the only step that uses the internet. Afterwards the app picks the model up automatically.

## 2. First launch

Run `scripts/start.sh` (or double-click `scripts\start.bat`). Your browser opens
`http://127.0.0.1:8765`. The address starts with 127.0.0.1, which means the app talks only to
your own computer.

Click **Load the synthetic demo course** to see a finished example. Its 12 papers are generated
(not real exams) and contain planted patterns: topics that alternate, topics that appear every
third paper, topics that became popular recently, and questions from an old syllabus.

Your data lives in `~/ExamPredictorData` (change it with `predictor --data-dir PATH serve`).
Back up that folder to keep your courses.

## 3. Creating a course

On the first screen, enter a course name (for example *Chemical Engineering Thermodynamics*) and
click **Create course**. Every course is independent. Papers from one course never influence
another course's predictions.

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

Open **5. Analyze & Predict** and press **Analyze & Predict**. A progress bar shows each step. A
12-paper course takes a few seconds.

What happens:

1. Every question is mapped to the syllabus and classified by type.
2. Repeated and reworded questions across years are linked.
3. The app builds a table of which topics appeared in which paper, with marks and formats.
4. **Backtesting.** About fifteen prediction methods each predict every past paper using only the
   papers before it. For 12 papers, papers 4 to 12 are predicted this way. This mirrors your real
   situation: predicting a paper you have not seen.
5. **Selection.** The method with the best average score is found, then the simplest method whose
   score is within one standard error of it is chosen. Small differences on a handful of papers
   are not trusted.
6. **Calibration.** If the held-out results support it, scores are converted into probabilities.
7. Explanations, question formulations, structure analysis and practice papers are produced.

Complex methods (logistic regression, random forests, gradient boosting, a hidden Markov model)
switch off automatically when there are not enough papers to train them. The Model performance
page lists each one with the reason. With only a few papers you will see:

> Advanced ML disabled because the historical sample is too small for reliable training.

That is the intended behaviour.

## 10. Understanding the predictions

**Summary cards** show how many papers were analysed, how many questions fell inside the syllabus,
which method was used and whether probabilities are calibrated. **Why this method** gives the
selection reasoning in plain language.

**Priority categories**

| Category | Meaning |
|---|---|
| Extremely High Priority | high probability (70% or more when calibrated) and at least three independent signals agree (frequency, recency, recurrence timing, semantic recurrence, marks) |
| High Priority | strong evidence with some uncertainty |
| Moderate Priority | some supporting evidence |
| Low Priority | weak evidence |
| Excluded | past questions outside the current syllabus |

**Percentages.** When the card says *Calibrated*, a topic shown at 70% appeared in about 70% of
similar cases in held-out papers, and the range (for example 66% to 74%) shows how precisely that
is known. When it says *Relative scores*, the numbers only order topics; they are not chances.

**Confidence** (High, Medium, Low) combines how many papers exist, how well the topic's questions
match the syllabus, and how wide the probability range is.

**Click a topic** to open its evidence:

* *Why this topic ranked here*: appearances in recent and all papers, when it last appeared, the
  gaps between appearances, how often topics in this course come back after the same gap, typical
  marks, common formats, repeated questions and where it sits in the syllabus.
* *Why it is not ranked higher*: for example "Recently repeated: topics reappear in the very next
  exam only 30% of the time in this course". These notes are learned from your papers, not from
  a fixed rule.
* *Signal contributions*: how much each family of evidence pushed the topic up (green) or down
  (red) in the explanation model.
* *Predicted question formulations*: examples of how the topic could be asked, labelled
  **PREDICTED QUESTION FORMULATION**. They reuse your course's own phrasing and syllabus terms.
  Numerical questions are shown as past numerical patterns; the app never invents numbers.
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
* **Model performance**: every method's backtest scores with standard errors, the selection
  reasoning, per-paper results, calibration quality, the ablation study (which kinds of evidence
  actually helped), the effect of the syllabus filter, format-forecast accuracy and exact-question
  recurrence results.
* **Predicted papers**: hypothetical papers following the historical structure (number of
  questions, sub-parts, short-notes question, unit balance). Paper A uses the top topics; B and C
  are other plausible combinations. Click **Generate other combinations** for more, or **Print**.

## 12. Search

**Search** finds past questions and syllabus items by keyword and by meaning. Filter by year, unit
or question type. Searching "Bernoulli" lists every related past question and the syllabus items
that mention it.

## 13. Exporting results

On the predictions page:

* **PDF report**: summary, ranked topics with probabilities, evidence for the top topics, predicted
  formulations, why-not notes, model comparison, ablation and excluded questions.
* **Excel workbook**: sheets for predictions, predicted questions, models, backtest folds, evidence
  and excluded questions.
* **CSV**: the ranking or the predicted questions as a single table.
* **JSON**: everything, for your own analysis.

From the command line: `predictor export RUN_ID --format pdf --out report.pdf`.

## 14. Settings

| Setting | What it changes |
|---|---|
| Syllabus strictness | which mapping statuses count (see section 8) |
| Clearly-in / probably-in / outside thresholds | the match scores behind A, B and D |
| Top-K for evaluation | how many topics the backtest scores; `auto` uses the typical number per paper |
| Papers before the first backtest target | warm-up papers before backtesting starts |
| Model selection rule | `one_se` (prefer simpler when the gap is within noise) or `best` |
| Recency half-life | how fast older papers lose weight in the Bayesian rate model |
| OCR language, resolution, page segmentation, deskew | text extraction from scans |
| Embedding backend | `auto`, the offline TF-IDF model, or the downloaded neural model |

Settings apply to the next analysis. All defaults and their explanations are in
`predictor/config/default.toml`.

## 15. Troubleshooting

| Problem | What to do |
|---|---|
| "OCR unavailable" on a scanned file | install Tesseract (section 1) and click **Reprocess** |
| Many "check pages" warnings | open the file's **Text** view; if the scan is poor, rescan at 300 dpi or type the questions in Review |
| A paper has no year | enter it in Review; the paper is then included |
| Questions merged or split wrongly | use Split and Merge in Review |
| A question is mapped to the wrong topic | click **Change** in Topic mapping; add an alias to the topic so similar wording maps correctly next time |
| Everything is "Relative scores" | calibration needs enough held-out papers; add more past papers |
| "Backtesting needs at least 4 exams" | add more papers; with 3 or fewer, only descriptive statistics are possible |
| The browser did not open | go to <http://127.0.0.1:8765> yourself |
| Port already in use | `predictor serve --port 8800` |

Logs are written to `~/ExamPredictorData/logs/predictor.log` (one JSON object per line).
