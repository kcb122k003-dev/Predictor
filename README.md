# Exam Predictor

A desktop application that ranks which syllabus topics, concepts and question forms are most
likely to appear in your next exam, based on past papers and the official course contents.

It runs entirely on your computer. Papers never leave the machine, there is no telemetry, and
nothing needs an internet connection after installation.

> Exam prediction is probabilistic. The app ranks topics by historical evidence; it cannot
> guarantee what the next paper will contain.

## What it does

1. **Reads your files.** Digital and scanned PDFs, Word files, photos and text. Scanned pages
   go through OCR with deskewing and contrast cleanup. Identical files, and the same paper
   uploaded twice in different formats, are detected so nothing is counted twice.
2. **Builds structured papers.** Year (Gregorian or Bikram Sambat), session, exam type, sections,
   questions, sub-questions, marks, OR alternatives, "attempt any N" instructions, MCQ options and
   question types. Every field can be corrected by hand.
3. **Treats the syllabus as a hard boundary.** Each question is matched to the course contents
   and labelled A (clearly in), B (probably in), C (uncertain) or D (outside). Questions outside
   the current syllabus never drive a prediction; they are listed separately with the reason.
4. **Tests prediction methods on your own history.** Each past paper is predicted using only the
   papers before it. About fifteen methods compete, from plain frequency to Bayesian, survival,
   Markov, logistic regression, gradient boosting, an HMM and an ensemble. Complex methods are
   switched off when there is too little data, and the simplest method within one standard error
   of the best is used.
5. **Explains every prediction.** Recent and total appearances, gaps, how often topics in this
   course reappear after a given gap, marks, formats, signal contributions, and "why not" notes
   for low-ranked topics. Percentages are shown as probabilities only when calibration was
   validated on held-out papers.
6. **Suggests grounded question formulations and practice papers**, built from your course's own
   wording and syllabus phrases. It never invents numbers for numerical questions.

## Quick start

Requirements: Python 3.10 or newer, about 1 GB of disk, 4 GB of RAM. Tesseract OCR is needed
only for scanned papers and photos.

**Linux and macOS**

```bash
scripts/install.sh      # creates .venv and installs everything
scripts/start.sh        # opens http://127.0.0.1:8765 in your browser
```

**Windows**: double-click `scripts\install.bat`, then `scripts\start.bat`.

**Manual install**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[ocr]"
predictor serve
```

To see the whole workflow before using your own files, click **Load the synthetic demo course**
on the first screen (or run `predictor demo`). The demo is generated data with planted patterns,
clearly labelled as such.

Install Tesseract for scanned papers: `sudo apt install tesseract-ocr` (Ubuntu/Debian),
`brew install tesseract` (macOS) or the installer from
<https://github.com/UB-Mannheim/tesseract/wiki> (Windows). Run `predictor doctor` to check.

## What results look like on the demo course

12 synthetic papers, 144 questions, 25 syllabus topics, analysed in about 7 seconds on a
4-core laptop:

| Method | NDCG@11 on 9 held-out papers |
|---|---:|
| Pooled hazard (time since last appearance) | 0.729 |
| Semantic soft recurrence | 0.716 |
| Ensemble | 0.713 |
| **Recent-window frequency (selected: simpler, within one standard error)** | **0.712** |
| Most frequent topics | 0.698 |
| Logistic regression | 0.653 |
| Random selection (exact expectation) | 0.461 |

* Syllabus alignment put 128 of 140 in-syllabus questions (91.4%) on the right topic with the
  offline model; all 4 questions from an old syllabus were marked "outside" and none of the 140
  in-syllabus questions was.
* The ablation study shows extra features did not help the logistic model on 12 papers. The app
  reports that instead of hiding it, and does not select the logistic model.
* Calibrated probabilities beat the base rate on held-out papers (Brier 0.199 vs 0.251).

Your own course will give different numbers; the Model performance page shows them.

## Documentation

* [User guide](docs/USER_GUIDE.md): installation, every screen, reading the results, troubleshooting.
* [Architecture and design analysis](docs/ARCHITECTURE.md): the specification review, contradictions
  and how they were resolved, data-size limits, pipeline, schema, backtesting, success metrics.
* [Developer guide](docs/DEVELOPER_GUIDE.md): code map, adding models, file formats or question
  types, tests, API.

## Command line

```text
predictor serve [--port 8765] [--no-browser]   start the app
predictor demo                                 create and analyse the demo course
predictor doctor                               check the installation
predictor ingest FILE... --kind exam|syllabus [--course ID]
predictor analyze COURSE_ID
predictor export RUN_ID --format pdf|xlsx|csv|json [--out FILE]
predictor models download                      optional neural embeddings (uses the internet once)
```

Add `--data-dir PATH` before the command to use another data folder (default `~/ExamPredictorData`).

## Privacy

* The server listens on 127.0.0.1 only.
* Uploaded files, the database and logs live in your data folder.
* No telemetry, no cloud OCR, no remote models. `allow_external_services` is off and nothing in the
  app uses the network. Downloading the optional neural model is a separate command you run yourself.

## Tests

```bash
pip install -e ".[ocr,dev]"
pytest                    # unit, integration and regression tests (about 1-2 minutes)
```

## Known limitations

* OCR reads printed text well, equations poorly and handwriting badly. Such regions are flagged for
  review; fix them in the Review screen.
* With fewer than about 6 papers, every prediction is highly uncertain and the app says so.
* Predicted question formulations are examples of likely forms, not the exam's wording.
