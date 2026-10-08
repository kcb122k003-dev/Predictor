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
4. **Tests prediction methods on your own history.** Each past paper from the second onward is
   predicted using only the papers before it. Nothing switches off because a course is small:
   there is no minimum number of papers for testing or for any of the thirteen components. They
   include recency-frequency, hierarchical Bayesian recurrence, semantic
   evidence from a bundled pretrained model, syllabus coverage, a general ranking model trained on
   simulated courses, a course-specific logistic model pulled toward that general model, and
   Markov, HMM and tree models. An evidence-aware ensemble weights each one by how much evidence
   supports it and how well it ranked earlier papers. The app marks each component ACTIVE, LIMITED,
   DOWNWEIGHTED, UNAVAILABLE (an input it needs does not exist) or REFERENCE (a baseline kept for
   comparison). A single method replaces the ensemble only if it beat the ensemble on earlier
   papers by more than one standard error.
5. **Explains every prediction.** Recent and total appearances, gaps, how often topics in this
   course reappear after a given gap, marks, formats, signal contributions, and "why not" notes
   for low-ranked topics. Each topic also shows a rank range, an uncertainty level and an evidence
   strength, because a high score does not mean the app is certain. Percentages are shown as
   probabilities only when calibration beat the base rate on later held-out papers by more than one
   standard error; otherwise you see relative scores.
6. **Suggests grounded question formulations and practice papers**, built from your course's own
   wording and syllabus phrases. It never invents numbers for numerical questions. Topics are
   ranked first, and each formulation must pass syllabus, topic, semantic and question-type checks
   before it is shown.

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

12 synthetic papers, 144 questions, 25 syllabus topics, analysed in about 10 seconds on a
4-core machine. The scores below compare each method's ranking with the topics the app itself
mapped the held-out questions to, which is all a real course can measure:

| Method | NDCG@11 on 11 held-out papers |
|---|---:|
| General ranking model (cross-course) | 0.758 |
| **Evidence-aware ensemble (final ranking)** | **0.754** |
| Semantic evidence (pretrained) | 0.749 |
| Hierarchical Bayesian recurrence | 0.724 |
| Recent-window frequency | 0.720 |
| Most frequent topics | 0.709 |
| Random selection (exact expectation) | 0.460 |

Against the generator's planted true topics, which the app cannot see, the ensemble is level with
plain frequency (slightly lower, within one standard error), so the demo does not show a gain
there. The gains over frequency are measured on simulated courses; see
[Low-data inference](docs/LOW_DATA_INFERENCE.md#8-results).

* The general model's small lead over the ensemble is within one standard error (0.013), so the
  app keeps the ensemble.
* Syllabus alignment put 127 of 140 in-syllabus questions (90.7%) on the right topic with the
  default hybrid of the pretrained model and TF-IDF (TF-IDF alone: 128). All 4 questions from an old
  syllabus were marked "outside" and none of the 140 in-syllabus questions was.
* In the staged ablation, only two steps improved NDCG by more than one standard error: adding
  Bayesian smoothing to frequency and recency, and the step to the full ensemble. The other steps changed it by
  less than 0.01, and the app labels them "not reliable" instead of hiding them.
* With 12 papers the run reports "Low-data advanced inference": the course logistic and tree models
  run as LIMITED with 2-3% of the weight each.
* Calibrated probabilities beat the base rate on 10 later held-out papers (Brier 0.182 vs 0.249).

Your own course will give different numbers; the Model performance page shows them.

## Documentation

* [User guide](docs/USER_GUIDE.md): installation, every screen, reading the results, troubleshooting.
* [Architecture and design analysis](docs/ARCHITECTURE.md): the specification review, contradictions
  and how they were resolved, pipeline, schema, backtesting, success metrics.
* [Low-data inference](docs/LOW_DATA_INFERENCE.md): how the ranking engine works with two, five or
  fifteen papers. It covers the Bayesian recurrence, the general and course models, ensemble weights,
  component statuses, calibration, rank intervals, ablation, the new database tables and migrations,
  measured results and limitations. It replaces the old minimum-paper rules.
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
predictor models download                      optional larger sentence-transformer (uses the internet once)
```

Add `--data-dir PATH` before the command to use another data folder (default `~/ExamPredictorData`).

You do not need `models download`: the bundled pretrained model works offline at every course size.
The download needs the `neural` extra (`pip install -e ".[neural]"`). Once the model is on disk, the
default settings use it for syllabus alignment and for finding reworded repeats of past questions.

## Privacy

* The server listens on 127.0.0.1 only.
* Uploaded files, the database and logs live in your data folder.
* No telemetry, no cloud OCR, no remote models. `allow_external_services` is off and nothing in the
  app uses the network.
* The pretrained semantic model (WordLlama, MIT licence) ships inside the `wordllama` package that
  installation adds. The app reads it from disk and downloads nothing. Downloading the optional
  larger sentence-transformer is a separate command you run yourself.
* The general ranking model can learn from the other real courses in your library, on your
  machine. One course's history never enters another course's statistics, and synthetic courses,
  including the demo, are never used for this.

## Tests

```bash
pip install -e ".[ocr,dev]"
pytest                    # unit, integration and regression tests (several minutes; integration tests run full analyses)
```

## Known limitations

* OCR reads printed text well, equations poorly and handwriting badly. Such regions are flagged for
  review; fix them in the Review screen.
* With one held-out paper the app cannot measure accuracy, and with fewer than about five the
  confidence interval is wider than most differences between methods. The app still ranks topics,
  shows wide rank ranges and says so.
* On the demo course the ensemble does not beat plain frequency against the planted true topics.
  Gains are measured on simulated courses, which are only as realistic as their generators.
* The general ranking model is trained on simulations. Until your library holds other real
  courses, it reflects generic examiner behaviour, not your institution's.
* The bundled pretrained model is a static word-embedding model. It found the right topic for 11 of
  26 reworded test questions, fewer than character n-grams (14), so reworded repeats are matched
  by character n-grams and word overlap unless you download a sentence-transformer.
* Predicted question formulations are examples of likely forms, not the exam's wording.
* The [full list](docs/LOW_DATA_INFERENCE.md#11-limitations) is in the low-data inference document.
