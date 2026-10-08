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
   there is no minimum number of papers for testing or for any of the thirteen components. A
   course with only a syllabus and no papers can be analysed too: the ranking then comes from the
   syllabus structure and the general model, and every topic gets the full rank range. The
   components include recency-frequency, hierarchical Bayesian recurrence, semantic evidence
   from a bundled pretrained model, syllabus coverage, a general ranking model trained on
   simulated courses, a course-specific logistic model pulled toward that general model, and
   Markov, HMM and tree models. An evidence-aware ensemble weights each one by how much evidence
   supports it and how well it ranked earlier papers. The app marks each component ACTIVE, LIMITED,
   DOWNWEIGHTED, UNAVAILABLE (an input it needs does not exist) or REFERENCE (a baseline kept for
   comparison). The ensemble stays the final ranking unless the best single method chosen on
   earlier papers (picked again before each held-out paper) beat it on the same held-out papers
   beyond a one-sided 95% t-bound of the paired differences (`ensemble.replace_confidence`).
5. **Explains every prediction.** Recent and total appearances, gaps, how often topics in this
   course reappear after a given gap, marks, formats, signal contributions, and "why not" notes
   for low-ranked topics. Each topic also shows a rank range, an uncertainty level and an evidence
   strength, because a high score does not mean the app is certain. Percentages are shown as
   probabilities only when the calibrated Brier score beat the base rate on later held-out papers
   beyond a one-sided 95% t-bound; otherwise you see relative scores.
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

12 synthetic papers, 144 questions, 25 syllabus topics, analysed in about 9 seconds on a
4-core machine. The scores below compare each method's ranking with the topics the app itself
mapped the held-out questions to, which is all a real course can measure:

| Method | NDCG@11 on 11 held-out papers (mean ± SE) |
|---|---:|
| General ranking model (cross-course) | 0.730 ± 0.022 |
| **Evidence-aware ensemble (final ranking)** | **0.717 ± 0.027** |
| Semantic evidence (pretrained) | 0.712 ± 0.021 |
| Best single method chosen on earlier papers | 0.707 ± 0.022 |
| Recent-window frequency | 0.687 ± 0.017 |
| Most frequent topics | 0.678 ± 0.024 |
| Hierarchical Bayesian recurrence | 0.673 ± 0.026 |
| Random selection (exact expectation) | 0.456 |

Against these labels the ensemble beat plain frequency by +0.039 ± 0.019 (better on 9 of 11
papers). Against the generator's planted true topics, which the app cannot see, the difference is
+0.004 ± 0.017, so the demo shows no improvement over frequency there. Part of the gain against
mapped labels comes from the semantic component, which shares the alignment's systematic choices.
On simulated courses with known outcomes, in-sample backtests put the ensemble 0.016-0.08 above
frequency at 3-20 papers. Scored strictly on the next paper, the gain is small: -0.008 to +0.032.
See [Low-data inference](docs/LOW_DATA_INFERENCE.md#8-results).

* The general model alone scored highest, but you can only pick it in hindsight. The app compares
  the ensemble with the best single method chosen on earlier papers: the ensemble led by
  +0.010 ± 0.018, so it stays the final ranking.
* Syllabus alignment put 131 of 140 in-syllabus questions (93.6%) on the right topic with the
  default hybrid of the pretrained model and TF-IDF. When it aligns a paper, the feedback pass uses
  questions from earlier papers only. All 4 questions from an old syllabus were marked "outside" and
  none of the 140 in-syllabus questions was.
* In the staged ablation, only adding semantic evidence changed NDCG reliably (+0.052 ± 0.019,
  outside the two-sided 95% t-interval). The app labels the other steps "not reliable" instead of
  hiding them; adding recency to frequency, for example, lowered NDCG by 0.015 ± 0.008.
* With 12 papers the run reports "Low-data advanced inference": the course logistic, random forest
  and gradient boosting models run as LIMITED with 2-4% of the weight each. The label is wording
  only; it switches nothing on or off.
* Calibrated probabilities beat the base rate on 10 later held-out papers: Brier 0.191 vs 0.248,
  a gain of 0.057 ± 0.005.

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
* On the demo course the ensemble does not beat plain frequency against the planted true topics
  (+0.004 ± 0.017). Gains show up mostly in in-sample backtests on simulated courses, which are
  only as realistic as their generators; next-paper gains there are small.
* The general ranking model is trained on simulations. Until your library holds other real
  courses, it reflects generic examiner behaviour, not your institution's. Set
  `models.use_simulated_prior = false` (or clear "Use the simulated cross-course prior" in
  Settings) to drop the simulations: the general model then learns only from other real courses
  in your library, and with none it is UNAVAILABLE and the other components share its weight.
* The bundled pretrained model is a static word-embedding model. It found the right topic for 11 of
  26 reworded test questions, fewer than character n-grams (14), so reworded repeats are matched
  by character n-grams and word overlap unless you download a sentence-transformer.
* Predicted question formulations are examples of likely forms, not the exam's wording.
* The [full list](docs/LOW_DATA_INFERENCE.md#11-limitations) is in the low-data inference document.
