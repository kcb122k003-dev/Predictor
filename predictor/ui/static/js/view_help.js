import { append, h } from "./dom.js";

export async function renderHelp(main) {
  const section = (title, ...paras) => h("div", { class: "card" }, h("h3", {}, title), paras.map(p => typeof p === "string" ? h("p", {}, p) : p));
  append(main, 
    h("h1", {}, "How it works"),
    h("div", { class: "banner warn" }, "Exam prediction is probabilistic. The app ranks topics by historical evidence; it cannot tell you what the next paper will contain."),
    section("1. Create a course and upload files",
      "Upload as many past papers as you have (PDF, Word, photos or scans) and the course contents (syllabus, outline). Scanned pages are read with OCR. Identical files and the same paper uploaded twice are detected so nothing is counted twice."),
    section("2. Review",
      "Check each paper's year and order: the order defines time for every recency calculation. Fix question boundaries, marks and text where the extraction went wrong. Items with a yellow border need a look."),
    section("3. Syllabus and mapping",
      "Every question is matched to the syllabus. Status A means clearly in the syllabus, B probably in, C uncertain, D outside. Only A and validated B questions count; questions outside the current syllabus are listed as excluded with the reason. You can override any mapping."),
    section("4. Analyze & Predict",
      "The app compares many prediction methods on your own history. For each past paper, every method predicts it using only the papers before it, and the results are scored. The simplest method that is within one standard error of the best is used for the next paper. Complex methods are switched off when there are too few papers to train them reliably, and the reason is shown."),
    section("5. Read the results",
      h("ul", { class: "evidence" },
        h("li", {}, "Priority categories group topics by evidence strength. Extremely High needs several independent signals agreeing."),
        h("li", {}, "Percentages are shown as probabilities only when calibration was validated on held-out papers. Otherwise they are relative scores."),
        h("li", {}, "Click a topic for its evidence, the past questions behind it, signal contributions and predicted question formulations."),
        h("li", {}, "Predicted question formulations reuse your course's own wording and syllabus phrases. Numerical questions are past numericals shown as patterns; the app never invents numbers."))),
    section("Privacy", "Everything runs on this computer. The server listens only on 127.0.0.1. No telemetry and no cloud processing. Downloading an optional neural model is a separate command you run yourself."),
  );
}
