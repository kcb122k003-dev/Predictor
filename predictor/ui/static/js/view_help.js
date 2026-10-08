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
      "The ranking combines several kinds of evidence: a pretrained language model, a cross-course model of how examiners behave, Bayesian recurrence of each topic, syllabus structure, timing patterns and models learned from this course. Each component's weight depends on how much evidence supports it (papers versus the number of parameters it estimates from your course) and on how well it predicted your earlier papers, where each past paper is predicted from the papers before it.",
      "Nothing is switched off because a course is small. With two or three papers the pretrained, cross-course and Bayesian parts carry most of the weight and every topic shows a wide rank range; course-specific learned models gain weight as papers accumulate and prove themselves. If picking the best single method on earlier papers beat the combined ranking on your later papers by a clear margin (beyond a one-sided 95% bound), that method is used instead and the reason is shown."),
    section("5. Read the results",
      h("ul", { class: "evidence" },
        h("li", {}, "Priority categories group topics by evidence strength. Extremely High needs several independent signals agreeing."),
        h("li", {}, "Percentages are shown as probabilities only when calibration was validated on held-out papers. Otherwise they are relative scores."),
        h("li", {}, "Each topic shows a rank range (how far it could move with the available papers) and its evidence strength (how much of its estimate comes from its own history rather than the prior)."),
        h("li", {}, "Click a topic for its evidence, the past questions behind it, each component's contribution, the Bayesian estimate with its credible interval, the most similar past questions and predicted question formulations."),
        h("li", {}, "Predicted question formulations reuse your course's own wording and syllabus phrases. Numerical questions are past numericals shown as patterns; the app never invents numbers."))),
    section("Privacy", "Everything runs on this computer. The server listens only on 127.0.0.1. No telemetry and no cloud processing. The bundled pretrained model ships with the app; downloading an optional larger model is a separate command you run yourself."),
  );
}
