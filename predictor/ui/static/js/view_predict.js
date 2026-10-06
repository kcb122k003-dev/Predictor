import { api } from "./api.js";
import { CATEGORY_COLORS, append, badge, clear, drawer, h, num, pct, statusBadge, table, toast } from "./dom.js";

const CATEGORIES = ["Extremely High Priority", "High Priority", "Moderate Priority", "Low Priority"];

export async function renderPredict(main, course, { refreshCourse }) {
  const head = h("div");
  const body = h("div");
  let timer = null;
  window.addEventListener("predictor:leave", () => clearTimeout(timer), { once: true });
  append(main, head, body);

  const runBtn = h("button", { class: "primary big" }, "Analyze & Predict");
  runBtn.addEventListener("click", start);
  append(head, h("div", { class: "row" }, h("h1", { style: { margin: 0 } }, "Predictions"), h("span", { class: "spacer" }), runBtn));

  async function start() {
    runBtn.disabled = true;
    try {
      const run = await api.post(`/api/courses/${course.id}/analyze`);
      poll(run.id);
    } catch (e) { toast(e.message, "error", 8000); runBtn.disabled = false; }
  }

  async function poll(runId) {
    const run = await api.get(`/api/runs/${runId}`);
    clear(body);
    if (run.status === "queued" || run.status === "running") {
      runBtn.disabled = true;
      append(body, h("div", { class: "card" }, h("p", {}, run.message || "Working..."),
        h("div", { class: "progress" }, h("div", { style: { width: `${Math.round(100 * (run.progress || 0))}%` } }))));
      timer = setTimeout(() => poll(runId), 1000);
      return;
    }
    runBtn.disabled = false;
    if (run.status === "error") {
      append(body, h("div", { class: "banner bad" }, h("strong", {}, "The analysis could not finish. "),
        h("pre", { class: "pagetext", style: { marginTop: "8px" } }, run.message)));
      return;
    }
    refreshCourse();
    await showResults();
  }

  async function showResults() {
    const res = await api.get(`/api/courses/${course.id}/results`);
    clear(body);
    if (!res.run) {
      append(body, h("div", { class: "card" },
        h("h3", {}, "No predictions yet"),
        h("p", {}, "Upload past papers and the course contents, check them in Review, then press Analyze & Predict."),
        h("p", { class: "muted" }, "The analysis maps every question to the syllabus, compares several prediction ",
          "methods on your own past papers (each past paper is predicted from the papers before it), picks the method ",
          "that held up best, and ranks the topics for the next paper.")));
      return;
    }
    const run = res.run;
    const s = run.summary;
    const preds = res.predictions;
    append(body, h("div", { class: "banner warn" }, s.disclaimer));
    for (const note of s.notes || []) append(body, h("div", { class: "banner warn" }, note));
    if (res.sufficiency && res.sufficiency.message) append(body, h("div", { class: "banner info" }, res.sufficiency.message));
    append(body, h("div", { class: "grid cols-4" },
      stat(s.exams, "past papers analysed"), stat(s.counted_questions, `of ${s.questions} questions inside the syllabus`),
      stat(s.selected_display, "prediction method used"),
      stat(s.calibrated ? "Calibrated" : "Relative scores", s.calibrated ? "percentages are validated probabilities" : "percentages are not probabilities")));
    append(body, h("div", { class: "card" }, h("p", { style: { margin: 0 } }, h("strong", {}, "Why this method: "), s.selection_reason),
      h("p", { class: "muted", style: { margin: "6px 0 0" } }, s.calibration_reason),
      h("div", { class: "row", style: { marginTop: "8px" } },
        h("a", { class: "btn small", href: `/api/runs/${run.id}/export?format=pdf` }, "PDF report"),
        h("a", { class: "btn small", href: `/api/runs/${run.id}/export?format=xlsx` }, "Excel workbook"),
        h("a", { class: "btn small", href: `/api/runs/${run.id}/export?format=csv&table=predictions` }, "CSV ranking"),
        h("a", { class: "btn small", href: `/api/runs/${run.id}/export?format=csv&table=questions` }, "CSV predicted questions"),
        h("a", { class: "btn small", href: `/api/runs/${run.id}/export?format=json` }, "JSON"),
        h("span", { class: "spacer" }), h("span", { class: "help" }, `Analysed in ${s.seconds}s with ${s.embedding_backend}`))));
    append(body, h("p", { class: "help" }, "Topic predictions are the main output. Exact wording is much harder to predict, so ",
      "predicted question formulations (inside each topic) are examples of likely forms, not the actual questions."));

    for (const cat of CATEGORIES) {
      const items = preds.filter(p => p.category === cat);
      if (!items.length) continue;
      const section = h("div", { class: "category" },
        h("h3", {}, h("span", { class: "dot", style: { background: CATEGORY_COLORS[cat] } }), cat, h("span", { class: "muted" }, `(${items.length})`)));
      for (const p of items) section.appendChild(card(p, run.id, s.calibrated));
      body.appendChild(section);
    }
    const excluded = (res.excluded && res.excluded.groups) || [];
    if (excluded.length) {
      body.appendChild(h("div", { class: "category" },
        h("h3", {}, h("span", { class: "dot", style: { background: CATEGORY_COLORS.Excluded } }), "Excluded: outside the current syllabus"),
        h("p", { class: "muted" }, "These past questions do not match the current course contents, so they cannot drive predictions."),
        table([
          { label: "Question", render: g => g.label },
          { label: "Papers", render: g => g.years.join(", ") },
          { label: "Why", render: g => g.reason },
        ], excluded)));
    }
  }

  function stat(v, label) {
    return h("div", { class: "stat" }, h("div", { class: "v" }, String(v ?? "")), h("div", { class: "l" }, label));
  }

  function card(p, runId, calibrated) {
    const f = p.facts || {};
    const tf = f.type_forecast || {};
    const value = calibrated && p.probability !== null
      ? h("div", {}, h("div", { class: "p" }, pct(p.probability)), h("div", { class: "help" }, `range ${pct(p.prob_low)} to ${pct(p.prob_high)}`))
      : h("div", {}, h("div", { class: "p" }, num(p.relative_score, 2)), h("div", { class: "help" }, "relative score"));
    const width = calibrated && p.probability !== null ? p.probability : p.relative_score;
    const el = h("div", { class: "topic-card" },
      h("div", { class: "rank" }, String(p.rank)),
      h("div", {},
        h("div", { class: "title" }, p.label),
        h("div", { class: "meta" },
          h("span", {}, `${f.recent_appearances ?? 0}/${f.recent_window ?? 0} recent papers`),
          h("span", {}, f.last_label ? `last: ${f.last_label}` : "never asked"),
          tf.format ? h("span", {}, `likely format: ${tf.format}`) : null,
          f.marks_mean ? h("span", {}, `marks ${f.marks_min}-${f.marks_max}`) : null,
          f.mapping_confidence ? h("span", {}, `syllabus match ${pct(f.mapping_confidence)}`) : null,
          badge(`${p.confidence} confidence`, p.confidence === "High" ? "ok" : p.confidence === "Low" ? "warn" : "info")),
        h("div", { class: "bar" }, h("div", { style: { width: `${Math.round(100 * (width || 0))}%`, background: CATEGORY_COLORS[p.category] } }))),
      h("div", { class: "prob" }, value));
    el.addEventListener("click", () => openTopic(runId, p.topic_id, calibrated));
    return el;
  }

  // Initial state: show an active run if one is going, else the latest results.
  const runs = await api.get(`/api/courses/${course.id}/runs`);
  const active = runs.find(r => r.status === "queued" || r.status === "running");
  if (active) poll(active.id); else await showResults();
}

export async function openTopic(runId, topicId, calibrated) {
  const d = await api.get(`/api/runs/${runId}/topics/${topicId}`);
  const p = d.prediction;
  const content = h("div");
  const add = (...items) => append(content, items);
  drawer(content);
  add(h("h2", {}, d.topic.path), h("p", { class: "muted" }, d.topic.concepts && d.topic.concepts.length ? `Concepts: ${d.topic.concepts.join("; ")}` : ""));
  if (p) {
    add(h("div", { class: "row" }, badge(p.category, "info"), badge(`${p.confidence} confidence`),
      calibrated && p.probability !== null ? badge(`${pct(p.probability)} (${pct(p.prob_low)} to ${pct(p.prob_high)})`, "ok")
        : badge(`relative score ${num(p.relative_score, 2)}`), badge(`rank ${p.rank}`)));
    add(h("h3", { style: { marginTop: "14px" } }, "Why this topic ranked here"),
      h("ul", { class: "evidence" }, p.evidence.map(e => h("li", {}, e))));
    if (p.why_not && p.why_not.length) {
      add(h("h3", {}, "Why it is not ranked higher"), h("ul", { class: "evidence" }, p.why_not.map(e => h("li", {}, e))));
    }
    const contrib = Object.entries(p.contributions || {});
    if (contrib.length) {
      const max = Math.max(...contrib.map(([, v]) => Math.abs(v)), 0.01);
      add(h("h3", {}, "Signal contributions"),
        h("p", { class: "help" }, `Additive contributions to the log-odds from ${p.contribution_source}. Green raises the topic, red lowers it.`),
        contrib.sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).map(([k, v]) => h("div", { class: "contrib" },
          h("span", {}, k),
          h("div", { class: "track" }, h("div", { class: "mid" }),
            v >= 0 ? h("div", { class: "pos", style: { width: `${50 * Math.abs(v) / max}%` } }) : h("div", { class: "neg", style: { width: `${50 * Math.abs(v) / max}%` } })),
          h("span", { class: "mono" }, (v >= 0 ? "+" : "") + v.toFixed(2)))));
    }
  }
  if (d.formulations.length) {
    add(h("h3", {}, "Predicted question formulations"),
      h("p", { class: "help" }, "Built only from this course's past wording and syllabus phrases. They show likely forms, not the actual exam questions."),
      d.formulations.map(f => h("div", { class: "formulation" },
        h("div", { class: "tag" }, "PREDICTED QUESTION FORMULATION"),
        h("div", {}, f.text),
        h("div", { class: "help" }, [`format: ${f.format}`, f.marks_low !== null ? `marks ${f.marks_low === f.marks_high ? f.marks_low : `${f.marks_low}-${f.marks_high}`}` : null,
          f.basis === "historical_variant" ? "reused past numerical (numbers not invented)" : "template from past openers + syllabus phrase",
          f.evidence_question_ids.length ? `based on ${f.evidence_question_ids.length} past question(s)` : null].filter(Boolean).join(" | ")),
        f.grounding && f.grounding.note ? h("div", { class: "help" }, f.grounding.note) : null)));
  }
  if (d.families.length) {
    add(h("h3", {}, "Recurring questions"),
      h("ul", { class: "evidence" }, d.families.slice(0, 6).map(f => h("li", {}, `${f.question_ids.length} similar question(s) in ${f.exams.join(", ")}`))));
  }
  add(h("h3", {}, `Past questions on this topic (${d.questions.length})`),
    d.questions.length ? table([
      { label: "Paper", render: q => h("div", {}, q.exam, h("div", { class: "help" }, q.path_label)) },
      { label: "Question", render: q => q.text },
      { label: "Marks", key: "marks", num: true },
      { label: "Type", render: q => (q.types || []).slice(0, 2).join(", ") },
      { label: "Mapping", render: q => h("div", {}, statusBadge(q.mapping.status), h("div", { class: "help" }, pct(q.mapping.confidence))) },
    ], d.questions) : h("p", { class: "muted" }, "None."));
  const refs = d.topic.source_refs || [];
  if (refs.length) {
    add(h("h3", {}, "Where it is in the course contents"),
      h("ul", { class: "evidence" }, refs.slice(0, 5).map(r => h("li", {}, `${r.file || ""}${r.page ? `, page ${r.page}` : ""}${r.line ? `, line ${r.line}` : ""}: `, h("span", { class: "muted" }, r.text || "")))));
  }
}
