import { api } from "./api.js";
import { CATEGORY_COLORS, append, badge, clear, componentBadge, drawer, h, num, pct, shareBar, statusBadge, table, toast } from "./dom.js";

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
        h("p", { class: "muted" }, "The analysis maps every question to the syllabus, then combines pretrained language ",
          "knowledge, a cross-course model, Bayesian recurrence and this course's own patterns. Each component is weighted ",
          "by how much evidence supports it and by how well it predicted your earlier papers (each past paper is predicted ",
          "from the papers before it). It works with any number of papers; fewer papers show up as wider uncertainty.")));
      return;
    }
    const run = res.run;
    const s = run.summary;
    const preds = res.predictions;
    append(body, h("div", { class: "banner warn" }, s.disclaimer));
    for (const note of s.notes || []) append(body, h("div", { class: "banner warn" }, note));
    if (res.evidence) append(body, inferenceStatus(res.evidence));
    else if (res.sufficiency && res.sufficiency.message) append(body, h("div", { class: "banner info" }, res.sufficiency.message));
    append(body, h("div", { class: "grid cols-4" },
      stat(s.exams, "past papers analysed"), stat(s.counted_questions, `of ${s.questions} questions inside the syllabus`),
      stat(s.selected_display, "prediction method used"),
      stat(s.calibrated ? "Calibrated" : "Relative scores", s.calibrated ? "percentages are validated probabilities" : "percentages are not probabilities")));
    append(body, h("div", { class: "card" }, h("p", { style: { margin: 0 } }, h("strong", {}, "Why this ranking: "), s.selection_reason),
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

  function inferenceStatus(ev) {
    const comps = (ev.components || []).filter(c => c.role === "component");
    const open = h("details", {}, h("summary", {}, `Components (${comps.filter(c => c.status !== "UNAVAILABLE").length} of ${comps.length} running)`),
      table([
        { label: "Component", render: c => h("div", {}, c.display, h("div", { class: "help" },
          c.scope === "global" ? "outside knowledge" : c.scope === "syllabus" ? "syllabus structure" : "estimated from this course")) },
        { label: "Status", render: c => componentBadge(c.status) },
        { label: "Weight", render: c => c.weight === null ? "" : pct(c.weight), num: true },
        { label: "Why", render: c => h("span", { class: "help" }, c.reason) },
      ], comps));
    const v = ev.validation || {};
    return h("div", { class: "card" },
      h("div", { class: "row" }, h("h3", { style: { margin: 0 } }, "Inference status"), badge(ev.mode, ev.mode === "Advanced inference" ? "ok" : "info"),
        ev.evidence_quality ? badge(`evidence quality: ${ev.evidence_quality.toLowerCase()}`) : null,
        ev.uncertainty && ev.uncertainty.overall ? badge(`prediction uncertainty: ${ev.uncertainty.overall.toLowerCase()}`, ev.uncertainty.overall === "High" ? "warn" : "") : null),
      h("p", { style: { margin: "8px 0 4px" } }, ev.message),
      v.message ? h("p", { class: "help", style: { margin: "0 0 6px" } }, v.message) : null,
      (ev.notes || []).map(n => h("p", { class: "help", style: { margin: "0 0 4px" } }, n)),
      open);
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
          h("span", { title: "Historical coverage: papers with the topic / all papers, and questions behind it" },
            `${f.appearances ?? 0}/${f.exams ?? 0} papers, ${f.questions_total ?? 0} question${f.questions_total === 1 ? "" : "s"}`),
          h("span", {}, f.last_label ? `last: ${f.last_label}` : "never asked"),
          tf.format ? h("span", {}, `likely format: ${tf.format}`) : null,
          f.marks_mean ? h("span", {}, `marks ${f.marks_min}-${f.marks_max}`) : null,
          f.mapping_confidence ? h("span", {}, `syllabus match ${pct(f.mapping_confidence)}`) : null,
          p.uncertainty && p.uncertainty.rank_low ? h("span", { title: "Plausible rank range given the available papers" },
            `rank range ${p.uncertainty.rank_low}-${p.uncertainty.rank_high}`) : null,
          p.evidence_strength ? h("span", { title: "How much of the estimate comes from this topic's own history" }, `evidence: ${p.evidence_strength.toLowerCase()}`) : null,
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
    const u = p.uncertainty || {};
    add(h("div", { class: "row" }, badge(p.category, "info"), badge(`${p.confidence} confidence`),
      calibrated && p.probability !== null ? badge(`${pct(p.probability)} (${pct(p.prob_low)} to ${pct(p.prob_high)})`, "ok")
        : badge(`relative score ${num(p.relative_score, 2)}`), badge(`rank ${p.rank}`),
      u.rank_low ? badge(`rank range ${u.rank_low}-${u.rank_high} (${(u.level || "").toLowerCase()} uncertainty)`) : null,
      p.evidence_strength ? badge(`evidence ${p.evidence_strength.toLowerCase()}`) : null));
    if ((p.facts || {}).evidence_summary) add(h("p", { class: "muted", style: { margin: "8px 0 0" } }, p.facts.evidence_summary));
    add(h("h3", { style: { marginTop: "14px" } }, "Why this topic ranked here"),
      h("ul", { class: "evidence" }, p.evidence.map(e => h("li", {}, e))));
    if (p.why_not && p.why_not.length) {
      add(h("h3", {}, "Why it is not ranked higher"), h("ul", { class: "evidence" }, p.why_not.map(e => h("li", {}, e))));
    }
    const f = p.facts || {};
    const shares = Object.entries(p.contributions || {}).sort((a, b) => b[1] - a[1]);
    if (shares.length) {
      const max = Math.max(...shares.map(([, v]) => v), 0.01);
      add(h("h3", {}, "Model contributions"),
        h("p", { class: "help" }, `Each component's share of the final score: ${p.contribution_source}. The shares add up to the topic's score.`),
        shares.map(([k, v]) => shareBar(k, v, max, v.toFixed(3))));
    }
    const b = f.bayes;
    if (b) {
      add(h("h3", {}, "Bayesian recurrence"),
        h("dl", { class: "facts" },
          h("dt", {}, "Chance of appearing"), h("dd", {}, `${pct(b.posterior_mean)} (median ${pct(b.posterior_median)})`),
          h("dt", {}, `${Math.round(100 * b.interval_level)}% credible interval`), h("dd", {}, `${pct(b.credible_interval[0])} to ${pct(b.credible_interval[1])}`),
          h("dt", {}, "Observed"), h("dd", {}, `${b.observed.appearances} of ${b.observed.exams} paper(s)`),
          h("dt", {}, "From the prior"), h("dd", {}, `${pct(b.prior_contribution)} (unit and course rate; shrinks as papers accumulate)`),
          h("dt", {}, "Effective sample size"), h("dd", {}, num(b.effective_sample_size, 1)),
          h("dt", {}, "Recency discount"), h("dd", {}, b.half_life ? `half-life ${b.half_life} papers` : "none")));
    }
    if (f.temporal_mode) {
      add(h("h3", {}, "Temporal pattern"),
        h("p", {}, `Used: ${f.temporal_mode}.`, f.temporal_log_bayes_factor !== undefined && f.temporal_log_bayes_factor !== null
          ? ` Gap hazard vs constant rate on earlier papers: log Bayes factor ${num(f.temporal_log_bayes_factor, 2)} (positive favours the hazard).` : ""));
    }
    if (f.semantic_evidence && f.semantic_evidence.length) {
      add(h("h3", {}, "Semantic evidence"),
        h("p", { class: "help" }, "Past in-syllabus questions closest in meaning to this topic (pretrained model)."),
        h("ul", { class: "evidence" }, f.semantic_evidence.map(e => h("li", {},
          `${e.exam}: "${e.text}" `, h("span", { class: "muted" }, `similarity ${num(e.similarity, 2)}${e.mapped_here ? ", mapped to this topic" : ""}`)))));
    }
    const contrib = Object.entries(p.signal_contributions || {});
    if (contrib.length) {
      const max = Math.max(...contrib.map(([, v]) => Math.abs(v)), 0.01);
      add(h("details", { style: { marginTop: "10px" } }, h("summary", {}, "Signal contributions (course-specific logistic model)"),
        h("p", { class: "help" }, `${p.signal_source}. Several signals measure similar things (frequency, recency, Bayesian rate), `,
          "so one can carry a large negative value while a related one carries a large positive value. Read the sum, ",
          "not individual signs; the model contributions above are the main explanation."),
        contrib.sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).map(([k, v]) => h("div", { class: "contrib" },
          h("span", {}, k),
          h("div", { class: "track" }, h("div", { class: "mid" }),
            v >= 0 ? h("div", { class: "pos", style: { width: `${50 * Math.abs(v) / max}%` } }) : h("div", { class: "neg", style: { width: `${50 * Math.abs(v) / max}%` } })),
          h("span", { class: "mono" }, (v >= 0 ? "+" : "") + v.toFixed(2))))));
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
        f.grounding && f.grounding.note ? h("div", { class: "help" }, f.grounding.note) : null,
        f.grounding && f.grounding.checks ? h("div", { class: "help" }, "Checks passed: syllabus wording, topic, semantic fit, question type.") : null)));
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
