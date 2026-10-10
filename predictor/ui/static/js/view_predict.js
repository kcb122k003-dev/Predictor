import { api } from "./api.js";
import { CATEGORY_COLORS, append, badge, clear, componentBadge, drawer, h, num, pct, table, toast } from "./dom.js";
import {
  SEPARATE_NOTE, formatEvidenceTag, freshnessBanner, illustrativeBlock, kindTag, priorityColor, priorityWord, rankRangeText,
  renderNodePanel, templateBlock,
} from "./topic_panel.js";

// Group headings. Runs made before the rename say "Extremely High Priority"; they are listed under Very High.
const CATEGORIES = ["Very High Priority", "High Priority", "Moderate Priority", "Low Priority"];
const OLD_CATEGORIES = { "Extremely High Priority": "Very High Priority" };
const plural = (n, word, many = `${word}s`) => `${n} ${n === 1 ? word : many}`;
const nodeLabel = (n) => `${n.number || ""} ${n.title || ""}`.trim();

export async function renderPredict(main, course, { refreshCourse }) {
  const head = h("div");
  const body = h("div");
  let timer = null;
  let nodes = null;           // explorer overview nodes (unit paths, Correct mapping topic list)
  let nodeById = new Map();
  let fresh = null;           // holder of the "results out of date" banner
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
    // The explorer overview gives each topic's unit path and the topic list for Correct mapping in the drawer.
    const [res, ov] = await Promise.all([
      api.get(`/api/courses/${course.id}/results`),
      api.get(`/api/courses/${course.id}/explorer`).catch(() => null),
    ]);
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
    nodes = ov ? ov.nodes : null;
    nodeById = new Map((nodes || []).map(n => [n.id, n]));
    const run = res.run;
    const s = run.summary;
    const preds = res.predictions;
    fresh = h("div");
    append(fresh, freshnessBanner(res.freshness, { courseId: course.id, onStart: start }));
    append(body, fresh, h("div", { class: "banner warn" }, s.disclaimer));
    if (res.evidence) {
      for (const note of s.notes || []) append(body, h("div", { class: "banner warn" }, note));
      append(body, inferenceStatus(res.evidence));
    } else {
      // A run from the engine before the low-data upgrade: its stored notes describe rules that no longer exist.
      append(body, h("div", { class: "banner info" }, "These results were produced by an earlier version of the engine. ",
        "Press Analyze & Predict to rank the topics with low-data advanced inference, rank ranges and evidence strength."));
    }
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
    append(body, h("p", { class: "help" }, "Topic predictions are the main output. Exact wording is much harder to predict, so an ",
      "illustrative practice question shows the general form a question could take, not the actual exam question. ",
      "Select a topic to see its full history, past questions and syllabus source."));

    for (const cat of CATEGORIES) {
      const items = preds.filter(p => (OLD_CATEGORIES[p.category] || p.category) === cat);
      if (!items.length) continue;
      const section = h("div", { class: "category" },
        h("h3", {}, h("span", { class: "dot", style: { background: CATEGORY_COLORS[cat] } }), cat, h("span", { class: "muted" }, `(${items.length})`)));
      for (const p of items) section.appendChild(card(p, s.calibrated));
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

  // Topic card: fields 1-5 always visible, the format details (6-8) in a <details> block. A click anywhere on the
  // card opens the topic panel, except on links, buttons and the details block.
  function card(p, calibrated) {
    const g = p.format_guide || null;
    const open = h("button", { class: "ex-link pr-title", "aria-haspopup": "dialog", title: "Show this topic's history, past questions and format guide",
      onclick: (e) => { e.stopPropagation(); openTopic(p.topic_id); } }, p.label);
    const path = unitPath(p.topic_id);
    const el = h("article", { class: "topic-card" },
      h("div", { class: "rank", title: "Rank" }, String(p.rank)),
      h("div", { class: "pr-body" },
        h("dl", { class: "pr-fields" },
          item("Predicted topic", open, path ? h("div", { class: "help" }, path) : null),
          item("Priority", priorityField(p)),
          item("Likelihood", likelihoodField(p, calibrated)),
          item("Historical support", historyField(p)),
          g ? item("General question format", h("strong", {}, g.display), " ", formatEvidenceTag(g),
            g.description ? h("div", {}, describeFormat(g)) : null) : null),
        g ? formatDetails(g) : null));
    el.addEventListener("click", (e) => {
      if (e.target.closest("a, button, details, input, select, label")) return;
      if (String(window.getSelection() || "").trim()) return;  // the student is selecting text
      openTopic(p.topic_id);
    });
    return el;
  }

  function unitPath(topicId) {
    const out = [];
    for (let n = nodeById.get((nodeById.get(topicId) || {}).parent_id); n; n = nodeById.get(n.parent_id)) out.unshift(nodeLabel(n));
    return out.join(" › ");
  }

  function item(label, ...value) {
    return [h("dt", {}, label), h("dd", {}, value)];
  }

  function priorityField(p) {
    const total = p.uncertainty && typeof p.uncertainty === "object" ? p.uncertainty.topics : null;
    return [h("span", { class: "badge pr-prio" }, h("span", { class: "ex-dot", style: { background: priorityColor(p) } }), priorityWord(p)),
      ` rank ${p.rank}${total ? ` of ${total}` : ""} `, kindTag("model")];
  }

  function likelihoodField(p, calibrated) {
    const cal = (p.calibrated ?? calibrated) && p.probability !== null && p.probability !== undefined;
    const known = (v) => v !== null && v !== undefined;
    const range = rankRangeText(p);
    const width = cal ? p.probability : p.relative_score;
    return [
      cal ? h("span", { title: SEPARATE_NOTE }, `${pct(p.probability)} chance of appearing`,
        known(p.prob_low) && known(p.prob_high) ? ` (range ${Math.round(100 * p.prob_low)}-${Math.round(100 * p.prob_high)}%)` : "")
        : h("span", {}, `Relative score ${num(p.relative_score, 2)} (not a probability)`, p.confidence ? ` · Confidence ${p.confidence}` : ""),
      " ", kindTag(cal ? "estimate" : "model"),
      range ? h("div", { class: "help" }, range) : null,
      h("div", { class: "bar" }, h("div", { style: { width: `${Math.round(100 * (width || 0))}%`, background: priorityColor(p) } }))];
  }

  // From the run's topic history; runs made before it existed only have the older facts.
  function historyField(p) {
    const hs = p.history || null;
    const f = p.facts || {};
    const a = hs ? hs.exam_frequency : f.appearances;
    const T = hs ? hs.usable_papers : f.exams;
    const n = hs ? hs.question_frequency : f.questions_total;
    const last = hs ? hs.last_label : f.last_label;
    if (a === undefined || a === null) return h("span", { class: "muted" }, "Not recorded for this analysis.");
    const top = hs && hs.top_format;
    if (a === 0) {
      return [`No historical evidence in the ${plural(T || 0, "usable paper")}. `, kindTag("fact"),
        h("div", { class: "help" }, "Not appearing before is not evidence that it will not be examined.")];
    }
    return [`Appeared in ${a} of ${T} usable papers; ${plural(n || 0, "question")}${last ? `; last appeared ${last}` : ""}. `, kindTag("fact"),
      top ? h("div", { class: "help" }, `Most common format: ${top.display} in ${plural(top.papers, "paper")}`) : null,
      hs && hs.provisional ? h("div", {}, h("span", { class: "badge warn", title: (hs.provisional_reasons || []).join("; ") }, "Counts provisional")) : null];
  }

  // The description repeats the format name ("Numerical problem: calculate ..."); the card shows the name already.
  function describeFormat(g) {
    const d = String(g.description || "");
    const prefix = `${g.display}:`.toLowerCase();
    if (!d.toLowerCase().startsWith(prefix)) return d;
    const rest = d.slice(prefix.length).trim();
    return rest ? rest[0].toUpperCase() + rest.slice(1) : d;
  }

  function formatDetails(g) {
    const il = g.illustrative || null;
    const first = il ? String(il.text || "").split("\n")[0] : "";
    const alts = (g.alternatives || []).slice(0, 2);
    return h("details", { class: "pr-details" },
      h("summary", {}, h("span", { class: "pr-sum-label" }, "Question format details"), first ? h("span", { class: "pr-sum-q" }, first) : null),
      il ? illustrativeBlock(il) : g.illustrative_note ? h("p", { class: "help" }, g.illustrative_note) : null,
      g.template ? templateBlock(g.template) : null,
      g.why ? h("p", {}, h("strong", {}, "Why this format: "), g.why) : null,
      g.marks_note ? h("p", { class: "help" }, g.marks_note) : null,
      g.reliability && g.reliability.text ? h("p", { class: "help" }, g.reliability.text) : null,
      alts.length ? h("div", {}, h("strong", {}, "Alternative formats"),
        h("ul", { class: "evidence" }, alts.map(x => h("li", {}, h("strong", {}, x.display), x.support ? `: ${x.support}` : "")))) : null);
  }

  // The same topic panel as the Syllabus Explorer, in a wide drawer. Links inside it open other syllabus items
  // in the same drawer; "Open in Syllabus Explorer" leaves this tab (the drawer closes on the route change).
  function openTopic(topicId) {
    const top = h("div");
    const box = h("div");
    const d = drawer(h("div", { class: "pr-drawer" }, top, box), { wide: true, label: "Topic details" });
    let token = 0;
    async function load(id) {
      const t = ++token;
      clear(box);
      box.appendChild(h("div", { class: "card ex-loading", role: "status" }, "Loading..."));
      try {
        const data = await api.get(`/api/courses/${course.id}/explorer/nodes/${id}`);
        if (t !== token || !d.panel.isConnected) return;
        renderNodePanel(box, data, { courseId: course.id, nodes, explorerLink: true, onNavigate: load, onChanged: changed });
        d.panel.scrollTop = 0;
      } catch (e) {
        if (t !== token) return;
        clear(box);
        box.appendChild(h("div", { class: "banner bad" }, `Could not load this topic: ${e.message}`));
      }
    }
    // A mapping was corrected or reset in the drawer: the results are out of date until the next analysis.
    async function changed() {
      let fr;
      try { fr = await api.get(`/api/courses/${course.id}/freshness`); } catch (e) { toast(e.message, "error"); return; }
      clear(top);
      append(top, freshnessBanner(fr, { courseId: course.id, onStart: () => { d.close(); start(); } }));
      if (fresh) { clear(fresh); append(fresh, freshnessBanner(fr, { courseId: course.id, onStart: start })); }
    }
    load(topicId);
  }

  // Initial state: show an active run if one is going, else the latest results.
  const runs = await api.get(`/api/courses/${course.id}/runs`);
  const active = runs.find(r => r.status === "queued" || r.status === "running");
  if (active) poll(active.id); else await showResults();
}
