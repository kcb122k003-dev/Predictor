// Shared topic panel: one syllabus node (unit, topic or sub-topic) with its history, past questions, format
// guide and syllabus source. Used by the Syllabus Explorer and by the Predictions tab drawer.
//
// Exports
//   renderNodePanel(container, data, opts) -> { showPaper(paperIndex), showQuestion(questionId) }
//     container          element to fill (it is cleared first).
//     data               response of GET /api/courses/{cid}/explorer/nodes/{nid}.
//     opts.courseId      course id (required).
//     opts.nodes         overview nodes from GET /api/courses/{cid}/explorer (optional). Used for the Correct
//                        mapping topic list and the children's priorities; fetched on demand when absent.
//     opts.onNavigate(id) open another node (default: go to #/course/{cid}/explorer/{id}).
//     opts.onChanged()   called after a mapping is corrected or reset. Without it the panel shows the
//                        "results out of date" banner (with Re-analyse now) at its own top.
//     opts.onReanalysed() called when a re-analysis started from the panel's own banner finishes.
//     opts.explorerLink  true to show an "Open in Syllabus Explorer" link (for use outside the explorer).
//   kindTag(kind)          visible tag saying what a value is: fact | estimate | model | illustrative | inferred | weak.
//   priorityWord(p)        "Very high" | "High" | "Moderate" | "Low" for a prediction (old runs: "Extremely High").
//   priorityColor(p)       colour token for a prediction's priority (from CATEGORY_COLORS).
//   likelihood(p)          { kind, text, note } likelihood wording; a probability only when calibrated.
//   SEPARATE_NOTE          why calibrated topic probabilities do not add up to 100%.
//   rankRangeText(p)       "Rank range: 20-24 of 25 (low uncertainty)", or "".
//   renderFormatGuide(g, { alternatives = 2 })  element for prediction.format_guide.
//   formatEvidenceTag(g)   kind tag for a format guide's evidence (established, weak or inferred).
//   templateBlock(t)       "General template" block with each [placeholder] marked (format_guide.template).
//   illustrativeBlock(il)  "Illustrative practice question" block (format_guide.illustrative).
//   freshnessBanner(freshness, { courseId, onDone, onStart })  stale-results banner with "Re-analyse now", or
//                          null. The button runs the analysis itself (then onDone), or calls onStart() instead.
//   openQuestionSource(questionId)   "View original" modal (PDF page image, or page text with the line marked).
//   openSyllabusSource(ref)          "View source passage" modal for a node.source_refs entry.
//   openMappingModal(question, { courseId, nodes, preset, onSaved })  "Correct mapping" modal.
//   syllabusOrder(nodes, roots)      node ids in syllabus (depth-first) order.

import { api } from "./api.js";
import { CATEGORY_COLORS, append, badge, clear, h, modal, num, pct, shareBar, statusBadge, toast } from "./dom.js";

// ------------------------------------------------------------------ small shared helpers
const KINDS = {
  fact: ["Historical fact", "Counted directly from the past papers you supplied."],
  estimate: ["Statistical estimate", "Estimated from the past papers by a statistical model; it comes with a range."],
  model: ["Model ranking", "The topic's position among this course's topics. It is not a probability."],
  illustrative: ["Illustrative", "Generated practice material. It is not a real exam question."],
  inferred: ["Inferred", "Not observed in your papers or documents; inferred from the syllabus or course patterns."],
  weak: ["Weak evidence", "Observed, but too few times to establish a pattern."],
};

export function kindTag(kind) {
  const [label, title] = KINDS[kind] || [kind, ""];
  return h("span", { class: `ex-kind ex-kind-${kind}`, title }, label);
}

export function priorityWord(p) {
  if (!p) return "";
  const raw = String(p.priority || p.category || "").replace(/\s*Priority$/i, "").trim();
  if (!raw) return "";
  if (/^(very|extremely) high$/i.test(raw)) return "Very high";
  return raw[0].toUpperCase() + raw.slice(1).toLowerCase();
}

const PRIORITY_KEYS = {
  "Very high": ["Very High Priority", "Extremely High Priority"], High: ["High Priority"],
  Moderate: ["Moderate Priority"], Low: ["Low Priority"],
};

export function priorityColor(p) {
  if (!p) return "transparent";
  for (const key of [p.category, ...(PRIORITY_KEYS[priorityWord(p)] || [])]) {
    if (key && CATEGORY_COLORS[key]) return CATEGORY_COLORS[key];
  }
  return "var(--border)";
}

export const SEPARATE_NOTE = "Each topic's probability is estimated separately; several topics appear in one paper, so they do not add up to 100%.";

export function likelihood(p) {
  if (!p) return null;
  if (p.calibrated && p.probability !== null && p.probability !== undefined) {
    const range = p.prob_low !== null && p.prob_low !== undefined
      ? ` (range ${Math.round(100 * p.prob_low)}-${Math.round(100 * p.prob_high)}%)` : "";
    return { kind: "estimate", text: `Estimated probability of appearing in the next examination: ${pct(p.probability)}${range}`, note: SEPARATE_NOTE };
  }
  const parts = [`Relative priority: ${priorityWord(p)}`];
  if (p.relative_score !== null && p.relative_score !== undefined) {
    parts.push(`Prediction score ${num(p.relative_score, 2)} (position among this course's topics; not a probability)`);
  }
  if (p.confidence) parts.push(`Confidence: ${p.confidence}`);
  return { kind: "model", text: parts.join(" · "), note: "" };
}

export function rankRangeText(p) {
  if (!p) return "";
  const u = p.uncertainty && typeof p.uncertainty === "object" ? p.uncertainty
    : { rank_low: p.rank_low, rank_high: p.rank_high, level: p.uncertainty };
  if (!u.rank_low) return "";
  const of = u.topics ? ` of ${u.topics}` : "";
  const level = u.level ? ` (${String(u.level).toLowerCase()} uncertainty)` : "";
  const range = u.rank_low === u.rank_high ? `${u.rank_low}` : `${u.rank_low}-${u.rank_high}`;
  return `Rank range: ${range}${of}${level}`;
}

export function syllabusOrder(nodes, roots) {
  const byId = new Map(nodes.map(n => [n.id, n]));
  const start = roots && roots.length ? roots : nodes.filter(n => !byId.has(n.parent_id)).map(n => n.id);
  const out = [];
  const seen = new Set();
  (function walk(ids) {
    for (const id of ids) {
      const n = byId.get(id);
      if (!n || seen.has(id)) continue;
      seen.add(id);
      out.push(id);
      walk(n.children || []);
    }
  })(start);
  for (const n of nodes) if (!seen.has(n.id)) out.push(n.id);
  return out;
}

const marksText = (v) => (v === null || v === undefined ? "" : String(Number.isInteger(v) ? v : +Number(v).toFixed(1)));
const plural = (n, word, many = `${word}s`) => `${n} ${n === 1 ? word : many}`;
const squash = (s) => String(s || "").replace(/\s+/g, " ").trim();
const letters = (s) => String(s || "").toLowerCase().replace(/[^a-z0-9]+/g, "");
const nodeLabel = (n) => `${n.number || ""} ${n.title || ""}`.trim();

function where(file, page, line) {
  return [file || "Unknown file", page ? `page ${page}` : null, line ? `line ${line}` : null].filter(Boolean).join(", ");
}

function section(title, ...children) {
  return h("section", { class: "card ex-section" }, h("h3", {}, title), children);
}

function tile(kind, label, value, sub) {
  return h("div", { class: "ex-tile" },
    h("div", { class: "ex-tile-head" }, h("span", { class: "l" }, label), kind ? kindTag(kind) : null),
    h("div", { class: "v" }, value), sub ? h("div", { class: "help" }, sub) : null);
}

function priorityChip(p) {
  if (!p) return null;
  return h("span", { class: "ex-prio" }, h("span", { class: "ex-dot", style: { background: priorityColor(p) } }), priorityWord(p));
}

// ------------------------------------------------------------------ panel
export function renderNodePanel(container, data, opts = {}) {
  const c = {
    data, node: data.node, stats: data.stats || null, p: data.prediction || null,
    questions: data.questions || [], courseId: opts.courseId, nodes: opts.nodes || null,
    navigate: opts.onNavigate || ((id) => { window.location.hash = `#/course/${opts.courseId}/explorer/${id}`; }),
    paperFilter: null, showPaper: () => {}, showQuestion: () => {},
  };
  c.T = c.stats ? c.stats.usable_papers : ((data.run || {}).usable_papers || 0);
  const root = h("div", { class: "ex-panel" });
  const top = h("div");
  c.changed = async () => {
    if (opts.onChanged) { opts.onChanged(); return; }
    try {
      const fr = await api.get(`/api/courses/${c.courseId}/freshness`);
      clear(top);
      append(top, freshnessBanner(fr, { courseId: c.courseId, onDone: opts.onReanalysed }));
    } catch (e) { toast(e.message, "error"); }
  };
  clear(container);
  append(container, root);
  append(root, top);
  if (opts.explorerLink) {
    append(root, h("p", { style: { margin: "0 0 8px" } },
      h("a", { class: "btn small", href: `#/course/${c.courseId}/explorer/${c.node.id}` }, "Open in Syllabus Explorer")));
  }
  append(root, overviewSection(c));
  if (!data.history_available) {
    append(root, h("div", { class: "banner info" }, data.message || "Counts appear after you run Analyze & Predict."),
      sourceSection(c));
    return { showPaper: () => {}, showQuestion: () => {} };
  }
  append(root, timelineSection(c), countsSection(c), formatsSection(c), questionsSection(c));
  if (c.p && c.p.format_guide) {
    append(root, section("Question format guide", renderFormatGuide(c.p.format_guide),
      h("p", { class: "help" }, "This guide describes the general form of a question. The real past questions are listed under Past questions above.")));
  }
  append(root, sourceSection(c));
  return { showPaper: (i) => c.showPaper(i), showQuestion: (id) => c.showQuestion(id) };
}

// A. Overview ---------------------------------------------------------
function overviewSection(c) {
  const n = c.node;
  const s = c.stats;
  const p = c.p;
  const out = h("section", { class: "card ex-section ex-overview" });
  const path = n.path || [];
  if (path.length) {
    append(out, h("nav", { class: "ex-crumbs", "aria-label": "Place in the syllabus" },
      path.map(a => [h("button", { class: "ex-link", onclick: () => c.navigate(a.id) }, nodeLabel(a)),
        h("span", { "aria-hidden": "true" }, "›")]),
      h("span", {}, nodeLabel(n))));
  }
  const original = n.original || nodeLabel(n);
  append(out, h("h2", { class: "ex-title" }, original));
  if (letters(original) !== letters(nodeLabel(n))) {
    append(out, h("p", { class: "muted", style: { margin: "0 0 6px" } },
      n.inferred ? "Name supplied by this app: " : "Short name in this app: ", h("strong", {}, nodeLabel(n))));
  }
  append(out, h("div", { class: "ex-tags" },
    badge(n.level_word || n.level || "item", n.level === "topic" ? "info" : ""),
    n.inferred ? h("span", { title: n.inferred_reason || "" }, kindTag("inferred"), " ",
      h("span", { class: "help" }, n.inferred_reason || "The name was inferred.")) : null,
    n.is_lab ? badge("lab or practical") : null,
    n.excluded ? badge("excluded from predictions", "bad") : null,
    (n.kinds || []).filter(k => k !== "lab").map(k => badge(k)),
    n.hours ? h("span", { class: "help" }, `${marksText(n.hours)} teaching hours`) : null));

  if (!s) return out;
  const a = s.exam_frequency;
  const tiles = [
    tile("fact", "Papers", `${a} of ${c.T} usable`,
      s.recent_window ? `${s.recent_hits} of the last ${s.recent_window}` : null),
    tile("fact", "Questions", String(s.question_frequency),
      s.secondary_questions > 0 ? `${s.secondary_questions} as secondary topic` : null),
    tile("fact", "Last appeared", s.last_label || "Not in the supplied papers", null),
    tile("fact", "Typical marks", s.marks && s.marks.known
      ? (s.marks.min === s.marks.max ? marksText(s.marks.min) : `${marksText(s.marks.min)}-${marksText(s.marks.max)}`) : "Not known",
    s.marks && s.marks.known ? `median ${marksText(s.marks.median)}, ${s.marks.known} with marks known` : null),
  ];
  if (p) {
    const word = priorityWord(p);
    tiles.push(tile("model", "Priority", a === 0 ? [word, " ", h("span", { class: "ex-v-note" }, "(no historical evidence)")] : word,
      p.rank ? `rank ${p.rank}${(p.uncertainty || {}).topics ? ` of ${p.uncertainty.topics}` : ""}` : null));
  }
  append(out, h("div", { class: "ex-tiles" }, tiles));
  if (s.provisional) {
    const reasons = s.provisional_reasons || [];
    append(out, h("details", { class: "ex-prov" },
      h("summary", {}, h("span", { class: "badge warn", title: reasons.join("; ") }, "Counts provisional")),
      reasons.length ? h("ul", { class: "evidence" }, reasons.map(r => h("li", {}, r))) : h("p", { class: "help" }, "Some inputs behind these counts are uncertain.")));
  }
  if (p) {
    const l = likelihood(p);
    append(out, h("div", { class: `ex-likely ${l.kind}` },
      h("div", { class: "ex-block-label" }, "Likelihood", kindTag(l.kind)),
      h("div", {}, l.text), l.note ? h("div", { class: "help" }, l.note) : null,
      rankRangeText(p) ? h("div", { class: "help" }, rankRangeText(p)) : null));
    if (p.priority_reason) append(out, h("p", { style: { margin: "6px 0 0" } }, h("strong", {}, "Why this priority: "), p.priority_reason));
    append(out, modelDetails(p));
    if (a === 0 && !/not evidence/i.test(p.priority_reason || "")) {
      append(out, h("p", { class: "help", style: { margin: "6px 0 0" } },
        "No past question on this item was found in the supplied papers. That is not evidence that it will not be examined."));
    }
  } else if (c.data.parent_prediction) {
    const pp = c.data.parent_prediction;
    append(out, h("p", { class: "help" }, "Predictions are made for whole topics. This item is part of a topic with ",
      priorityChip(pp), ` priority (rank ${pp.rank}).`));
  } else if (n.level !== "group") {
    append(out, h("p", { class: "help" }, n.excluded ? "This item is excluded, so it has no prediction." : "No prediction for this item in the latest analysis."));
  }
  if (n.level === "group") append(out, groupBlock(c));
  return out;
}

// The model's own explanation of the ranking, collapsed: evidence lines, component shares, the Bayesian
// recurrence estimate, the temporal pattern, the closest past questions and the logistic signal contributions.
function modelDetails(p) {
  const f = p.facts || {};
  const evidence = p.evidence || [];
  const whyNot = p.why_not || [];
  const shares = Object.entries(p.contributions || {}).sort((x, y) => y[1] - x[1]);
  const signals = Object.entries(p.signal_contributions || {}).sort((x, y) => Math.abs(y[1]) - Math.abs(x[1]));
  const b = f.bayes;
  const semantic = f.semantic_evidence || [];
  if (!evidence.length && !shares.length && !b) return null;
  const maxShare = Math.max(...shares.map(([, v]) => v), 0.01);
  const maxSignal = Math.max(...signals.map(([, v]) => Math.abs(v)), 0.01);
  return h("details", { class: "ex-more ex-model" },
    h("summary", {}, "How the model ranked it ", kindTag("model")),
    evidence.length ? [h("h4", {}, "Why it ranked here"), h("ul", { class: "evidence" }, evidence.map(e => h("li", {}, e)))] : null,
    whyNot.length ? [h("h4", {}, "Why it is not ranked higher"), h("ul", { class: "evidence" }, whyNot.map(e => h("li", {}, e)))] : null,
    shares.length ? [h("h4", {}, "Model contributions"),
      h("p", { class: "help" }, `Each component's share of the final score: ${p.contribution_source}. The shares add up to the topic's score.`),
      shares.map(([k, v]) => shareBar(k, v, maxShare, v.toFixed(3)))] : null,
    b ? [h("h4", {}, "Bayesian recurrence ", kindTag("estimate")),
      h("dl", { class: "facts" },
        h("dt", {}, "Chance of appearing"), h("dd", {}, `${pct(b.posterior_mean)} (median ${pct(b.posterior_median)})`),
        h("dt", {}, `${Math.round(100 * b.interval_level)}% credible interval`), h("dd", {}, `${pct(b.credible_interval[0])} to ${pct(b.credible_interval[1])}`),
        h("dt", {}, "Observed"), h("dd", {}, `${b.observed.appearances} of ${b.observed.exams} paper(s)`),
        h("dt", {}, "From the prior"), h("dd", {}, `${pct(b.prior_contribution)} (unit and course rate; shrinks as papers accumulate)`),
        h("dt", {}, "Effective sample size"), h("dd", {}, num(b.effective_sample_size, 1)),
        h("dt", {}, "Recency discount"), h("dd", {}, b.half_life ? `half-life ${b.half_life} papers` : "none")),
      h("p", { class: "help" }, "This rate is one input to the ranking. It is not the topic's final likelihood.")] : null,
    f.temporal_mode ? [h("h4", {}, "Temporal pattern"),
      h("p", {}, `Used: ${f.temporal_mode}.`, f.temporal_log_bayes_factor !== undefined && f.temporal_log_bayes_factor !== null
        ? ` Gap hazard vs constant rate on earlier papers: log Bayes factor ${num(f.temporal_log_bayes_factor, 2)} (positive favours the hazard).` : "")] : null,
    semantic.length ? [h("h4", {}, "Semantic evidence"),
      h("p", { class: "help" }, "Past in-syllabus questions closest in meaning to this topic (pretrained model)."),
      h("ul", { class: "evidence" }, semantic.map(e => h("li", {},
        `${e.exam}: "${e.text}" `, h("span", { class: "muted" }, `similarity ${num(e.similarity, 2)}${e.mapped_here ? ", mapped to this topic" : ""}`))))] : null,
    signals.length ? h("details", { style: { marginTop: "10px" } }, h("summary", {}, "Signal contributions (course-specific logistic model)"),
      h("p", { class: "help" }, `${p.signal_source}. Several signals measure similar things (frequency, recency, Bayesian rate), `,
        "so one can carry a large negative value while a related one carries a large positive value. Read the sum, ",
        "not individual signs; the model contributions above are the main explanation."),
      signals.map(([k, v]) => h("div", { class: "contrib" },
        h("span", {}, k),
        h("div", { class: "track" }, h("div", { class: "mid" }),
          h("div", { class: v >= 0 ? "pos" : "neg", style: { width: `${50 * Math.abs(v) / maxSignal}%` } })),
        h("span", { class: "mono" }, (v >= 0 ? "+" : "") + v.toFixed(2))))) : null);
}

function groupBlock(c) {
  const kids = c.data.children || [];
  const preds = c.data.child_predictions || [];
  const predById = new Map(preds.map(p => [p.topic_id, p]));
  const nodeById = new Map((c.nodes || []).map(n => [n.id, n]));
  const prio = (id) => predById.get(id) || (nodeById.get(id) || {}).prediction || null;
  const box = h("div", { style: { marginTop: "10px" } });
  if (preds.length) {
    append(box, h("div", { class: "ex-block-label" }, "Topics in this group by model ranking", kindTag("model")),
      h("ol", { class: "ex-childpreds" }, preds.map(p => h("li", {},
        h("button", { class: "ex-link", onclick: () => c.navigate(p.topic_id) }, p.label), " ",
        h("span", { class: "help" }, `rank ${p.rank}`), " ", priorityChip(p)))));
  }
  if (kids.length) {
    const rows = kids.map(k => {
      const st = k.stats || {};
      const tr = h("tr", { class: "clickable", onclick: () => c.navigate(k.id) },
        h("td", {}, h("button", { class: "ex-link", onclick: (e) => { e.stopPropagation(); c.navigate(k.id); } }, nodeLabel(k))),
        h("td", { class: "num" }, st.usable_papers !== undefined ? `${st.exam_frequency}/${st.usable_papers}` : ""),
        h("td", { class: "num" }, st.question_frequency ?? ""),
        h("td", {}, st.last_label || (st.usable_papers !== undefined ? "Not in the supplied papers" : "")),
        h("td", {}, priorityChip(prio(k.id))));
      return tr;
    });
    append(box, h("h4", { style: { marginTop: "12px" } }, "Items in this group"),
      h("div", { class: "table-wrap" }, h("table", { class: "data ex-kids" },
        h("thead", {}, h("tr", {}, h("th", {}, "Title"), h("th", { class: "num" }, "Papers"), h("th", { class: "num" }, "Questions"),
          h("th", {}, "Last appeared"), h("th", {}, "Priority"))),
        h("tbody", {}, rows))));
  }
  return box;
}

// B. History timeline ---------------------------------------------------
function timelineSection(c) {
  const tl = c.data.timeline || [];
  const qById = new Map(c.questions.map(q => [q.id, q]));
  const rows = tl.map(t => {
    const label = t.label || String(t.year || "");
    if (t.kind === "missing") {
      return h("tr", { class: "ex-missing" }, h("td", {}, label), h("td", { colspan: "3" }, "No paper in the supplied data"));
    }
    if (t.kind === "excluded") {
      return h("tr", { class: "ex-missing" }, h("td", {}, label), h("td", { colspan: "3" }, `Paper excluded: ${t.reason || t.excluded_kind || ""}`));
    }
    const qs = (t.questions || []).map(id => qById.get(id)).filter(Boolean);
    const go = () => c.showPaper(t.index);
    return h("tr", { class: "clickable", onclick: go },
      h("td", {}, h("button", { class: "ex-link", title: "Show this paper's questions", onclick: (e) => { e.stopPropagation(); go(); } }, label),
        t.undated ? h("div", { class: "help" }, "Undated: placed by upload order") : null,
        (t.provisional || []).length ? h("div", { class: "help" }, `Provisional: ${t.provisional.join("; ")}`) : null),
      h("td", {}, t.appeared ? badge("Yes", "ok") : badge("No")),
      h("td", {}, qs.length ? qs.map(q => `${q.question || "?"}${q.role === "secondary" ? " (secondary)" : ""}`).join(", ")
        : (t.questions || []).length ? String(t.questions.length) : ""),
      h("td", {}, (t.formats || []).join(", ")));
  });
  const missing = tl.filter(t => t.kind === "missing").map(t => t.label || t.year);
  const excluded = c.data.excluded_papers || [];
  return section(["History in past papers ", kindTag("fact")],
    h("p", { class: "help" }, "Select a paper to see its questions on this item."),
    h("div", { class: "table-wrap" }, h("table", { class: "data ex-timeline" },
      h("thead", {}, h("tr", {}, h("th", {}, "Year / paper"), h("th", {}, "Appeared?"), h("th", {}, "Questions"), h("th", {}, "Formats"))),
      h("tbody", {}, rows))),
    h("p", { class: "help", style: { marginTop: "8px" } },
      `${plural(c.T, "usable paper")}; missing years: ${missing.length ? missing.join(", ") : "none"}; excluded papers: `,
      excluded.length ? excluded.map(e => `${e.label} (${e.kind})`).join(", ") : "none", "."));
}

// C. Three counts --------------------------------------------------------
function countsSection(c) {
  const s = c.stats;
  const r = s.repetition || {};
  const box = (title, value, sub, def) => h("div", { class: "ex-tile" },
    h("div", { class: "ex-tile-head" }, h("span", { class: "l" }, title), kindTag("fact")),
    h("div", { class: "v" }, value), sub ? h("div", { class: "help" }, sub) : null, def);
  return section("How often it was asked",
    h("div", { class: "ex-counts" },
      box("Exam frequency", `${s.exam_frequency} of ${c.T} papers`,
        s.secondary_only_papers > 0 ? `${s.secondary_only_papers} only as a secondary topic` : null,
        h("p", { class: "help" }, "Papers with at least one question on it.")),
      box("Question frequency", plural(s.question_frequency, "question"),
        `${s.primary_questions} primary, ${s.secondary_questions} secondary`,
        h("p", { class: "help" }, "Primary: it is the question's main topic. Secondary: the question also covers it.")),
      box("Repetition", `${(r.exact || 0) + (r.paraphrase || 0)} repeated`, "Each question compared with the earlier papers:",
        h("ul", { class: "ex-replist" },
          h("li", {}, h("strong", {}, plural(r.exact || 0, "exact or near-exact repeat")), ": almost the same words as an earlier question"),
          h("li", {}, h("strong", {}, plural(r.paraphrase || 0, "paraphrase")), ": the same question in other words"),
          h("li", {}, h("strong", {}, `${r.concept || 0} same concept`), ": tests the same idea in a different way"),
          h("li", {}, h("strong", {}, `${r.new || 0} different or new`), ": no close earlier question")))),
    h("p", { class: "help", style: { marginTop: "8px" } },
      "Several similar questions in one paper count once for papers, and once each for questions."));
}

// D. Question formats ----------------------------------------------------
function formatsSection(c) {
  const s = c.stats;
  const labels = s.labels || [];
  const fams = s.families || [];
  if (!labels.length && !fams.length) {
    return section("Question formats", h("p", { class: "muted" }, "No past questions, so no format has been observed for this item."));
  }
  const top = fams[0];
  return section(["Question formats ", kindTag("fact")],
    labels.length ? h("div", { class: "table-wrap" }, h("table", { class: "data" },
      h("thead", {}, h("tr", {}, h("th", {}, "Format"), h("th", { class: "num" }, "Questions"), h("th", { class: "num" }, "Papers containing it"))),
      h("tbody", {}, labels.map(l => h("tr", {}, h("td", {}, l.display), h("td", { class: "num" }, l.questions), h("td", { class: "num" }, l.papers)))))) : null,
    s.multi_label_questions > 0 ? h("p", { class: "help" }, "A question can have several formats, so these counts overlap.") : null,
    top ? h("p", { style: { marginTop: "8px" } }, h("strong", {}, `Most common format: ${top.display}`),
      `, observed in ${top.papers} of ${c.T} usable papers (${top.papers} of the ${s.exam_frequency} papers containing this item).`) : null,
    fams.length > 1 ? h("p", {}, "Other observed formats: ", fams.slice(1).map(f => `${f.display} (${plural(f.papers, "paper")})`).join(", "), ".") : null,
    h("p", { class: "help" }, "The most frequent past format is not certain to appear next time."));
}

// E. Historical questions -------------------------------------------------
function questionsSection(c) {
  const box = h("section", { class: "card ex-section ex-questions" });
  const paperLabel = (i) => ((c.data.papers || []).find(p => p.index === i) || {}).label || `paper ${i + 1}`;
  function render() {
    clear(box);
    const list = c.paperFilter === null ? c.questions : c.questions.filter(q => q.paper_index === c.paperFilter);
    append(box, h("h3", {}, `Past questions (${c.questions.length})`, kindTag("fact")),
      h("p", { class: "help" }, "Newest first, in the original wording."));
    if (c.paperFilter !== null) {
      append(box, h("div", { class: "banner info ex-filter-note" }, `Showing questions from ${paperLabel(c.paperFilter)} only. `,
        h("button", { class: "small", onclick: () => { c.paperFilter = null; render(); } }, "Show all papers")));
    }
    if (!list.length) append(box, h("p", { class: "muted" }, c.paperFilter === null ? "No counted past questions on this item." : "No counted question on this item in that paper."));
    for (const q of list) append(box, questionItem(c, q));
    const unc = c.data.uncounted || [];
    if (unc.length) {
      append(box, h("details", { class: "ex-more" },
        h("summary", {}, `Possible matches not counted (uncertain mapping) (${unc.length})`),
        h("p", { class: "help" }, "These questions might belong here, but the match was too uncertain to count. Correct the mapping if you know where they belong."),
        unc.map(u => h("div", { class: "ex-q" },
          h("div", { class: "ex-qhead" }, h("strong", {}, u.paper), u.question ? h("span", { class: "mono" }, `Q ${u.question}`) : null, u.status ? statusBadge(u.status) : null),
          h("div", { class: "ex-qtext" }, u.text || "(text not available)"),
          (u.matches || []).length ? h("div", { class: "help" }, "Possible topics: ",
            u.matches.map(m => `${m.label} (${m.status}, score ${num(m.score, 2)})`).join("; ")) : null,
          h("div", { class: "ex-actions" },
            u.text ? h("button", { class: "small", onclick: () => openQuestionSource(u.id) }, "View original") : null,
            h("button", { class: "small", onclick: () => correct(c, { id: u.id, paper: u.paper, question: u.question, text: u.text, manual: false }) }, "Correct mapping"))))));
    }
    const rel = c.data.related || [];
    if (rel.length) {
      append(box, h("details", { class: "ex-more" },
        h("summary", {}, `Related questions mapped to other topics (not counted) (${rel.length})`),
        h("p", { class: "help" }, "Past questions closest in meaning to this item. They are counted for the topic they are mapped to, not here."),
        rel.map(r => h("div", { class: "ex-q" },
          h("div", { class: "ex-qhead" }, h("strong", {}, r.exam), h("span", { class: "help" }, `similarity ${num(r.similarity, 2)}`)),
          h("div", { class: "ex-qtext" }, r.text),
          h("div", { class: "ex-actions" }, h("button", { class: "small", onclick: () => openQuestionSource(r.question_id) }, "View original"))))));
    }
  }
  c.showPaper = (index) => {
    c.paperFilter = index;
    render();
    box.scrollIntoView({ behavior: "smooth", block: "start" });
  };
  c.showQuestion = (qid) => {
    if (c.paperFilter !== null) { c.paperFilter = null; render(); }
    const el = box.querySelector(`[data-qid="${qid}"]`);
    if (!el) { openQuestionSource(qid); return; }
    el.scrollIntoView({ behavior: "smooth", block: "center" });
    el.classList.add("flash");
    setTimeout(() => el.classList.remove("flash"), 1600);
  };
  render();
  return box;
}

function questionItem(c, q) {
  const raw = !q.missing && q.raw_text && squash(q.raw_text) !== squash(q.text);
  const earlier = q.earlier || [];
  return h("article", { class: "ex-q", "data-qid": q.id },
    h("div", { class: "ex-qhead" },
      h("strong", {}, q.paper), q.question ? h("span", { class: "mono" }, `Q ${q.question}`) : null,
      q.marks !== null && q.marks !== undefined ? h("span", { class: "muted" }, `${marksText(q.marks)} marks`) : null,
      badge(q.role === "secondary" ? "secondary topic" : "primary topic", q.role === "secondary" ? "" : "info"),
      q.status ? statusBadge(q.status) : null,
      q.confidence !== null && q.confidence !== undefined ? h("span", { class: "help", title: "How strongly the question matches this syllabus item" }, `match ${pct(q.confidence)}`) : null,
      q.manual ? badge("set by you", "warn") : null,
      q.is_optional ? h("span", { class: "help" }, "choice question") : null),
    (q.labels || []).length ? h("div", { class: "ex-tags", style: { margin: "4px 0 0" } }, q.labels.map(l => badge(l))) : null,
    q.missing ? h("p", { class: "muted" }, "The text of this question is no longer stored (the paper may have been processed again). Re-analyse to refresh.")
      : h("blockquote", { class: "ex-qtext" }, q.text),
    raw ? h("details", {}, h("summary", { class: "help" }, "As extracted"), h("pre", { class: "pagetext" }, q.raw_text)) : null,
    h("div", { class: "ex-qmeta" },
      q.relation_text ? h("div", {}, q.relation_text, earlier.length ? ": " : "",
        earlier.map((e, i) => [i ? ", " : "", h("button", { class: "ex-link", onclick: () => c.showQuestion(e.id) }, `${e.paper} ${e.question || ""}`.trim())])) : null,
      (q.other_topics || []).length ? h("div", {}, "Also mapped to: ",
        q.other_topics.map((t, i) => [i ? ", " : "", h("button", { class: "ex-link", onclick: () => c.navigate(t.id) }, t.label), ` (${t.role})`])) : null,
      q.file ? h("div", {}, `Source: ${where(q.file, q.page, null)}`) : null,
      (q.provisional || []).length ? h("div", {}, h("span", { class: "badge warn" }, "Provisional"), " ", q.provisional.join("; ")) : null),
    h("div", { class: "ex-actions" },
      q.missing ? null : h("button", { class: "small", onclick: () => openQuestionSource(q.id) }, "View original"),
      h("button", { class: "small", onclick: () => correct(c, q) }, "Correct mapping")));
}

function correct(c, q) {
  const n = c.node;
  const preset = { primary: null, secondary: [] };
  if (n.level !== "group" && q.role) {
    const others = q.other_topics || [];
    if (q.role === "primary") preset.primary = n.id;
    else { preset.secondary.push(n.id); const prim = others.find(t => t.role === "primary"); if (prim) preset.primary = prim.id; }
    for (const t of others) if (t.role === "secondary" && t.id !== preset.primary) preset.secondary.push(t.id);
  }
  openMappingModal(q, { courseId: c.courseId, nodes: c.nodes, preset, onSaved: c.changed });
}

// F. Format guide ---------------------------------------------------------
function withPlaceholders(text) {
  return String(text || "").split(/(\[[^\]]+\])/).filter(Boolean)
    .map(part => (/^\[[^\]]+\]$/.test(part) ? h("span", { class: "ex-ph" }, part) : part));
}

export function templateBlock(t) {
  const ph = t.placeholders || [];
  return h("div", { class: "ex-template" },
    h("div", { class: "ex-block-label" }, "General template"),
    h("p", { class: "ex-template-text" }, withPlaceholders(t.text)),
    ph.length ? h("p", { class: "help" }, "Fill in: ", ph.map(x => `${x.name}${x.unit ? ` (${x.unit})` : ""}${x.example ? `, for example ${x.example}` : ""}`).join("; "), ".") : null,
    t.note ? h("p", { class: "help" }, t.note) : null);
}

export function illustrativeBlock(il) {
  const src = il.source;
  return h("div", { class: "ex-illustrative" },
    h("div", { class: "ex-block-label" }, "Illustrative practice question - not a real exam question", kindTag("illustrative")),
    h("p", { class: "ex-illustrative-text" }, il.text),
    il.basis === "past_values" && src && !String(il.note || "").includes(src.exam)
      ? h("p", { class: "help" }, `The values come from ${src.exam} question ${src.question}, a real past paper.`) : null,
    il.marks_low !== null && il.marks_low !== undefined ? h("p", { class: "help" },
      `Marks in similar past questions: ${il.marks_low === il.marks_high ? marksText(il.marks_low) : `${marksText(il.marks_low)}-${marksText(il.marks_high)}`}.`) : null,
    il.note ? h("p", { class: "help" }, il.note) : null);
}

export function formatEvidenceTag(g) {
  return kindTag(g.evidence === "established" ? "fact" : g.evidence === "weak" ? "weak" : "inferred");
}

export function renderFormatGuide(g, { alternatives = 2 } = {}) {
  const tag = formatEvidenceTag(g);
  const alts = (g.alternatives || []).slice(0, alternatives);
  return h("div", { class: "ex-guide" },
    h("div", { class: "ex-guide-head" }, h("strong", {}, `Suggested format: ${g.display}`), tag),
    g.description ? h("p", { style: { margin: "6px 0" } }, g.description) : null,
    g.why ? h("p", {}, h("strong", {}, "Why this format: "), g.why) : null,
    g.marks_note ? h("p", { class: "help" }, g.marks_note) : null,
    g.reliability && g.reliability.text ? h("p", { class: "help" }, g.reliability.text) : null,
    g.template ? templateBlock(g.template) : null,
    g.illustrative ? illustrativeBlock(g.illustrative) : g.illustrative_note ? h("p", { class: "help" }, g.illustrative_note) : null,
    alts.length ? h("div", { class: "ex-alts" }, h("h4", { style: { margin: "12px 0 4px" } }, "Other formats seen for this item"),
      alts.map(a => h("div", { class: "ex-alt" },
        h("div", {}, h("strong", {}, a.display), a.support ? h("span", { class: "help" }, ` - ${a.support}`) : null),
        a.description ? h("p", { style: { margin: "4px 0" } }, a.description) : null,
        a.template ? templateBlock(a.template) : null,
        a.illustrative && typeof a.illustrative === "object" ? illustrativeBlock(a.illustrative) : null))) : null);
}

// G. Syllabus source -------------------------------------------------------
function sourceSection(c) {
  const n = c.node;
  const refs = n.source_refs || [];
  const kids = c.data.children || [];
  const confs = c.questions.map(q => q.confidence).filter(v => v !== null && v !== undefined);
  const mean = confs.length ? confs.reduce((x, y) => x + y, 0) / confs.length : null;
  const unc = c.data.uncounted || [];
  const s = c.stats;
  return section("Syllabus source",
    refs.length ? refs.map(r => h("div", { class: "ex-ref" },
      h("div", { class: "help" }, where(r.file, r.page, r.line)),
      h("pre", { class: "ex-passage" }, r.passage || r.text || ""),
      r.file_id ? h("button", { class: "small", onclick: () => openSyllabusSource(r) }, "View source passage") : null))
      : h("p", { class: "muted" }, n.inferred ? `No source location: ${n.inferred_reason || "the name was inferred."}` : "No source location was recorded for this item."),
    n.description ? h("p", {}, n.description) : null,
    (n.concepts || []).length ? h("p", {}, h("strong", {}, "Concepts: "), n.concepts.join("; ")) : null,
    kids.length ? h("p", {}, h("strong", {}, "Related sub-topics: "),
      kids.map((k, i) => [i ? ", " : "", h("button", { class: "ex-link", onclick: () => c.navigate(k.id) }, nodeLabel(k))])) : null,
    mean !== null ? h("p", {}, h("strong", {}, "Mapping confidence: "), `mean ${pct(mean)} over ${plural(confs.length, "question")}`) : null,
    unc.length ? h("p", { class: "help" }, `Ambiguity: ${plural(unc.length, "question")} might belong here, but the mapping was too uncertain to count. See Past questions.`) : null,
    s && s.provisional ? h("p", { class: "help" }, `Ambiguity: the counts are provisional (${(s.provisional_reasons || []).join("; ") || "uncertain inputs"}).`) : null);
}

// ------------------------------------------------------------------ source modals
// fraction: where the passage roughly sits on the page (line / lines), used to scroll it into view.
function pageImage(fileId, page, highlight, fallback, fraction = null) {
  const url = `/api/files/${fileId}/pages/${page}/image?highlight=${encodeURIComponent(highlight || "")}`;
  const holder = h("div", { class: "ex-page" });
  const img = h("img", { class: "ex-page-img", src: url, alt: `Page ${page} with the passage highlighted` });
  img.addEventListener("load", () => {
    const box = img.closest(".modal, .drawer");
    if (!box || fraction === null || fraction < 0.3) return;
    const top = img.getBoundingClientRect().top - box.getBoundingClientRect().top + box.scrollTop;
    box.scrollTop = Math.max(0, top + fraction * img.clientHeight - box.clientHeight / 2);
  });
  img.addEventListener("error", () => {
    clear(holder);
    append(holder, h("p", { class: "help" }, "The page image could not be made; showing the extracted text instead."), fallback());
  });
  holder.appendChild(img);
  return holder;
}

function pageText(text, line, span = 1) {
  const pre = h("pre", { class: "pagetext ex-pagetext" });
  const lines = String(text || "").split("\n");
  let mark = null;
  lines.forEach((ln, i) => {
    const no = i + 1;
    const hit = line && no >= line && no < line + span;
    if (hit) {
      const m = h("mark", { class: "ex-hl" }, ln);
      if (!mark) mark = m;
      pre.appendChild(m);
    } else pre.appendChild(document.createTextNode(ln));
    if (i < lines.length - 1) pre.appendChild(document.createTextNode("\n"));
  });
  if (mark) requestAnimationFrame(() => mark.scrollIntoView({ block: "center" }));
  return pre;
}

export async function openQuestionSource(qid) {
  let s;
  try { s = await api.get(`/api/questions/${qid}/source`); } catch (e) { toast(e.message, "error"); return; }
  const fallback = () => (s.page_text ? pageText(s.page_text, s.line, Math.max(1, (s.raw_text || "").split("\n").length))
    : h("pre", { class: "pagetext" }, s.raw_text || s.text || ""));
  const lineCount = s.page_text ? s.page_text.split("\n").length : 0;
  const fraction = lineCount && s.line ? s.line / lineCount : null;
  const view = s.is_pdf && s.file_id && s.page ? pageImage(s.file_id, s.page, (s.text || "").slice(0, 60), fallback, fraction) : fallback();
  modal(`${s.exam || "Past paper"}, question ${s.question || ""}`.trim(), h("div", {},
    h("p", { class: "help" }, where(s.file, s.page, s.line), s.is_pdf ? ". The question is highlighted on the page." : ". The question's line is highlighted."),
    view,
    h("details", { style: { marginTop: "8px" } }, h("summary", {}, "Question text as stored"), h("pre", { class: "pagetext" }, s.text || ""))), { wide: true });
}

export async function openSyllabusSource(ref) {
  const title = `Syllabus source: ${where(ref.file, ref.page, null)}`;
  const fallback = () => h("pre", { class: "pagetext" }, ref.passage || ref.text || "");
  let text = null;
  if (ref.file_id && ref.page) {
    try {
      const pages = await api.get(`/api/files/${ref.file_id}/pages`);
      text = (pages.find(p => p.page === ref.page) || {}).text || null;
    } catch (_) { /* fall back to the stored passage */ }
  }
  if (ref.is_pdf && ref.file_id && ref.page) {
    const lineCount = text ? text.split("\n").length : 0;
    modal(title, h("div", {}, h("p", { class: "help" }, "The syllabus passage is highlighted on the page."),
      pageImage(ref.file_id, ref.page, ref.text || "", fallback, lineCount && ref.line ? ref.line / lineCount : null)), { wide: true });
    return;
  }
  modal(title, h("div", {}, h("p", { class: "help" }, text ? "The syllabus line is highlighted." : "Extracted passage."),
    text ? pageText(text, ref.line, Math.max(1, String(ref.text || "").split("\n").length)) : fallback()), { wide: true });
}

// ------------------------------------------------------------------ correct mapping
export async function openMappingModal(q, { courseId, nodes, preset = {}, onSaved } = {}) {
  if (!nodes) {
    try { nodes = (await api.get(`/api/courses/${courseId}/explorer`)).nodes; } catch (e) { toast(e.message, "error"); return; }
  }
  const byId = new Map(nodes.map(n => [n.id, n]));
  const pathOf = (n) => {
    const out = [];
    for (let p = byId.get(n.parent_id); p; p = byId.get(p.parent_id)) out.unshift(nodeLabel(p));
    return out.join(" › ");
  };
  const state = { primary: preset.primary && byId.has(preset.primary) ? preset.primary : null,
    secondary: new Set((preset.secondary || []).filter(id => byId.has(id) && id !== preset.primary)), outside: false };
  // The current topics are listed first so they are visible when the modal opens.
  const current = new Set([state.primary, ...state.secondary].filter(Boolean));
  const ordered = syllabusOrder(nodes);
  const choices = [...ordered.filter(id => current.has(id)), ...ordered.filter(id => !current.has(id))]
    .map(id => byId.get(id)).filter(n => n.level !== "group" && !n.excluded)
    .map(n => ({ n, path: pathOf(n), hay: `${nodeLabel(n)} ${n.original || ""} ${pathOf(n)} ${(n.concepts || []).join(" ")}`.toLowerCase() }));
  const name = `ex-primary-${q.id}`;
  const search = h("input", { type: "search", placeholder: "Search topics by number or name", "aria-label": "Search topics", style: { width: "100%" } });
  const list = h("div", { class: "ex-pick", role: "group", "aria-label": "Topics" });
  const summary = h("p", { class: "help", "aria-live": "polite" });
  const outside = h("input", { type: "checkbox" });
  const status = h("select", {}, h("option", { value: "A" }, "A: clearly in the syllabus"), h("option", { value: "B" }, "B: probably in the syllabus"));

  function renderSummary() {
    if (state.outside) { summary.textContent = "The question will be marked as outside the syllabus and will not count for any topic."; return; }
    const lbl = (id) => nodeLabel(byId.get(id));
    summary.textContent = `Primary topic: ${state.primary ? lbl(state.primary) : "none chosen yet"}. Secondary: ${state.secondary.size ? [...state.secondary].map(lbl).join("; ") : "none"}.`;
  }
  function renderList() {
    clear(list);
    const term = search.value.trim().toLowerCase();
    const hits = choices.filter(ch => !term || ch.hay.includes(term));
    for (const ch of hits.slice(0, 200)) {
      const id = ch.n.id;
      const radio = h("input", { type: "radio", name, value: id, checked: state.primary === id, disabled: state.outside });
      const check = h("input", { type: "checkbox", checked: state.secondary.has(id), disabled: state.outside });
      radio.addEventListener("change", () => { state.primary = id; state.secondary.delete(id); renderList(); renderSummary(); });
      check.addEventListener("change", () => {
        if (check.checked) { state.secondary.add(id); if (state.primary === id) state.primary = null; } else state.secondary.delete(id);
        renderList(); renderSummary();
      });
      list.appendChild(h("div", { class: `ex-pick-row${state.primary === id ? " chosen" : ""}` },
        h("div", { class: "ex-pick-name" }, nodeLabel(ch.n), ch.path ? h("div", { class: "help" }, ch.path) : null),
        h("label", {}, radio, "Primary"), h("label", {}, check, "Secondary")));
    }
    if (!hits.length) list.appendChild(h("p", { class: "muted", style: { padding: "8px" } }, "No topic matches your search."));
    else if (hits.length > 200) list.appendChild(h("p", { class: "help", style: { padding: "8px" } }, `Showing 200 of ${hits.length}. Refine the search to see the rest.`));
  }
  let timer = null;
  search.addEventListener("input", () => { clearTimeout(timer); timer = setTimeout(renderList, 120); });
  outside.addEventListener("change", () => { state.outside = outside.checked; status.disabled = outside.checked; search.disabled = outside.checked; renderList(); renderSummary(); });

  const cancel = h("button", {}, "Cancel");
  const reset = q.manual ? h("button", {}, "Use automatic mapping") : null;
  const save = h("button", { class: "primary" }, "Save");
  const m = modal("Correct mapping", h("div", { class: "grid ex-mapping" },
    h("div", {}, h("div", { class: "help" }, [q.paper, q.question ? `question ${q.question}` : null].filter(Boolean).join(", ")),
      h("blockquote", { class: "ex-qtext" }, q.text || "(text not available)"),
      h("p", { class: "help" }, q.manual ? "You set the current mapping yourself." : "The current mapping was made automatically.")),
    h("label", { class: "field" }, h("span", {}, "Find a topic"), search),
    list, summary,
    h("div", { class: "row" }, h("label", { class: "row", style: { gap: "6px" } }, outside, "Outside the syllabus (no topic)"),
      h("span", { class: "spacer" }), h("label", { class: "row", style: { gap: "6px" } }, "Status", status)),
    h("div", { class: "row end" }, cancel, reset, save)));
  cancel.addEventListener("click", () => m.close());
  save.addEventListener("click", async () => {
    if (!state.outside && !state.primary) { toast("Choose a primary topic, or mark the question as outside the syllabus.", "error"); return; }
    const body = state.outside ? { topic_ids: [], status: "D" } : { topic_ids: [state.primary, ...state.secondary], status: status.value };
    save.disabled = true;
    try {
      await api.put(`/api/questions/${q.id}/mapping`, body);
      m.close();
      toast("Saved. The counts and predictions update after you re-analyse.");
      if (onSaved) onSaved();
    } catch (e) { toast(e.message, "error", 8000); save.disabled = false; }
  });
  if (reset) {
    reset.addEventListener("click", async () => {
      try {
        await api.del(`/api/questions/${q.id}/mapping`);
        m.close();
        toast("Your correction was removed. The automatic mapping is used from the next analysis.");
        if (onSaved) onSaved();
      } catch (e) { toast(e.message, "error", 8000); }
    });
  }
  renderList();
  renderSummary();
  search.focus();
}

// ------------------------------------------------------------------ freshness
export function freshnessBanner(fr, { courseId, onDone, onStart } = {}) {
  if (!fr || !fr.stale) return null;
  const btn = h("button", { class: "primary small" }, "Re-analyse now");
  const progress = h("div");
  const el = h("div", { class: "banner warn ex-stale", role: "status" },
    h("p", { style: { margin: 0 } }, h("strong", {}, "Results out of date. "),
      fr.message || "Your data changed after the last analysis. Re-analyse to update the counts and predictions."),
    h("div", { class: "row", style: { marginTop: "8px" } }, btn), progress);
  btn.addEventListener("click", () => (onStart ? onStart() : reanalyse(courseId, btn, progress, onDone || (() => {
    clear(el);
    el.className = "banner info";
    append(el, "Analysis finished. Open this item again to see the updated counts.");
  }))));
  return el;
}

async function reanalyse(courseId, btn, progress, onDone) {
  let timer = null;
  let stopped = false;
  window.addEventListener("predictor:leave", () => { stopped = true; clearTimeout(timer); }, { once: true });
  btn.disabled = true;
  let run;
  try { run = await api.post(`/api/courses/${courseId}/analyze`); } catch (e) { toast(e.message, "error", 8000); btn.disabled = false; return; }
  async function poll() {
    if (stopped) return;
    let r;
    try { r = await api.get(`/api/runs/${run.id}`); } catch (e) { toast(e.message, "error"); btn.disabled = false; return; }
    if (stopped) return;
    clear(progress);
    if (r.status === "queued" || r.status === "running") {
      append(progress, h("p", { class: "help", style: { margin: "8px 0 4px" } }, r.message || "Working..."),
        h("div", { class: "progress" }, h("div", { style: { width: `${Math.round(100 * (r.progress || 0))}%` } })));
      timer = setTimeout(poll, 1000);
      return;
    }
    btn.disabled = false;
    if (r.status === "error") {
      append(progress, h("p", { style: { margin: "8px 0 0" } }, `The analysis could not finish: ${r.message || "unknown error"}`));
      return;
    }
    toast("Analysis finished. The counts and predictions are up to date.");
    onDone();
  }
  poll();
}
