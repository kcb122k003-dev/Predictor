import { api } from "./api.js";
import { append, badge, clear, h, modal, table, toast } from "./dom.js";

const TYPE_OPTIONS = ["definition", "conceptual_explanation", "short_answer", "long_theory", "derivation", "numerical",
  "problem_solving", "compare_contrast", "diagram", "explain_with_example", "application", "design",
  "analytical_reasoning", "mcq", "true_false", "fill_blank", "mixed", "unclassified"];

export async function renderReview(main, course, { refreshCourse }) {
  const examsBox = h("div");
  const treeBox = h("div");
  let selected = null;
  let onlyReview = false;
  append(main, 
    h("h1", {}, "Review papers"),
    h("p", { class: "muted" }, "Check the year and order of each paper, then the extracted questions. Fix anything the ",
      "parser got wrong. Your edits are kept when the analysis runs again."),
    examsBox, treeBox);

  async function loadExams() {
    const exams = await api.get(`/api/courses/${course.id}/exams`);
    clear(examsBox);
    if (!exams.length) { examsBox.appendChild(h("div", { class: "empty" }, "No papers yet. Upload them first.")); return exams; }
    examsBox.appendChild(table([
      { label: "Paper", render: e => h("div", { style: { minWidth: "200px" } }, h("strong", {}, e.label || `Exam ${e.id}`),
          h("div", { class: "help" }, e.file || ""),
          e.exclusion_reason ? h("div", { class: "help", style: { color: "var(--warn)" } }, e.exclusion_reason) : null,
          (e.warnings || []).map(w => h("div", { class: "help" }, w))) },
      { label: "Year", render: e => editNum(e, "year", conf(e, "year")) },
      { label: "Calendar", render: e => editSelect(e, "calendar", ["AD", "BS", "unknown"]) },
      { label: "Session", render: e => editText(e, "session", 8) },
      { label: "Type", render: e => editText(e, "exam_type", 9) },
      { label: "Order", render: e => editNum(e, "order_index", null, 0.01) },
      { label: "Full marks", render: e => editNum(e, "full_marks") },
      { label: "Questions", render: e => h("div", {}, String(e.questions), e.needs_review ? h("div", {}, badge(`${e.needs_review} to review`, "warn")) : null), num: true },
      { label: "Include", render: e => {
          const box = h("input", { type: "checkbox" });
          box.checked = e.include_in_analysis;
          box.addEventListener("click", (ev) => ev.stopPropagation());
          box.addEventListener("change", async () => { await api.patch(`/api/exams/${e.id}`, { include_in_analysis: box.checked }); loadExams(); refreshCourse(); });
          return box;
        } },
    ], exams, {
      onRowClick: (e) => { selected = e.id; loadTree(); highlight(); },
      rowClass: (e) => e.id === selected ? "selected" : "",
    }));
    examsBox.appendChild(h("p", { class: "help" }, "Order decides the time sequence used for recency and backtesting. ",
      "It is computed from year and session; edit it when two papers share a year. Click a row to review its questions."));
    return exams;
  }

  function highlight() {
    examsBox.querySelectorAll("tr").forEach(tr => tr.classList.remove("selected"));
  }

  function conf(e, key) {
    const c = (e.metadata_confidence || {})[key];
    if (!c) return null;
    return c.confidence < 0.6 ? badge(`low confidence`, "warn") : null;
  }

  function editNum(e, key, extra, step = 1) {
    const input = h("input", { type: "number", step, value: e[key] ?? "", style: { width: key === "order_index" ? "96px" : "74px" } });
    input.addEventListener("click", ev => ev.stopPropagation());
    input.addEventListener("change", async () => {
      const v = input.value === "" ? null : Number(input.value);
      await api.patch(`/api/exams/${e.id}`, { [key]: key === "year" && v !== null ? Math.round(v) : v });
      toast("Saved."); loadExams(); refreshCourse();
    });
    return h("div", {}, input, extra);
  }

  function editText(e, key, size) {
    const input = h("input", { value: e[key] || "", style: { width: `${size * 9}px` } });
    input.addEventListener("click", ev => ev.stopPropagation());
    input.addEventListener("change", async () => { await api.patch(`/api/exams/${e.id}`, { [key]: input.value }); toast("Saved."); loadExams(); });
    return input;
  }

  function editSelect(e, key, options) {
    const sel = h("select", {}, options.map(o => h("option", { value: o }, o)));
    sel.value = e[key] || options[0];
    sel.addEventListener("click", ev => ev.stopPropagation());
    sel.addEventListener("change", async () => { await api.patch(`/api/exams/${e.id}`, { [key]: sel.value }); loadExams(); });
    return sel;
  }

  async function loadTree() {
    clear(treeBox);
    if (!selected) return;
    const data = await api.get(`/api/exams/${selected}/questions`);
    const filter = h("input", { type: "checkbox" });
    filter.checked = onlyReview;
    filter.addEventListener("change", () => { onlyReview = filter.checked; loadTree(); });
    const pagesBtn = h("button", { class: "small" }, "Show extracted text");
    pagesBtn.addEventListener("click", () => modal("Extracted text", h("div", {}, data.pages.map(p => h("div", {},
      h("h4", {}, `Page ${p.page} (${p.method})`), h("pre", { class: "pagetext" }, p.text)))), { wide: true }));
    const addBtn = h("button", { class: "small" }, "Add question");
    addBtn.addEventListener("click", () => addQuestion(null));
    append(treeBox, h("div", { class: "card" },
      h("div", { class: "row" }, h("h3", { style: { margin: 0 } }, "Questions"), h("span", { class: "spacer" }),
        h("label", { class: "row" }, filter, "Only items needing review"), pagesBtn, addBtn),
      data.sections.length ? h("p", { class: "help" }, `Sections: ${data.sections.map(s => `${s.label}${s.attempt_count ? ` (attempt any ${s.attempt_count})` : ""}`).join(", ")}`) : null,
      h("div", { class: "qtree" }, data.questions.map(q => node(q)))));
  }

  function hasReview(q) { return q.needs_review || (q.children || []).some(hasReview); }

  function node(q) {
    if (onlyReview && !hasReview(q)) return null;
    const text = h("div", { class: "qtext" }, q.text || h("span", { class: "muted" }, "(stem only)"));
    const marks = q.marks !== null && q.marks !== undefined ? badge(`${q.marks} marks${q.marks_source === "distributed" ? " (shared)" : ""}`, "info") : badge("no marks");
    const types = (q.types || []).map(t => badge(t.replaceAll("_", " ")));
    const flags = (q.flags || []).filter(f => !f.startsWith("attempt_any")).map(f => badge(f.replaceAll("_", " "), "warn"));
    const actions = h("div", { class: "qactions" },
      h("button", { class: "small", onclick: () => edit(q) }, "Edit"),
      q.is_leaf ? h("button", { class: "small", onclick: () => split(q) }, "Split") : null,
      q.is_leaf ? h("button", { class: "small", onclick: () => merge(q) }, "Merge with next") : null,
      h("button", { class: "small", onclick: () => addQuestion(q) }, "Add sub-part"),
      q.needs_review ? h("button", { class: "small", onclick: async () => { await api.patch(`/api/questions/${q.id}`, { needs_review: false }); loadTree(); loadExams(); } }, "Looks right") : null,
      h("button", { class: "small danger", onclick: async () => { if (confirm("Delete this question and its sub-parts?")) { await api.del(`/api/questions/${q.id}`); loadTree(); loadExams(); } } }, "Delete"));
    return h("div", { class: `qnode ${q.needs_review ? "review" : ""}` },
      h("div", { class: "qhead" }, h("span", { class: "qlabel" }, q.path_label), text),
      h("div", { class: "row", style: { marginTop: "4px" } }, marks, q.or_group ? badge(`OR group ${q.or_group}`, "info") : null,
        q.is_optional ? badge("optional") : null, types, flags,
        q.page ? h("span", { class: "help" }, `page ${q.page}`) : null,
        q.equations && q.equations.length ? h("span", { class: "help mono" }, q.equations.map(e => e.raw).join("; ")) : null),
      q.options && q.options.length ? h("ol", { type: "a" }, q.options.map(o => h("li", {}, o))) : null,
      actions,
      q.children && q.children.length ? h("div", { class: "children" }, q.children.map(c => node(c))) : null);
  }

  function edit(q) {
    const text = h("textarea", { rows: 4 }, q.text || "");
    const marks = h("input", { type: "number", step: "0.5", value: q.marks ?? "" });
    const types = h("select", { multiple: true, size: 6 }, TYPE_OPTIONS.map(t => h("option", { value: t }, t.replaceAll("_", " "))));
    for (const opt of types.options) opt.selected = (q.types || []).includes(opt.value);
    const optional = h("input", { type: "checkbox" });
    optional.checked = q.is_optional;
    const save = h("button", { class: "primary" }, "Save");
    const m = modal(`Edit ${q.path_label}`, h("div", { class: "grid" },
      h("label", { class: "field" }, h("span", {}, "Question text"), text),
      h("div", { class: "row" }, h("label", { class: "field" }, h("span", {}, "Marks"), marks),
        h("label", { class: "field" }, h("span", {}, "Question types (Ctrl/Cmd for several)"), types),
        h("label", { class: "row" }, optional, "Optional (student chooses)")),
      h("div", { class: "row end" }, save)));
    save.addEventListener("click", async () => {
      const chosen = [...types.selectedOptions].map(o => o.value);
      const body = { text: text.value, marks: marks.value === "" ? null : Number(marks.value), is_optional: optional.checked };
      if (JSON.stringify(chosen) !== JSON.stringify(q.types || [])) body.question_types = chosen;
      await api.patch(`/api/questions/${q.id}`, body);
      m.close(); toast("Saved."); loadTree(); loadExams();
    });
  }

  function split(q) {
    const area = h("textarea", { rows: 8 }, q.text);
    const go = h("button", { class: "primary" }, "Split");
    const m = modal(`Split ${q.path_label}`, h("div", {},
      h("p", { class: "muted" }, "Put each part on its own line. Each line becomes a separate question."), area,
      h("div", { class: "row end", style: { marginTop: "8px" } }, go)));
    go.addEventListener("click", async () => {
      const parts = area.value.split("\n").map(s => s.trim()).filter(Boolean);
      await api.post(`/api/questions/${q.id}/split`, { parts });
      m.close(); loadTree(); loadExams();
    });
  }

  async function merge(q) {
    await api.post(`/api/questions/${q.id}/merge-next`);
    loadTree(); loadExams();
  }

  function addQuestion(parent) {
    const label = h("input", { placeholder: parent ? "e.g. c" : "e.g. 7", size: 6 });
    const text = h("textarea", { rows: 4 });
    const marks = h("input", { type: "number", step: "0.5" });
    const go = h("button", { class: "primary" }, "Add");
    const m = modal(parent ? `Add a sub-part to ${parent.path_label}` : "Add a question", h("div", { class: "grid" },
      h("div", { class: "row" }, h("label", { class: "field" }, h("span", {}, "Label"), label),
        h("label", { class: "field" }, h("span", {}, "Marks"), marks)),
      h("label", { class: "field" }, h("span", {}, "Text"), text), h("div", { class: "row end" }, go)));
    go.addEventListener("click", async () => {
      await api.post(`/api/exams/${selected}/questions`, { parent_id: parent ? parent.id : null, label: label.value || "?",
        text: text.value, marks: marks.value === "" ? null : Number(marks.value) });
      m.close(); loadTree(); loadExams();
    });
  }

  const exams = await loadExams();
  if (exams && exams.length) { selected = exams[0].id; await loadTree(); }
}
