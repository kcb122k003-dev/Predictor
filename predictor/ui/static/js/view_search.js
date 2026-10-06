import { api } from "./api.js";
import { append, clear, field, h, statusBadge, table } from "./dom.js";

export async function renderSearch(main, course) {
  const q = h("input", { placeholder: "e.g. Bernoulli, entropy, venturimeter", style: { width: "100%" } });
  const year = h("input", { type: "number", placeholder: "any" });
  const qtype = h("select", {}, h("option", { value: "" }, "any type"),
    ["definition", "conceptual_explanation", "derivation", "numerical", "compare_contrast", "diagram", "short_answer", "mcq"]
      .map(t => h("option", { value: t }, t.replaceAll("_", " "))));
  const unit = h("select", {}, h("option", { value: "" }, "any unit"));
  const go = h("button", { class: "primary" }, "Search");
  const out = h("div");
  append(main, h("h1", {}, "Search"),
    h("p", { class: "muted" }, "Finds past questions and syllabus items by keyword and by meaning."),
    h("div", { class: "card" }, h("div", { class: "grid cols-4" }, h("div", { style: { gridColumn: "span 2" } }, field("Search for", q)),
      field("Year", year), field("Question type", qtype)), h("div", { class: "row", style: { marginTop: "8px" } }, field("Unit", unit), h("span", { class: "spacer" }), go)),
    out);

  const syl = await api.get(`/api/courses/${course.id}/syllabus`);
  const current = syl.versions.find(v => v.is_current);
  for (const u of current ? current.topics : []) unit.appendChild(h("option", { value: u.id }, `${u.number || ""} ${u.title}`.trim()));

  async function run() {
    const params = new URLSearchParams({ q: q.value });
    if (year.value) params.set("year", year.value);
    if (qtype.value) params.set("qtype", qtype.value);
    if (unit.value) params.set("unit", unit.value);
    const res = await api.get(`/api/courses/${course.id}/search?${params}`);
    clear(out);
    append(out, h("h3", {}, `Questions (${res.questions.length})`),
      res.questions.length ? table([
        { label: "Paper", render: r => h("div", {}, r.exam, h("div", { class: "help" }, r.label)) },
        { label: "Question", key: "text" }, { label: "Marks", key: "marks", num: true },
        { label: "Topic", render: r => h("div", {}, r.topic || "", r.status ? h("div", {}, statusBadge(r.status)) : null) },
        { label: "Match", render: r => Object.entries(r.match).map(([k, v]) => `${k} ${v.toFixed(2)}`).join(", ") },
      ], res.questions) : h("p", { class: "muted" }, "No matching questions."),
      h("h3", { style: { marginTop: "16px" } }, `Syllabus items (${res.topics.length})`),
      res.topics.length ? table([
        { label: "Item", key: "path" }, { label: "Concepts", render: r => (r.concepts || []).join("; ") },
        { label: "Source", render: r => (r.source_refs || []).map(s => `${s.file || ""}${s.page ? ` p.${s.page}` : ""}`).join(", ") },
      ], res.topics) : h("p", { class: "muted" }, "No matching syllabus items."));
  }
  go.addEventListener("click", run);
  q.addEventListener("keydown", (e) => { if (e.key === "Enter") run(); });
  q.focus();
}
