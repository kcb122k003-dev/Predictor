import { api } from "./api.js";
import { append, badge, clear, h, modal, pct, statusBadge, table, toast } from "./dom.js";

export async function renderMapping(main, course) {
  const filter = h("select", {},
    h("option", { value: "" }, "All statuses"), h("option", { value: "A" }, "A clearly in syllabus"),
    h("option", { value: "B" }, "B probably in syllabus"), h("option", { value: "C" }, "C uncertain"),
    h("option", { value: "D" }, "D outside syllabus"), h("option", { value: "unmapped" }, "Not mapped yet"));
  const box = h("div");
  append(main, h("h1", {}, "Topic mapping"),
    h("p", { class: "muted" }, "Each question is matched to the syllabus by meaning and shared terms. Only A and ",
      "validated B mappings count toward predictions (see the strictness setting). Mappings are refreshed every time ",
      "you run the analysis; your manual choices are kept."),
    h("div", { class: "row", style: { marginBottom: "10px" } }, filter), box);
  filter.addEventListener("change", load);

  let topics = [];
  async function loadTopics() {
    const syl = await api.get(`/api/courses/${course.id}/syllabus`);
    const current = syl.versions.find(v => v.is_current) || syl.versions[0];
    topics = [];
    (function walk(list, path) { for (const n of list) { const p = [...path, n.title]; topics.push({ id: n.id, label: `${n.number || ""} ${p.join(" > ")}`.trim() }); walk(n.children || [], p); } })(current ? current.topics : [], []);
  }

  async function load() {
    const rows = await api.get(`/api/courses/${course.id}/mappings${filter.value ? `?status=${filter.value}` : ""}`);
    clear(box);
    if (!rows.length) { box.appendChild(h("div", { class: "empty" }, "Nothing to show. Run the analysis once to map questions to the syllabus.")); return; }
    const counts = {};
    for (const r of rows) counts[r.status] = (counts[r.status] || 0) + 1;
    box.appendChild(h("div", { class: "row", style: { marginBottom: "8px" } }, Object.entries(counts).map(([s, n]) => h("span", {}, statusBadge(s), ` ${n}`))));
    box.appendChild(table([
      { label: "Paper", render: r => h("div", {}, r.question.exam, h("div", { class: "help" }, r.question.path_label)) },
      { label: "Question", render: r => h("div", { style: { maxWidth: "520px" } }, r.question.text) },
      { label: "Status", render: r => h("div", {}, statusBadge(r.status), r.mappings[0] && r.mappings[0].method === "manual" ? h("div", {}, badge("set by you", "warn")) : null, r.included ? null : h("div", { class: "help" }, "paper excluded")) },
      { label: "Topic", render: r => r.mappings.length ? h("div", {}, r.mappings.map(m => h("div", {}, m.topic || "(none)", m.rank > 1 ? h("span", { class: "help" }, " (also)") : null))) : "" },
      { label: "Match", render: r => r.mappings[0] ? h("div", {}, pct(r.mappings[0].confidence), h("div", { class: "help" }, (r.mappings[0].matched_terms || []).slice(0, 5).join(", "))) : "" },
      { label: "", render: r => h("button", { class: "small", onclick: (e) => { e.stopPropagation(); override(r); } }, "Change") },
    ], rows, { onRowClick: (r) => explain(r) }));
  }

  function explain(r) {
    const m = r.mappings[0];
    modal(`${r.question.exam} ${r.question.path_label}`, h("div", {},
      h("p", {}, r.question.text),
      m ? h("div", {},
        h("p", {}, statusBadge(m.status), " ", (m.evidence || {}).reason || ""),
        h("ul", { class: "evidence" },
          h("li", {}, `Matched syllabus item: ${m.topic || "none"}`),
          h("li", {}, `Syllabus text: ${m.evidence_text || ""}`),
          h("li", {}, `Semantic similarity ${m.semantic.toFixed(3)}, keyword coverage ${pct(m.keyword)}, match strength ${pct(m.confidence)}`),
          (m.matched_terms || []).length ? h("li", {}, `Shared terms: ${m.matched_terms.join(", ")}`) : null,
          ((m.evidence || {}).unknown_terms || []).length ? h("li", {}, `Terms not found in the syllabus: ${m.evidence.unknown_terms.join(", ")}`) : null,
          (m.evidence || {}).historical_syllabus ? h("li", {}, `Matches the '${m.evidence.historical_syllabus.version}' syllabus topic '${m.evidence.historical_syllabus.topic}'`) : null))
        : h("p", { class: "muted" }, "Not mapped yet.")));
  }

  function override(r) {
    const sel = h("select", { multiple: true, size: 12, style: { width: "100%" } }, topics.map(t => h("option", { value: t.id }, t.label)));
    for (const opt of sel.options) opt.selected = r.mappings.some(m => String(m.topic_id) === opt.value && m.method === "manual");
    const status = h("select", {}, h("option", { value: "A" }, "A in syllabus"), h("option", { value: "B" }, "B probably in syllabus"), h("option", { value: "D" }, "D outside syllabus"));
    const save = h("button", { class: "primary" }, "Save");
    const reset = h("button", {}, "Use automatic mapping");
    const m = modal("Change mapping", h("div", { class: "grid" }, h("p", {}, r.question.text),
      h("label", { class: "field" }, h("span", {}, "Topic(s), first is primary (Ctrl/Cmd for several)"), sel),
      h("label", { class: "field" }, h("span", {}, "Status"), status), h("div", { class: "row end" }, reset, save)));
    save.addEventListener("click", async () => {
      const ids = [...sel.selectedOptions].map(o => Number(o.value));
      if (!ids.length && status.value !== "D") { toast("Pick a topic, or choose 'outside syllabus'.", "error"); return; }
      await api.put(`/api/questions/${r.question.id}/mapping`, { topic_ids: ids, status: status.value });
      m.close(); toast("Saved. Run the analysis again to update predictions."); load();
    });
    reset.addEventListener("click", async () => { await api.del(`/api/questions/${r.question.id}/mapping`); m.close(); toast("Manual mapping removed."); load(); });
  }

  await loadTopics();
  await load();
}
