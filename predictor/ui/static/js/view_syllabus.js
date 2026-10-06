import { api } from "./api.js";
import { append, badge, clear, h, modal, toast } from "./dom.js";

export async function renderSyllabus(main, course, { refreshCourse }) {
  const box = h("div");
  append(main, h("h1", {}, "Syllabus"),
    h("p", { class: "muted" }, "The syllabus is the hard boundary for every prediction. Units hold topics; topics are ",
      "the main prediction unit; the finest items are concepts. Edit titles, add aliases (other names the papers ",
      "use), or exclude items that are not examinable."), box);

  async function load() {
    const data = await api.get(`/api/courses/${course.id}/syllabus`);
    clear(box);
    const rebuild = h("button", {}, "Rebuild from documents");
    rebuild.addEventListener("click", async () => {
      if (!confirm("Re-parse all course-content files? Topics you edited block this.")) return;
      try { const r = await api.post(`/api/courses/${course.id}/syllabus/rebuild`); toast(`${r.topics_added} topics rebuilt.`); load(); }
      catch (e) { toast(e.message, "error", 8000); }
    });
    const add = h("button", { class: "primary" }, "Add unit");
    add.addEventListener("click", () => addTopic(null));
    box.appendChild(h("div", { class: "row", style: { marginBottom: "10px" } }, add, rebuild));
    if (!data.versions.length || !data.versions.some(v => v.topics.length)) {
      box.appendChild(h("div", { class: "empty" }, "No syllabus yet. Upload course contents in step 1 or add units by hand."));
      return;
    }
    for (const v of data.versions) {
      box.appendChild(h("div", { class: "card" },
        h("div", { class: "row" }, h("h3", { style: { margin: 0 } }, v.is_current ? "Current syllabus" : `Historical: ${v.label}`),
          badge(`${v.counts.units} units`), badge(`${v.counts.topics} topics`, "info"), badge(`${v.counts.concepts} concepts`)),
        v.is_current ? null : h("p", { class: "help" }, "Questions that match only a historical syllabus are excluded from predictions and explained as such."),
        h("div", { class: "ttree" }, h("ul", {}, v.topics.map(t => node(t, v))))));
    }
  }

  function node(t, v) {
    const levelBadge = badge(t.level, t.level === "topic" ? "info" : "");
    const meta = [];
    if (t.hours) meta.push(`${t.hours} h`);
    if (t.marks_weight) meta.push(`${t.marks_weight} marks`);
    const src = (t.source_refs || [])[0];
    return h("li", {},
      h("div", { class: `tnode ${t.excluded ? "excluded" : ""}` },
        h("strong", {}, `${t.number || ""} ${t.title}`.trim()), levelBadge,
        meta.length ? h("span", { class: "muted" }, meta.join(", ")) : null,
        (t.kinds || []).map(k => badge(k)), t.user_edited ? badge("edited", "warn") : null,
        (t.aliases || []).length ? h("span", { class: "help" }, `also called: ${t.aliases.join(", ")}`) : null,
        h("span", { class: "spacer" }),
        h("button", { class: "small", onclick: () => edit(t, v) }, "Edit"),
        h("button", { class: "small", onclick: () => addTopic(t) }, "Add child"),
        h("button", { class: "small", onclick: async () => { await api.patch(`/api/topics/${t.id}`, { excluded: !t.excluded }); load(); } }, t.excluded ? "Include" : "Exclude"),
        h("button", { class: "small danger", onclick: async () => { if (confirm(`Delete '${t.title}' and everything under it?`)) { await api.del(`/api/topics/${t.id}`); load(); refreshCourse(); } } }, "Delete")),
      (t.concepts || []).length ? h("div", { class: "help", style: { marginLeft: "8px" } }, `Concepts: ${t.concepts.join("; ")}`) : null,
      src ? h("div", { class: "help", style: { marginLeft: "8px" } }, `Source: ${src.file || ""}${src.page ? `, page ${src.page}` : ""}`) : null,
      t.children && t.children.length ? h("ul", {}, t.children.map(c => node(c, v))) : null);
  }

  function edit(t, v) {
    const title = h("input", { value: t.title, style: { width: "100%" } });
    const number = h("input", { value: t.number || "", size: 6 });
    const hours = h("input", { type: "number", step: "0.5", value: t.hours ?? "" });
    const concepts = h("textarea", { rows: 3 }, (t.concepts || []).join("; "));
    const aliases = h("textarea", { rows: 2 }, (t.aliases || []).join("; "));
    const parent = h("select", {}, h("option", { value: "" }, "(top level)"));
    const flat = [];
    (function walk(list, depth) { for (const n of list) { flat.push([n, depth]); walk(n.children || [], depth + 1); } })(v.topics, 0);
    for (const [n, d] of flat) if (n.id !== t.id) parent.appendChild(h("option", { value: n.id }, `${"  ".repeat(d)}${n.number || ""} ${n.title}`));
    parent.value = t.parent_id || "";
    const save = h("button", { class: "primary" }, "Save");
    const m = modal(`Edit ${t.title}`, h("div", { class: "grid" },
      h("div", { class: "row" }, h("label", { class: "field" }, h("span", {}, "Number"), number),
        h("label", { class: "field", style: { flex: 1 } }, h("span", {}, "Title"), title),
        h("label", { class: "field" }, h("span", {}, "Teaching hours"), hours)),
      h("label", { class: "field" }, h("span", {}, "Concepts (separate with ;)"), concepts),
      h("label", { class: "field" }, h("span", {}, "Aliases: other names papers use for this (separate with ;)"), aliases),
      h("label", { class: "field" }, h("span", {}, "Parent"), parent),
      h("div", { class: "row end" }, save)));
    save.addEventListener("click", async () => {
      const split = s => s.split(";").map(x => x.trim()).filter(Boolean);
      const body = { title: title.value, number: number.value, hours: hours.value === "" ? null : Number(hours.value),
        concepts: split(concepts.value), aliases: split(aliases.value) };
      if (String(parent.value || "") !== String(t.parent_id || "")) body.parent_id = parent.value ? Number(parent.value) : null;
      try { await api.patch(`/api/topics/${t.id}`, body); m.close(); load(); }
      catch (e) { toast(e.message, "error"); }
    });
  }

  function addTopic(parentNode) {
    const title = h("input", { style: { width: "100%" } });
    const number = h("input", { size: 6 });
    const go = h("button", { class: "primary" }, "Add");
    const m = modal(parentNode ? `Add under ${parentNode.title}` : "Add a unit", h("div", { class: "grid" },
      h("div", { class: "row" }, h("label", { class: "field" }, h("span", {}, "Number"), number),
        h("label", { class: "field", style: { flex: 1 } }, h("span", {}, "Title"), title)), h("div", { class: "row end" }, go)));
    go.addEventListener("click", async () => {
      await api.post(`/api/courses/${course.id}/topics`, { title: title.value, number: number.value, parent_id: parentNode ? parentNode.id : null });
      m.close(); load(); refreshCourse();
    });
  }

  await load();
}
