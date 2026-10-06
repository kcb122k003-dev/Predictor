import { api } from "./api.js";
import { append, badge, clear, field, h, modal, pct, table, toast } from "./dom.js";

const ACCEPT = ".pdf,.docx,.png,.jpg,.jpeg,.tif,.tiff,.bmp,.webp,.txt,.md";

export async function renderFiles(main, course, { refreshCourse }) {
  const list = h("div");
  let timer = null;
  window.addEventListener("predictor:leave", () => clearTimeout(timer), { once: true });

  const version = h("select", {}, h("option", { value: "current" }, "Current syllabus"),
    h("option", { value: "__new" }, "A historical syllabus..."));

  append(main, 
    h("h1", {}, "Upload files"),
    h("p", { class: "muted" }, "Files stay on this computer. Scanned PDFs and photos are read with OCR. ",
      "Identical files are detected and not counted twice."),
    h("div", { class: "grid cols-2" },
      h("div", {}, h("h3", {}, "Past exam papers"), dropzone("exam", "Drop past papers here",
        "PDF, DOCX, images or text. Any number of files.")),
      h("div", {}, h("h3", {}, "Course contents"), dropzone("syllabus", "Drop syllabus or course outline here",
        "Syllabus, course outline, topic lists. Several documents are merged."),
        h("div", { class: "row", style: { marginTop: "8px" } }, field("These documents describe", version)))),
    h("h2", { style: { marginTop: "22px" } }, "Files"), list);

  function dropzone(kind, title, help) {
    const input = h("input", { type: "file", multiple: true, accept: ACCEPT, style: { display: "none" } });
    const zone = h("div", { class: "dropzone" }, h("strong", {}, title), help, input);
    zone.addEventListener("click", () => input.click());
    zone.addEventListener("dragover", (e) => { e.preventDefault(); zone.classList.add("over"); });
    zone.addEventListener("dragleave", () => zone.classList.remove("over"));
    zone.addEventListener("drop", (e) => { e.preventDefault(); zone.classList.remove("over"); upload(kind, e.dataTransfer.files); });
    input.addEventListener("change", () => upload(kind, input.files));
    return zone;
  }

  async function upload(kind, files) {
    if (!files || !files.length) return;
    const form = new FormData();
    for (const f of files) form.append("files", f);
    form.append("kind", kind);
    let label = "current";
    if (kind === "syllabus" && version.value === "__new") {
      label = prompt("Name this historical syllabus (for example 'Syllabus 2015'):", "historical") || "historical";
    }
    form.append("syllabus_version", label);
    toast(`Uploading ${files.length} file(s)...`);
    const res = await api.upload(`/api/courses/${course.id}/files`, form);
    for (const r of res) {
      if (r.error) toast(`${r.filename}: ${r.error}`, "error", 8000);
      else if (r.duplicate) toast(r.message, "info", 7000);
    }
    await load();
  }

  async function load() {
    const files = await api.get(`/api/courses/${course.id}/files`);
    clear(list);
    if (!files.length) { list.appendChild(h("div", { class: "empty" }, "No files yet.")); return; }
    list.appendChild(table([
      { label: "File", render: f => h("div", {}, f.filename, h("div", { class: "muted" }, f.kind === "exam" ? "Exam paper" : "Course contents")) },
      { label: "Status", render: f => statusOf(f) },
      { label: "Pages", key: "pages", num: true },
      { label: "Read with", render: f => readWith(f.summary) },
      { label: "Result", render: f => resultOf(f) },
      { label: "", render: f => h("div", { class: "row" },
          h("button", { class: "small", onclick: (e) => { e.stopPropagation(); showPages(f); } }, "Text"),
          h("button", { class: "small", onclick: async (e) => { e.stopPropagation(); await api.post(`/api/files/${f.id}/reprocess`); toast("Reprocessing..."); load(); } }, "Reprocess"),
          h("button", { class: "small danger", onclick: async (e) => {
            e.stopPropagation();
            if (!confirm(`Delete ${f.filename}? Its parsed exam is removed too.`)) return;
            await api.del(`/api/files/${f.id}`); load(); refreshCourse();
          } }, "Delete")) },
    ], files));
    const busy = files.some(f => f.status === "pending" || f.status === "processing");
    if (busy) timer = setTimeout(load, 1500);
    else refreshCourse();
  }

  function statusOf(f) {
    if (f.status === "done") return badge("done", "ok");
    if (f.status === "error") return h("div", {}, badge("error", "bad"), h("div", { class: "help" }, f.error));
    return badge(f.status, "info");
  }

  function readWith(s) {
    if (!s || !s.file_type) return "";
    const parts = [s.file_type.toUpperCase()];
    if (s.ocr_pages) parts.push(`OCR on ${s.ocr_pages} page(s), mean confidence ${Math.round(s.mean_ocr_confidence || 0)}%`);
    return h("div", {}, parts.join(", "),
      (s.flagged_pages || []).length ? h("div", {}, badge(`check pages ${s.flagged_pages.join(", ")}`, "warn")) : null);
  }

  function resultOf(f) {
    const s = f.summary || {};
    if (f.kind === "exam") {
      if (!f.exam_id) return "";
      const bits = [h("a", { href: `#/course/${course.id}/review` }, f.exam_label || "exam"), ` ${s.questions ?? ""} questions`];
      if (s.needs_review) bits.push(" ", badge(`${s.needs_review} to review`, "warn"));
      if (f.included === false) bits.push(h("div", { class: "help" }, f.exclusion_reason));
      return h("div", {}, bits);
    }
    return h("div", {}, `${s.topics_added ?? 0} topics added`,
      (s.warnings || []).slice(0, 2).map(w => h("div", { class: "help" }, w)));
  }

  async function showPages(f) {
    const pages = await api.get(`/api/files/${f.id}/pages`);
    modal(f.filename, h("div", {}, pages.map(p => h("div", {},
      h("h4", {}, `Page ${p.page} (${p.method}${p.confidence ? `, OCR confidence ${Math.round(p.confidence)}%` : ""})`,
        " ", (p.flags || []).map(fl => badge(fl.replaceAll("_", " "), "warn"))),
      h("pre", { class: "pagetext" }, p.text)))), { wide: true });
  }

  await load();
}

