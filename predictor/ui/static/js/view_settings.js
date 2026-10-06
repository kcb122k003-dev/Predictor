import { api } from "./api.js";
import { append, badge, h, toast } from "./dom.js";

// Settings shown in the form. Everything else stays at its default (see config/default.toml).
const FIELDS = [
  ["Syllabus boundary", [
    ["alignment.strictness", "Syllabus strictness", "number", "0 = lenient (uncertain mappings count), 0.5 = clearly in + validated probably in, 1 = only clearly in.", 0.05],
    ["alignment.thresholds.tfidf.clearly_in", "Clearly-in threshold (offline model)", "number", "Combined match score for status A.", 0.01],
    ["alignment.thresholds.tfidf.probably_in", "Probably-in threshold (offline model)", "number", "Score for status B.", 0.01],
    ["alignment.thresholds.tfidf.outside", "Outside threshold (offline model)", "number", "Below this a question is outside the syllabus.", 0.01],
  ]],
  ["Prediction", [
    ["models.top_k", "Top-K for evaluation", "text", "'auto' uses the typical number of topics per paper.", null],
    ["models.min_train_exams", "Papers before the first backtest target", "number", "Backtesting starts after this many papers.", 1],
    ["models.selection_rule", "Model selection rule", "select:one_se,best", "one_se prefers the simpler model when the difference is within one standard error.", null],
    ["temporal.default_half_life", "Recency half-life (papers)", "number", "How quickly older papers lose weight in the Bayesian rate model.", 0.25],
    ["models.logistic_C", "Logistic regularisation C", "number", "Smaller = stronger regularisation.", 0.05],
  ]],
  ["Text extraction", [
    ["ocr.enabled", "Use OCR for scanned pages and images", "checkbox", "", null],
    ["ocr.language", "OCR language(s)", "text", "Tesseract language codes, for example eng or eng+nep.", null],
    ["ocr.dpi", "OCR resolution (dpi)", "number", "300 is a good default; 400 helps small print.", 50],
    ["ocr.psm", "Page segmentation mode", "number", "4 keeps question lines intact on most papers; 3 is fully automatic.", 1],
    ["ocr.deskew", "Straighten tilted scans", "checkbox", "", null],
  ]],
  ["Semantic model", [
    ["embeddings.backend", "Embedding backend", "select:auto,tfidf,sentence-transformers", "auto uses a downloaded neural model if present, otherwise the offline TF-IDF model.", null],
    ["embeddings.model_name", "Neural model name", "text", "Downloaded once with: predictor models download", null],
  ]],
];

function getPath(obj, path) { return path.split(".").reduce((o, k) => (o || {})[k], obj); }
function setPath(obj, path, value) {
  const keys = path.split(".");
  let o = obj;
  for (const k of keys.slice(0, -1)) o = (o[k] = o[k] || {});
  o[keys[keys.length - 1]] = value;
}

export async function renderSettings(main) {
  const [settings, health] = await Promise.all([api.get("/api/settings"), api.get("/api/health")]);
  const eff = settings.effective;
  const inputs = [];
  append(main, h("h1", {}, "Settings"),
    h("div", { class: "card" }, h("h3", {}, "This computer"),
      h("p", {}, "OCR: ", health.ocr.available ? badge(`Tesseract ${health.ocr.version}`, "ok") : badge("not available", "warn"),
        health.ocr.available ? "" : ` ${health.ocr.reason}`),
      h("p", {}, "Neural embeddings: ", health.embeddings.neural_available ? badge("ready", "ok") : badge("not installed", "info"),
        health.embeddings.neural_available ? "" : ` The offline TF-IDF model is used. ${health.embeddings.neural_reason}`),
      h("p", {}, "External services: ", badge(health.external_services ? "allowed" : "off", health.external_services ? "warn" : "ok"),
        " Nothing is sent over the network."),
      h("p", { class: "help" }, `Data folder: ${health.data_dir}`)));
  for (const [group, fields] of FIELDS) {
    const card = h("div", { class: "card" }, h("h3", {}, group));
    for (const [path, label, type, help, step] of fields) {
      const value = getPath(eff, path);
      let input;
      if (type === "checkbox") { input = h("input", { type: "checkbox" }); input.checked = !!value; }
      else if (type.startsWith("select:")) { input = h("select", {}, type.slice(7).split(",").map(o => h("option", { value: o }, o))); input.value = value; }
      else { input = h("input", { type, step: step ?? undefined, value: value ?? "" }); }
      inputs.push({ path, type, input, original: value });
      card.appendChild(h("div", { class: "row", style: { marginBottom: "8px", alignItems: "flex-start" } },
        h("label", { class: "field", style: { minWidth: "300px" } }, h("span", {}, label), input), h("span", { class: "help", style: { maxWidth: "520px" } }, help)));
    }
    main.appendChild(card);
  }
  const save = h("button", { class: "primary" }, "Save settings");
  const reset = h("button", {}, "Reset to defaults");
  append(main, h("div", { class: "row" }, save, reset),
    h("p", { class: "help" }, "Changes apply to the next analysis. Per-course overrides can be set through the API (PATCH /api/courses/{id})."));
  save.addEventListener("click", async () => {
    const overrides = JSON.parse(JSON.stringify(settings.overrides || {}));
    for (const { path, type, input, original } of inputs) {
      let v = type === "checkbox" ? input.checked : input.value;
      if (type === "number") v = Number(v);
      if (path === "models.top_k" && /^\d+$/.test(String(v))) v = Number(v);
      if (v !== original) setPath(overrides, path, v);
    }
    try { await api.put("/api/settings", overrides); toast("Settings saved."); }
    catch (e) { toast(e.message, "error", 8000); }
  });
  reset.addEventListener("click", async () => {
    if (!confirm("Remove all your setting changes?")) return;
    await api.put("/api/settings", {});
    toast("Defaults restored."); window.location.reload();
  });
}
