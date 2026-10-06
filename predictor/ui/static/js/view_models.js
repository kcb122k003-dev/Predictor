import { api } from "./api.js";
import { append, badge, h, num, plot, table } from "./dom.js";

export async function renderModels(main, course) {
  append(main, h("h1", {}, "Model performance"));
  const results = await api.get(`/api/courses/${course.id}/results`);
  if (!results.run) { append(main, h("div", { class: "empty" }, "Run the analysis first.")); return; }
  const runId = results.run.id;
  const s = results.run.summary;
  const [models, artifacts] = await Promise.all([api.get(`/api/runs/${runId}/artifacts/models`), Promise.all(
    ["charts", "ablation", "calibration", "syllabus_filter", "type_forecast", "families", "sufficiency"].map(k =>
      api.get(`/api/runs/${runId}/artifacts/${k}`).catch(() => ({}))))]);
  const [charts, ablation, calibration, filterCheck, typeFc, families, sufficiency] = artifacts;
  const metric = (models.primary || "ndcg").toUpperCase();

  append(main, h("p", { class: "muted" }, `Every model predicted each past paper using only the papers before it (time-ordered backtest on `,
    `${models.targets.length} papers: ${models.targets.join(", ")}). K = ${models.k} topics, the typical number per paper.`),
    h("div", { class: "card" }, h("p", { style: { margin: 0 } }, h("strong", {}, "Selected: "), s.selected_display, ". ", models.reason),
      h("p", { class: "help", style: { margin: "6px 0 0" } }, `Leakage audit: ${models.leakage_audit && models.leakage_audit.passed ?
        `passed for ${models.leakage_audit.models_checked} models (scrambling later papers did not change earlier predictions)` : "not run"}.`)));

  const rows = models.table.filter(r => !r.hidden);
  append(main, h("div", { class: "card" }, h("h3", {}, "Model comparison"),
    table([
      { label: "Model", render: r => h("div", {}, r.display, r.selected ? h("span", {}, " ", badge("selected", "ok")) : null,
          h("div", { class: "help" }, r.description || "")) },
      { label: `${metric}@K`, render: r => r.mean.ndcg === undefined || r.mean.ndcg === null ? "" : `${num(r.mean.ndcg)} ± ${num(r.se.ndcg)}`, num: true },
      { label: "Recall@K", render: r => num(r.mean.recall), num: true },
      { label: "Precision@K", render: r => num(r.mean.precision), num: true },
      { label: "Hit@1", render: r => num(r.mean["hit@1"], 2), num: true },
      { label: "Hit@3", render: r => num(r.mean["hit@3"], 2), num: true },
      { label: "Hit@5", render: r => num(r.mean["hit@5"], 2), num: true },
      { label: "MRR", render: r => num(r.mean.mrr), num: true },
      { label: "Status", render: r => r.enabled ? (r.fallback_folds ? badge(`fallback on ${r.fallback_folds} fold(s)`, "warn") : badge("enabled", "ok"))
          : h("div", {}, badge("disabled", "warn"), h("div", { class: "help" }, r.gate_reason)) },
    ], rows, { rowClass: r => r.selected ? "selected" : "" }),
    h("p", { class: "help" }, "± is the standard error across held-out papers. Random selection is the exact expected score of a random ranking.")));

  const mc = charts.model_comparison;
  if (mc) {
    const el = h("div", { class: "chart" });
    append(main, h("div", { class: "card" }, el));
    plot(el, [{ type: "bar", x: mc.models, y: mc.values, error_y: { type: "data", array: mc.errors.map(e => e ?? 0), visible: true },
      marker: { color: mc.names.map(n => n === mc.selected ? "#1f7a4d" : n === "random" ? "#999" : "#3b6ea8") } }],
      { title: mc.title, xaxis: { tickangle: -30 } });
  }
  const bf = charts.backtest_folds;
  if (bf) {
    const el = h("div", { class: "chart" });
    append(main, h("div", { class: "card" }, el));
    const keep = ["random", "frequency", "last_exam", "ewma", results.run.summary.selected_model, "ensemble", "logistic"];
    plot(el, Object.entries(bf.models).filter(([k]) => keep.includes(k)).map(([k, v]) => ({ type: "scatter", mode: "lines+markers", name: v.display, x: bf.x, y: v.values })),
      { title: bf.title });
  }

  append(main, h("div", { class: "card" }, h("h3", {}, "Probability calibration"),
    h("p", {}, calibration.reason || "Not available."),
    calibration.nested_brier !== undefined && calibration.nested_brier !== null ? h("p", { class: "muted" },
      `Held-out Brier score ${num(calibration.nested_brier)} vs ${num(calibration.climatology_brier)} for always predicting the base rate; expected calibration error ${num(calibration.ece)}.`) : null,
    (calibration.reliability || []).length ? (() => {
      const el = h("div", { class: "chart", style: { height: "320px" } });
      setTimeout(() => plot(el, [
        { type: "scatter", mode: "lines", x: [0, 1], y: [0, 1], name: "perfect", line: { dash: "dot", color: "#999" } },
        { type: "scatter", mode: "lines+markers", name: "held-out", x: calibration.reliability.map(b => b.mean_predicted), y: calibration.reliability.map(b => b.observed_rate),
          text: calibration.reliability.map(b => `${b.count} topic-papers`) }],
        { title: "Reliability: predicted vs observed", xaxis: { range: [0, 1], title: "predicted" }, yaxis: { range: [0, 1], title: "observed" } }), 0);
      return el;
    })() : null));

  const abl = h("div", { class: "card" }, h("h3", {}, "Ablation study"));
  if (ablation.available) {
    append(abl, h("p", { class: "muted" }, ablation.note),
      h("h4", {}, "Adding signals one family at a time"),
      table([{ label: "Variant", key: "variant" }, { label: ablation.metric, render: r => `${num(r.mean)} ± ${num(r.se)}`, num: true },
        { label: "Change", render: r => r.delta === undefined ? "" : `${r.delta >= 0 ? "+" : ""}${num(r.delta)} ± ${num(r.delta_se)}`, num: true },
        { label: "Papers better / worse", render: r => r.folds_better === undefined ? "" : `${r.folds_better} / ${r.folds_worse}` }], ablation.staged),
      h("h4", { style: { marginTop: "12px" } }, "Removing one signal family"),
      table([{ label: "Variant", key: "variant" }, { label: ablation.metric, render: r => `${num(r.mean)} ± ${num(r.se)}`, num: true },
        { label: "Change vs all signals", render: r => r.delta === undefined ? "" : `${r.delta >= 0 ? "+" : ""}${num(r.delta)} ± ${num(r.delta_se)}`, num: true }], ablation.leave_one_out));
  } else append(abl, h("p", {}, ablation.reason || "Not available."));
  append(main, abl);

  if (filterCheck.available) {
    append(main, h("div", { class: "card" }, h("h3", {}, "Effect of the syllabus filter"),
      h("p", {}, `${filterCheck.model}: ${filterCheck.metric} ${num(filterCheck.with_filter.mean)} with the filter and `,
        `${num(filterCheck.without_filter.mean)} without it (change ${num(filterCheck.without_filter.delta)} ± ${num(filterCheck.without_filter.delta_se)}). `,
        `Without the filter, ${filterCheck.extra_appearances_without_filter} extra topic appearances would have been counted.`),
      h("p", { class: "help" }, filterCheck.note)));
  }
  append(main, h("div", { class: "grid cols-2" },
    h("div", { class: "card" }, h("h3", {}, "Question format forecast"), h("p", {}, typeFc.reason || ""),
      typeFc.accuracy ? table([{ label: "Method", render: r => r[0] }, { label: "Top-1 format accuracy", render: r => r[1].mean === null ? "" : `${Math.round(100 * r[1].mean)}% ± ${Math.round(100 * (r[1].se || 0))}`, num: true }],
        Object.entries(typeFc.accuracy)) : null,
      Object.entries(typeFc.gated || {}).map(([k, v]) => h("p", { class: "help" }, `${k}: ${v}`))),
    h("div", { class: "card" }, h("h3", {}, "Exact question recurrence"),
      families.backtest ? h("p", {}, `Exact-question recall@${families.backtest.k}: ${num(families.backtest.model.mean)} for the topic-aware family ranking vs `,
        `${num(families.backtest.recency_baseline.mean)} for recency alone, over ${families.backtest.folds} papers.`) : null,
      families.backtest ? h("p", { class: "help" }, families.backtest.note) : null)));
  if (sufficiency && sufficiency.models) {
    append(main, h("div", { class: "card" }, h("h3", {}, "Data sufficiency"),
      h("p", {}, `${sufficiency.exams} papers, ${sufficiency.folds} backtest folds. Mode: ${sufficiency.mode}.`),
      sufficiency.message ? h("p", {}, sufficiency.message) : null,
      table([{ label: "Model", key: "model" }, { label: "Used?", render: r => r.enabled ? "yes" : "no" }, { label: "Reason", key: "reason" }], sufficiency.models)));
  }
  if (models.concept_table) {
    const sel = models.concept_table.find(r => r.selected);
    const rnd = models.concept_table.find(r => r.model === "random");
    if (sel) append(main, h("div", { class: "card" }, h("h3", {}, "Concept level"),
      h("p", {}, `At the finest syllabus level, ${sel.display} reaches recall@K ${num(sel.mean.recall)} (random ${num(rnd ? rnd.mean.recall : null)}).`)));
  }
}
