import { api } from "./api.js";
import { append, badge, componentBadge, h, num, pct, plot, table } from "./dom.js";

export async function renderModels(main, course) {
  append(main, h("h1", {}, "Model performance"));
  const results = await api.get(`/api/courses/${course.id}/results`);
  if (!results.run) { append(main, h("div", { class: "empty" }, "Run the analysis first.")); return; }
  const runId = results.run.id;
  const s = results.run.summary;
  const [models, artifacts] = await Promise.all([api.get(`/api/runs/${runId}/artifacts/models`), Promise.all(
    ["charts", "ablation", "calibration", "syllabus_filter", "type_forecast", "families", "evidence"].map(k =>
      api.get(`/api/runs/${runId}/artifacts/${k}`).catch(() => ({}))))]);
  const [charts, ablation, calibration, filterCheck, typeFc, families, evidence] = artifacts;
  const metric = (models.primary || "ndcg").toUpperCase();

  append(main, h("p", { class: "muted" }, `Every model and component predicted each past paper using only the papers before it (rolling-origin backtest on `,
    `${models.targets.length} held-out paper(s)${models.targets.length ? ": " + models.targets.join(", ") : ""}). K = ${models.k} topics, the typical number per paper.`),
    h("div", { class: "card" }, h("p", { style: { margin: 0 } }, h("strong", {}, "Final ranking: "), s.selected_display, ". ", models.reason),
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
      { label: "Status", render: r => h("div", {}, componentBadge(r.status), r.weight !== null && r.weight !== undefined && r.role === "component" ? ` weight ${pct(r.weight)}` : "",
          h("div", { class: "help" }, r.status_reason || ""), r.notes && r.notes.length ? h("div", { class: "help" }, r.notes.join(" ")) : null) },
    ], rows, { rowClass: r => r.selected ? "selected" : "" }),
    h("p", { class: "help" }, "± is the standard error across held-out papers. Random selection is the exact expected score of a random ranking. ",
      "No component is switched off for lack of data: each runs on every fold and earns ensemble weight from its reliability and measured skill.")));

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
    const keep = ["random", "frequency", "recency", "beta_binomial", "general", results.run.summary.selected_model, "ensemble"];
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
      h("h4", {}, "Adding components one family at a time"),
      table([{ label: "Variant", key: "variant" },
        { label: `NDCG@${ablation.k}`, render: r => `${num(r.ndcg)} ± ${num(r.ndcg_se)}`, num: true },
        { label: "Change", render: r => r.delta === undefined || r.delta === null ? "" : `${r.delta >= 0 ? "+" : ""}${num(r.delta)} ± ${num(r.delta_se)}${r.reliable ? "" : " (not reliable)"}`, num: true },
        { label: "Hit@1", render: r => num(r["hit@1"], 2), num: true }, { label: "Hit@3", render: r => num(r["hit@3"], 2), num: true },
        { label: "Hit@5", render: r => num(r["hit@5"], 2), num: true }, { label: `Recall@${ablation.k}`, render: r => num(r.recall), num: true },
        { label: `Precision@${ablation.k}`, render: r => num(r.precision), num: true },
        { label: "Concept recall", render: r => num(r.concept_recall), num: true },
        { label: "Exact repeats", render: r => num(r.exact_recurrence_recall), num: true },
        { label: "Papers better / worse", render: r => r.folds_better === undefined ? "" : `${r.folds_better} / ${r.folds_worse}` }], ablation.staged),
      h("h4", { style: { marginTop: "12px" } }, "Removing one component"),
      table([{ label: "Variant", key: "variant" }, { label: `NDCG@${ablation.k}`, render: r => `${num(r.ndcg)} ± ${num(r.ndcg_se)}`, num: true },
        { label: "Change vs full ensemble", render: r => r.delta === undefined || r.delta === null ? "" : `${r.delta >= 0 ? "+" : ""}${num(r.delta)} ± ${num(r.delta_se)}`, num: true }], ablation.leave_one_out));
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
  if (evidence && evidence.profile) {
    const pr = evidence.profile;
    const v = evidence.validation || {};
    const gm = evidence.general_model || {};
    const gc = evidence.generation_checks || {};
    const fact = (k, val) => [h("dt", {}, k), h("dd", {}, val === null || val === undefined || val === "" ? "-" : String(val))];
    append(main, h("div", { class: "card" }, h("div", { class: "row" }, h("h3", { style: { margin: 0 } }, "Evidence and diagnostics"),
        badge(evidence.mode, evidence.mode === "Advanced inference" ? "ok" : "info")),
      h("p", {}, evidence.message),
      h("div", { class: "grid cols-2" },
        h("div", {}, h("h4", {}, "Evidence profile"), h("dl", { class: "facts" },
          fact("Papers (exam-level evidence)", `${pr.exams}${pr.exams_without_year ? ` (${pr.exams_without_year} without a year)` : ""}`),
          fact("Effective papers", pr.effective_exams),
          fact("Years", pr.year_span ? `${pr.year_span[0]}-${pr.year_span[1]}${pr.calendar_gaps ? `, ${pr.calendar_gaps} missing year(s)` : ""}` : "unknown"),
          fact("In-syllabus questions (language evidence)", `${pr.questions} of ${pr.questions_total}`),
          fact("Questions with marks", pr.questions_with_marks),
          fact("Topics observed", `${pr.topics_observed} of ${pr.topics} (${pr.units} unit(s))`),
          fact("Topics per paper", num(pr.mean_topics_per_exam, 1)),
          fact("Coverage spread", num(pr.coverage_entropy, 2)),
          fact("Question formats seen", pr.formats_observed),
          fact("Syllabus weights from", pr.syllabus_weights),
          fact("Pretrained semantic model", pr.pretrained ? "available" : `unavailable (${pr.pretrained_note})`),
          fact("Other real courses in library", `${gm.real_courses || 0} (general model: ${gm.source || "simulation"})`),
          fact("Overall evidence quality", evidence.evidence_quality),
          fact("Prediction uncertainty", evidence.uncertainty ? `${evidence.uncertainty.overall} (median rank range ${num(evidence.uncertainty.median_rank_range, 0)} positions; ` +
            Object.entries(evidence.uncertainty.levels || {}).map(([k, n]) => `${n} ${k.toLowerCase()}`).join(", ") + ")" : "")),
          (evidence.notes || []).map(n => h("p", { class: "help" }, n))),
        h("div", {}, h("h4", {}, "Validation on held-out papers"), h("dl", { class: "facts" },
          fact("Held-out folds", v.folds),
          fact(`Mean ${v.metric || "NDCG"}`, v.mean === undefined ? "" : `${num(v.mean)} ± ${num(v.se)}`),
          fact("95% interval", v.ci95 && v.ci95[0] !== null ? `${num(v.ci95[0])} to ${num(v.ci95[1])}` : "needs two or more folds"),
          fact("Fold-to-fold spread (SD)", num(v.sd))),
          h("p", { class: "help" }, v.message || ""),
          (v.baselines || []).length ? table([{ label: "Compared with", key: "display" },
            { label: "Their mean", render: r => num(r.baseline_mean), num: true },
            { label: "Difference", render: r => `${r.difference >= 0 ? "+" : ""}${num(r.difference)} ± ${num(r.difference_se)}`, num: true },
            { label: "Papers better / worse", render: r => `${r.folds_better} / ${r.folds_worse}` }], v.baselines) : null,
          h("p", { class: "help" }, `Rank intervals: ${evidence.uncertainty ? evidence.uncertainty.jackknife_replicates : 0} leave-one-paper-out rankings and `,
            `${evidence.uncertainty ? evidence.uncertainty.posterior_draws : 0} posterior draws`,
            evidence.uncertainty && evidence.uncertainty.held_fixed && evidence.uncertainty.held_fixed.length ? ` (held fixed: ${evidence.uncertainty.held_fixed.join(", ")})` : "", "."),
          gc.checked ? h("p", { class: "help" }, `Generated formulations checked: ${gc.checked}; rejected for topic ${gc.rejected.topic}, semantic fit ${gc.rejected.semantic}, question type ${gc.rejected.type}.`) : null)),
      h("h4", { style: { marginTop: "12px" } }, "Ensemble components"),
      table([{ label: "Component", key: "display" }, { label: "Status", render: r => componentBadge(r.status) },
        { label: "Weight", render: r => r.weight === null ? "" : pct(r.weight), num: true },
        { label: "Reliability", render: r => num(r.reliability, 2), num: true }, { label: "Skill", render: r => num(r.skill, 2), num: true },
        { label: "Why", render: r => h("span", { class: "help" }, r.reason) }], (evidence.components || []).filter(r => r.role !== "baseline"))));
  }
  if (models.concept_table) {
    const sel = models.concept_table.find(r => r.selected);
    const rnd = models.concept_table.find(r => r.model === "random");
    if (sel) append(main, h("div", { class: "card" }, h("h3", {}, "Concept level"),
      h("p", {}, `At the finest syllabus level, ${sel.display} reaches recall@K ${num(sel.mean.recall)} (random ${num(rnd ? rnd.mean.recall : null)}).`)));
  }
}
