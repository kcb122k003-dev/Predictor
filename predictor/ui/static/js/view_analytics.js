import { api } from "./api.js";
import { append, clear, h, modal, plot, table } from "./dom.js";

export async function renderAnalytics(main, course) {
  append(main, h("h1", {}, "Analytics"));
  const results = await api.get(`/api/courses/${course.id}/results`);
  if (!results.run) { append(main, h("div", { class: "empty" }, "Run the analysis first.")); return; }
  const runId = results.run.id;
  const [charts, coverage, structure, rotation] = await Promise.all([
    api.get(`/api/runs/${runId}/artifacts/charts`), api.get(`/api/runs/${runId}/artifacts/coverage`),
    api.get(`/api/runs/${runId}/artifacts/structure`), api.get(`/api/runs/${runId}/artifacts/rotation`)]);
  append(main, h("p", { class: "muted" }, "Click a bar or a cell to list the questions behind it."));

  const tabs = [["timeline", "Timeline"], ["frequency", "Frequency"], ["recency", "Recency"], ["intervals", "Recurrence"],
    ["cooc", "Co-occurrence"], ["types", "Question types"], ["coverage", "Unit coverage"], ["structure", "Paper structure"]];
  const bar = h("div", { class: "tabs" });
  const area = h("div");
  append(main, bar, area);
  for (const [key, label] of tabs) {
    const b = h("button", {}, label);
    b.addEventListener("click", () => { bar.querySelectorAll("button").forEach(x => x.classList.remove("active")); b.classList.add("active"); show(key); });
    bar.appendChild(b);
  }
  bar.firstChild.classList.add("active");
  show("timeline");

  async function showQuestions(ids, title) {
    if (!ids || !ids.length) return;
    const qs = await api.get(`/api/questions?ids=${ids.join(",")}`);
    modal(title, table([
      { label: "Paper", render: q => h("div", {}, q.exam, h("div", { class: "help" }, q.path_label)) },
      { label: "Question", key: "text" }, { label: "Marks", key: "marks", num: true },
      { label: "Type", render: q => (q.types || []).slice(0, 2).join(", ") },
    ], qs), { wide: true });
  }

  function chartBox(height = 420) { const d = h("div", { class: "chart", style: { height: `${height}px` } }); area.appendChild(d); return d; }

  function show(key) {
    clear(area);
    if (key === "timeline") {
      const c = charts.timeline;
      const el = chartBox(Math.max(380, 22 * c.y.length + 120));
      plot(el, [{ type: "heatmap", x: c.x, y: c.y, z: c.z, customdata: c.marks, colorscale: [[0, "rgba(128,128,128,0.08)"], [1, "#2f5f9e"]],
        showscale: false, xgap: 2, ygap: 2, hovertemplate: "%{y}<br>%{x}<br>marks %{customdata}<extra></extra>" }],
        { title: c.title, yaxis: { autorange: "reversed", tickfont: { size: 11 } }, margin: { l: 280, r: 10, t: 40, b: 80 } },
        (ev) => { const pt = ev.points[0]; const yi = c.y.indexOf(pt.y); const xi = c.x.indexOf(pt.x); showQuestions(c.questions[yi][xi], `${pt.y} in ${pt.x}`); });
    } else if (key === "frequency") {
      const c = charts.frequency_by_exam;
      if (c) {
        plot(chartBox(), c.series.map(sr => ({ type: "bar", name: sr.name, x: c.x, y: sr.values, customdata: sr.questions })),
          { barmode: "stack", title: c.title }, (ev) => { const pt = ev.points[0]; showQuestions(pt.customdata, `${pt.data.name}, ${pt.x}`); });
      }
      const m = charts.marks_distribution;
      if (m) {
        const centers = m.counts.map((_, i) => (m.edges[i] + m.edges[i + 1]) / 2);
        plot(chartBox(300), [{ type: "bar", x: centers, y: m.counts, marker: { color: "#3b6ea8" } }], { title: m.title, xaxis: { title: "marks" } });
      }
      const tc = coverage.topics;
      if (tc && tc.available) {
        append(area, h("div", { class: "card" }, h("h3", {}, "Coverage"),
          h("p", {}, `On average ${tc.mean_topics_per_exam} topics per paper.`),
          tc.always_tested.length ? h("p", {}, `Tested in 90% or more of papers: ${tc.always_tested.join("; ")}`) : null,
          tc.rarely_tested.length ? h("p", {}, `Rarely tested (15% or less): ${tc.rarely_tested.join("; ")}`) : null,
          tc.never_tested.length ? h("p", {}, `Never tested: ${tc.never_tested.join("; ")}`) : null));
      }
    } else if (key === "recency") {
      const c = charts.recency_importance;
      plot(chartBox(Math.max(380, 20 * c.labels.length + 100)), [
        { type: "bar", orientation: "h", name: "recency-weighted", y: c.labels, x: c.values, marker: { color: "#2f5f9e" } },
        { type: "bar", orientation: "h", name: "all-time frequency", y: c.labels, x: c.frequency, marker: { color: "#9db4d3" } }],
        { title: c.title, barmode: "group", yaxis: { autorange: "reversed" }, margin: { l: 280, r: 10, t: 40, b: 40 } });
      const t = charts.time_since_last;
      plot(chartBox(Math.max(380, 20 * t.labels.length + 100)), [{ type: "bar", orientation: "h", y: t.labels,
        x: t.values.map(v => v ?? 0), marker: { color: t.never.map(n => n ? "#aaa" : "#b8641e") } }],
        { title: `${t.title} (grey = never)`, yaxis: { autorange: "reversed" }, margin: { l: 280, r: 10, t: 40, b: 40 } });
    } else if (key === "intervals") {
      const c = charts.recurrence_intervals;
      plot(chartBox(Math.max(380, 22 * c.items.length + 100)), c.items.map(it => ({ type: "box", x: it.gaps, name: it.label, boxpoints: "all", orientation: "h", showlegend: false })),
        { title: c.title, margin: { l: 280, r: 10, t: 40, b: 40 } });
      const rows = (rotation.items || []);
      append(area, h("div", { class: "card" }, h("h3", {}, "Rotation test"),
        h("p", { class: "muted" }, "A rotation is reported only when the gaps between appearances are more regular than random placement would produce (permutation test)."),
        rows.length ? table([{ label: "Topic", key: "label" }, { label: "Gaps", render: r => r.gaps.join(", ") },
          { label: "Period", key: "period", num: true }, { label: "p-value", key: "p_value", num: true },
          { label: "Rotation?", render: r => r.significant ? "yes" : "no evidence" }], rows) : h("p", {}, "Not enough repeated appearances to test.")));
    } else if (key === "cooc") {
      const c = charts.cooccurrence_network;
      const ex = [], ey = [];
      for (const e of c.edges) { const a = c.nodes[e.a], b = c.nodes[e.b]; ex.push(a.x, b.x, null); ey.push(a.y, b.y, null); }
      plot(chartBox(560), [
        { type: "scatter", mode: "lines", x: ex, y: ey, line: { color: "rgba(120,140,170,0.45)", width: 1 }, hoverinfo: "skip", showlegend: false },
        { type: "scatter", mode: "markers+text", x: c.nodes.map(n => n.x), y: c.nodes.map(n => n.y), text: c.nodes.map(n => n.label.slice(0, 28)),
          textposition: "top center", textfont: { size: 9 }, marker: { size: c.nodes.map(n => 6 + 2 * n.size), color: c.nodes.map(n => n.unit), colorscale: "Viridis" },
          hovertext: c.nodes.map(n => `${n.label}: ${n.size} papers`), hoverinfo: "text", showlegend: false }],
        { title: c.title, xaxis: { visible: false }, yaxis: { visible: false, scaleanchor: "x" } });
      const co = coverage.cooccurrence;
      if (co && co.available) {
        append(area, h("div", { class: "card" }, h("h3", {}, "Statistically tested pairs"), h("p", { class: "muted" }, co.note),
          co.significant.length ? table([{ label: "Topic A", key: "a_label" }, { label: "Topic B", key: "b_label" },
            { label: "Together", key: "together", num: true }, { label: "Lift", key: "lift", num: true }, { label: "q-value", key: "q_value", num: true }], co.significant)
            : h("p", {}, `None of the ${co.tested_pairs} tested pairs is significant after correction. The network above is descriptive only.`)));
      }
    } else if (key === "types") {
      const c = charts.type_trends;
      plot(chartBox(), c.series.map(sr => ({ type: "bar", name: sr.name, x: c.x, y: sr.values })), { barmode: "stack", title: c.title });
    } else if (key === "coverage") {
      const c = charts.unit_coverage;
      if (c) plot(chartBox(), c.series.map(sr => ({ type: "scatter", mode: "lines", stackgroup: "one", name: sr.name, x: c.x, y: sr.values })),
        { title: c.title, yaxis: { tickformat: ".0%" } });
      const u = coverage.units;
      if (u && u.available) append(area, h("div", { class: "card" }, h("p", {}, `Mean balance index: ${u.mean_balance}. `, h("span", { class: "muted" }, u.note))));
    } else if (key === "structure") {
      append(area, h("div", { class: "card" }, h("h3", {}, "Historical structure"),
        table([{ label: "Property", key: "label" }, { label: "Typical", render: r => String(r.typical) },
          { label: "Recent share", render: r => `${Math.round(100 * r.share_recent)}%` }, { label: "Confidence", key: "level" },
          { label: "Per paper", render: r => r.values.map(v => v === true ? "yes" : v === false ? "no" : v ?? "-").join(", ") }], structure.patterns)),
        h("div", { class: "card" }, h("h3", {}, "Hypotheses about the next paper (speculative)"),
          h("ul", { class: "evidence" }, structure.hypotheses.map(x => h("li", {}, x)))));
    }
  }
}
