import { api } from "./api.js";
import { append, clear, h } from "./dom.js";

export async function renderPaper(main, course) {
  append(main, h("h1", {}, "Predicted papers"));
  const results = await api.get(`/api/courses/${course.id}/results`);
  if (!results.run) { append(main, h("div", { class: "empty" }, "Run the analysis first.")); return; }
  const runId = results.run.id;
  const data = await api.get(`/api/runs/${runId}/artifacts/papers`);
  let seed = 7;
  const box = h("div");
  const regen = h("button", {}, "Generate other combinations");
  regen.addEventListener("click", async () => {
    seed += 1;
    const res = await api.post(`/api/runs/${runId}/papers`, { seed, variants: 3 });
    draw(res.papers);
  });
  append(main, 
    h("div", { class: "banner warn" }, "These are hypothetical papers assembled from historical structure and topic rankings. ",
      "None of them is the actual exam. Use them for practice and to plan revision."),
    h("div", { class: "row no-print", style: { marginBottom: "12px" } }, regen, h("button", { onclick: () => window.print() }, "Print")),
    data.structure_hypotheses && data.structure_hypotheses.length ? h("details", { class: "card no-print" },
      h("summary", {}, "Structure assumptions used"), h("ul", { class: "evidence" }, data.structure_hypotheses.map(x => h("li", {}, x)))) : null,
    box);
  draw(data.papers);

  function draw(papers) {
    clear(box);
    for (const p of papers) {
      const el = h("div", { class: "paper" }, h("h2", {}, p.title), h("p", { class: "help" }, p.strategy), h("p", { class: "help" }, p.disclaimer));
      for (const q of p.questions) {
        if (q.short_notes) {
          el.appendChild(h("div", { class: "pq" }, h("strong", {}, `${q.number}. `), q.instruction,
            h("div", { class: "parts" }, q.parts.map(pt => h("div", {}, `${pt.label}) ${pt.text}`)))));
          continue;
        }
        const parts = q.parts.length === 1
          ? [h("div", {}, item(q.parts[0], false))]
          : q.parts.map(pt => h("div", {}, `${pt.label}) `, item(pt, true)));
        el.appendChild(h("div", { class: "pq" }, h("strong", {}, `${q.number}. `), h("div", { class: "parts" }, parts)));
        if (q.or_alternative) {
          el.appendChild(h("div", { class: "or" }, "OR"));
          el.appendChild(h("div", { class: "pq" }, h("strong", {}, `${q.number}. `), h("div", { class: "parts" }, item(q.or_alternative, false))));
        }
      }
      el.appendChild(h("p", { class: "help", style: { marginTop: "12px" } }, `Units covered: ${p.units_covered.join("; ")}`));
      box.appendChild(el);
    }
  }

  function item(pt) {
    return h("span", {}, pt.text, pt.marks ? h("span", { class: "muted" }, ` [${pt.marks}]`) : null,
      h("div", { class: "help" }, `${pt.topic}${pt.basis === "historical_variant" ? " | past numerical pattern" : ""}`));
  }
}
