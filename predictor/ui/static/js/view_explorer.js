// Syllabus Explorer: the course contents as a navigable tree (left) and the selected node's panel (right).
// Routes: #/explorer (course list), #/course/{cid}/explorer and #/course/{cid}/explorer/{nodeId}.

import { api } from "./api.js";
import { append, badge, clear, h, toast } from "./dom.js";
import { freshnessBanner, priorityColor, priorityWord, rankRangeText, renderNodePanel, syllabusOrder } from "./topic_panel.js";

const PRIORITIES = ["Very high", "High", "Moderate", "Low"];
const SORTS = [["syllabus", "Syllabus order"], ["frequency", "Historical frequency"], ["rank", "Likelihood (rank)"], ["last", "Last appearance"]];
const nodeLabel = (n) => `${n.number || ""} ${n.title || ""}`.trim();

function when(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

// ------------------------------------------------------------------ #/explorer
export async function renderExplorerCourses(main) {
  const courses = await api.get("/api/explorer/courses");
  append(main, h("h1", {}, "Syllabus Explorer"),
    h("p", { class: "muted" }, "Browse a course's syllabus. For every unit and topic you see how often it appeared in your ",
      "past papers, the questions asked in their original wording, their formats and the current prediction. Choose a course."));
  if (!courses.length) {
    append(main, h("div", { class: "empty" }, "No courses yet. ", h("a", { href: "#/" }, "Create a course"), " first."));
    return;
  }
  const grid = h("div", { class: "grid cols-3" });
  for (const c of courses) {
    const fr = c.freshness || {};
    grid.appendChild(h("a", { class: "card ex-course", href: `#/course/${c.id}/explorer` },
      h("h3", {}, c.name), c.code ? h("div", { class: "muted" }, c.code) : null,
      h("p", { style: { margin: "8px 0 4px" } }, `${c.nodes} syllabus items, ${c.papers_included} of ${c.papers} papers included`),
      h("p", { class: "help", style: { margin: "0 0 8px" } }, c.run ? `Last analysis: ${when(c.run.finished_at)}` : "Not analysed yet"),
      h("div", { class: "row" }, c.run ? (fr.stale ? badge("results out of date", "warn") : badge("analysed", "ok")) : badge("no counts yet"),
        c.is_synthetic ? badge("demo data") : null)));
  }
  main.appendChild(grid);
}

// ------------------------------------------------------------------ #/course/{cid}/explorer[/{nodeId}]
export async function renderExplorer(main, course, { sub } = {}) {
  const cid = course.id;
  let alive = true;
  window.addEventListener("predictor:leave", () => { alive = false; }, { once: true });

  const banners = h("div");
  const intro = h("p", { class: "muted" });
  const listBox = h("div", { class: "ex-tree", role: "navigation", "aria-label": "Syllabus items" });
  const countLine = h("span", { "aria-live": "polite" });
  const panel = h("div", { class: "ex-main" });
  const state = { q: "", unit: "", history: "", format: "", priority: "", evidence: "", sort: "syllabus" };
  const expanded = new Set();
  let ov = null;
  let byId = new Map();
  let order = [];
  let pos = new Map();
  let selected = null;
  let nodeToken = 0;

  // Controls -------------------------------------------------------
  const search = h("input", { type: "search", placeholder: "Search titles, numbers, wording, concepts", "aria-label": "Search the syllabus" });
  const selUnit = h("select", { "aria-label": "Unit or group" });
  const selHistory = h("select", { "aria-label": "History" });
  const selFormat = h("select", { "aria-label": "Format" });
  const selPriority = h("select", { "aria-label": "Priority" },
    h("option", { value: "" }, "Any priority"), PRIORITIES.map(p => h("option", { value: p }, p)));
  const selEvidence = h("select", { "aria-label": "Evidence" },
    h("option", { value: "" }, "Any evidence"),
    h("option", { value: "strong" }, "Strong historical evidence"),
    h("option", { value: "uncertain" }, "High model uncertainty"));
  const selSort = h("select", { "aria-label": "Sort" }, SORTS.map(([v, l]) => h("option", { value: v }, `Sort: ${l}`)));
  const expandAll = h("button", { class: "small" }, "Expand all");
  const collapseAll = h("button", { class: "small" }, "Collapse all");
  const clearBtn = h("button", { class: "small" }, "Clear filters");
  const filtersBox = h("details", { class: "ex-filterbox" }, h("summary", {}, "Filters"),
    h("div", { class: "ex-filter-grid" }, selUnit, selHistory, selFormat, selPriority, selEvidence, clearBtn));

  let searchTimer = null;
  search.addEventListener("input", () => { clearTimeout(searchTimer); searchTimer = setTimeout(() => { state.q = search.value.trim().toLowerCase(); renderList(); }, 120); });
  for (const [el, key] of [[selUnit, "unit"], [selHistory, "history"], [selFormat, "format"], [selPriority, "priority"], [selEvidence, "evidence"], [selSort, "sort"]]) {
    el.addEventListener("change", () => { state[key] = el.value; renderList(); });
  }
  expandAll.addEventListener("click", () => { for (const n of ov.nodes) if ((n.children || []).length) expanded.add(n.id); renderList(); });
  collapseAll.addEventListener("click", () => { expanded.clear(); renderList(); });
  clearBtn.addEventListener("click", () => {
    Object.assign(state, { q: "", unit: "", history: "", format: "", priority: "", evidence: "" });
    search.value = "";
    for (const el of [selUnit, selHistory, selFormat, selPriority, selEvidence]) el.value = "";
    renderList();
  });

  const treePane = h("aside", { class: "ex-tree-pane", "aria-label": "Syllabus tree" },
    h("div", { class: "ex-filters" }, search, filtersBox, selSort),
    h("div", { class: "ex-toolbar" }, countLine, h("span", { class: "spacer" }), expandAll, collapseAll),
    listBox);
  append(main, h("h1", {}, "Syllabus Explorer"), intro, banners, h("div", { class: "ex-layout" }, treePane, panel));

  // Data -------------------------------------------------------------
  async function loadOverview(first) {
    const data = await api.get(`/api/courses/${cid}/explorer`);
    if (!alive) return false;
    ov = data;
    byId = new Map(ov.nodes.map(n => [n.id, n]));
    order = syllabusOrder(ov.nodes, ov.roots);
    pos = new Map(order.map((id, i) => [id, i]));
    if (first) for (const n of ov.nodes) if (n.depth <= 1 && (n.children || []).length) expanded.add(n.id);
    fillSelects();
    renderIntro();
    renderBanners();
    renderList();
    return true;
  }

  function fillSelects() {
    const recent = ((ov.nodes.find(n => n.stats) || {}).stats || {}).recent_window || 3;
    clear(selUnit);
    append(selUnit, h("option", { value: "" }, "All units"),
      (ov.roots || []).map(id => byId.get(id)).filter(n => n && (n.children || []).length)
        .map(n => h("option", { value: n.id }, nodeLabel(n))));
    clear(selHistory);
    append(selHistory, h("option", { value: "" }, "Any history"),
      h("option", { value: "all" }, `Appeared in each of the last ${recent} papers`),
      h("option", { value: "any" }, `Appeared in any of the last ${recent} papers`),
      h("option", { value: "never" }, "Never appeared in the supplied papers"),
      h("option", { value: "has" }, "Has past questions"));
    clear(selFormat);
    append(selFormat, h("option", { value: "" }, "Any format"), (ov.families || []).map(f => h("option", { value: f }, `Format: ${f}`)));
    selUnit.value = state.unit;
    selHistory.value = state.history;
    selFormat.value = state.format;
  }

  function renderIntro() {
    const ex = ov.excluded_papers || [];
    clear(intro);
    append(intro, ov.run
      ? `${ov.usable_papers} usable past papers${ex.length ? `, ${ex.length} excluded` : ""}${(ov.missing_years || []).length ? `, no paper for ${ov.missing_years.join(", ")}` : ""}. `
      : "", "Choose a unit or topic to see its exam history, past questions, formats and prediction.");
  }

  function renderBanners() {
    clear(banners);
    if (!ov.run) {
      append(banners, h("div", { class: "banner info" }, "This course has not been analysed yet. The counts, past questions and ",
        "predictions appear after you run ", h("a", { href: `#/course/${cid}/predict` }, "Analyze & Predict"), ". You can still browse the syllabus."));
      return;
    }
    const fr = ov.freshness;
    if (fr && fr.stale) append(banners, freshnessBanner(fr, { courseId: cid, onDone: reloadAll }));
    else if (fr && fr.message) append(banners, h("div", { class: "banner info" }, fr.message));
  }

  async function reloadAll() {
    try {
      if (!(await loadOverview(false))) return;
      if (selected) showNode(selected, { scroll: false });
    } catch (e) { toast(e.message, "error"); }
  }

  async function refreshFreshness() {
    try {
      const fr = await api.get(`/api/courses/${cid}/freshness`);
      if (!alive || !ov) return;
      ov.freshness = fr;
      renderBanners();
    } catch (e) { toast(e.message, "error"); }
  }

  // Filtering ------------------------------------------------------------
  const filtering = () => !!(state.q || state.unit || state.history || state.format || state.priority || state.evidence);

  function ancestors(id) {
    const out = [];
    for (let n = byId.get((byId.get(id) || {}).parent_id); n; n = byId.get(n.parent_id)) out.push(n.id);
    return out;
  }

  function matches(n) {
    const s = n.stats;
    const p = n.prediction;
    if (state.q) {
      const hay = `${n.number || ""} ${n.title || ""} ${n.original || ""} ${n.label || ""} ${(n.concepts || []).join(" ")}`.toLowerCase();
      if (!hay.includes(state.q)) return false;
    }
    if (state.unit && n.id !== Number(state.unit) && !ancestors(n.id).includes(Number(state.unit))) return false;
    if (state.history) {
      if (!s) return false;
      if (state.history === "all" && !(s.recent_window > 0 && s.recent_hits >= s.recent_window)) return false;
      if (state.history === "any" && !(s.recent_hits > 0)) return false;
      if (state.history === "never" && s.exam_frequency !== 0) return false;
      if (state.history === "has" && !(s.question_frequency > 0)) return false;
    }
    if (state.format && !(s && (s.families || []).some(f => f.display === state.format && f.papers >= 2))) return false;
    if (state.priority && priorityWord(p) !== state.priority) return false;
    if (state.evidence === "strong" && !(s && p && s.exam_frequency >= 3 && ["Strong", "Moderate"].includes(p.evidence_strength))) return false;
    if (state.evidence === "uncertain" && !(p && (p.uncertainty === "High" || (p.uncertainty || {}).level === "High"))) return false;
    return true;
  }

  // Tree / list rendering -------------------------------------------------
  function row(n, { caption = null, dim = false } = {}) {
    const s = n.stats;
    const p = n.prediction;
    const marks = [];
    if (s && s.exam_frequency === 0) marks.push(h("span", { class: "ex-mark" }, "no past questions"));
    if (n.inferred) marks.push(h("span", { class: "ex-mark inferred", title: n.inferred_reason || "" }, "inferred name"));
    if (n.excluded) marks.push(h("span", { class: "ex-mark excluded" }, "excluded"));
    if (n.is_lab) marks.push(h("span", { class: "ex-mark lab" }, "lab"));
    const tip = [nodeLabel(n), p ? `${priorityWord(p)} priority, rank ${p.rank}` : null, s ? `appeared in ${s.exam_frequency} of ${s.usable_papers} usable papers` : null].filter(Boolean).join("; ");
    return h("button", {
      class: `ex-row${n.id === selected ? " active" : ""}${n.excluded ? " excluded" : ""}${dim ? " dim" : ""}`,
      "data-id": n.id, "aria-current": n.id === selected ? "true" : null, title: tip,
      onclick: () => select(n.id),
    },
    h("span", { class: "ex-dot", style: { background: p ? priorityColor(p) : "transparent" } }),
    h("span", { class: "ex-row-title" }, n.number ? h("span", { class: "ex-num" }, n.number) : null, n.number ? " " : null, n.title),
    s ? h("span", { class: "ex-count" }, `${s.exam_frequency}/${s.usable_papers} papers`) : h("span"),
    caption ? h("span", { class: "ex-caption" }, caption) : null,
    marks.length ? h("span", { class: "ex-marks" }, marks) : null);
  }

  function treeLevel(ids, ul, visible, matchSet) {
    for (const id of ids) {
      const n = byId.get(id);
      if (!n || (visible && !visible.has(id))) continue;
      const kids = (n.children || []).filter(k => byId.has(k));
      const li = h("li");
      const wrap = h("div", { class: "ex-row-wrap" });
      li.appendChild(wrap);
      if (kids.length) {
        const open = visible ? true : expanded.has(id);
        const childUl = h("ul", { class: "ex-children" });
        childUl.hidden = !open;
        const toggle = h("button", { class: "ex-toggle", "aria-expanded": open ? "true" : "false", "aria-label": `${open ? "Collapse" : "Expand"} ${nodeLabel(n)}` }, open ? "▾" : "▸");
        toggle.addEventListener("click", () => {
          const nowOpen = childUl.hidden;
          if (nowOpen) {
            expanded.add(id);
            if (!childUl.dataset.built) { treeLevel(kids, childUl, visible, matchSet); childUl.dataset.built = "1"; }
          } else expanded.delete(id);
          childUl.hidden = !nowOpen;
          toggle.textContent = nowOpen ? "▾" : "▸";
          toggle.setAttribute("aria-expanded", nowOpen ? "true" : "false");
          toggle.setAttribute("aria-label", `${nowOpen ? "Collapse" : "Expand"} ${nodeLabel(n)}`);
        });
        append(wrap, toggle, row(n, { dim: matchSet && !matchSet.has(id) }));
        if (open) { treeLevel(kids, childUl, visible, matchSet); childUl.dataset.built = "1"; }
        li.appendChild(childUl);
      } else {
        append(wrap, h("span", { class: "ex-toggle-space" }), row(n, { dim: matchSet && !matchSet.has(id) }));
      }
      ul.appendChild(li);
    }
  }

  function renderList() {
    if (!ov) return;
    clear(listBox);
    const active = filtering();
    const matchSet = active ? new Set(ov.nodes.filter(matches).map(n => n.id)) : null;
    const tree = state.sort === "syllabus";
    expandAll.hidden = !tree || active;
    collapseAll.hidden = !tree || active;
    filtersBox.querySelector("summary").textContent = `Filters${["unit", "history", "format", "priority", "evidence"].filter(k => state[k]).length ? ` (${["unit", "history", "format", "priority", "evidence"].filter(k => state[k]).length} on)` : ""}`;
    if (tree) {
      let visible = null;
      if (matchSet) {
        visible = new Set();
        for (const id of matchSet) { visible.add(id); for (const a of ancestors(id)) visible.add(a); }
      }
      const ul = h("ul", { class: "ex-root" });
      treeLevel(ov.roots && ov.roots.length ? ov.roots : order.filter(id => !byId.has(byId.get(id).parent_id)), ul, visible, matchSet);
      listBox.appendChild(ul);
      countLine.textContent = matchSet ? `${matchSet.size} of ${ov.nodes.length} items match` : `${ov.nodes.length} items`;
      if (matchSet && !matchSet.size) listBox.appendChild(h("p", { class: "muted ex-none" }, ov.run ? "No item matches these filters." : "No item matches. History filters need an analysis first."));
      return;
    }
    const pick = ov.nodes.filter(n => n.level !== "group" && (!matchSet || matchSet.has(n.id)));
    const key = {
      frequency: (a, b) => ((b.stats || {}).exam_frequency ?? -1) - ((a.stats || {}).exam_frequency ?? -1)
        || ((b.stats || {}).question_frequency ?? -1) - ((a.stats || {}).question_frequency ?? -1),
      rank: (a, b) => ((a.prediction || {}).rank ?? Infinity) - ((b.prediction || {}).rank ?? Infinity),
      last: (a, b) => ((b.stats || {}).last_index ?? -1) - ((a.stats || {}).last_index ?? -1),
    }[state.sort];
    pick.sort((a, b) => key(a, b) || pos.get(a.id) - pos.get(b.id));
    const ul = h("ul", { class: "ex-root ex-flat" });
    for (const n of pick) {
      const path = ancestors(n.id).reverse().map(id => nodeLabel(byId.get(id))).join(" › ");
      const extra = state.sort === "rank" && n.prediction ? `rank ${n.prediction.rank}` : state.sort === "last" && n.stats ? (n.stats.last_label || "never in the supplied papers") : "";
      ul.appendChild(h("li", {}, h("div", { class: "ex-row-wrap" }, row(n, { caption: [path, extra].filter(Boolean).join(" · ") || null }))));
    }
    listBox.appendChild(ul);
    countLine.textContent = `${pick.length} topics`;
    if (!pick.length) listBox.appendChild(h("p", { class: "muted ex-none" }, "No topic matches these filters."));
  }

  function markSelected() {
    for (const el of listBox.querySelectorAll(".ex-row.active")) { el.classList.remove("active"); el.removeAttribute("aria-current"); }
    if (!selected || !ov) return;
    let grew = false;
    for (const a of ancestors(selected)) if (!expanded.has(a)) { expanded.add(a); grew = true; }
    if (grew && state.sort === "syllabus" && !filtering()) renderList();
    const el = listBox.querySelector(`.ex-row[data-id="${selected}"]`);
    if (!el) return;
    el.classList.add("active");
    el.setAttribute("aria-current", "true");
    const box = listBox.getBoundingClientRect();
    const r = el.getBoundingClientRect();
    if (r.top < box.top || r.bottom > box.bottom) listBox.scrollTop += r.top - box.top - box.height / 3;
  }

  // Selection and the main panel ---------------------------------------------
  function select(id) {
    const target = `#/course/${cid}/explorer/${id}`;
    if (window.location.hash !== target) window.location.hash = target;
    else if (selected !== id) showNode(id);
  }

  function renderWelcome() {
    clear(panel);
    const top = ov ? ov.nodes.filter(n => n.prediction).sort((a, b) => a.prediction.rank - b.prediction.rank).slice(0, 5) : [];
    append(panel, h("div", { class: "card" },
      h("h3", {}, "Choose a syllabus item"),
      h("p", {}, "Select a unit or topic on the left. You will see how often it appeared in the past papers, every past question on it ",
        "in the original wording, the formats it was asked in, a suggested question format and where it sits in the syllabus."),
      h("p", { class: "help" }, "Tags such as Historical fact, Statistical estimate and Model ranking tell you what kind of value you are reading. See Help for details."),
      top.length ? [h("h4", {}, "Highest ranked topics"), h("ol", { class: "ex-childpreds" }, top.map(n => h("li", {},
        h("button", { class: "ex-link", onclick: () => select(n.id) }, nodeLabel(n)), " ",
        h("span", { class: "help" }, `${priorityWord(n.prediction)} priority. ${rankRangeText(n.prediction)}`))))] : null));
  }

  async function showNode(id, { scroll = true } = {}) {
    selected = id;
    markSelected();
    if (!id) { renderWelcome(); return; }
    const token = ++nodeToken;
    clear(panel);
    panel.appendChild(h("div", { class: "card ex-loading", role: "status" }, "Loading..."));
    const narrow = window.matchMedia("(max-width: 980px)").matches;
    try {
      const data = await api.get(`/api/courses/${cid}/explorer/nodes/${id}`);
      if (!alive || token !== nodeToken) return;
      renderNodePanel(panel, data, { courseId: cid, nodes: ov ? ov.nodes : null, onNavigate: select, onChanged: refreshFreshness });
      if (scroll && narrow) panel.scrollIntoView({ block: "start" });
      else if (scroll && panel.getBoundingClientRect().top < 0) panel.scrollIntoView({ block: "start" });
    } catch (e) {
      if (!alive || token !== nodeToken) return;
      clear(panel);
      panel.appendChild(h("div", { class: "banner bad" }, `Could not load this item: ${e.message}`));
    }
  }

  function update(nextSub) {
    if (!alive) return;
    const id = Number(nextSub) || null;
    if (id !== selected) showNode(id);
  }

  panel.appendChild(h("div", { class: "card ex-loading", role: "status" }, "Loading..."));
  if (await loadOverview(true)) showNode(Number(sub) || null);
  return { update };
}
