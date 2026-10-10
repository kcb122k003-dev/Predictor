// Small DOM helpers. Text is always inserted with textContent, never as HTML,
// because extracted exam text is untrusted input.

export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") el.className = value;
    else if (key === "style" && typeof value === "object") Object.assign(el.style, value);
    else if (key.startsWith("on") && typeof value === "function") el.addEventListener(key.slice(2), value);
    else if (key === "dataset") Object.assign(el.dataset, value);
    else if (value === true) el.setAttribute(key, "");
    else el.setAttribute(key, String(value));
  }
  append(el, ...children);
  return el;
}

// Null-safe append: skips null/undefined/false and flattens nested arrays.
export function append(el, ...children) {
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    el.appendChild(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return el;
}

export function clear(el) {
  while (el.firstChild) el.removeChild(el.firstChild);
  return el;
}

export function toast(message, kind = "info", ms = 4500) {
  const box = document.getElementById("toasts");
  const t = h("div", { class: `toast ${kind}` }, message);
  box.appendChild(t);
  setTimeout(() => t.remove(), ms);
}

export function pct(v, digits = 0) {
  if (v === null || v === undefined || Number.isNaN(v)) return "";
  return `${(100 * v).toFixed(digits)}%`;
}

export function num(v, digits = 3) {
  if (v === null || v === undefined || Number.isNaN(v)) return "";
  return Number(v).toFixed(digits);
}

export function badge(text, kind = "") {
  return h("span", { class: `badge ${kind}` }, text);
}

// Inference component status (never "off": a component is active, limited, downweighted or lacks an input).
export function componentBadge(status) {
  const kinds = { ACTIVE: "ok", LIMITED: "info", DOWNWEIGHTED: "warn", UNAVAILABLE: "", REFERENCE: "" };
  const titles = {
    ACTIVE: "Running with enough evidence for its parameters",
    LIMITED: "Running and contributing, with high uncertainty (little course evidence)",
    DOWNWEIGHTED: "Running, but earlier papers showed it ranks worse than the other components",
    UNAVAILABLE: "An input it needs does not exist for this course",
    REFERENCE: "Baseline kept for comparison",
  };
  return h("span", { class: `badge ${kinds[status] ?? ""}`, title: titles[status] || "" }, status || "");
}

// Horizontal share bar (0..max) with a label and value.
export function shareBar(label, value, max, text) {
  const w = max > 0 ? Math.max(0, Math.min(100, 100 * value / max)) : 0;
  return h("div", { class: "share" }, h("span", {}, label),
    h("div", { class: "track" }, h("div", { class: "fill", style: { width: `${w}%` } })),
    h("span", { class: "mono" }, text ?? value.toFixed(2)));
}

export function statusBadge(status) {
  const labels = { A: "A clearly in", B: "B probably in", C: "C uncertain", D: "D outside", unmapped: "unmapped" };
  return h("span", { class: `badge status-${status}`, title: labels[status] || status }, labels[status] || status);
}

export function table(columns, rows, { onRowClick, rowClass } = {}) {
  const thead = h("thead", {}, h("tr", {}, columns.map(c => h("th", { class: c.num ? "num" : "" }, c.label))));
  const tbody = h("tbody");
  for (const row of rows) {
    const tr = h("tr", { class: [onRowClick ? "clickable" : "", rowClass ? rowClass(row) : ""].join(" ") });
    for (const c of columns) {
      const value = c.render ? c.render(row) : row[c.key];
      tr.appendChild(h("td", { class: c.num ? "num" : "" }, value ?? ""));
    }
    if (onRowClick) tr.addEventListener("click", () => onRowClick(row, tr));
    tbody.appendChild(tr);
  }
  return h("div", { class: "table-wrap" }, h("table", { class: "data" }, thead, tbody));
}

export function modal(title, body, { wide = false } = {}) {
  const backdrop = h("div", { class: "drawer-backdrop" });
  const box = h("div", { class: "modal", style: wide ? { width: "min(1100px, 96vw)" } : {} },
    h("button", { class: "close-x small", onclick: () => close() }, "Close"),
    h("h2", {}, title), body);
  function close() { backdrop.remove(); box.remove(); document.removeEventListener("keydown", onKey); }
  function onKey(e) { if (e.key === "Escape") close(); }
  backdrop.addEventListener("click", close);
  document.addEventListener("keydown", onKey);
  document.body.append(backdrop, box);
  return { close, box };
}

// Side drawer. It closes on Escape, on a backdrop click and when the route (hash) changes, because it lives
// outside the page's main element.
export function drawer(body, { wide = false, label = "Details" } = {}) {
  const backdrop = h("div", { class: "drawer-backdrop" });
  const panel = h("div", { class: `drawer${wide ? " wide" : ""}`, role: "dialog", "aria-label": label },
    h("button", { class: "close-x small", onclick: () => close() }, "Close"), body);
  function close() {
    backdrop.remove(); panel.remove();
    document.removeEventListener("keydown", onKey);
    window.removeEventListener("hashchange", close);
  }
  function onKey(e) { if (e.key === "Escape" && !document.querySelector(".modal")) close(); }
  backdrop.addEventListener("click", close);
  document.addEventListener("keydown", onKey);
  window.addEventListener("hashchange", close);
  document.body.append(backdrop, panel);
  return { close, panel };
}

export function field(label, input, help) {
  return h("label", { class: "field" }, h("span", {}, label), input, help ? h("span", { class: "help" }, help) : null);
}

export function hasPlotly() {
  return typeof window.Plotly !== "undefined";
}

export function plotTheme() {
  const css = getComputedStyle(document.documentElement);
  const color = css.getPropertyValue("--text").trim() || "#1c2330";
  const grid = css.getPropertyValue("--border").trim() || "#d9dee6";
  return {
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { color, size: 12, family: "system-ui, sans-serif" },
    xaxis: { gridcolor: grid, zerolinecolor: grid, automargin: true },
    yaxis: { gridcolor: grid, zerolinecolor: grid, automargin: true },
    margin: { l: 50, r: 20, t: 30, b: 50 }, legend: { orientation: "h" },
  };
}

export function plot(el, data, layout = {}, onClick) {
  if (!hasPlotly()) {
    el.appendChild(h("p", { class: "muted" }, "Charts need the plotly package (pip install plotly)."));
    return;
  }
  const base = plotTheme();
  const merged = { ...base, ...layout, xaxis: { ...base.xaxis, ...(layout.xaxis || {}) }, yaxis: { ...base.yaxis, ...(layout.yaxis || {}) } };
  window.Plotly.newPlot(el, data, merged, { responsive: true, displaylogo: false });
  if (onClick) el.on("plotly_click", onClick);
}

// "Extremely High Priority" is the label used by runs made before it was renamed to "Very High Priority".
export const CATEGORY_COLORS = {
  "Very High Priority": "var(--cat-1)", "Extremely High Priority": "var(--cat-1)", "High Priority": "var(--cat-2)",
  "Moderate Priority": "var(--cat-3)", "Low Priority": "var(--cat-4)", "Excluded": "var(--bad)",
};
