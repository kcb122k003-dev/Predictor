import { api } from "./api.js";
import { clear, h, toast } from "./dom.js";
import { renderCourses } from "./view_courses.js";
import { renderFiles } from "./view_files.js";
import { renderReview } from "./view_review.js";
import { renderSyllabus } from "./view_syllabus.js";
import { renderMapping } from "./view_mapping.js";
import { renderPredict } from "./view_predict.js";
import { renderAnalytics } from "./view_analytics.js";
import { renderModels } from "./view_models.js";
import { renderPaper } from "./view_paper.js";
import { renderSearch } from "./view_search.js";
import { renderSettings } from "./view_settings.js";
import { renderHelp } from "./view_help.js";

const TABS = [
  ["files", "1. Upload files"],
  ["review", "2. Review papers"],
  ["syllabus", "3. Syllabus"],
  ["mapping", "4. Topic mapping"],
  ["predict", "5. Analyze & Predict"],
  null,
  ["analytics", "Analytics"],
  ["models", "Model performance"],
  ["paper", "Predicted papers"],
  ["search", "Search"],
];
const VIEWS = {
  files: renderFiles, review: renderReview, syllabus: renderSyllabus, mapping: renderMapping,
  predict: renderPredict, analytics: renderAnalytics, models: renderModels, paper: renderPaper, search: renderSearch,
};

let currentCourse = null;
let renderToken = 0;

export function navigate(hash) { window.location.hash = hash; }

async function route() {
  const token = ++renderToken;
  const hash = window.location.hash.replace(/^#/, "") || "/";
  const parts = hash.split("/").filter(Boolean);
  const main = document.getElementById("main");
  const side = document.getElementById("side");
  const layout = document.getElementById("layout");
  clear(main);
  // Stop pollers from the previous view.
  window.dispatchEvent(new Event("predictor:leave"));
  try {
    if (parts[0] === "course" && parts[1]) {
      const id = Number(parts[1]);
      if (!currentCourse || currentCourse.id !== id) currentCourse = await api.get(`/api/courses/${id}`);
      if (token !== renderToken) return;
      const tab = parts[2] || (currentCourse.last_run && currentCourse.last_run.status === "done" ? "predict" : "files");
      document.getElementById("course-name").textContent = currentCourse.name;
      layout.classList.remove("no-side");
      renderSide(side, id, tab);
      const view = VIEWS[tab] || renderFiles;
      await view(main, currentCourse, { refreshCourse });
    } else {
      currentCourse = null;
      document.getElementById("course-name").textContent = "";
      layout.classList.add("no-side");
      clear(side);
      if (parts[0] === "settings") await renderSettings(main);
      else if (parts[0] === "help") await renderHelp(main);
      else await renderCourses(main);
    }
  } catch (err) {
    console.error(err);
    main.appendChild(h("div", { class: "banner bad" }, `Could not load this page: ${err.message}`));
  }
}

async function refreshCourse() {
  if (!currentCourse) return null;
  currentCourse = await api.get(`/api/courses/${currentCourse.id}`);
  return currentCourse;
}

function renderSide(side, id, active) {
  clear(side);
  for (const tab of TABS) {
    if (tab === null) { side.appendChild(h("div", { class: "sep" })); continue; }
    const [key, label] = tab;
    side.appendChild(h("a", { href: `#/course/${id}/${key}`, class: key === active ? "active" : "" }, label));
  }
}

window.addEventListener("hashchange", route);
window.addEventListener("unhandledrejection", (e) => toast(e.reason?.message || String(e.reason), "error"));
route();
