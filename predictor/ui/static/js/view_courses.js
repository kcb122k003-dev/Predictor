import { api } from "./api.js";
import { append, badge, field, h, toast } from "./dom.js";

export async function renderCourses(main) {
  const courses = await api.get("/api/courses");
  const name = h("input", { placeholder: "e.g. Chemical Engineering Thermodynamics", style: { width: "100%" } });
  const code = h("input", { placeholder: "Course code (optional)" });
  const create = h("button", { class: "primary" }, "Create course");
  create.addEventListener("click", async () => {
    if (!name.value.trim()) { toast("Give the course a name.", "error"); return; }
    const c = await api.post("/api/courses", { name: name.value, code: code.value });
    window.location.hash = `#/course/${c.id}/files`;
  });
  const demo = h("button", {}, "Load the synthetic demo course");
  demo.addEventListener("click", async () => {
    demo.disabled = true;
    demo.textContent = "Creating demo...";
    try {
      const c = await api.post("/api/demo");
      toast("Demo course created. Its files are being processed.");
      window.location.hash = `#/course/${c.id}/files`;
    } finally { demo.disabled = false; }
  });

  append(main, 
    h("h1", {}, "Your courses"),
    h("p", { class: "muted" }, "Each course is a separate project with its own papers, syllabus and predictions. ",
      "Data from one course never affects another."),
    h("div", { class: "card" },
      h("h3", {}, "New course"),
      h("div", { class: "grid cols-2" }, field("Course name", name), field("Code", code)),
      h("div", { class: "row", style: { marginTop: "10px" } }, create, h("span", { class: "spacer" }), demo)),
  );
  if (!courses.length) {
    main.appendChild(h("div", { class: "empty" }, "No courses yet. Create one above, or load the demo to see how it works."));
    return;
  }
  const grid = h("div", { class: "grid cols-3" });
  for (const c of courses) {
    const run = c.last_run;
    const status = !run ? badge("not analysed") : run.status === "done" ? badge("analysed", "ok")
      : run.status === "error" ? badge("analysis failed", "bad") : badge(run.status, "info");
    grid.appendChild(h("a", { class: "card", href: `#/course/${c.id}`, style: { display: "block", color: "inherit" } },
      h("h3", {}, c.name), c.code ? h("div", { class: "muted" }, c.code) : null,
      h("p", { class: "muted", style: { marginTop: "8px" } },
        `${c.exams_included} of ${c.exams} papers included, ${c.topics} syllabus topics`),
      status));
  }
  main.appendChild(grid);
}
