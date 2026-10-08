"""Command-line interface: ``predictor serve | demo | doctor | ingest | analyze | export | courses | models``."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
import threading
import webbrowser
from pathlib import Path

from . import __version__


def _ctx(args):
    from .services.context import AppContext

    return AppContext.create(args.data_dir)


def cmd_serve(args) -> int:
    import uvicorn

    from .app.server import create_app

    ctx = _ctx(args)
    host = args.host or ctx.settings.app.host
    port = int(args.port or ctx.settings.app.port)
    if host not in ("127.0.0.1", "localhost", "::1"):
        print(f"Warning: listening on {host} makes the app reachable from other computers on your network.")
    url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{port}/"
    print(f"Exam Predictor {__version__}\nData folder: {ctx.data_dir}\nOpen {url} in your browser. Press Ctrl+C to stop.")
    if ctx.settings.app.open_browser and not args.no_browser:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    uvicorn.run(create_app(ctx), host=host, port=port, log_level="warning")
    return 0


def cmd_demo(args) -> int:
    import tempfile

    from .demo.generator import generate_demo
    from .services.analysis_service import AnalysisService
    from .services.course_service import CourseService
    from .services.ingest_service import IngestService

    ctx = _ctx(args)
    ingest = IngestService(ctx)
    cid = CourseService(ctx).create("Fluid Mechanics (synthetic demo)", "CE 501",
                                    "Generated example data with planted patterns. Not real exams.",
                                    is_synthetic=True)
    with tempfile.TemporaryDirectory() as tmp:
        generate_demo(Path(tmp))
        for kind, folder in (("syllabus", "syllabus"), ("exam", "exams")):
            for path in sorted((Path(tmp) / folder).iterdir()):
                res = ingest.add_file(cid, path, path.name, kind)
                if res.duplicate:
                    print(f"  {path.name}: {res.message}")
                    continue
                out = ingest.process_file(res.file_id)
                print(f"  {path.name}: {out.get('questions', out.get('topics_added', ''))} "
                      f"{'questions' if kind == 'exam' else 'topics'}"
                      + (" (duplicate paper, excluded)" if out.get("duplicate_of") else ""))
    service = AnalysisService(ctx)
    summary = service.run(service.create_run(cid))
    print(f"\nCourse {cid} analysed in {summary['seconds']}s. Final ranking: {summary['selected_display']}.")
    print(f"{summary.get('inference_mode', '')}: {summary.get('inference_message', '')}")
    print(summary["selection_reason"])
    print("Start the app with: predictor serve")
    return 0


def cmd_doctor(args) -> int:
    ok = True
    print(f"Exam Predictor {__version__}")
    print(f"Python {platform.python_version()} on {platform.system()} {platform.machine()}")
    if sys.version_info < (3, 10):
        print("  PROBLEM: Python 3.10 or newer is required.")
        ok = False
    for mod, need in (("numpy", True), ("scipy", True), ("sklearn", True), ("sqlalchemy", True), ("fastapi", True),
                      ("uvicorn", True), ("pymupdf", True), ("docx", True), ("rapidfuzz", True), ("reportlab", True),
                      ("openpyxl", True), ("plotly", False), ("pytesseract", False), ("cv2", False),
                      ("sentence_transformers", False)):
        try:
            importlib.import_module(mod)
            print(f"  ok       {mod}")
        except Exception:
            print(f"  {'MISSING ' if need else 'optional'} {mod}")
            ok = ok and not need
    ctx = _ctx(args)
    ocr = ctx.ocr
    print(f"  OCR: {'Tesseract ' + str(ocr.version()) if ocr.available else 'not available - ' + ocr.unavailable_reason}")
    from .embeddings.backends import get_pretrained, neural_available

    neural, why = neural_available(ctx.settings)
    pre, pre_why = get_pretrained(ctx.settings)
    print(f"  Pretrained semantic model: {'ready (' + pre.name + ', bundled, offline)' if pre else 'unavailable (' + pre_why + ')'}")
    print(f"  Larger neural embeddings: {'ready' if neural else 'not used (' + why + ')'}; the offline TF-IDF model always works")
    print(f"  Data folder: {ctx.data_dir} ({'writable' if _writable(ctx.data_dir) else 'NOT writable'})")
    print("All required components are present." if ok else "Some required components are missing (see above).")
    return 0 if ok else 1


def _writable(path: Path) -> bool:
    try:
        probe = path / ".write_test"
        probe.write_text("x")
        probe.unlink()
        return True
    except OSError:
        return False


def cmd_courses(args) -> int:
    from .services.course_service import CourseService

    for c in CourseService(_ctx(args)).list():
        run = c["last_run"] or {}
        print(f"{c['id']:>4}  {c['name']}  ({c['exams_included']}/{c['exams']} papers, {c['topics']} topics, "
              f"last run: {run.get('status', 'none')})")
    return 0


def cmd_ingest(args) -> int:
    from .services.course_service import CourseService
    from .services.ingest_service import IngestService

    ctx = _ctx(args)
    course_id = args.course
    if course_id is None:
        course_id = CourseService(ctx).create(args.name or "New course")
        print(f"Created course {course_id}")
    service = IngestService(ctx)
    for f in args.files:
        path = Path(f)
        res = service.add_file(course_id, path, path.name, args.kind)
        if res.duplicate:
            print(f"{path.name}: {res.message}")
            continue
        out = service.process_file(res.file_id)
        print(f"{path.name}: {json.dumps({k: v for k, v in out.items() if k != 'warnings'}, default=str)}")
    return 0


def cmd_analyze(args) -> int:
    from .services.analysis_service import AnalysisService

    service = AnalysisService(_ctx(args))
    summary = service.run(service.create_run(args.course))
    print(json.dumps({k: summary.get(k) for k in ("exams", "questions", "topics", "inference_mode", "inference_message",
                                                   "selected_display", "selection_reason", "calibration_reason",
                                                   "seconds")}, indent=2))
    return 0


def cmd_export(args) -> int:
    from .export.service import ExportService

    service = ExportService(_ctx(args))
    fmt = args.format
    data = {"pdf": lambda: service.pdf(args.run), "xlsx": lambda: service.xlsx(args.run),
            "json": lambda: service.json(args.run), "csv": lambda: service.csv(args.run, args.table)}[fmt]()
    out = Path(args.out or f"run{args.run}.{fmt}")
    out.write_bytes(data)
    print(f"Wrote {out}")
    return 0


def cmd_models(args) -> int:
    from .config.settings import load_settings
    from .embeddings.backends import download_model

    settings = load_settings(args.data_dir, {"embeddings": {"model_name": args.model}} if args.model else None)
    print(f"Downloading {settings.embeddings.model_name}. This is the only command that uses the internet.")
    path = download_model(settings)
    print(f"Saved to {path}. The app will use it automatically (embedding backend 'auto').")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="predictor", description="Local exam topic and question forecasting.")
    p.add_argument("--data-dir", help="Folder for the database and uploaded files (default ~/ExamPredictorData)")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command")
    s = sub.add_parser("serve", help="Start the app in your browser")
    s.add_argument("--host")
    s.add_argument("--port", type=int)
    s.add_argument("--no-browser", action="store_true")
    s.set_defaults(func=cmd_serve)
    sub.add_parser("demo", help="Create and analyse the synthetic demo course").set_defaults(func=cmd_demo)
    sub.add_parser("doctor", help="Check the installation").set_defaults(func=cmd_doctor)
    sub.add_parser("courses", help="List courses").set_defaults(func=cmd_courses)
    i = sub.add_parser("ingest", help="Add files to a course")
    i.add_argument("files", nargs="+")
    i.add_argument("--course", type=int)
    i.add_argument("--name", help="Name for a new course when --course is not given")
    i.add_argument("--kind", choices=["exam", "syllabus"], required=True)
    i.set_defaults(func=cmd_ingest)
    a = sub.add_parser("analyze", help="Run Analyze & Predict for a course")
    a.add_argument("course", type=int)
    a.set_defaults(func=cmd_analyze)
    e = sub.add_parser("export", help="Export a run")
    e.add_argument("run", type=int)
    e.add_argument("--format", choices=["pdf", "xlsx", "csv", "json"], default="pdf")
    e.add_argument("--table", default="predictions")
    e.add_argument("--out")
    e.set_defaults(func=cmd_export)
    m = sub.add_parser("models", help="Manage optional neural models")
    msub = m.add_subparsers(dest="action")
    d = msub.add_parser("download", help="Download the sentence-transformers model (uses the internet once)")
    d.add_argument("--model")
    d.set_defaults(func=cmd_models)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        if args.command is None:  # plain "predictor" starts the app
            args = parser.parse_args((["--data-dir", args.data_dir] if args.data_dir else []) + ["serve"])
        else:
            parser.print_help()
            return 0
    return int(args.func(args) or 0)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
