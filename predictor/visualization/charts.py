"""Chart-ready data for the analytics dashboard (spec section 51).

Each chart keeps the question ids behind every point so a click in the UI can list the
underlying questions.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..evaluation.backtest import BacktestReport
from ..temporal import dynamics as dyn
from ..temporal.panel import FORMATS, Panel


def build_charts(panel: Panel, cell_questions: dict[tuple[int, int], list[int]], unit_labels: list[str],
                 report: BacktestReport, leaf_marks: list[float], settings) -> dict[str, Any]:
    T, K = panel.Y.shape
    exams = [e.label for e in panel.exams]
    labels = panel.item_labels
    charts: dict[str, Any] = {}

    charts["timeline"] = {
        "title": "Topic appearance timeline", "x": exams, "y": labels,
        "z": panel.Y.T.astype(int).tolist(), "marks": panel.marks.T.round(1).tolist(),
        "questions": [[cell_questions.get((t, k), []) for t in range(T)] for k in range(K)],
    }
    if panel.item_unit is not None and len(unit_labels):
        series = []
        for u, name in enumerate(unit_labels):
            cols = [k for k in range(K) if int(panel.item_unit[k]) == u]
            counts = panel.Y[:, cols].sum(axis=1) if cols else np.zeros(T)
            qids = [[q for k in cols for q in cell_questions.get((t, k), [])] for t in range(T)]
            series.append({"name": name, "values": counts.astype(int).tolist(), "questions": qids})
        charts["frequency_by_exam"] = {"title": "Topics tested per exam, by unit", "x": exams, "series": series}
    if T:
        h = float(settings.temporal.default_half_life)
        ew = dyn.ewma(panel.Y, h)
        order = np.argsort(-ew)
        charts["recency_importance"] = {
            "title": f"Recency-weighted topic importance (half-life {h:g} exams)",
            "labels": [labels[i] for i in order], "values": ew[order].round(3).tolist(),
            "frequency": panel.Y.mean(axis=0)[order].round(3).tolist(), "item_ids": [panel.item_ids[i] for i in order]}
        since = dyn.since_last(panel.Y)
        order = np.argsort(-since)
        charts["time_since_last"] = {
            "title": "Exams since last appearance", "labels": [labels[i] for i in order],
            "values": [int(v) if v <= T else None for v in since[order]],
            "never": [bool(v > T) for v in since[order]], "item_ids": [panel.item_ids[i] for i in order]}
        gaps = []
        for k in range(K):
            idx = np.flatnonzero(panel.Y[:, k])
            if idx.size >= 2:
                gaps.append({"label": labels[k], "item_id": panel.item_ids[k], "gaps": np.diff(idx).astype(int).tolist()})
        charts["recurrence_intervals"] = {"title": "Gaps between appearances (exams)", "items": gaps}
        lift = dyn.same_exam_lift(panel.Y)
        counts = panel.Y.sum(axis=0)
        nodes = []
        unit_of = panel.item_unit if panel.item_unit is not None else np.zeros(K, dtype=int)
        n_units = max(int(unit_of.max()) + 1 if K else 1, 1)
        for k in range(K):
            angle = 2 * math.pi * (k / max(K, 1))
            radius = 1.0 + 0.15 * (int(unit_of[k]) % 2)
            nodes.append({"id": panel.item_ids[k], "label": labels[k], "x": round(radius * math.cos(angle), 4),
                          "y": round(radius * math.sin(angle), 4), "size": int(counts[k]),
                          "unit": int(unit_of[k]), "units": n_units})
        edges = []
        for i in range(K):
            for j in range(i + 1, K):
                together = int((panel.Y[:, i] * panel.Y[:, j]).sum())
                if together >= 3 and lift[i, j] >= 1.3:
                    edges.append({"a": i, "b": j, "lift": round(float(lift[i, j]), 3), "together": together})
        edges.sort(key=lambda e: -e["lift"])
        charts["cooccurrence_network"] = {"title": "Topics that appear in the same exam more often than chance",
                                          "nodes": nodes, "edges": edges[:60]}
        fmt = panel.formats.sum(axis=1)  # (T, F)
        charts["type_trends"] = {"title": "Question formats per exam", "x": exams,
                                 "series": [{"name": f, "values": fmt[:, j].astype(int).tolist()}
                                            for j, f in enumerate(FORMATS) if fmt[:, j].sum()]}
        if panel.item_unit is not None and len(unit_labels):
            U = len(unit_labels)
            m = np.zeros((T, U))
            for k in range(K):
                m[:, int(panel.item_unit[k])] += panel.marks[:, k]
            tot = m.sum(axis=1, keepdims=True)
            share = np.divide(m, tot, out=np.zeros_like(m), where=tot > 0)
            charts["unit_coverage"] = {"title": "Share of marks by unit", "x": exams,
                                       "series": [{"name": unit_labels[u], "values": share[:, u].round(3).tolist()}
                                                  for u in range(U)]}
    vals = [v for v in leaf_marks if v]
    if vals:
        hist, edges = np.histogram(vals, bins=min(12, max(3, len(set(vals)))))
        charts["marks_distribution"] = {"title": "Marks per question", "counts": hist.tolist(),
                                        "edges": [round(float(e), 2) for e in edges]}
    rows = [r for r in report.summary_table() if r["enabled"] and r["mean"]]
    primary = report.primary
    charts["model_comparison"] = {
        "title": f"Backtest {primary.upper()}@{report.k} by model", "metric": primary,
        "models": [r["display"] for r in rows], "names": [r["model"] for r in rows],
        "values": [r["mean"].get(primary) for r in rows], "errors": [r["se"].get(primary) for r in rows],
        "selected": report.selected}
    folds = {}
    for name, mr in report.models.items():
        if mr.hidden or not mr.enabled or not mr.fold_metrics:
            continue
        folds[name] = {"display": mr.display,
                       "values": [round(mr.fold_metrics[t][primary], 4) for t in report.targets if t in mr.fold_metrics]}
    charts["backtest_folds"] = {"title": f"{primary.upper()}@{report.k} on each held-out exam",
                                "x": [panel.exams[t].label for t in report.targets], "models": folds}
    if report.calibration is not None and report.calibration.reliability:
        charts["calibration"] = {"title": "Calibration on held-out exams", "bins": report.calibration.reliability}
    return charts
