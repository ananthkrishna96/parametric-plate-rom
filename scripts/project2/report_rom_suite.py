#!/usr/bin/env python3
"""Build Project-2 ROM suite post-processing tables and plots.

The ROM suite runner writes outputs in the production layout

    <rom_root>/suites/intrusive/<CASE>/<method>/...
    <rom_root>/suites/nonintrusive/<CASE>/<method>/...

Older/temporary tests may use a legacy flat layout

    <rom_root>/suites/<CASE>/intrusive/<method>/...
    <rom_root>/suites/<CASE>/nonintrusive/<method>/...

This script supports both layouts and writes report-ready CSV, Markdown,
LaTeX, JSON, and optional plots under

    <rom_root>/suites/reports/<CASE>/
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - plotting is optional at runtime
    plt = None

try:
    from paramplate.io.data_locations import load_data_locations
except Exception:  # pragma: no cover - fallback for standalone use
    load_data_locations = None

DEFAULT_CASE = "CASE_02_SUMMER_1MP1TP__PANEL_MONOLITHIC_NV0_NH0__BC_FREE_EDGE__LOAD_PATCH"


@dataclass(frozen=True)
class ReportRow:
    suite: str
    method: str
    field: str
    rank: int | None
    coeff_test_error: float | None = None
    field_test_error: float | None = None
    projection_test_error: float | None = None
    train_error: float | None = None
    source_file: str = ""

    def metric_value(self) -> float:
        """Return the primary field/projection test error used for ranking."""
        if self.field_test_error is not None:
            return self.field_test_error
        if self.projection_test_error is not None:
            return self.projection_test_error
        return math.inf

    def as_dict(self) -> dict[str, Any]:
        primary = self.metric_value()
        return {
            "suite": self.suite,
            "method": self.method,
            "field": self.field,
            "rank": self.rank,
            "coeff_test_error": self.coeff_test_error,
            "field_test_error": self.field_test_error,
            "projection_test_error": self.projection_test_error,
            "train_error": self.train_error,
            "primary_test_error": None if math.isinf(primary) else primary,
            "source_file": self.source_file,
        }


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def _row_from_flat_item(item: dict[str, Any], source_file: Path) -> ReportRow:
    """Parse an already-flat row-like summary dictionary."""
    method = str(item.get("method", source_file.parent.name.replace("_", "-")))
    suite = str(item.get("suite", _infer_suite_from_path(source_file)))
    field = str(item.get("field", item.get("field_name", item.get("name", "unknown"))))

    # Intrusive projection summaries usually call the metric test_error or
    # relative_error_test; non-intrusive summaries usually call it
    # field_test_error or field_relative_error_test.
    field_test = _as_float(item.get("field_test_error"))
    if field_test is None:
        field_test = _as_float(item.get("field_relative_error_test"))

    projection_test = None
    if field_test is None:
        projection_test = _as_float(item.get("test_error"))
    if projection_test is None and field_test is None:
        projection_test = _as_float(item.get("relative_error_test"))

    train_error = _as_float(item.get("train_error"))
    if train_error is None:
        train_error = _as_float(item.get("field_relative_error_train"))
    if train_error is None:
        train_error = _as_float(item.get("relative_error_train"))

    coeff_error = _as_float(item.get("coeff_test_error"))
    if coeff_error is None:
        coeff_error = _as_float(item.get("coefficient_relative_error_test"))

    return ReportRow(
        suite=suite,
        method=method,
        field=field,
        rank=_as_int(item.get("rank")),
        coeff_test_error=coeff_error,
        field_test_error=field_test,
        projection_test_error=projection_test,
        train_error=train_error,
        source_file=str(source_file),
    )


def _rows_from_nested_method_summary(obj: dict[str, Any], source_file: Path) -> list[ReportRow]:
    """Parse the actual run_rom_suite.py JSON shape.

    Current Project-2 method summaries are written as one JSON file per method:

        {
          "suite": "intrusive" | "nonintrusive",
          "method": "pod-projected" | "pod-gpr" | ...,
          "w": { ... metrics ... },
          "theta": { ... metrics ... } | null
        }

    The field metric dictionaries are not flat rows, so they need to inherit
    the suite/method metadata from the parent object.
    """
    rows: list[ReportRow] = []
    parent_suite = str(obj.get("suite", _infer_suite_from_path(source_file)))
    parent_method = str(obj.get("method", source_file.parent.name.replace("_", "-")))

    for field_key in ("w", "theta"):
        metrics = obj.get(field_key)
        if not isinstance(metrics, dict):
            continue
        flat = dict(metrics)
        flat.setdefault("suite", parent_suite)
        flat.setdefault("method", parent_method)
        flat.setdefault("field", flat.get("field_name", field_key))
        rows.append(_row_from_flat_item(flat, source_file))
    return rows


def _rows_from_summary_object(obj: Any, source_file: Path) -> list[ReportRow]:
    """Normalize several possible summary JSON shapes into ReportRow objects."""
    rows: list[ReportRow] = []

    if isinstance(obj, dict):
        # First handle the real Project-2 per-method summary shape.
        rows.extend(_rows_from_nested_method_summary(obj, source_file))
        if rows:
            return rows

        if "rows" in obj and isinstance(obj["rows"], list):
            candidates = obj["rows"]
        elif "results" in obj and isinstance(obj["results"], list):
            candidates = obj["results"]
        elif all(k in obj for k in ("method", "field")) or all(k in obj for k in ("method", "field_name")):
            candidates = [obj]
        else:
            candidates = []
            for value in obj.values():
                if isinstance(value, list):
                    candidates.extend(x for x in value if isinstance(x, dict))
    elif isinstance(obj, list):
        candidates = [x for x in obj if isinstance(x, dict)]
    else:
        candidates = []

    for item in candidates:
        if isinstance(item, dict):
            rows.append(_row_from_flat_item(item, source_file))
    return rows


def _infer_suite_from_path(path: Path) -> str:
    parts = set(path.parts)
    if "intrusive" in parts:
        return "intrusive"
    if "nonintrusive" in parts:
        return "nonintrusive"
    return "unknown"


def case_summary_files(suite_root: Path, case: str) -> list[Path]:
    """Return summary JSON files for a case in production and legacy layouts."""
    patterns = [
        # Production layout used by run_rom_suite.py:
        f"intrusive/{case}/*/intrusive_projection_summary.json",
        f"nonintrusive/{case}/*/nonintrusive_summary.json",
        f"intrusive/{case}/**/*summary*.json",
        f"nonintrusive/{case}/**/*summary*.json",
        # Legacy flat/test layout:
        f"{case}/intrusive/*/intrusive_projection_summary.json",
        f"{case}/nonintrusive/*/nonintrusive_summary.json",
        f"{case}/**/*summary*.json",
    ]
    files: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for path in sorted(suite_root.glob(pattern)):
            if path in seen or not path.is_file():
                continue
            # Avoid re-ingesting already-generated report JSON files.
            if "reports" in path.parts:
                continue
            seen.add(path)
            files.append(path)
    return files


def collect_suite_rows(suite_root: Path, case: str) -> list[ReportRow]:
    rows: list[ReportRow] = []
    for path in case_summary_files(suite_root, case):
        try:
            rows.extend(_rows_from_summary_object(_read_json(path), path))
        except Exception as exc:
            print(f"[warning] could not parse {path}: {exc}", file=sys.stderr)

    # Deduplicate by suite/method/field/source. Method-specific summaries are kept.
    dedup: dict[tuple[str, str, str, str], ReportRow] = {}
    for row in rows:
        key = (row.suite, row.method, row.field, row.source_file)
        dedup[key] = row
    return list(dedup.values())


def _sort_key(row: ReportRow) -> tuple[str, str, float, str]:
    return (row.field, "0" if row.suite == "intrusive" else "1", row.metric_value(), row.method)


def ranked_rows(rows: Iterable[ReportRow]) -> list[ReportRow]:
    return sorted(rows, key=_sort_key)


def write_csv(rows: list[ReportRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(ReportRow("", "", "", None).as_dict().keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.as_dict())


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if math.isinf(value) or math.isnan(value):
            return ""
        return f"{value:.6e}"
    return str(value)


def write_markdown(rows: list[ReportRow], path: Path, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["field", "suite", "method", "rank", "coeff_test_error", "field/projection_test_error", "train_error"]
    lines = [
        f"# {title}",
        "",
        f"Generated UTC: `{_utc_now()}`",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for r in rows:
        lines.append("| " + " | ".join([
            r.field,
            r.suite,
            r.method,
            _fmt(r.rank),
            _fmt(r.coeff_test_error),
            _fmt(r.metric_value()),
            _fmt(r.train_error),
        ]) + " |")
    lines.append("")
    lines.append("## Recommended interpretation")
    lines.append("")
    lines.append("Use `pod-projected` as the projection/truncation lower-bound for the current POD basis. Among non-intrusive methods, choose the method with the smallest primary test error for the target field, while also checking stability across cases and basis sizes.")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_latex(rows: list[ReportRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        r"\begin{tabular}{llllrr}",
        r"\toprule",
        r"Field & Suite & Method & Rank & Coeff. error & Field/proj. error \\",
        r"\midrule",
    ]
    for r in rows:
        lines.append(
            f"{r.field} & {r.suite} & {r.method} & {_fmt(r.rank)} & {_fmt(r.coeff_test_error)} & {_fmt(r.metric_value())} \\\\" 
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def write_plots(rows: list[ReportRow], out_dir: Path) -> None:
    if plt is None:
        print("[warning] matplotlib is unavailable; skipping plots", file=sys.stderr)
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for field in sorted({r.field for r in rows}):
        sub = [r for r in rows if r.field == field and math.isfinite(r.metric_value())]
        if not sub:
            continue
        sub = sorted(sub, key=lambda r: r.metric_value())
        labels = [r.method for r in sub]
        values = [r.metric_value() for r in sub]
        fig = plt.figure(figsize=(8, 4.5))
        ax = fig.add_subplot(111)
        ax.bar(labels, values)
        ax.set_yscale("log")
        ax.set_ylabel("Relative test error")
        ax.set_title(f"ROM comparison for {field}")
        ax.tick_params(axis="x", rotation=35)
        fig.tight_layout()
        fig.savefig(out_dir / f"rom_suite_{field}_test_error.png", dpi=300)
        fig.savefig(out_dir / f"rom_suite_{field}_test_error.pdf")
        plt.close(fig)


def load_pod_basis_decision(pod_dir: Path) -> dict[str, Any]:
    decision: dict[str, Any] = {"pod_dir": str(pod_dir), "fields": {}}
    summary_path = pod_dir / "pod_summary.json"
    if summary_path.exists():
        try:
            decision["pod_summary"] = _read_json(summary_path)
        except Exception as exc:
            decision["pod_summary_error"] = str(exc)

    for field, filename in [("w", "pod_w.npz"), ("theta", "pod_theta.npz")]:
        path = pod_dir / filename
        if not path.exists():
            continue
        info: dict[str, Any] = {"artifact": str(path)}
        try:
            data = np.load(path)
            saved_rank = None
            for key in ("singular_values", "eigenvalues", "energy", "cumulative_energy", "basis"):
                if key in data:
                    arr = np.asarray(data[key])
                    info[f"{key}_shape"] = list(arr.shape)
                    if key == "basis" and arr.ndim >= 2:
                        saved_rank = int(arr.shape[1])
                    if arr.ndim == 1:
                        info[f"{key}_first10"] = [float(x) for x in arr[:10]]
            if "cumulative_energy" in data:
                ce = np.asarray(data["cumulative_energy"], dtype=float)
            elif "energy" in data:
                energy = np.asarray(data["energy"], dtype=float)
                ce = np.cumsum(energy) / max(float(np.sum(energy)), 1e-300)
            elif "singular_values" in data:
                sv = np.asarray(data["singular_values"], dtype=float)
                e = sv**2
                ce = np.cumsum(e) / max(float(np.sum(e)), 1e-300)
            else:
                ce = None
            if ce is not None and ce.size:
                for tol in (0.999, 0.9999, 0.99999):
                    idx = int(np.searchsorted(ce, tol, side="left") + 1)
                    info[f"rank_for_energy_{tol}"] = min(idx, int(ce.size))
                if saved_rank is not None:
                    info["saved_rank"] = saved_rank
                    info["retained_energy_at_saved_rank"] = float(ce[min(saved_rank, int(ce.size)) - 1])
        except Exception as exc:
            info["error"] = str(exc)
        decision["fields"][field] = info
    return decision


def infer_roots(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    if args.suite_root:
        suite_root = Path(args.suite_root).expanduser().resolve()
    elif load_data_locations is not None:
        loc = load_data_locations(data_root=args.data_root)
        suite_root = loc.paper2_rom_root / "suites"
    else:
        raise SystemExit("Provide --suite-root, or run inside an installed paramplate repository.")

    if args.pod_dir:
        pod_dir = Path(args.pod_dir).expanduser().resolve()
    elif load_data_locations is not None:
        loc = load_data_locations(data_root=args.data_root)
        pod_dir = loc.paper2_rom_root / "pod_bases" / args.case
    else:
        pod_dir = suite_root.parent / "pod_bases" / args.case

    out_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else suite_root / "reports" / args.case
    return suite_root, pod_dir, out_dir


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build Project-2 ROM suite reporting artifacts.")
    p.add_argument("--case", default=DEFAULT_CASE)
    p.add_argument("--data-root", default=None, help="Optional external paramplate data root.")
    p.add_argument("--suite-root", default=None, help="Override suite output root containing intrusive/<case>/ and nonintrusive/<case>/.")
    p.add_argument("--pod-dir", default=None, help="Override POD artifact directory for the case.")
    p.add_argument("--output-dir", default=None, help="Report output directory. Default: <suite_root>/reports/<case>.")
    p.add_argument("--no-plots", action="store_true", help="Skip PNG/PDF plots.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    suite_root, pod_dir, out_dir = infer_roots(args)
    if not suite_root.exists():
        raise SystemExit(f"Suite root not found: {suite_root}")

    rows = ranked_rows(collect_suite_rows(suite_root, args.case))
    if not rows:
        checked = [str(p) for p in [suite_root / "intrusive" / args.case, suite_root / "nonintrusive" / args.case, suite_root / args.case]]
        raise SystemExit("No suite summary rows found. Checked case locations:\n  " + "\n  ".join(checked))

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, out_dir / "rom_suite_comparison.csv")
    write_markdown(rows, out_dir / "rom_suite_comparison.md", f"Project-2 ROM suite comparison: {args.case}")
    write_latex(rows, out_dir / "rom_suite_comparison_table.tex")
    if not args.no_plots:
        write_plots(rows, out_dir)

    decision = {
        "created_utc": _utc_now(),
        "case": args.case,
        "suite_root": str(suite_root),
        "production_case_dirs": {
            "intrusive": str(suite_root / "intrusive" / args.case),
            "nonintrusive": str(suite_root / "nonintrusive" / args.case),
        },
        "pod_basis_decision": load_pod_basis_decision(pod_dir),
        "ranking": [r.as_dict() for r in rows],
    }
    (out_dir / "rom_suite_report.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")

    print(f"collected rows: {len(rows)}")
    print(f"wrote: {out_dir / 'rom_suite_comparison.csv'}")
    print(f"wrote: {out_dir / 'rom_suite_comparison.md'}")
    print(f"wrote: {out_dir / 'rom_suite_comparison_table.tex'}")
    print(f"wrote: {out_dir / 'rom_suite_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
