#!/usr/bin/env python3
"""
Common thermomechanical snapshot-generation workflow for Project 2.

This module is designed to be placed next to an importable solver file that defines
GeneralMultiphysicsSolver from the current THERMOMECHANICAL_FOM_ROM implementation.
The default expected solver module is THERMOMECHANICAL_FOM_ROM.py.
"""

from __future__ import annotations

import argparse
import csv
import copy
import importlib
import importlib.util
import json
import math
import os
import shutil
import sys
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from time import perf_counter as clock
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import pandas as pd
except Exception:
    pd = None

try:
    from scipy.stats import qmc
except Exception:
    qmc = None

try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
except Exception:
    PCA = None
    StandardScaler = None


# -----------------------------------------------------------------------------
# Configuration containers
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class ParamRange:
    name: str
    lower: float
    upper: float
    unit: str = ""
    scale: str = "linear"       # "linear" or "log"
    role: str = "TP"            # "MP" or "TP"
    description: str = ""

    def validate(self) -> None:
        if not np.isfinite(self.lower) or not np.isfinite(self.upper):
            raise ValueError(f"Non-finite range for {self.name}: {self.lower}, {self.upper}")
        if self.upper <= self.lower:
            raise ValueError(f"Invalid range for {self.name}: upper must exceed lower")
        if self.scale not in ("linear", "log"):
            raise ValueError(f"Unsupported scale '{self.scale}' for parameter {self.name}")
        if self.scale == "log" and self.lower <= 0.0:
            raise ValueError(f"Log-scale parameter {self.name} must have lower > 0")
        if self.role not in ("MP", "TP"):
            raise ValueError(f"role for {self.name} must be MP or TP")


@dataclass
class CaseConfig:
    case_id: int
    case_name: str
    filename: str
    season: str
    mp_tp_label: str
    physical_meaning: str
    recommended_n_snapshots: int
    expected_solver_difficulty: str

    # Mechanical solver uses study_case=1 with mu_mech=[Dx,Dy,Dxy,Ds,ks,f].
    study_case: int = 1
    nominal_mech: Dict[str, float] = field(default_factory=lambda: dict(
        Dx=6.0e3, Dy=6.0e3, Dxy=2.5e3, Ds=1.8e3, ks=5.0e5, f=-3.0e3
    ))
    nominal_thermal: Dict[str, float] = field(default_factory=lambda: dict(
        T_amb=305.15, T_sub=306.15, h_con=10.0, eps_r=0.90, q_s=255.0,
        h_c_cont=150.0, h_c_gap=7.0, eta_c=20.0,
        kx=0.40, ky=0.40, kz=0.40,
        alpha1=1.3e-4, alpha2=1.3e-4, rho=950.0,
    ))
    parameter_ranges: List[ParamRange] = field(default_factory=list)

    bc_type: str = "free_edge"          # supported by solver: free_edge, simply_supported
    load_type: str = "patch"            # supported by solver: uniform, patch, multi_patch, linear
    n_vert: int = 0
    n_horiz: int = 0

    plate_resolution: int = 64
    plate_degree: int = 2
    heat_nx: int = 64
    heat_ny: int = 32
    heat_nz: int = 16
    heat_degree: int = 1
    theta_cg_degree: int = 1
    theta_quadrature: int = 20

    solve_mode: str = "coupled"
    omega: float = 0.7
    tol_w: float = 1.0e-5
    tol_theta: float = 1.0e-5
    max_coupling_iters: int = 25

    def validate(self) -> None:
        if self.season not in ("summer", "winter"):
            raise ValueError("season must be 'summer' or 'winter'")
        if self.solve_mode not in ("coupled", "one_way", "mechanical"):
            raise ValueError("solve_mode must be coupled, one_way, or mechanical")
        for pr in self.parameter_ranges:
            pr.validate()
        for key in ["Dx", "Dy", "Dxy", "Ds", "ks", "f"]:
            if key not in self.nominal_mech:
                raise ValueError(f"nominal_mech missing {key}")
        required_th = [
            "T_amb", "T_sub", "h_con", "eps_r", "q_s", "h_c_cont", "h_c_gap",
            "eta_c", "kx", "ky", "kz", "alpha1", "alpha2", "rho",
        ]
        for key in required_th:
            if key not in self.nominal_thermal:
                raise ValueError(f"nominal_thermal missing {key}")


# -----------------------------------------------------------------------------
# Solver import and command-line setup
# -----------------------------------------------------------------------------

def import_solver_class(solver_module: str = "THERMOMECHANICAL_FOM_ROM"):
    """Import GeneralMultiphysicsSolver from a module name or a .py path."""
    solver_module = os.path.expanduser(str(solver_module))
    if solver_module.endswith(".py") or os.path.sep in solver_module:
        path = Path(solver_module).resolve()
        if not path.exists():
            raise FileNotFoundError(
                f"Solver module path not found: {path}\n"
                "Create THERMOMECHANICAL_FOM_ROM.py from the current solver code, "
                "or pass --solver-module /path/to/your_solver.py."
            )
        spec = importlib.util.spec_from_file_location(path.stem, str(path))
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import solver module from {path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[path.stem] = mod
        spec.loader.exec_module(mod)
    else:
        mod = importlib.import_module(solver_module)
    if not hasattr(mod, "GeneralMultiphysicsSolver"):
        raise AttributeError(f"{solver_module} does not define GeneralMultiphysicsSolver")
    return mod.GeneralMultiphysicsSolver


def build_arg_parser(config: CaseConfig) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=f"Thermomechanical snapshot generator: {config.case_name}")
    p.add_argument("--n-snapshots", type=int, default=config.recommended_n_snapshots)
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--n-vert", type=int, default=config.n_vert)
    p.add_argument("--n-horiz", type=int, default=config.n_horiz)
    p.add_argument("--panel-type", type=str, default="auto", choices=["auto", "monolithic", "contact1", "contact2"], help="Optional shortcut: monolithic=(0,0), contact1=(1,0), contact2=(2,0). If not auto, it overrides --n-vert/--n-horiz.")
    p.add_argument("--bc-type", type=str, default=config.bc_type, choices=["free_edge", "simply_supported"])
    p.add_argument("--load-type", type=str, default=config.load_type, choices=["uniform", "patch", "multi_patch", "linear"])
    p.add_argument("--plate-resolution", type=int, default=config.plate_resolution)
    p.add_argument("--plate-degree", type=int, default=config.plate_degree)
    p.add_argument("--heat-nx", type=int, default=config.heat_nx)
    p.add_argument("--heat-ny", type=int, default=config.heat_ny)
    p.add_argument("--heat-nz", type=int, default=config.heat_nz)
    p.add_argument("--heat-degree", type=int, default=config.heat_degree)
    p.add_argument("--theta-cg-degree", type=int, default=config.theta_cg_degree)
    p.add_argument("--theta-quadrature", type=int, default=config.theta_quadrature)
    p.add_argument("--solve-mode", type=str, default=config.solve_mode, choices=["coupled", "one_way", "mechanical"])
    p.add_argument("--omega", type=float, default=config.omega)
    p.add_argument("--tol-w", type=float, default=config.tol_w)
    p.add_argument("--tol-theta", type=float, default=config.tol_theta)
    p.add_argument("--max-coupling-iters", type=int, default=config.max_coupling_iters)
    p.add_argument("--output-root", type=str, default="THERMO_SNAPSHOTS")
    p.add_argument("--solver-module", type=str, default="THERMOMECHANICAL_FOM_ROM")
    p.add_argument("--sampling", type=str, default="lhs", choices=["lhs", "random"])
    p.add_argument("--resume", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--strict", action="store_true", help="Stop immediately on failed or non-converged sample.")
    p.add_argument("--make-plots", action="store_true")
    p.add_argument("--plots-only", action="store_true", help="Regenerate diagnostics from an existing archive.")
    p.add_argument("--save-field-plots", action="store_true", help="Save representative FEniCS field plots during solving.")
    p.add_argument("--status", action="store_true", help="Print a clean status report for the selected case/panel/BC/load and exit without solving.")
    p.add_argument("--watch-status", type=float, default=0.0, help="With --status, refresh the status report every N seconds; 0 prints once.")
    p.add_argument("--status-tail-lines", type=int, default=20, help="With --status, show this many recent lines from status.log.")
    return p


# -----------------------------------------------------------------------------
# Sampling and parameter decoding
# -----------------------------------------------------------------------------

def _unit_lhs(n: int, d: int, seed: int) -> np.ndarray:
    if qmc is not None:
        return qmc.LatinHypercube(d=d, seed=seed, optimization="random-cd").random(n)
    rng = np.random.default_rng(seed)
    u = np.zeros((n, d), dtype=float)
    for j in range(d):
        pts = (np.arange(n) + rng.random(n)) / n
        rng.shuffle(pts)
        u[:, j] = pts
    return u


def sample_parameters(config: CaseConfig, n: int, seed: int, method: str = "lhs") -> np.ndarray:
    config.validate()
    d = len(config.parameter_ranges)
    if n < 1:
        raise ValueError("n must be >= 1")
    if d == 0:
        return np.zeros((n, 0), dtype=float)
    if method == "lhs":
        u = _unit_lhs(n, d, seed)
    elif method == "random":
        u = np.random.default_rng(seed).random((n, d))
    else:
        raise ValueError(f"Unsupported sampling method: {method}")

    x = np.zeros_like(u, dtype=float)
    for j, pr in enumerate(config.parameter_ranges):
        if pr.scale == "log":
            lo, hi = math.log10(pr.lower), math.log10(pr.upper)
            x[:, j] = 10.0 ** (lo + u[:, j] * (hi - lo))
        else:
            x[:, j] = pr.lower + u[:, j] * (pr.upper - pr.lower)
    return x


def decode_sample(config: CaseConfig, sample: Sequence[float]) -> Tuple[List[float], Dict[str, float], Dict[str, Any]]:
    """Map sampled variables to the actual solver inputs.

    The current solver's mechanical branch is study_case=1 with
    mu_mech=[Dx, Dy, Dxy, Ds, ks, f].  Derived variables D_eff, D_ratio,
    Dxy_ratio, and Ds_factor are decoded here without changing the solver API.
    """
    mech = dict(config.nominal_mech)
    therm = dict(config.nominal_thermal)
    sampled = {pr.name: float(v) for pr, v in zip(config.parameter_ranges, sample)}

    for name, val in sampled.items():
        if name in ("Dx", "Dy", "Dxy", "Ds", "ks", "f"):
            mech[name] = val
        elif name == "rho":
            therm["rho"] = val
        elif name in therm:
            therm[name] = val

    # Derived orthotropic parametrization.  D_ratio means Dx/Dy, and D_eff=sqrt(Dx*Dy).
    if "D_eff" in sampled or "D_ratio" in sampled:
        D_eff_nom = math.sqrt(float(mech["Dx"]) * float(mech["Dy"]))
        ratio_nom = float(mech["Dx"]) / float(mech["Dy"])
        D_eff = float(sampled.get("D_eff", D_eff_nom))
        D_ratio = float(sampled.get("D_ratio", ratio_nom))
        mech["Dx"] = D_eff * math.sqrt(D_ratio)
        mech["Dy"] = D_eff / math.sqrt(D_ratio)

    D_eff_current = math.sqrt(float(mech["Dx"]) * float(mech["Dy"]))
    if "Dxy_ratio" in sampled:
        mech["Dxy"] = float(sampled["Dxy_ratio"]) * D_eff_current
    if "Ds_factor" in sampled:
        mech["Ds"] = float(sampled["Ds_factor"]) * D_eff_current

    # Safety check for the 2x2 bending block.  The chosen ranges should already satisfy this.
    limit = 0.95 * D_eff_current
    stability_adjustment = None
    if abs(float(mech["Dxy"])) >= limit:
        old = float(mech["Dxy"])
        mech["Dxy"] = math.copysign(limit, old)
        stability_adjustment = {"Dxy_clipped_from": old, "Dxy_clipped_to": mech["Dxy"]}

    mu_mech = [float(mech[k]) for k in ["Dx", "Dy", "Dxy", "Ds", "ks", "f"]]
    meta = {
        "sampled_parameters": sampled,
        "decoded_mechanical": {k: float(mech[k]) for k in ["Dx", "Dy", "Dxy", "Ds", "ks", "f"]},
        "decoded_thermal": {k: float(therm[k]) for k in therm},
        "stability_adjustment": stability_adjustment,
    }
    return mu_mech, therm, meta


# -----------------------------------------------------------------------------
# Filesystem helpers and JSON/CSV utilities
# -----------------------------------------------------------------------------

def json_safe(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(data), indent=2, sort_keys=True))


def _safe_name_token(value: Any) -> str:
    """Return a filesystem-safe uppercase token."""
    txt = str(value).strip().upper()
    out = []
    for ch in txt:
        if ch.isalnum():
            out.append(ch)
        elif ch in ("_", "-"):
            out.append("_")
        else:
            out.append("_")
    token = "".join(out)
    while "__" in token:
        token = token.replace("__", "_")
    return token.strip("_") or "UNNAMED"


def panel_label(n_vert: int, n_horiz: int) -> str:
    """Panel label used in folder/archive names."""
    nv, nh = int(n_vert), int(n_horiz)
    if nv == 0 and nh == 0:
        return "MONOLITHIC_NV0_NH0"
    if nv == 1 and nh == 0:
        return "CONTACT_NV1_NH0"
    if nv == 2 and nh == 0:
        return "CONTACT_NV2_NH0"
    return f"CUSTOM_NV{nv}_NH{nh}"


def apply_panel_type_shortcut(args) -> None:
    """Apply --panel-type shortcut without removing the original --n-vert/--n-horiz controls."""
    pt = str(getattr(args, "panel_type", "auto")).lower()
    if pt == "monolithic":
        args.n_vert, args.n_horiz = 0, 0
    elif pt == "contact1":
        args.n_vert, args.n_horiz = 1, 0
    elif pt == "contact2":
        args.n_vert, args.n_horiz = 2, 0


def build_full_case_run_name(config: CaseConfig) -> str:
    """Folder/archive stem: case type + panel type + BC + load."""
    base = str(config.case_name).split("__PANEL_")[0]
    return "__".join([
        _safe_name_token(base),
        "PANEL_" + panel_label(config.n_vert, config.n_horiz),
        "BC_" + _safe_name_token(config.bc_type),
        "LOAD_" + _safe_name_token(config.load_type),
    ])


def final_snapshot_archive_name(config: CaseConfig) -> str:
    """Primary final archive filename with case/panel/BC/load encoded."""
    return "SNAPSHOTS__" + build_full_case_run_name(config) + ".npz"


def find_snapshot_archive(out_dir: Path, config: Optional[CaseConfig] = None) -> Path:
    """Find the primary named archive, with old snapshots.npz fallback.

    If config is provided and the named archive does not exist yet, return the
    planned named archive path unless an older compatibility snapshots.npz is the
    only existing archive.
    """
    if config is not None:
        named = out_dir / final_snapshot_archive_name(config)
        compat = out_dir / "snapshots.npz"
        if named.exists() or not compat.exists():
            return named
    named_candidates = sorted(out_dir.glob("SNAPSHOTS__*.npz"))
    if named_candidates:
        return named_candidates[0]
    return out_dir / "snapshots.npz"


def case_output_dir(output_root: str, config: CaseConfig) -> Path:
    return Path(output_root).expanduser().resolve() / build_full_case_run_name(config)


def prepare_output_dir(out_dir: Path, overwrite: bool, resume: bool, plots_only: bool) -> None:
    if plots_only:
        if not out_dir.exists():
            raise FileNotFoundError(f"Cannot use --plots-only; output directory does not exist: {out_dir}")
        return
    existing_archives = sorted(out_dir.glob("SNAPSHOTS__*.npz")) + ([out_dir / "snapshots.npz"] if (out_dir / "snapshots.npz").exists() else [])
    if existing_archives and not overwrite and not resume:
        final_archive = existing_archives[0]
        raise FileExistsError(
            f"Archive already exists: {final_archive}\n"
            "Use --resume to continue from tmp files or --overwrite to delete and restart."
        )
    if overwrite and out_dir.exists() and not resume:
        shutil.rmtree(out_dir)
    (out_dir / "tmp").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)


def save_parameter_table(out_dir: Path, config: CaseConfig, X: np.ndarray) -> None:
    names = [p.name for p in config.parameter_ranges]
    np.save(out_dir / "parameter_matrix.npy", X)
    with open(out_dir / "parameter_matrix.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(names)
        for row in X:
            w.writerow([f"{float(v):.16e}" for v in row])


def sample_tmp_path(out_dir: Path, i: int) -> Path:
    return out_dir / "tmp" / f"sample_{i:06d}.npz"


def failure_tmp_path(out_dir: Path, i: int) -> Path:
    return out_dir / "tmp" / f"sample_{i:06d}_FAILED.json"


def _format_duration(seconds: Optional[float]) -> str:
    """Return a compact human-readable duration."""
    if seconds is None:
        return "unknown"
    try:
        seconds = float(seconds)
    except Exception:
        return "unknown"
    if not np.isfinite(seconds):
        return "unknown"
    seconds = max(0.0, seconds)
    if seconds < 60.0:
        return f"{seconds:.1f}s"
    minutes = seconds / 60.0
    if minutes < 60.0:
        return f"{minutes:.1f}min"
    hours = minutes / 60.0
    return f"{hours:.2f}h"


def _sample_index_from_path(path: Path) -> int:
    try:
        return int(path.name.split("_")[1].split(".")[0])
    except Exception:
        return -1


def _safe_load_sample_metadata(path: Path) -> Tuple[Optional[Dict[str, Any]], Optional[np.ndarray]]:
    """Read metadata safely, including while another process may be writing files."""
    try:
        data = np.load(path, allow_pickle=True)
        try:
            meta = json.loads(str(data["metadata_json"].item()))
        except Exception:
            meta = {"sample_index": _sample_index_from_path(path)}
        try:
            sample = np.asarray(data["sample"], dtype=float)
        except Exception:
            sample = None
        return meta, sample
    except Exception:
        return None, None


def collect_case_status(out_dir: Path, config: CaseConfig, n_expected: int) -> Dict[str, Any]:
    """Collect a clean progress summary from tmp sample files and failure logs."""
    tmp_dir = out_dir / "tmp"
    sample_files = sorted(tmp_dir.glob("sample_*.npz")) if tmp_dir.exists() else []
    failed_files = sorted(tmp_dir.glob("sample_*_FAILED.json")) if tmp_dir.exists() else []

    completed = []
    for fp in sample_files:
        meta, sample = _safe_load_sample_metadata(fp)
        if meta is None:
            continue
        idx = int(meta.get("sample_index", _sample_index_from_path(fp)))
        completed.append({
            "index": idx,
            "file": fp.name,
            "sample": None if sample is None else sample.tolist(),
            "converged": meta.get("converged", None),
            "stopped_by_max_iters": meta.get("stopped_by_max_iters", None),
            "coupled_iterations": meta.get("coupled_iterations", None),
            "err_w": meta.get("err_w", None),
            "err_theta": meta.get("err_theta", None),
            "wall_time_sec": meta.get("wall_time_sec", None),
            "termination_reason": meta.get("termination_reason", None),
        })

    completed = sorted(completed, key=lambda r: int(r.get("index", -1)))
    completed_count = len(completed)
    failed_count = len(failed_files)
    converged_count = sum(1 for r in completed if bool(r.get("converged")) and not bool(r.get("stopped_by_max_iters")))
    nonconverged_count = completed_count - converged_count

    times = []
    for r in completed:
        try:
            t = float(r.get("wall_time_sec"))
            if np.isfinite(t):
                times.append(t)
        except Exception:
            pass
    avg_time = float(np.mean(times)) if times else None
    total_time = float(np.sum(times)) if times else None
    remaining = max(0, int(n_expected) - completed_count - failed_count)
    eta = None if avg_time is None else avg_time * remaining

    final_archive = find_snapshot_archive(out_dir, config)
    return {
        "case_name": config.case_name,
        "out_dir": str(out_dir),
        "target_snapshots": int(n_expected),
        "completed_count": int(completed_count),
        "failed_count": int(failed_count),
        "converged_count": int(converged_count),
        "nonconverged_count": int(nonconverged_count),
        "remaining_not_saved": int(remaining),
        "progress_fraction": float(completed_count / max(int(n_expected), 1)),
        "avg_sample_time_sec": avg_time,
        "total_completed_wall_time_sec": total_time,
        "eta_remaining_sec": eta,
        "final_archive_exists": bool(final_archive.exists()),
        "final_archive": str(final_archive),
        "summary_csv_exists": bool((out_dir / "summary.csv").exists()),
        "failed_csv_exists": bool((out_dir / "failed_samples.csv").exists()),
        "latest_completed": completed[-1] if completed else None,
        "completed_samples": completed,
        "failed_files": [fp.name for fp in failed_files],
        "status_files": {
            "latest_text": str(out_dir / "latest_status.txt"),
            "latest_json": str(out_dir / "_STATUS.json"),
            "log": str(out_dir / "status.log"),
        },
    }


def format_case_status(status: Dict[str, Any], tail_lines: Optional[List[str]] = None) -> str:
    n = int(status["target_snapshots"])
    done = int(status["completed_count"])
    failed = int(status["failed_count"])
    remaining = int(status["remaining_not_saved"])
    pct = 100.0 * float(status["progress_fraction"])
    latest = status.get("latest_completed")

    lines = []
    lines.append("=" * 96)
    lines.append(f"SNAPSHOT STATUS  : {status['case_name']}")
    lines.append(f"OUTPUT           : {status['out_dir']}")
    lines.append(f"PRIMARY ARCHIVE  : {status.get('final_archive', 'unknown')}")
    lines.append(f"PROGRESS         : {done}/{n} completed ({pct:.1f}%) | failed={failed} | remaining_not_saved={remaining}")
    lines.append(f"CONVERGENCE      : converged={status['converged_count']} | nonconverged_saved={status['nonconverged_count']}")
    lines.append(f"TIME             : avg/sample={_format_duration(status.get('avg_sample_time_sec'))} | completed_wall={_format_duration(status.get('total_completed_wall_time_sec'))} | ETA≈{_format_duration(status.get('eta_remaining_sec'))}")
    lines.append(f"FINAL ARCHIVE    : {'YES' if status.get('final_archive_exists') else 'NO'}")
    if latest:
        sample = latest.get("sample") or []
        sample_txt = np.array2string(np.asarray(sample, dtype=float), precision=5) if len(sample) else "[]"
        err_w = latest.get("err_w", np.nan)
        err_theta = latest.get("err_theta", np.nan)
        try:
            err_w_txt = f"{float(err_w):.3e}"
        except Exception:
            err_w_txt = "nan"
        try:
            err_theta_txt = f"{float(err_theta):.3e}"
        except Exception:
            err_theta_txt = "nan"
        lines.append("-" * 96)
        lines.append(
            f"LATEST COMPLETED : sample {int(latest.get('index', -1)) + 1:04d}/{n:04d} "
            f"values={sample_txt} converged={latest.get('converged')} "
            f"iters={latest.get('coupled_iterations')} "
            f"err_w={err_w_txt} err_theta={err_theta_txt} "
            f"time={_format_duration(latest.get('wall_time_sec'))}"
        )
    else:
        lines.append("LATEST COMPLETED : none yet")
    if done < n:
        lines.append(f"NEXT / RUNNING   : sample {done + 1:04d}/{n:04d} is next or currently running")
    else:
        lines.append("NEXT / RUNNING   : all target samples have temporary files")
    if tail_lines:
        lines.append("-" * 96)
        lines.append("RECENT status.log:")
        lines.extend(line.rstrip("\n") for line in tail_lines)
    lines.append("=" * 96)
    return "\n".join(lines)


def print_case_status(out_dir: Path, config: CaseConfig, n_expected: int, tail_lines: int = 0) -> Dict[str, Any]:
    status = collect_case_status(out_dir, config, n_expected)
    tail = None
    log_path = out_dir / "status.log"
    if tail_lines and log_path.exists():
        try:
            tail = log_path.read_text().splitlines()[-int(tail_lines):]
        except Exception:
            tail = None
    print(format_case_status(status, tail_lines=tail), flush=True)
    return status


def write_status_files(
    out_dir: Path,
    config: CaseConfig,
    n_expected: int,
    event: str,
    current_index: Optional[int] = None,
    current_sample: Optional[Sequence[float]] = None,
    note: Optional[str] = None,
) -> Dict[str, Any]:
    """Write _STATUS.json, latest_status.txt, and append status.log."""
    out_dir.mkdir(parents=True, exist_ok=True)
    status = collect_case_status(out_dir, config, n_expected)
    status["event"] = str(event)
    status["event_utc"] = datetime.utcnow().isoformat() + "Z"
    if current_index is not None:
        status["current_index"] = int(current_index)
    if current_sample is not None:
        status["current_sample"] = np.asarray(current_sample, dtype=float).tolist()
    if note is not None:
        status["note"] = str(note)

    write_json(out_dir / "_STATUS.json", status)
    (out_dir / "latest_status.txt").write_text(format_case_status(status) + "\n")

    sample_txt = ""
    if current_sample is not None:
        sample_txt = " sample=" + np.array2string(np.asarray(current_sample, dtype=float), precision=5)
    progress = f"completed={status['completed_count']}/{status['target_snapshots']} failed={status['failed_count']} remaining={status['remaining_not_saved']}"
    eta = f"eta≈{_format_duration(status.get('eta_remaining_sec'))}"
    line = f"[{status['event_utc']}] {event} {progress} {eta}"
    if current_index is not None:
        line += f" current={int(current_index) + 1}/{status['target_snapshots']}"
    line += sample_txt
    if note:
        line += f" note={note}"
    with open(out_dir / "status.log", "a", buffering=1) as f:
        f.write(line + "\n")
    return status


def print_compact_progress(out_dir: Path, config: CaseConfig, n_expected: int) -> None:
    st = collect_case_status(out_dir, config, n_expected)
    print(
        f"    progress: completed={st['completed_count']}/{st['target_snapshots']} "
        f"failed={st['failed_count']} remaining={st['remaining_not_saved']} "
        f"avg={_format_duration(st.get('avg_sample_time_sec'))} "
        f"ETA≈{_format_duration(st.get('eta_remaining_sec'))}",
        flush=True,
    )


# -----------------------------------------------------------------------------
# Solver setup and sample solve wrappers
# -----------------------------------------------------------------------------

def configure_solver(SolverClass, config: CaseConfig, args, out_dir: Path):
    solver = SolverClass(study_case=int(config.study_case))
    solver.bc_type = args.bc_type
    solver.load_type = args.load_type
    solver.size = int(args.plate_resolution)
    solver.degree = int(args.plate_degree)
    solver.define_domain(int(args.n_vert), int(args.n_horiz), plot_subdomains=False)
    solver.output_dir = str(out_dir)
    return solver


def _vec_stats(vec: Optional[np.ndarray]) -> Dict[str, Optional[float]]:
    if vec is None or len(vec) == 0:
        return {"min": None, "max": None, "mean": None, "rms": None, "l2": None}
    v = np.asarray(vec, dtype=float).ravel()
    return {
        "min": float(np.min(v)),
        "max": float(np.max(v)),
        "mean": float(np.mean(v)),
        "rms": float(np.sqrt(np.mean(v * v))),
        "l2": float(np.linalg.norm(v)),
    }


def _field_vec(field) -> Optional[np.ndarray]:
    if field is None:
        return None
    try:
        return field.vector().get_local().copy()
    except Exception:
        return None


def _fenics_dim(field) -> int:
    if field is None:
        return 0
    try:
        return int(field.function_space().dim())
    except Exception:
        return 0


def save_field_plot(field, fig_path: Path, title: str) -> None:
    if field is None:
        return
    try:
        from dolfin import plot as fenics_plot
        fig = plt.figure(figsize=(6.4, 4.8))
        p = fenics_plot(field)
        plt.colorbar(p)
        plt.title(title)
        plt.xlabel("x")
        plt.ylabel("y")
        fig.tight_layout()
        fig.savefig(fig_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
    except Exception:
        plt.close("all")


def solve_one_sample(solver, config: CaseConfig, sample: Sequence[float], args, i: int, out_dir: Path) -> Dict[str, Any]:
    mu_mech, mu_th, meta = decode_sample(config, sample)
    solver.set_rom_thermal_parameters(**mu_th)

    solve_mode = str(args.solve_mode)
    t0 = clock()
    heat = None
    theta_field = None
    w_field = None
    out = {}

    if solve_mode == "coupled":
        out = solver.solve_rom_sample(
            mu_mech,
            thermal_on=True,
            coupled_on=True,
            heat_nx=int(args.heat_nx),
            heat_ny=int(args.heat_ny),
            heat_nz=int(args.heat_nz),
            heat_degree=int(args.heat_degree),
            Nz_quad_T1=int(args.theta_quadrature),
            T1_cg_degree=int(args.theta_cg_degree),
            coupling_omega=float(args.omega),
            coupling_tol_w=float(args.tol_w),
            coupling_tol_T1=float(args.tol_theta),
            coupling_max_iters=int(args.max_coupling_iters),
            coupling_verbose=False,
            return_mode="global",
            return_coupled_dict=True,
        )
        heat = out.get("heat", None)
        theta_field = out.get("T1", None)
        w_field = out.get("w", None)
        converged = bool(out.get("converged", False))
        stopped_by_max_iters = bool(out.get("stopped_by_max_iters", False))
        iters = int(out.get("iters", -1))
        err_w = float(out.get("err_w", np.nan))
        err_theta = float(out.get("err_T1", np.nan))
        termination_reason = str(out.get("termination_reason", "unknown"))
    elif solve_mode == "one_way":
        heat = solver.solve_rom_sample(
            mu_mech,
            thermal_on=True,
            coupled_on=False,
            heat_nx=int(args.heat_nx),
            heat_ny=int(args.heat_ny),
            heat_nz=int(args.heat_nz),
            heat_degree=int(args.heat_degree),
            Nz_quad_T1=int(args.theta_quadrature),
            T1_cg_degree=int(args.theta_cg_degree),
            return_mode="global",
        )
        w_field = getattr(solver, "w_contact_global", None)
        theta_field = getattr(solver, "T1_from_heat", None)
        converged = True
        stopped_by_max_iters = False
        iters = 1
        err_w = np.nan
        err_theta = np.nan
        termination_reason = "one_way_completed"
    elif solve_mode == "mechanical":
        solver.update_thermal_parameters(enable=False, T1_expr=None)
        w_field = solver.offline_solve_kirchhoff_problem(mu_mech, return_mode="global", thermal_on=False)
        theta_field = None
        heat = None
        converged = True
        stopped_by_max_iters = False
        iters = 0
        err_w = np.nan
        err_theta = np.nan
        termination_reason = "mechanical_completed"
    else:
        raise ValueError(f"Unsupported solve_mode: {solve_mode}")

    dt = clock() - t0
    w_vec = _field_vec(w_field)
    theta_vec = _field_vec(theta_field)

    heat_dofs = 0
    T_ref = np.nan
    if heat is not None:
        try:
            heat_dofs = int(heat["Vt"].dim())
        except Exception:
            heat_dofs = int(out.get("heat_dofs", 0)) if isinstance(out, dict) else 0
        try:
            T_ref = float(heat.get("T_ref", np.nan))
        except Exception:
            T_ref = np.nan

    w_dofs = _fenics_dim(w_field)
    theta_dofs = _fenics_dim(theta_field)
    total_dofs = int(w_dofs + heat_dofs + theta_dofs)

    sample_meta = {
        **meta,
        "sample_index": int(i),
        "case_name": config.case_name,
        "season": config.season,
        "mp_tp_label": config.mp_tp_label,
        "solve_mode": solve_mode,
        "bc_type": args.bc_type,
        "load_type": args.load_type,
        "n_vert": int(args.n_vert),
        "n_horiz": int(args.n_horiz),
        "plate_resolution": int(args.plate_resolution),
        "plate_degree": int(args.plate_degree),
        "heat_nx": int(args.heat_nx),
        "heat_ny": int(args.heat_ny),
        "heat_nz": int(args.heat_nz),
        "heat_degree": int(args.heat_degree),
        "theta_cg_degree": int(args.theta_cg_degree),
        "theta_quadrature": int(args.theta_quadrature),
        "omega": float(args.omega),
        "tol_w": float(args.tol_w),
        "tol_theta": float(args.tol_theta),
        "max_coupling_iters": int(args.max_coupling_iters),
        "wall_time_sec": float(dt),
        "converged": bool(converged),
        "stopped_by_max_iters": bool(stopped_by_max_iters),
        "coupled_iterations": int(iters),
        "err_w": err_w,
        "err_theta": err_theta,
        "termination_reason": termination_reason,
        "w_dofs": int(w_dofs),
        "heat_dofs": int(heat_dofs),
        "theta_dofs": int(theta_dofs),
        "total_coupled_dofs": int(total_dofs),
        "T_ref": T_ref,
        "w_stats": _vec_stats(w_vec),
        "theta_stats": _vec_stats(theta_vec),
    }

    tmp = sample_tmp_path(out_dir, i)
    np.savez_compressed(
        tmp,
        w=np.asarray(w_vec if w_vec is not None else [], dtype=float),
        theta=np.asarray(theta_vec if theta_vec is not None else [], dtype=float),
        sample=np.asarray(sample, dtype=float),
        mu_mech=np.asarray(mu_mech, dtype=float),
        mu_th=np.asarray([mu_th[k] for k in sorted(mu_th.keys())], dtype=float),
        mu_th_names=np.asarray(sorted(mu_th.keys()), dtype=str),
        convergence=np.asarray([iters, float(converged), float(stopped_by_max_iters), err_w, err_theta], dtype=float),
        dofs=np.asarray([w_dofs, heat_dofs, theta_dofs, total_dofs], dtype=float),
        heat_discretization=np.asarray([args.heat_nx, args.heat_ny, args.heat_nz, args.heat_degree, args.theta_quadrature, args.theta_cg_degree], dtype=float),
        metadata_json=np.asarray(json.dumps(json_safe(sample_meta))),
    )

    if args.save_field_plots and args.make_plots:
        if i in (0, max(0, int(args.n_snapshots) // 2), int(args.n_snapshots) - 1):
            fig_dir = out_dir / "figures"
            save_field_plot(w_field, fig_dir / f"field_w_sample_{i:06d}.png", f"w, sample {i}")
            save_field_plot(theta_field, fig_dir / f"field_theta_sample_{i:06d}.png", f"theta/T1, sample {i}")

    return sample_meta


# -----------------------------------------------------------------------------
# Consolidation and diagnostics
# -----------------------------------------------------------------------------

def load_completed_samples(out_dir: Path, n_expected: int) -> List[Tuple[int, Dict[str, Any], Dict[str, np.ndarray]]]:
    rows = []
    for i in range(n_expected):
        p = sample_tmp_path(out_dir, i)
        if not p.exists():
            continue
        data = np.load(p, allow_pickle=True)
        try:
            meta = json.loads(str(data["metadata_json"].item()))
        except Exception:
            meta = {"sample_index": i}
        rows.append((i, meta, {k: data[k] for k in data.files}))
    return rows


def consolidate(out_dir: Path, config: CaseConfig, X: np.ndarray) -> Path:
    rows = load_completed_samples(out_dir, len(X))
    if not rows:
        raise RuntimeError("No completed samples found; cannot consolidate.")

    w_list, theta_list, samples, mech, conv, dofs, heatdisc, meta_json = [], [], [], [], [], [], [], []
    thermal_values, thermal_names = [], None
    for i, meta, data in rows:
        w = np.asarray(data["w"], dtype=float)
        th = np.asarray(data["theta"], dtype=float)
        if w.size == 0:
            continue
        w_list.append(w)
        theta_list.append(th)
        samples.append(np.asarray(data["sample"], dtype=float))
        mech.append(np.asarray(data["mu_mech"], dtype=float))
        thermal_values.append(np.asarray(data["mu_th"], dtype=float))
        thermal_names = np.asarray(data["mu_th_names"], dtype=str)
        conv.append(np.asarray(data["convergence"], dtype=float))
        dofs.append(np.asarray(data["dofs"], dtype=float))
        heatdisc.append(np.asarray(data["heat_discretization"], dtype=float))
        meta_json.append(json.dumps(json_safe(meta)))

    theta_array = np.vstack(theta_list) if theta_list and all(t.size == theta_list[0].size for t in theta_list) else np.asarray(theta_list, dtype=object)
    archive = out_dir / final_snapshot_archive_name(config)
    np.savez_compressed(
        archive,
        snapshots=np.vstack(w_list),
        theta_snapshots=theta_array,
        parameters=np.vstack(samples),
        parameter_names=np.asarray([p.name for p in config.parameter_ranges], dtype=str),
        parameter_units=np.asarray([p.unit for p in config.parameter_ranges], dtype=str),
        parameter_roles=np.asarray([p.role for p in config.parameter_ranges], dtype=str),
        mech_parameters=np.vstack(mech),
        mech_parameter_names=np.asarray(["Dx", "Dy", "Dxy", "Ds", "ks", "f"], dtype=str),
        thermal_parameters=np.vstack(thermal_values),
        thermal_parameter_names=np.asarray(thermal_names, dtype=str),
        convergence=np.vstack(conv),
        convergence_names=np.asarray(["iters", "converged", "stopped_by_max_iters", "err_w", "err_theta"], dtype=str),
        snapshot_dofs=np.vstack(dofs),
        snapshot_dof_names=np.asarray(["w_dofs", "heat_dofs", "theta_dofs", "total_coupled_dofs"], dtype=str),
        heat_discretizations=np.vstack(heatdisc),
        heat_discretization_names=np.asarray(["heat_nx", "heat_ny", "heat_nz", "heat_degree", "theta_quadrature", "theta_cg_degree"], dtype=str),
        metadata_json=np.asarray(meta_json, dtype=str),
        case_config_json=np.asarray(json.dumps(json_safe(asdict(config))), dtype=str),
    )

    # Preserve the old Paper-1-style filename for compatibility, but make the
    # named archive the primary returned file.  A hardlink avoids duplicate disk
    # storage on normal filesystems; copy is the safe fallback.
    compat_archive = out_dir / "snapshots.npz"
    if compat_archive.resolve() != archive.resolve():
        try:
            if compat_archive.exists() or compat_archive.is_symlink():
                compat_archive.unlink()
            os.link(archive, compat_archive)
        except Exception:
            shutil.copy2(archive, compat_archive)

    write_summary_csv(out_dir, rows)
    return archive


def write_summary_csv(out_dir: Path, rows: List[Tuple[int, Dict[str, Any], Dict[str, np.ndarray]]]) -> None:
    summary_path = out_dir / "summary.csv"
    cols = [
        "sample_index", "converged", "stopped_by_max_iters", "coupled_iterations",
        "err_w", "err_theta", "termination_reason", "wall_time_sec",
        "w_min", "w_max", "w_rms", "theta_min", "theta_max", "theta_rms",
        "w_dofs", "heat_dofs", "theta_dofs", "total_coupled_dofs",
    ]
    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for i, meta, _data in rows:
            ws = meta.get("w_stats", {}) or {}
            ts = meta.get("theta_stats", {}) or {}
            writer.writerow({
                "sample_index": i,
                "converged": meta.get("converged", None),
                "stopped_by_max_iters": meta.get("stopped_by_max_iters", None),
                "coupled_iterations": meta.get("coupled_iterations", None),
                "err_w": meta.get("err_w", None),
                "err_theta": meta.get("err_theta", None),
                "termination_reason": meta.get("termination_reason", None),
                "wall_time_sec": meta.get("wall_time_sec", None),
                "w_min": ws.get("min", None),
                "w_max": ws.get("max", None),
                "w_rms": ws.get("rms", None),
                "theta_min": ts.get("min", None),
                "theta_max": ts.get("max", None),
                "theta_rms": ts.get("rms", None),
                "w_dofs": meta.get("w_dofs", None),
                "heat_dofs": meta.get("heat_dofs", None),
                "theta_dofs": meta.get("theta_dofs", None),
                "total_coupled_dofs": meta.get("total_coupled_dofs", None),
            })


def write_failed_csv(out_dir: Path) -> None:
    failed = sorted((out_dir / "tmp").glob("sample_*_FAILED.json"))
    path = out_dir / "failed_samples.csv"
    with open(path, "w", newline="") as f:
        cols = ["sample_index", "error", "traceback_file"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for fp in failed:
            item = json.loads(fp.read_text())
            w.writerow({
                "sample_index": item.get("sample_index"),
                "error": item.get("error"),
                "traceback_file": str(fp.name),
            })


def plot_parameter_diagnostics(out_dir: Path, config: CaseConfig, X: np.ndarray) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True, parents=True)
    names = [p.name for p in config.parameter_ranges]
    if X.size == 0 or X.shape[1] == 0:
        return

    # Histograms.
    ncols = min(3, X.shape[1])
    nrows = int(math.ceil(X.shape[1] / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.2 * nrows), squeeze=False)
    for j, name in enumerate(names):
        ax = axes[j // ncols][j % ncols]
        ax.hist(X[:, j], bins=min(20, max(5, X.shape[0] // 3)))
        ax.set_title(name)
        ax.grid(True, linestyle=":", alpha=0.4)
    for j in range(X.shape[1], nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.tight_layout()
    fig.savefig(fig_dir / "parameter_histograms.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # PCA/coverage plot.
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    if X.shape[1] == 1:
        jitter = np.random.default_rng(123).normal(0.0, 0.02, X.shape[0])
        sc = ax.scatter(X[:, 0], jitter, c=X[:, 0], s=28, edgecolors="black", linewidths=0.3)
        ax.set_xlabel(names[0])
        ax.set_ylabel("jitter")
        fig.colorbar(sc, ax=ax, label=names[0])
    else:
        if PCA is not None and StandardScaler is not None:
            Z = StandardScaler().fit_transform(X)
            pca = PCA(n_components=2).fit(Z)
            Y = pca.transform(Z)
            c = np.linalg.norm(Y - Y.mean(axis=0), axis=1)
            sc = ax.scatter(Y[:, 0], Y[:, 1], c=c, s=28, edgecolors="black", linewidths=0.3)
            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
            fig.colorbar(sc, ax=ax, label="distance from center")
        else:
            sc = ax.scatter(X[:, 0], X[:, 1], c=np.arange(X.shape[0]), s=28, edgecolors="black", linewidths=0.3)
            ax.set_xlabel(names[0])
            ax.set_ylabel(names[1])
            fig.colorbar(sc, ax=ax, label="sample index")
    ax.set_title("Parameter-space coverage")
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(fig_dir / "parameter_space_pca.png", dpi=260, bbox_inches="tight")
    plt.close(fig)

    # Pair plot for first up to five parameters.
    m = min(5, X.shape[1])
    if m >= 2:
        fig, axes = plt.subplots(m, m, figsize=(2.4 * m, 2.4 * m), squeeze=False)
        for r in range(m):
            for c in range(m):
                ax = axes[r][c]
                if r == c:
                    ax.hist(X[:, r], bins=min(16, max(5, X.shape[0] // 4)))
                else:
                    ax.scatter(X[:, c], X[:, r], s=10)
                if r == m - 1:
                    ax.set_xlabel(names[c], fontsize=8)
                if c == 0:
                    ax.set_ylabel(names[r], fontsize=8)
                ax.tick_params(labelsize=7)
                ax.grid(True, linestyle=":", alpha=0.25)
        fig.tight_layout()
        fig.savefig(fig_dir / "parameter_pairplot_first5.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def plot_snapshot_diagnostics(out_dir: Path, archive: Path) -> None:
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True, parents=True)
    data = np.load(archive, allow_pickle=True)
    S = np.asarray(data["snapshots"], dtype=float)
    conv = np.asarray(data["convergence"], dtype=float)
    theta = np.asarray(data["theta_snapshots"], dtype=object)

    # Norm diagnostics.
    norms = np.linalg.norm(S, axis=1)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(np.arange(len(norms)), norms, marker="o", linewidth=1.2)
    ax.set_xlabel("sample index")
    ax.set_ylabel("||w||_2")
    ax.set_title("Displacement snapshot vector norms")
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(fig_dir / "snapshot_w_norms.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Singular values and cumulative energy.
    if S.shape[0] >= 2:
        Sc = S - S.mean(axis=0, keepdims=True)
        _, svals, _ = np.linalg.svd(Sc, full_matrices=False)
        energy = np.cumsum(svals**2) / max(np.sum(svals**2), 1e-300)
        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        ax.semilogy(np.arange(1, len(svals) + 1), svals / max(svals[0], 1e-300), marker="o")
        ax.set_xlabel("mode index")
        ax.set_ylabel("normalized singular value")
        ax.set_title("Snapshot singular-value decay")
        ax.grid(True, which="both", linestyle=":", alpha=0.4)
        fig.tight_layout()
        fig.savefig(fig_dir / "singular_value_decay.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.2, 4.4))
        ax.plot(np.arange(1, len(energy) + 1), energy, marker="o")
        ax.set_ylim(0.0, 1.005)
        ax.set_xlabel("mode index")
        ax.set_ylabel("cumulative energy")
        ax.set_title("Snapshot cumulative POD/PCA energy")
        ax.grid(True, linestyle=":", alpha=0.4)
        fig.tight_layout()
        fig.savefig(fig_dir / "snapshot_energy.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    # Coupled iteration histogram and convergence flags.
    if conv.size:
        iters = conv[:, 0]
        flags = conv[:, 1] > 0.5
        fig, ax = plt.subplots(figsize=(6.6, 4.2))
        ax.hist(iters[np.isfinite(iters)], bins=range(int(np.nanmin(iters)), int(np.nanmax(iters)) + 2))
        ax.set_xlabel("coupled iterations")
        ax.set_ylabel("count")
        ax.set_title("Coupled-iteration count histogram")
        ax.grid(True, linestyle=":", alpha=0.4)
        fig.tight_layout()
        fig.savefig(fig_dir / "coupled_iteration_histogram.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5.4, 4.0))
        ax.bar(["converged", "not converged"], [int(flags.sum()), int((~flags).sum())])
        ax.set_ylabel("count")
        ax.set_title("Convergence status")
        ax.grid(True, axis="y", linestyle=":", alpha=0.4)
        fig.tight_layout()
        fig.savefig(fig_dir / "convergence_status.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    # Coefficient-vector representative plots, not a geometric field plot.
    sel = sorted(set([0, len(S)//2, len(S)-1]))
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for i in sel:
        if 0 <= i < len(S):
            ax.plot(S[i], linewidth=0.8, label=f"sample {i}")
    ax.set_xlabel("DoF index")
    ax.set_ylabel("w coefficient")
    ax.set_title("Representative displacement coefficient vectors")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(fig_dir / "representative_w_vectors.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def run_diagnostics(out_dir: Path, config: CaseConfig, X: np.ndarray) -> None:
    archive = find_snapshot_archive(out_dir, config)
    if X is None or X.size == 0:
        param_file = out_dir / "parameter_matrix.npy"
        if param_file.exists():
            X = np.load(param_file)
    if X is not None and X.size:
        plot_parameter_diagnostics(out_dir, config, X)
    if archive.exists():
        plot_snapshot_diagnostics(out_dir, archive)


# -----------------------------------------------------------------------------
# Main run function called by case scripts
# -----------------------------------------------------------------------------

def run_case_from_cli(config: CaseConfig) -> None:
    parser = build_arg_parser(config)
    args = parser.parse_args()
    apply_panel_type_shortcut(args)
    config.validate()

    # CLI overrides applied to config only for metadata/reporting.
    run_config = copy.deepcopy(config)
    run_config.bc_type = args.bc_type
    run_config.load_type = args.load_type
    run_config.n_vert = int(args.n_vert)
    run_config.n_horiz = int(args.n_horiz)
    run_config.plate_resolution = int(args.plate_resolution)
    run_config.plate_degree = int(args.plate_degree)
    run_config.heat_nx = int(args.heat_nx)
    run_config.heat_ny = int(args.heat_ny)
    run_config.heat_nz = int(args.heat_nz)
    run_config.heat_degree = int(args.heat_degree)
    run_config.theta_cg_degree = int(args.theta_cg_degree)
    run_config.theta_quadrature = int(args.theta_quadrature)
    run_config.solve_mode = str(args.solve_mode)
    run_config.omega = float(args.omega)
    run_config.tol_w = float(args.tol_w)
    run_config.tol_theta = float(args.tol_theta)
    run_config.max_coupling_iters = int(args.max_coupling_iters)

    base_case_name = str(run_config.case_name).split("__PANEL_")[0]
    run_config.case_name = build_full_case_run_name(run_config)

    out_dir = case_output_dir(args.output_root, run_config)

    if args.status:
        while True:
            print_case_status(out_dir, run_config, int(args.n_snapshots), tail_lines=int(args.status_tail_lines))
            if float(args.watch_status) <= 0.0:
                return
            sleep(float(args.watch_status))

    prepare_output_dir(out_dir, args.overwrite, args.resume, args.plots_only)

    print("=" * 90)
    print(f"BASE CASE TYPE    : {base_case_name}")
    print(f"RUN/FOLDER NAME   : {run_config.case_name}")
    print(f"SEASON / MP-TP    : {run_config.season} / {run_config.mp_tp_label}")
    print(f"SNAPSHOTS         : {args.n_snapshots}")
    print(f"SOLVE MODE        : {args.solve_mode}  (default campaign mode is coupled)")
    print(f"PANELS            : {panel_label(args.n_vert, args.n_horiz)}")
    print(f"BC / LOAD         : {args.bc_type} / {args.load_type}")
    print(f"PLATE             : resolution={args.plate_resolution}, degree={args.plate_degree}")
    print(f"HEAT              : ({args.heat_nx},{args.heat_ny},{args.heat_nz}), pT={args.heat_degree}")
    print(f"THETA/T1          : cg_degree={args.theta_cg_degree}, quadrature={args.theta_quadrature}")
    print(f"OUTPUT            : {out_dir}")
    print(f"FINAL ARCHIVE     : {out_dir / final_snapshot_archive_name(run_config)}")
    print(f"PARAMETERS        : {[p.name for p in run_config.parameter_ranges]}")
    print("=" * 90)

    if args.plots_only:
        X = np.load(out_dir / "parameter_matrix.npy") if (out_dir / "parameter_matrix.npy").exists() else np.empty((0, 0))
        run_diagnostics(out_dir, run_config, X)
        print(f"Regenerated plots in {out_dir / 'figures'}")
        return

    X_file = out_dir / "parameter_matrix.npy"
    if args.resume and X_file.exists():
        X = np.load(X_file)
        if len(X) != args.n_snapshots:
            raise ValueError(f"--resume found {len(X)} samples, but --n-snapshots={args.n_snapshots}")
        print(f"Loaded existing parameter matrix: {X_file}")
    else:
        X = sample_parameters(run_config, int(args.n_snapshots), int(args.seed), args.sampling)
        save_parameter_table(out_dir, run_config, X)

    write_json(out_dir / "metadata.json", {
        "created_utc": datetime.utcnow().isoformat() + "Z",
        "base_case_type": base_case_name,
        "full_run_name": run_config.case_name,
        "primary_archive": final_snapshot_archive_name(run_config),
        "case_config": asdict(run_config),
        "cli_args": vars(args),
        "note": "theta in the manuscript is stored as theta_snapshots; the code solver calls it T1.",
    })

    write_status_files(out_dir, run_config, int(args.n_snapshots), event="run_start", note="snapshot generation started")

    SolverClass = import_solver_class(args.solver_module)
    solver = configure_solver(SolverClass, run_config, args, out_dir)

    failure_count = 0
    for i, sample in enumerate(X):
        tmp = sample_tmp_path(out_dir, i)
        if args.resume and tmp.exists():
            print(f"[{i+1:04d}/{len(X):04d}] already complete: {tmp.name}")
            write_status_files(out_dir, run_config, len(X), event="sample_resume_skip", current_index=i, current_sample=sample)
            print_compact_progress(out_dir, run_config, len(X))
            continue
        print(f"[{i+1:04d}/{len(X):04d}] running sample = {np.array2string(sample, precision=5)}")
        write_status_files(out_dir, run_config, len(X), event="sample_running", current_index=i, current_sample=sample)
        try:
            meta = solve_one_sample(solver, run_config, sample, args, i, out_dir)
            ok = bool(meta.get("converged", False)) and not bool(meta.get("stopped_by_max_iters", False))
            print(
                f"    done: converged={meta['converged']} iters={meta['coupled_iterations']} "
                f"err_w={meta['err_w']:.3e} err_theta={meta['err_theta']:.3e} "
                f"time={meta['wall_time_sec']:.1f}s"
            )
            write_status_files(out_dir, run_config, len(X), event="sample_done", current_index=i, current_sample=sample, note=f"converged={meta.get('converged')} iters={meta.get('coupled_iterations')}")
            print_compact_progress(out_dir, run_config, len(X))
            if args.solve_mode == "coupled" and not ok:
                failure_count += 1
                if args.strict:
                    raise RuntimeError(f"Sample {i} did not converge in coupled mode: {meta['termination_reason']}")
        except Exception as exc:
            failure_count += 1
            fail = {
                "sample_index": int(i),
                "sample": np.asarray(sample, dtype=float).tolist(),
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
            write_json(failure_tmp_path(out_dir, i), fail)
            print(f"    FAILED: {exc}")
            write_status_files(out_dir, run_config, len(X), event="sample_failed", current_index=i, current_sample=sample, note=repr(exc))
            print_compact_progress(out_dir, run_config, len(X))
            if args.strict:
                write_failed_csv(out_dir)
                raise

    archive = consolidate(out_dir, run_config, X)
    write_status_files(out_dir, run_config, len(X), event="run_consolidated", note=f"archive={archive.name}")
    write_failed_csv(out_dir)
    if args.make_plots:
        run_diagnostics(out_dir, run_config, X)

    print("=" * 90)
    print(f"Completed case: {run_config.case_name}")
    print(f"Final archive : {archive}")
    print(f"Failures/nonconverged count noted during run: {failure_count}")
    print(f"Summary CSV   : {out_dir / 'summary.csv'}")
    print(f"Failed CSV    : {out_dir / 'failed_samples.csv'}")
    print("=" * 90)
