#!/usr/bin/env python3
"""
Baseline-aware gap-filling addon runner and merger for thermomechanical SNAPGEN.

Purpose
-------
This script is meant to be placed in the same SNAPGEN folder as:

    THERMOMECHANICAL_FOM_ROM.py
    THERMOMECHANICAL_SNAPGEN_COMMON.py
    THERMOMECHANICAL_SNAPGEN_CASES.py
    RUN_SNAPGEN_MONOLITHIC_FREE_EDGE_PATCH_MPTP_COUNTS.py

It is designed for the situation:

    1. A baseline campaign already exists in THERMO_SNAPSHOTS/.
    2. You later want extra snapshots without rerunning the old baseline samples.
    3. The extra addon samples should be baseline-aware: they should fill gaps in
       the already-sampled parameter space as much as possible.
    4. After the addon finishes, baseline + addon are merged into a clean final
       ROM-ready archive tree.

Recommended use after the current baseline finishes
---------------------------------------------------

Dry plan:

    python THERMO_SNAPGEN_BASELINE_AWARE_ADDON.py --mode plan

Generate gap-filling addon parameters only:

    python THERMO_SNAPGEN_BASELINE_AWARE_ADDON.py --mode generate-parameters --run-flag overwrite

Run the addon snapshots in the background:

    nohup python -u THERMO_SNAPGEN_BASELINE_AWARE_ADDON.py \
      --mode run-addon \
      --run-flag overwrite \
      > RUN_LOGS/NOHUP_ADDON_GAPFILL_DOUBLE.out 2>&1 &

Resume addon if interrupted:

    nohup python -u THERMO_SNAPGEN_BASELINE_AWARE_ADDON.py \
      --mode run-addon \
      --run-flag resume \
      > RUN_LOGS/NOHUP_ADDON_GAPFILL_DOUBLE_RESUME.out 2>&1 &

After addon finishes, merge baseline + addon:

    python THERMO_SNAPGEN_BASELINE_AWARE_ADDON.py --mode merge

All in one long run:

    nohup python -u THERMO_SNAPGEN_BASELINE_AWARE_ADDON.py \
      --mode all \
      --run-flag overwrite \
      > RUN_LOGS/NOHUP_ADDON_GAPFILL_DOUBLE_ALL.out 2>&1 &

Default campaign
----------------
The defaults match the baseline campaign currently being run:

    panel type  : monolithic
    n_vert      : 0
    n_horiz     : 0
    BC          : free_edge
    load        : patch
    solve mode  : coupled
    baseline    : THERMO_SNAPSHOTS
    addon       : THERMO_SNAPSHOTS_ADDON_GAPFILL_DOUBLE
    merged      : THERMO_SNAPSHOTS_MERGED_DOUBLE

Default addon counts are the same as the baseline official MP/TP counts:

    CASE 01,06: 100
    CASE 02,03,07,08: 150
    CASE 04,05,09,10: 200

Therefore the default produces a doubled final dataset:
    baseline 1600 + addon 1600 = merged 3200 snapshots.

How the addon sampling works
----------------------------
For each case:

    1. Read the baseline parameter matrix from the existing baseline output.
    2. Normalize all parameters to [0,1]^d using the exact case ranges.
       Log-scale parameters are normalized in log10-space.
    3. Generate a large candidate pool.
    4. Use maximin / farthest-point selection:
       each new addon point maximizes its distance from all old baseline points
       and already-selected addon points.
    5. Save the selected addon matrix as parameter_matrix.npy/csv in the addon
       output folder.
    6. Run the existing THERMOMECHANICAL_SNAPGEN_CASES.py with --resume so it
       uses that precomputed parameter_matrix.npy.

This script does not modify THERMOMECHANICAL_SNAPGEN_COMMON.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Non-interactive plotting if diagnostics are called.
import matplotlib
matplotlib.use("Agg")

try:
    from scipy.stats import qmc
except Exception:
    qmc = None

# Existing SNAPGEN modules.  This script must be placed next to them.
from THERMOMECHANICAL_SNAPGEN_CASES import build_case_configs
from THERMOMECHANICAL_SNAPGEN_COMMON import (
    CaseConfig,
    ParamRange,
    build_full_case_run_name,
    case_output_dir,
    final_snapshot_archive_name,
    find_snapshot_archive,
    panel_label,
    run_diagnostics,
    save_parameter_table,
    write_json,
)

SNAPGEN_SCRIPT = "THERMOMECHANICAL_SNAPGEN_CASES.py"
FOM_FILE = "THERMOMECHANICAL_FOM_ROM.py"
COMMON_FILE = "THERMOMECHANICAL_SNAPGEN_COMMON.py"
CASES_FILE = "THERMOMECHANICAL_SNAPGEN_CASES.py"

DEFAULT_CASES = list(range(1, 11))
DEFAULT_N_SNAPSHOTS_BY_CASE = {
    1: 100,
    2: 150,
    3: 150,
    4: 200,
    5: 200,
    6: 100,
    7: 150,
    8: 150,
    9: 200,
    10: 200,
}

SAMPLEWISE_KEYS = [
    "snapshots",
    "theta_snapshots",
    "parameters",
    "mech_parameters",
    "thermal_parameters",
    "convergence",
    "snapshot_dofs",
    "heat_discretizations",
    "metadata_json",
]

LABEL_KEYS = [
    "parameter_names",
    "parameter_units",
    "parameter_roles",
    "mech_parameter_names",
    "thermal_parameter_names",
    "convergence_names",
    "snapshot_dof_names",
    "heat_discretization_names",
]


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------

def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def banner(title: str) -> str:
    line = "=" * 100
    return f"{line}\n{title}\n{line}"


def parse_cases(text: str) -> List[int]:
    raw = str(text).replace(",", " ").split()
    out = [int(x) for x in raw]
    for c in out:
        if c < 1 or c > 10:
            raise ValueError(f"Invalid case {c}. Use cases 1..10.")
    return out


def parse_count_map(text: str) -> Dict[int, int]:
    """
    Parse count map like:
        "1:100,2:150,3:150"
    """
    result: Dict[int, int] = {}
    text = str(text or "").strip()
    if not text:
        return result
    for part in text.replace(",", " ").split():
        if ":" not in part:
            raise ValueError(f"Invalid --addon-counts item '{part}'. Expected CASE:COUNT.")
        k, v = part.split(":", 1)
        cid = int(k)
        n = int(v)
        if cid < 1 or cid > 10:
            raise ValueError(f"Invalid case in --addon-counts: {cid}")
        if n < 0:
            raise ValueError(f"Addon count must be >=0 for case {cid}")
        result[cid] = n
    return result


def addon_count_for_case(args: argparse.Namespace, case_id: int) -> int:
    explicit = parse_count_map(args.addon_counts)
    if int(case_id) in explicit:
        return int(explicit[int(case_id)])
    if int(args.addon_n_snapshots) >= 0:
        return int(args.addon_n_snapshots)
    base = int(DEFAULT_N_SNAPSHOTS_BY_CASE[int(case_id)])
    return int(round(float(args.addon_factor) * base))


def runtime_config(base: CaseConfig, args: argparse.Namespace) -> CaseConfig:
    cfg = deepcopy(base)
    cfg.bc_type = str(args.bc_type)
    cfg.load_type = str(args.load_type)
    if args.panel_type == "monolithic":
        cfg.n_vert, cfg.n_horiz = 0, 0
    elif args.panel_type == "contact1":
        cfg.n_vert, cfg.n_horiz = 1, 0
    elif args.panel_type == "contact2":
        cfg.n_vert, cfg.n_horiz = 2, 0
    else:
        cfg.n_vert, cfg.n_horiz = int(args.n_vert), int(args.n_horiz)

    cfg.plate_resolution = int(args.plate_resolution)
    cfg.plate_degree = int(args.plate_degree)
    cfg.heat_nx = int(args.heat_nx)
    cfg.heat_ny = int(args.heat_ny)
    cfg.heat_nz = int(args.heat_nz)
    cfg.heat_degree = int(args.heat_degree)
    cfg.theta_cg_degree = int(args.theta_cg_degree)
    cfg.theta_quadrature = int(args.theta_quadrature)
    cfg.solve_mode = str(args.solve_mode)
    cfg.omega = float(args.omega)
    cfg.tol_w = float(args.tol_w)
    cfg.tol_theta = float(args.tol_theta)
    cfg.max_coupling_iters = int(args.max_coupling_iters)
    return cfg


def run_name_for(config: CaseConfig) -> str:
    return build_full_case_run_name(config)


def expected_archive_path(root: str, config: CaseConfig) -> Path:
    out_dir = case_output_dir(root, config)
    return find_snapshot_archive(out_dir, config)


def load_archive(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Archive not found: {path}")
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def array_to_str_list(arr: np.ndarray) -> List[str]:
    return [str(x) for x in np.asarray(arr).tolist()]


def ensure_same_string_array(a: np.ndarray, b: np.ndarray, key: str) -> None:
    aa = array_to_str_list(a)
    bb = array_to_str_list(b)
    if aa != bb:
        raise ValueError(f"Archive metadata mismatch for {key}:\n  baseline={aa}\n  addon={bb}")


def safe_json(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe_json(v) for v in obj]
    return obj


def write_text_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", buffering=1) as f:
        f.write(text.rstrip("\n") + "\n")


# -----------------------------------------------------------------------------
# Parameter-space normalization and gap-filling
# -----------------------------------------------------------------------------

def normalize_parameters(config: CaseConfig, X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"Parameter matrix must be 2D, got shape {X.shape}.")
    if X.shape[1] != len(config.parameter_ranges):
        raise ValueError(
            f"Parameter dimension mismatch for {config.case_name}: "
            f"matrix has {X.shape[1]}, case has {len(config.parameter_ranges)}."
        )

    U = np.zeros_like(X, dtype=float)
    for j, pr in enumerate(config.parameter_ranges):
        if pr.scale == "log":
            lo = math.log10(float(pr.lower))
            hi = math.log10(float(pr.upper))
            U[:, j] = (np.log10(X[:, j]) - lo) / (hi - lo)
        else:
            lo = float(pr.lower)
            hi = float(pr.upper)
            U[:, j] = (X[:, j] - lo) / (hi - lo)
    return U


def denormalize_parameters(config: CaseConfig, U: np.ndarray) -> np.ndarray:
    U = np.asarray(U, dtype=float)
    if U.ndim != 2:
        raise ValueError(f"Unit matrix must be 2D, got shape {U.shape}.")
    X = np.zeros_like(U, dtype=float)
    for j, pr in enumerate(config.parameter_ranges):
        uj = np.clip(U[:, j], 0.0, 1.0)
        if pr.scale == "log":
            lo = math.log10(float(pr.lower))
            hi = math.log10(float(pr.upper))
            X[:, j] = 10.0 ** (lo + uj * (hi - lo))
        else:
            lo = float(pr.lower)
            hi = float(pr.upper)
            X[:, j] = lo + uj * (hi - lo)
    return X


def candidate_unit_points(n: int, d: int, seed: int) -> np.ndarray:
    if d < 1:
        return np.zeros((n, 0), dtype=float)
    if qmc is not None:
        try:
            return qmc.LatinHypercube(d=d, seed=int(seed), optimization="random-cd").random(n)
        except Exception:
            return qmc.LatinHypercube(d=d, seed=int(seed)).random(n)
    rng = np.random.default_rng(int(seed))
    return rng.random((n, d))


def select_gapfill_unit_points(
    U_old: np.ndarray,
    n_add: int,
    *,
    seed: int,
    candidate_pool_size: int,
    duplicate_tol: float = 1.0e-10,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Maximin/farthest-point selection in normalized parameter space.

    U_old is already normalized to [0,1]^d.
    """
    U_old = np.asarray(U_old, dtype=float)
    if U_old.ndim != 2:
        raise ValueError(f"U_old must be 2D, got shape {U_old.shape}.")
    n_old, d = U_old.shape
    n_add = int(n_add)
    if n_add < 0:
        raise ValueError("n_add must be >= 0")
    if n_add == 0:
        return np.zeros((0, d), dtype=float), {"n_old": n_old, "n_add": 0, "method": "none"}

    if d == 0:
        raise ValueError("Cannot gap-fill a zero-dimensional parameter case.")

    # Exact 1D interval bisection gives better gap-filling than a finite candidate pool.
    if d == 1:
        pts = sorted(float(x) for x in U_old[:, 0] if np.isfinite(x))
        pts = [min(1.0, max(0.0, x)) for x in pts]
        selected: List[float] = []
        all_pts = sorted([0.0] + pts + [1.0])
        for _ in range(n_add):
            gaps = [(all_pts[i+1] - all_pts[i], all_pts[i], all_pts[i+1]) for i in range(len(all_pts)-1)]
            gaps.sort(reverse=True, key=lambda item: item[0])
            gap, a, b = gaps[0]
            x = 0.5 * (a + b)
            selected.append(x)
            all_pts.append(x)
            all_pts.sort()
        U_new = np.asarray(selected, dtype=float).reshape(-1, 1)
        D_old = np.abs(U_new - U_old.T)
        min_to_old = D_old.min(axis=1) if n_old else np.ones(n_add)
        if n_add > 1:
            D_pair = np.abs(U_new - U_new.T)
            D_pair[D_pair == 0.0] = np.inf
            min_pair = D_pair.min(axis=1)
        else:
            min_pair = np.asarray([np.inf])
        return U_new, {
            "method": "1D_largest_interval_bisection",
            "n_old": int(n_old),
            "n_add": int(n_add),
            "dimension": int(d),
            "candidate_pool_size": None,
            "min_distance_new_to_old": float(np.min(min_to_old)),
            "mean_distance_new_to_old": float(np.mean(min_to_old)),
            "min_pairwise_new_distance": float(np.min(min_pair)),
            "duplicate_tol": float(duplicate_tol),
        }

    # Candidate-based farthest-point selection.
    M = int(candidate_pool_size)
    if M < n_add:
        M = max(n_add, 10 * n_add)
    C = candidate_unit_points(M, d, seed=seed)

    # Initial minimum squared distance to old baseline points.
    if n_old > 0:
        min_d2 = np.full(M, np.inf, dtype=float)
        # Blocked distance update to keep memory controlled.
        block = 4096
        for start in range(0, n_old, block):
            O = U_old[start:start+block]
            d2 = ((C[:, None, :] - O[None, :, :]) ** 2).sum(axis=2)
            min_d2 = np.minimum(min_d2, d2.min(axis=1))
    else:
        min_d2 = np.full(M, np.inf, dtype=float)

    # Avoid exact duplicates if any candidate lands on old point.
    min_d2[min_d2 <= duplicate_tol**2] = -np.inf

    selected_idx: List[int] = []
    selected_pts: List[np.ndarray] = []

    for _ in range(n_add):
        idx = int(np.argmax(min_d2))
        if not np.isfinite(min_d2[idx]):
            raise RuntimeError(
                "Could not select enough gap-fill candidates. Increase --candidate-pool-size "
                "or reduce --duplicate-tol."
            )
        selected_idx.append(idx)
        p = C[idx].copy()
        selected_pts.append(p)

        # Update candidate distance to the newly selected point.
        d2_new = ((C - p[None, :]) ** 2).sum(axis=1)
        min_d2 = np.minimum(min_d2, d2_new)
        min_d2[idx] = -np.inf

    U_new = np.vstack(selected_pts)

    # Reports.
    D_old_min = []
    if n_old:
        for p in U_new:
            D_old_min.append(float(np.sqrt(((U_old - p[None, :]) ** 2).sum(axis=1).min())))
    else:
        D_old_min = [float("nan")] * len(U_new)

    if len(U_new) > 1:
        D = ((U_new[:, None, :] - U_new[None, :, :]) ** 2).sum(axis=2)
        D[D == 0.0] = np.inf
        pair_min = np.sqrt(D.min(axis=1))
    else:
        pair_min = np.asarray([np.inf])

    report = {
        "method": "candidate_maximin_farthest_point",
        "n_old": int(n_old),
        "n_add": int(n_add),
        "dimension": int(d),
        "candidate_pool_size": int(M),
        "seed": int(seed),
        "min_distance_new_to_old": float(np.nanmin(D_old_min)),
        "mean_distance_new_to_old": float(np.nanmean(D_old_min)),
        "min_pairwise_new_distance": float(np.min(pair_min)),
        "mean_pairwise_new_min_distance": float(np.mean(pair_min[np.isfinite(pair_min)])) if np.isfinite(pair_min).any() else float("inf"),
        "duplicate_tol": float(duplicate_tol),
    }
    return U_new, report


def load_baseline_parameter_matrix(baseline_root: str, config: CaseConfig) -> Tuple[np.ndarray, Path, Optional[Path]]:
    out_dir = case_output_dir(baseline_root, config)
    param_path = out_dir / "parameter_matrix.npy"
    archive_path = find_snapshot_archive(out_dir, config)

    if param_path.exists():
        X = np.load(param_path)
        return np.asarray(X, dtype=float), param_path, archive_path if archive_path.exists() else None

    if archive_path.exists():
        data = np.load(archive_path, allow_pickle=True)
        if "parameters" not in data.files:
            raise KeyError(f"Archive exists but has no 'parameters' key: {archive_path}")
        X = np.asarray(data["parameters"], dtype=float)
        return X, archive_path, archive_path

    raise FileNotFoundError(
        f"No baseline parameter_matrix.npy or archive found for {config.case_name}\n"
        f"Expected folder: {out_dir}"
    )


def check_baseline_ready(args: argparse.Namespace, config: CaseConfig, n_baseline_expected: Optional[int] = None) -> Dict[str, Any]:
    X_base, source, archive = load_baseline_parameter_matrix(args.baseline_root, config)
    names = [p.name for p in config.parameter_ranges]

    if X_base.shape[1] != len(names):
        raise ValueError(
            f"Baseline parameter dimension mismatch for {config.case_name}: "
            f"got {X_base.shape[1]}, expected {len(names)}."
        )

    if not np.all(np.isfinite(X_base)):
        raise ValueError(f"Baseline parameter matrix has NaN/Inf for {config.case_name}: {source}")

    U = normalize_parameters(config, X_base)
    outside = np.logical_or(U < -1.0e-8, U > 1.0 + 1.0e-8)
    if np.any(outside):
        raise ValueError(
            f"Baseline parameter matrix contains values outside current case ranges for {config.case_name}."
        )

    final_archive_exists = bool(archive and Path(archive).exists())
    if args.require_baseline_archive and not final_archive_exists:
        raise FileNotFoundError(f"Baseline final archive required but not found for {config.case_name}")

    if n_baseline_expected is not None and int(n_baseline_expected) > 0 and X_base.shape[0] != int(n_baseline_expected):
        raise ValueError(
            f"Baseline count mismatch for {config.case_name}: "
            f"found {X_base.shape[0]}, expected {n_baseline_expected}."
        )

    return {
        "case_name": config.case_name,
        "run_name": run_name_for(config),
        "parameter_names": names,
        "baseline_count": int(X_base.shape[0]),
        "baseline_parameter_source": str(source),
        "baseline_archive": None if archive is None else str(archive),
        "baseline_archive_exists": final_archive_exists,
    }


def generate_addon_parameter_matrix(args: argparse.Namespace, config: CaseConfig, n_add: int) -> Tuple[np.ndarray, Dict[str, Any]]:
    X_base, source, archive = load_baseline_parameter_matrix(args.baseline_root, config)
    U_base = normalize_parameters(config, X_base)

    pool = int(args.candidate_pool_size)
    if pool <= 0:
        pool = max(
            int(args.min_candidate_pool),
            int(args.candidate_multiplier) * max(1, int(n_add)),
        )
    pool = min(pool, int(args.max_candidate_pool))

    U_new, report = select_gapfill_unit_points(
        U_base,
        int(n_add),
        seed=int(args.seed) + 1009 * int(config.case_id),
        candidate_pool_size=pool,
        duplicate_tol=float(args.duplicate_tol),
    )
    X_new = denormalize_parameters(config, U_new)

    # Final duplicate/near-duplicate diagnostics in normalized space.
    U_combined = np.vstack([U_base, U_new])
    if U_new.shape[0] > 0:
        d_old = []
        for p in U_new:
            d_old.append(float(np.sqrt(((U_base - p[None, :]) ** 2).sum(axis=1).min())))
        min_new_to_old = float(np.min(d_old))
    else:
        min_new_to_old = float("nan")

    report.update({
        "case_id": int(config.case_id),
        "case_name": str(config.case_name),
        "run_name": run_name_for(config),
        "baseline_root": str(args.baseline_root),
        "addon_root": str(args.addon_root),
        "baseline_parameter_source": str(source),
        "baseline_archive": None if archive is None else str(archive),
        "parameter_names": [p.name for p in config.parameter_ranges],
        "parameter_units": [p.unit for p in config.parameter_ranges],
        "parameter_scales": [p.scale for p in config.parameter_ranges],
        "parameter_ranges": [
            dict(name=p.name, lower=float(p.lower), upper=float(p.upper), unit=p.unit, scale=p.scale, role=p.role, description=p.description)
            for p in config.parameter_ranges
        ],
        "baseline_count": int(X_base.shape[0]),
        "addon_count": int(X_new.shape[0]),
        "combined_count_if_merged": int(U_combined.shape[0]),
        "min_new_to_baseline_normalized_distance": min_new_to_old,
        "created_utc": utc_now(),
        "sampling_note": "Addon points were selected by baseline-aware gap-filling in normalized parameter space.",
    })

    return X_new, report


def write_addon_parameter_files(args: argparse.Namespace, config: CaseConfig, n_add: int, *, force: bool) -> Path:
    out_dir = case_output_dir(args.addon_root, config)

    if force and out_dir.exists():
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tmp").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    param_path = out_dir / "parameter_matrix.npy"
    report_path = out_dir / "addon_gapfill_report.json"

    if param_path.exists() and not force:
        X = np.load(param_path)
        if len(X) != int(n_add):
            raise ValueError(
                f"Existing addon parameter matrix length mismatch for {config.case_name}: "
                f"{len(X)} exists but requested {n_add}. Use --run-flag overwrite to regenerate."
            )
        return param_path

    X_new, report = generate_addon_parameter_matrix(args, config, int(n_add))
    save_parameter_table(out_dir, config, X_new)
    write_json(report_path, report)

    # Additional normalized matrix for diagnostics/debugging.
    U_new = normalize_parameters(config, X_new)
    np.save(out_dir / "parameter_matrix_normalized.npy", U_new)

    # Human-readable report.
    lines = []
    lines.append(f"Addon gap-fill parameter report for {run_name_for(config)}")
    lines.append(f"created_utc: {report['created_utc']}")
    lines.append(f"baseline_count: {report['baseline_count']}")
    lines.append(f"addon_count: {report['addon_count']}")
    lines.append(f"combined_count_if_merged: {report['combined_count_if_merged']}")
    lines.append(f"method: {report['method']}")
    lines.append(f"min_new_to_baseline_normalized_distance: {report['min_new_to_baseline_normalized_distance']:.6e}")
    lines.append(f"min_pairwise_new_distance: {report.get('min_pairwise_new_distance', float('nan')):.6e}")
    lines.append(f"parameter_names: {report['parameter_names']}")
    (out_dir / "addon_gapfill_report.txt").write_text("\n".join(lines) + "\n")

    return param_path


# -----------------------------------------------------------------------------
# Running the addon using the existing case generator
# -----------------------------------------------------------------------------

def build_addon_case_command(args: argparse.Namespace, case_id: int, n_add: int) -> List[str]:
    cmd = [
        sys.executable,
        "-u",
        SNAPGEN_SCRIPT,
        "--case", str(case_id),
        "--panel-type", args.panel_type,
        "--n-vert", str(args.n_vert),
        "--n-horiz", str(args.n_horiz),
        "--bc-type", args.bc_type,
        "--load-type", args.load_type,
        "--n-snapshots", str(n_add),
        "--seed", str(args.seed),  # kept in metadata; parameter_matrix.npy controls actual samples under --resume
        "--plate-resolution", str(args.plate_resolution),
        "--plate-degree", str(args.plate_degree),
        "--heat-nx", str(args.heat_nx),
        "--heat-ny", str(args.heat_ny),
        "--heat-nz", str(args.heat_nz),
        "--heat-degree", str(args.heat_degree),
        "--theta-cg-degree", str(args.theta_cg_degree),
        "--theta-quadrature", str(args.theta_quadrature),
        "--solve-mode", args.solve_mode,
        "--omega", str(args.omega),
        "--tol-w", str(args.tol_w),
        "--tol-theta", str(args.tol_theta),
        "--max-coupling-iters", str(args.max_coupling_iters),
        "--output-root", args.addon_root,
        "--solver-module", args.solver_module,
        "--sampling", "lhs",  # not used if precomputed parameter_matrix.npy exists under --resume
        "--resume",
    ]
    if args.make_plots:
        cmd.append("--make-plots")
    if args.save_field_plots:
        cmd.append("--save-field-plots")
    if args.strict:
        cmd.append("--strict")
    return cmd


def run_logged_subprocess(cmd: List[str], cwd: Path, case_log: Path, campaign_log: Path, env: Dict[str, str]) -> int:
    pretty = " ".join(shlex.quote(x) for x in cmd)
    header = f"\n[{now()}] COMMAND:\n{pretty}\n"
    print(header, flush=True)
    write_text_log(case_log, header)
    write_text_log(campaign_log, header)

    proc = subprocess.Popen(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        print(line, flush=True)
        write_text_log(case_log, line)
        write_text_log(campaign_log, line)
    return int(proc.wait())


def run_addon_cases(args: argparse.Namespace, cases: Dict[int, CaseConfig], selected_cases: List[int]) -> int:
    cwd = Path.cwd().resolve()
    log_root = Path(args.log_root).resolve()
    log_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    campaign_log = log_root / f"CAMPAIGN_ADDON_GAPFILL_{args.campaign_name}_{timestamp}.log"

    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("PYTHONUNBUFFERED", "1")

    write_text_log(campaign_log, banner("BASELINE-AWARE ADDON RUN START"))
    write_text_log(campaign_log, f"Started local : {now()}")
    write_text_log(campaign_log, f"Started UTC   : {utc_now()}")
    write_text_log(campaign_log, f"Working dir   : {cwd}")
    write_text_log(campaign_log, f"Python        : {sys.executable}")
    write_text_log(campaign_log, f"Baseline root : {args.baseline_root}")
    write_text_log(campaign_log, f"Addon root    : {args.addon_root}")
    write_text_log(campaign_log, f"Cases         : {selected_cases}")
    write_text_log(campaign_log, f"Panel/BC/load : {args.panel_type} / {args.bc_type} / {args.load_type}")
    write_text_log(campaign_log, f"Run flag      : {args.run_flag}")

    print(banner("BASELINE-AWARE ADDON RUN START"), flush=True)
    print(f"Campaign log: {campaign_log}", flush=True)

    failed_cases: List[int] = []

    for cid in selected_cases:
        cfg = runtime_config(cases[cid], args)
        n_add = addon_count_for_case(args, cid)
        if n_add == 0:
            msg = f"CASE {cid}: addon_count=0, skipping."
            print(msg, flush=True)
            write_text_log(campaign_log, msg)
            continue

        # For clean addon run: remove addon output and regenerate parameter matrix.
        # For resume: keep existing matrix/tmp files, generate only if matrix missing.
        force_params = args.run_flag == "overwrite"
        param_path = write_addon_parameter_files(args, cfg, n_add, force=force_params)

        case_log = Path(args.log_root).resolve() / f"ADDON_CASE_{cid}_{args.panel_type}_{args.bc_type}_{args.load_type}_{n_add}SNAPS.log"
        write_text_log(case_log, banner(f"RUNNING ADDON CASE {cid}"))
        write_text_log(case_log, f"Started: {now()}")
        write_text_log(case_log, f"Parameter matrix: {param_path}")
        write_text_log(case_log, f"Addon output: {case_output_dir(args.addon_root, cfg)}")

        cmd = build_addon_case_command(args, cid, n_add)
        if args.dry_run:
            txt = " ".join(shlex.quote(x) for x in cmd)
            print(txt, flush=True)
            write_text_log(campaign_log, txt)
            continue

        t0 = time.time()
        rc = run_logged_subprocess(cmd, cwd, case_log, campaign_log, env)
        dt = time.time() - t0

        msg = f"ADDON CASE {cid} finished at {now()} with exit status {rc}; wall_time={dt/3600:.3f} h"
        print(msg, flush=True)
        write_text_log(case_log, msg)
        write_text_log(campaign_log, msg)

        if rc != 0:
            failed_cases.append(cid)
            if not args.continue_on_error:
                write_text_log(campaign_log, f"Stopping because case {cid} failed.")
                return rc

    write_text_log(campaign_log, banner("BASELINE-AWARE ADDON RUN FINISHED"))
    write_text_log(campaign_log, f"Failed cases: {failed_cases}")
    print(f"Addon campaign finished. Failed cases: {failed_cases}", flush=True)
    return 1 if failed_cases else 0


# -----------------------------------------------------------------------------
# Merge baseline + addon archives
# -----------------------------------------------------------------------------

def concatenate_key(key: str, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    # metadata_json is a 1D string/object array; all other samplewise keys are sample-axis arrays.
    if key == "metadata_json":
        return np.concatenate([np.asarray(A), np.asarray(B)], axis=0)

    if key == "theta_snapshots":
        # Usually 2D.  If object arrays appear, concatenate as object arrays.
        try:
            return np.concatenate([A, B], axis=0)
        except Exception:
            return np.asarray(list(A) + list(B), dtype=object)

    return np.concatenate([A, B], axis=0)


def summarize_npz_keys(data: Dict[str, np.ndarray]) -> Dict[str, Any]:
    out = {}
    for k, v in data.items():
        try:
            out[k] = {"shape": list(np.shape(v)), "dtype": str(np.asarray(v).dtype)}
        except Exception:
            out[k] = {"shape": "unknown", "dtype": "unknown"}
    return out


def merge_summary_csv(baseline_dir: Path, addon_dir: Path, merged_dir: Path, n_base: int) -> None:
    out = merged_dir / "summary.csv"
    rows: List[Dict[str, Any]] = []
    fieldnames: List[str] = []

    def read_rows(path: Path, source: str, offset: int) -> None:
        nonlocal fieldnames
        if not path.exists():
            return
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for local_idx, row in enumerate(reader):
                row = dict(row)
                row["source_block"] = source
                try:
                    local_sample_index = int(row.get("sample_index", local_idx))
                except Exception:
                    local_sample_index = local_idx
                row["local_sample_index"] = local_sample_index
                row["global_sample_index"] = offset + local_sample_index
                rows.append(row)
                for k in row.keys():
                    if k not in fieldnames:
                        fieldnames.append(k)

    read_rows(baseline_dir / "summary.csv", "baseline", 0)
    read_rows(addon_dir / "summary.csv", "addon", int(n_base))

    if not rows:
        return

    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def merge_failed_csv(baseline_dir: Path, addon_dir: Path, merged_dir: Path) -> None:
    out = merged_dir / "failed_samples.csv"
    rows: List[Dict[str, Any]] = []
    fieldnames = ["source_block", "sample_index", "error", "traceback_file"]

    for source, folder in [("baseline", baseline_dir), ("addon", addon_dir)]:
        path = folder / "failed_samples.csv"
        if not path.exists():
            continue
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rr = {"source_block": source}
                rr.update(dict(row))
                rows.append(rr)

    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def write_merged_parameter_tables(merged_dir: Path, names: Sequence[str], X: np.ndarray) -> None:
    np.save(merged_dir / "parameter_matrix.npy", X)
    with (merged_dir / "parameter_matrix.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(list(names))
        for row in np.asarray(X, dtype=float):
            w.writerow([f"{float(v):.16e}" for v in row])


def merge_one_case(args: argparse.Namespace, config: CaseConfig) -> Dict[str, Any]:
    baseline_dir = case_output_dir(args.baseline_root, config)
    addon_dir = case_output_dir(args.addon_root, config)
    merged_dir = case_output_dir(args.merged_root, config)
    merged_dir.mkdir(parents=True, exist_ok=True)
    (merged_dir / "figures").mkdir(parents=True, exist_ok=True)

    base_archive_path = find_snapshot_archive(baseline_dir, config)
    addon_archive_path = find_snapshot_archive(addon_dir, config)

    base = load_archive(base_archive_path)
    addon = load_archive(addon_archive_path)

    for key in LABEL_KEYS:
        if key in base and key in addon:
            ensure_same_string_array(base[key], addon[key], key)

    # Critical dimensional compatibility.
    if base["snapshots"].shape[1:] != addon["snapshots"].shape[1:]:
        raise ValueError(
            f"Displacement snapshot shape mismatch for {config.case_name}: "
            f"{base['snapshots'].shape} vs {addon['snapshots'].shape}"
        )

    if "theta_snapshots" in base and "theta_snapshots" in addon:
        if np.asarray(base["theta_snapshots"]).dtype != object and np.asarray(addon["theta_snapshots"]).dtype != object:
            if base["theta_snapshots"].shape[1:] != addon["theta_snapshots"].shape[1:]:
                raise ValueError(
                    f"Theta snapshot shape mismatch for {config.case_name}: "
                    f"{base['theta_snapshots'].shape} vs {addon['theta_snapshots'].shape}"
                )

    X_base = np.asarray(base["parameters"], dtype=float)
    X_add = np.asarray(addon["parameters"], dtype=float)
    U_base = normalize_parameters(config, X_base)
    U_add = normalize_parameters(config, X_add)

    # Duplicate / closeness report.
    min_add_to_base = []
    for p in U_add:
        min_add_to_base.append(float(np.sqrt(((U_base - p[None, :]) ** 2).sum(axis=1).min())))
    min_add_to_base_val = float(np.min(min_add_to_base)) if min_add_to_base else float("nan")

    duplicate_count = int(np.sum(np.asarray(min_add_to_base) <= float(args.duplicate_tol))) if min_add_to_base else 0

    merged: Dict[str, Any] = {}
    for key in SAMPLEWISE_KEYS:
        if key in base and key in addon:
            merged[key] = concatenate_key(key, base[key], addon[key])

    for key in LABEL_KEYS:
        if key in base:
            merged[key] = base[key]

    # Preserve case config, but add merge metadata.
    if "case_config_json" in base:
        merged["case_config_json"] = base["case_config_json"]

    merge_meta = {
        "created_utc": utc_now(),
        "case_id": int(config.case_id),
        "case_name": str(config.case_name),
        "run_name": run_name_for(config),
        "baseline_archive": str(base_archive_path),
        "addon_archive": str(addon_archive_path),
        "merged_root": str(args.merged_root),
        "baseline_count": int(X_base.shape[0]),
        "addon_count": int(X_add.shape[0]),
        "merged_count": int(X_base.shape[0] + X_add.shape[0]),
        "parameter_names": [p.name for p in config.parameter_ranges],
        "min_addon_to_baseline_distance_normalized": min_add_to_base_val,
        "duplicate_tol": float(args.duplicate_tol),
        "duplicate_count_at_tol": duplicate_count,
        "baseline_npz_keys": summarize_npz_keys(base),
        "addon_npz_keys": summarize_npz_keys(addon),
        "note": "Merged archive = baseline samples followed by baseline-aware addon samples.",
    }
    merged["merge_metadata_json"] = np.asarray(json.dumps(safe_json(merge_meta)), dtype=str)

    archive = merged_dir / final_snapshot_archive_name(config)
    np.savez_compressed(archive, **merged)

    # Compatibility file snapshots.npz.
    compat = merged_dir / "snapshots.npz"
    try:
        if compat.exists() or compat.is_symlink():
            compat.unlink()
        os.link(archive, compat)
    except Exception:
        shutil.copy2(archive, compat)

    X_merged = np.vstack([X_base, X_add])
    param_names = array_to_str_list(base["parameter_names"]) if "parameter_names" in base else [p.name for p in config.parameter_ranges]
    write_merged_parameter_tables(merged_dir, param_names, X_merged)

    write_json(merged_dir / "merge_metadata.json", merge_meta)
    merge_summary_csv(baseline_dir, addon_dir, merged_dir, n_base=X_base.shape[0])
    merge_failed_csv(baseline_dir, addon_dir, merged_dir)

    if args.make_plots:
        run_diagnostics(merged_dir, config, X_merged)

    return merge_meta


def merge_cases(args: argparse.Namespace, cases: Dict[int, CaseConfig], selected_cases: List[int]) -> int:
    reports = []
    for cid in selected_cases:
        cfg = runtime_config(cases[cid], args)
        print(banner(f"MERGING CASE {cid}: {run_name_for(cfg)}"), flush=True)
        report = merge_one_case(args, cfg)
        reports.append(report)
        print(
            f"Merged CASE {cid}: baseline={report['baseline_count']} "
            f"addon={report['addon_count']} merged={report['merged_count']} "
            f"duplicates@tol={report['duplicate_count_at_tol']}",
            flush=True,
        )

    root = Path(args.merged_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "MERGED_DATASET_SUMMARY.json", {
        "created_utc": utc_now(),
        "baseline_root": str(args.baseline_root),
        "addon_root": str(args.addon_root),
        "merged_root": str(args.merged_root),
        "cases": reports,
        "total_baseline": int(sum(r["baseline_count"] for r in reports)),
        "total_addon": int(sum(r["addon_count"] for r in reports)),
        "total_merged": int(sum(r["merged_count"] for r in reports)),
    })
    return 0


# -----------------------------------------------------------------------------
# CLI modes
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Baseline-aware gap-filling addon snapshot runner and merger."
    )

    p.add_argument("--mode", choices=["plan", "generate-parameters", "run-addon", "merge", "all"], default="plan")
    p.add_argument("--cases", type=str, default="1 2 3 4 5 6 7 8 9 10")

    p.add_argument("--baseline-root", type=str, default="THERMO_SNAPSHOTS")
    p.add_argument("--addon-root", type=str, default="THERMO_SNAPSHOTS_ADDON_GAPFILL_DOUBLE")
    p.add_argument("--merged-root", type=str, default="THERMO_SNAPSHOTS_MERGED_DOUBLE")

    p.add_argument("--panel-type", type=str, default="monolithic", choices=["monolithic", "contact1", "contact2", "auto"])
    p.add_argument("--n-vert", type=int, default=0)
    p.add_argument("--n-horiz", type=int, default=0)
    p.add_argument("--bc-type", type=str, default="free_edge", choices=["free_edge", "simply_supported"])
    p.add_argument("--load-type", type=str, default="patch", choices=["patch", "linear", "multi_patch", "uniform"])

    # Addon-count controls.
    p.add_argument("--addon-factor", type=float, default=1.0, help="Addon count = addon_factor * baseline official count. Default 1.0 doubles the final dataset.")
    p.add_argument("--addon-n-snapshots", type=int, default=-1, help="Override addon count for every selected case. -1 means use addon-factor/schedule.")
    p.add_argument("--addon-counts", type=str, default="", help="Per-case addon counts, e.g. '1:100,2:150,3:150'. Overrides addon-factor for listed cases.")

    # Gap-fill controls.
    p.add_argument("--seed", type=int, default=3100, help="Seed for candidate pool generation. Does not reproduce the baseline.")
    p.add_argument("--candidate-multiplier", type=int, default=250, help="candidate_pool_size = max(min_pool, multiplier*n_add) unless explicit --candidate-pool-size is set.")
    p.add_argument("--min-candidate-pool", type=int, default=20000)
    p.add_argument("--max-candidate-pool", type=int, default=120000)
    p.add_argument("--candidate-pool-size", type=int, default=0, help="Explicit candidate pool size. 0 means automatic.")
    p.add_argument("--duplicate-tol", type=float, default=1.0e-10)
    p.add_argument("--require-baseline-archive", action="store_true", default=True)
    p.add_argument("--allow-missing-baseline-archive", action="store_false", dest="require_baseline_archive")

    # Solver/discretization controls copied from baseline runner.
    p.add_argument("--plate-resolution", type=int, default=64)
    p.add_argument("--plate-degree", type=int, default=2)
    p.add_argument("--heat-nx", type=int, default=64)
    p.add_argument("--heat-ny", type=int, default=32)
    p.add_argument("--heat-nz", type=int, default=16)
    p.add_argument("--heat-degree", type=int, default=1)
    p.add_argument("--theta-cg-degree", type=int, default=1)
    p.add_argument("--theta-quadrature", type=int, default=20)
    p.add_argument("--solve-mode", choices=["coupled", "one_way", "mechanical"], default="coupled")
    p.add_argument("--omega", type=float, default=0.7)
    p.add_argument("--tol-w", type=str, default="1e-5")
    p.add_argument("--tol-theta", type=str, default="1e-5")
    p.add_argument("--max-coupling-iters", type=int, default=25)
    p.add_argument("--solver-module", type=str, default="THERMOMECHANICAL_FOM_ROM")

    # Run/log controls.
    p.add_argument("--run-flag", choices=["overwrite", "resume"], default="resume")
    p.add_argument("--log-root", type=str, default="RUN_LOGS")
    p.add_argument("--campaign-name", type=str, default="MONOLITHIC_FREE_EDGE_PATCH_GAPFILL_ADDON")
    p.add_argument("--make-plots", action="store_true", default=True)
    p.add_argument("--no-make-plots", action="store_false", dest="make_plots")
    p.add_argument("--save-field-plots", action="store_true")
    p.add_argument("--strict", action="store_true")
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def precheck_files() -> None:
    missing = [f for f in [FOM_FILE, COMMON_FILE, CASES_FILE, SNAPGEN_SCRIPT] if not Path(f).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required SNAPGEN files in {Path.cwd()}: {missing}")


def print_plan(args: argparse.Namespace, cases: Dict[int, CaseConfig], selected_cases: List[int]) -> None:
    print(banner("BASELINE-AWARE ADDON PLAN"))
    print(f"Working folder : {Path.cwd().resolve()}")
    print(f"Baseline root  : {args.baseline_root}")
    print(f"Addon root     : {args.addon_root}")
    print(f"Merged root    : {args.merged_root}")
    print(f"Panel/BC/load  : {args.panel_type} / {args.bc_type} / {args.load_type}")
    print(f"Cases          : {selected_cases}")
    print(f"Mode           : {args.mode}")
    print("-" * 100)
    print(f"{'Case':>4}  {'Run name':<78} {'base':>6} {'add':>6} {'final':>7}  {'status'}")
    print("-" * 100)

    total_base = 0
    total_add = 0
    for cid in selected_cases:
        cfg = runtime_config(cases[cid], args)
        n_add = addon_count_for_case(args, cid)
        try:
            info = check_baseline_ready(args, cfg)
            n_base = int(info["baseline_count"])
            status = "baseline OK"
        except Exception as exc:
            n_base = 0
            status = f"BASELINE ISSUE: {exc}"
        total_base += n_base
        total_add += n_add
        print(f"{cid:>4}  {run_name_for(cfg):<78} {n_base:>6} {n_add:>6} {n_base+n_add:>7}  {status}")
    print("-" * 100)
    print(f"TOTAL baseline={total_base}, addon={total_add}, final_if_merged={total_base + total_add}")
    print("=" * 100)


def generate_parameters_for_cases(args: argparse.Namespace, cases: Dict[int, CaseConfig], selected_cases: List[int]) -> None:
    for cid in selected_cases:
        cfg = runtime_config(cases[cid], args)
        n_add = addon_count_for_case(args, cid)
        if n_add == 0:
            print(f"CASE {cid}: addon_count=0, skipping parameter generation.")
            continue
        force = args.run_flag == "overwrite"
        print(banner(f"GENERATING GAP-FILL PARAMETERS FOR CASE {cid}: {run_name_for(cfg)}"))
        info = check_baseline_ready(args, cfg)
        print(f"Baseline source: {info['baseline_parameter_source']}")
        print(f"Baseline count : {info['baseline_count']}")
        print(f"Addon count    : {n_add}")
        param_path = write_addon_parameter_files(args, cfg, n_add, force=force)
        print(f"Addon parameter matrix written: {param_path}")


def main() -> int:
    args = build_parser().parse_args()
    selected_cases = parse_cases(args.cases)
    cases = build_case_configs()

    precheck_files()

    if args.mode == "plan":
        print_plan(args, cases, selected_cases)
        return 0

    if args.mode == "generate-parameters":
        generate_parameters_for_cases(args, cases, selected_cases)
        return 0

    if args.mode == "run-addon":
        return run_addon_cases(args, cases, selected_cases)

    if args.mode == "merge":
        return merge_cases(args, cases, selected_cases)

    if args.mode == "all":
        rc = run_addon_cases(args, cases, selected_cases)
        if rc != 0:
            print("Addon run failed; merge not started.", flush=True)
            return rc
        return merge_cases(args, cases, selected_cases)

    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    raise SystemExit(main())
