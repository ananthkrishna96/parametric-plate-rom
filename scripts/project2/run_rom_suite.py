#!/usr/bin/env python3
"""Run Project-2 ROM suites from existing POD preprocessing artifacts.

This script is the unified entry point after the common foundation has been
built:

    data-loader + POD preprocessing

It can run/plan:

    intrusive      : POD-projected now; POD-Galerkin is capability-disabled.
    nonintrusive   : PODI-RBF, PODI-linear, legacy-aligned POD-GPR, POD-NN, and POD-AE.
"""

from __future__ import annotations

# Match the original notebook's conservative thread policy before importing
# NumPy/scikit-learn/PyTorch-backed modules.  This is a stability fix only and
# does not alter any ROM method settings.
import os

for _var in [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]:
    os.environ.setdefault(_var, "1")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from tabulate import tabulate

from paramplate.io.data_locations import load_data_locations
from paramplate.rom.datasets import load_project2_dataset_by_name
from paramplate.rom.intrusive import (
    evaluate_pod_projection,
    load_pod_artifact,
    write_intrusive_projection_summary,
)
from paramplate.rom.nonintrusive import (
    evaluate_coefficient_regressor,
    make_coefficient_regressor,
    write_nonintrusive_result,
)
from paramplate.rom.suites import (
    ALL_METHODS,
    method_names_for_suite,
    methods_for_suite,
    validate_method_list,
)


def _default_pod_dir(case: str) -> Path:
    loc = load_data_locations()
    return loc.paper2_rom_root / "pod_bases" / case


def _suite_output_root(suite: str, case: str) -> Path:
    loc = load_data_locations()
    return loc.paper2_rom_root / "suites" / suite / case


def _load_parameters_for_suite(pod_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load raw train/test parameters for legacy-aligned non-intrusive models.

    The common POD preprocessing stores both scaled parameters and raw parameters.
    The original THERMOMECHANICAL_FOM_ROM.py non-intrusive classes fit their own
    StandardScaler objects internally.  Therefore the suite runner must pass raw
    parameters here, not the already scaled arrays.
    """
    with np.load(pod_dir / "parameters_scaled.npz", allow_pickle=True) as z:
        train_idx = np.asarray(z["train_indices"], dtype=int)
        test_idx = np.asarray(z["test_indices"], dtype=int)
        if "raw_parameters" in z:
            raw = np.asarray(z["raw_parameters"], dtype=np.float64)
            if raw.ndim == 1:
                raw = raw.reshape(-1, 1)
            return raw[train_idx], raw[test_idx], train_idx, test_idx
        return (
            np.asarray(z["train_scaled"], dtype=np.float64),
            np.asarray(z["test_scaled"], dtype=np.float64),
            train_idx,
            test_idx,
        )


def _print_registry(suite: str) -> None:
    rows = [
        [m.suite, m.name, m.family, m.status, "yes" if m.default_enabled else "no", m.description]
        for m in methods_for_suite(suite, include_disabled=True)
    ]
    print(tabulate(rows, headers=["suite", "method", "family", "status", "default", "description"], tablefmt="github"))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def run_intrusive(case: str, pod_dir: Path, *, write: bool, overwrite: bool) -> list[dict]:
    dataset = load_project2_dataset_by_name(case, load_theta=True)
    pod_w = load_pod_artifact(pod_dir / "pod_w.npz")
    w_metrics = evaluate_pod_projection(dataset.snapshots, pod_w, field_name="w")

    theta_metrics = None
    if dataset.theta_snapshots is not None and (pod_dir / "pod_theta.npz").exists():
        pod_theta = load_pod_artifact(pod_dir / "pod_theta.npz")
        theta_metrics = evaluate_pod_projection(dataset.theta_snapshots, pod_theta, field_name="theta")

    rows = [{
        "suite": "intrusive",
        "method": "pod-projected",
        "field": "w",
        "rank": w_metrics.rank,
        "test_error": w_metrics.relative_error_test,
        "train_error": w_metrics.relative_error_train,
    }]
    if theta_metrics is not None:
        rows.append({
            "suite": "intrusive",
            "method": "pod-projected",
            "field": "theta",
            "rank": theta_metrics.rank,
            "test_error": theta_metrics.relative_error_test,
            "train_error": theta_metrics.relative_error_train,
        })

    print("\nIntrusive suite:")
    print(tabulate(rows, headers="keys", tablefmt="github", floatfmt=".6e"))

    if write:
        out = _suite_output_root("intrusive", case) / "pod_projected"
        path = write_intrusive_projection_summary(
            out,
            case_name=case,
            source_archive=dataset.source_path,
            w_metrics=w_metrics,
            theta_metrics=theta_metrics,
            overwrite=overwrite,
        )
        print(f"wrote: {path}")

    print("\nPOD-Galerkin: capable but disabled by default; wrap the legacy FEniCS residual/Jacobian before enabling.")
    return rows


def run_nonintrusive(
    case: str,
    pod_dir: Path,
    methods: Iterable[str],
    *,
    kernel: str,
    write: bool,
    overwrite: bool,
    seed: int,
    max_iter: int | None = None,
) -> list[dict]:
    dataset = load_project2_dataset_by_name(case, load_theta=True)
    P_train, P_test, train_idx, test_idx = _load_parameters_for_suite(pod_dir)
    pod_w = load_pod_artifact(pod_dir / "pod_w.npz")
    Xw_train = np.asarray(dataset.snapshots, dtype=np.float64)[train_idx]
    Xw_test = np.asarray(dataset.snapshots, dtype=np.float64)[test_idx]

    pod_theta = None
    Xt_train = Xt_test = None
    if dataset.theta_snapshots is not None and (pod_dir / "pod_theta.npz").exists():
        pod_theta = load_pod_artifact(pod_dir / "pod_theta.npz")
        Xt_train = np.asarray(dataset.theta_snapshots, dtype=np.float64)[train_idx]
        Xt_test = np.asarray(dataset.theta_snapshots, dtype=np.float64)[test_idx]

    all_rows: list[dict] = []
    for method in methods:
        if method == "pod-galerkin" or method == "pod-projected":
            continue
        print(f"\nNon-intrusive method: {method}")

        config_override = None
        if max_iter is not None and method == "pod-nn":
            # Optional smoke-test cap only.  Default None preserves legacy notebook epochs.
            config_override = {
                "coeff_epochs": int(max_iter),
                "field_epochs": max(1, min(275, int(max_iter) // 10)),
                "log_every": max(1, min(50, int(max_iter))),
            }
        elif max_iter is not None and method == "pod-ae":
            # Optional smoke-test cap only.  Default None preserves legacy notebook epochs.
            config_override = {
                "ae_epochs": int(max_iter),
                "latent_epochs": int(max_iter),
                "e2e_epochs": max(1, min(500, int(max_iter) // 5)),
                "log_every": max(1, min(50, int(max_iter))),
            }

        w_model = make_coefficient_regressor(method, kernel=kernel, seed=seed, n_basis=pod_w.rank, config=config_override)
        w_metrics, Cw_train_pred, Cw_test_pred = evaluate_coefficient_regressor(
            method=method,
            field_name="w",
            regressor=w_model,
            parameters_train=P_train,
            parameters_test=P_test,
            coefficients_train=pod_w.train_coefficients,
            coefficients_test=pod_w.test_coefficients,
            pod=pod_w,
            snapshots_train=Xw_train,
            snapshots_test=Xw_test,
        )

        theta_metrics = None
        theta_predictions = None
        theta_model = None
        if pod_theta is not None and Xt_train is not None and Xt_test is not None:
            theta_model = make_coefficient_regressor(method, kernel=kernel, seed=seed, n_basis=pod_theta.rank, config=config_override)
            theta_metrics, Ct_train_pred, Ct_test_pred = evaluate_coefficient_regressor(
                method=method,
                field_name="theta",
                regressor=theta_model,
                parameters_train=P_train,
                parameters_test=P_test,
                coefficients_train=pod_theta.train_coefficients,
                coefficients_test=pod_theta.test_coefficients,
                pod=pod_theta,
                snapshots_train=Xt_train,
                snapshots_test=Xt_test,
            )
            theta_predictions = (Ct_train_pred, Ct_test_pred)

        method_rows = [{
            "suite": "nonintrusive",
            "method": method,
            "field": "w",
            "rank": w_metrics.rank,
            "coeff_test_error": w_metrics.coefficient_relative_error_test,
            "field_test_error": w_metrics.field_relative_error_test,
        }]
        if theta_metrics is not None:
            method_rows.append({
                "suite": "nonintrusive",
                "method": method,
                "field": "theta",
                "rank": theta_metrics.rank,
                "coeff_test_error": theta_metrics.coefficient_relative_error_test,
                "field_test_error": theta_metrics.field_relative_error_test,
            })

        print(tabulate(method_rows, headers="keys", tablefmt="github", floatfmt=".6e"))
        all_rows.extend(method_rows)

        if write:
            out = _suite_output_root("nonintrusive", case) / method.replace("-", "_")
            path = write_nonintrusive_result(
                out,
                case_name=case,
                method=method,
                source_archive=dataset.source_path,
                w_metrics=w_metrics,
                w_predictions=(Cw_train_pred, Cw_test_pred),
                w_model=w_model,
                theta_metrics=theta_metrics,
                theta_predictions=theta_predictions,
                theta_model=theta_model,
                overwrite=overwrite,
            )
            print(f"wrote: {path}")

    if write and all_rows:
        root = _suite_output_root("nonintrusive", case)
        _write_csv(root / "nonintrusive_suite_summary.csv", all_rows)
    return all_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true", help="List ROM suite registry and exit.")
    parser.add_argument("--suite", choices=["common", "intrusive", "nonintrusive", "all"], default="all")
    parser.add_argument("--case", default="CASE_02_SUMMER_1MP1TP__PANEL_MONOLITHIC_NV0_NH0__BC_FREE_EDGE__LOAD_PATCH")
    parser.add_argument("--pod-dir", default=None, help="Existing POD preprocessing directory. Defaults to data_root/.../pod_bases/<case>.")
    parser.add_argument("--methods", default=None, help="Comma-separated method list. Defaults to suite default methods.")
    parser.add_argument("--kernel", default="thin_plate_spline", help="RBF/GPR kernel selector where applicable.")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--max-iter", type=int, default=None,
                        help="Optional cap for NN/AE epochs for quick smoke runs. Omit to preserve legacy notebook defaults.")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.list:
        _print_registry(args.suite)
        return 0

    if args.suite == "common":
        _print_registry("common")
        print("\nCommon foundation is already represented by check_rom_dataset.py and build_pod_dataset.py.")
        return 0

    pod_dir = Path(args.pod_dir).expanduser().resolve() if args.pod_dir else _default_pod_dir(args.case)
    if not (pod_dir / "pod_w.npz").exists():
        raise FileNotFoundError(
            f"POD preprocessing artifacts not found in {pod_dir}. "
            "Run scripts/project2/build_pod_dataset.py first."
        )

    if args.methods:
        methods = validate_method_list([x.strip() for x in args.methods.split(",") if x.strip()], suite=args.suite)
    else:
        methods = method_names_for_suite(args.suite, include_disabled=False)

    print(f"case    : {args.case}")
    print(f"pod_dir : {pod_dir}")
    print(f"suite   : {args.suite}")
    print(f"methods : {', '.join(methods)}")
    print(f"write   : {args.write}")

    rows: list[dict] = []
    if args.suite in {"intrusive", "all"}:
        rows.extend(run_intrusive(args.case, pod_dir, write=args.write, overwrite=args.overwrite))
    if args.suite in {"nonintrusive", "all"}:
        rows.extend(run_nonintrusive(
            args.case,
            pod_dir,
            [m for m in methods if m in method_names_for_suite("nonintrusive", include_disabled=False)],
            kernel=args.kernel,
            write=args.write,
            overwrite=args.overwrite,
            seed=args.seed,
            max_iter=args.max_iter,
        ))

    if rows:
        print("\nCombined suite summary:")
        print(tabulate(rows, headers="keys", tablefmt="github", floatfmt=".6e"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
