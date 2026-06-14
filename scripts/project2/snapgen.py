#!/usr/bin/env python3
"""Unified Project-2 thermomechanical SNAPGEN command wrapper.

This is the clean repo-facing entry point for Project-2 snapshot generation.

It intentionally delegates to the non-lossy legacy SNAPGEN scripts preserved in:
    legacy/project2/snapgen/

Why this design:
    - no legacy functionality is removed;
    - source code is kept in Git;
    - generated snapshots/logs are written outside Git, under a data root;
    - old one-case, campaign, status, addon, and Paper-1-compat workflows remain usable.

Typical use from repository root:
    python scripts/project2/snapgen.py setup-data-layout --link-legacy
    python scripts/project2/snapgen.py list-cases
    python scripts/project2/snapgen.py run-case --case 1 --n-snapshots 2 --overwrite
    python scripts/project2/snapgen.py run-campaign --cases "1 2 3" --resume
    python scripts/project2/snapgen.py status
    python scripts/project2/snapgen.py addon --mode plan
    python scripts/project2/snapgen.py manifest
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence


CASE_NAMES = {
    1: "CASE_01_SUMMER_0MP1TP",
    2: "CASE_02_SUMMER_1MP1TP",
    3: "CASE_03_SUMMER_1MP2TP",
    4: "CASE_04_SUMMER_2MP2TP",
    5: "CASE_05_SUMMER_2MP3TP",
    6: "CASE_06_WINTER_0MP1TP",
    7: "CASE_07_WINTER_1MP1TP",
    8: "CASE_08_WINTER_1MP2TP",
    9: "CASE_09_WINTER_2MP2TP",
    10: "CASE_10_WINTER_2MP3TP",
}

DEFAULT_COUNTS = {
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


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, *here.parents]:
        if (parent / "pyproject.toml").exists() and (parent / "src").exists():
            return parent
    raise RuntimeError("Could not locate repository root from this script path.")


def default_data_root(root: Path) -> Path:
    return Path(os.environ.get("PARAMPLATE_DATA_ROOT", root.parent / "parametric-plate-rom-data")).expanduser().resolve()


def legacy_snapgen_dir(root: Path) -> Path:
    return root / "legacy" / "project2" / "snapgen"


def paper2_root(data_root: Path) -> Path:
    return data_root / "paper2_thermomechanical"


def baseline_root(data_root: Path) -> Path:
    return paper2_root(data_root) / "campaigns" / "baseline"


def addon_root(data_root: Path) -> Path:
    return paper2_root(data_root) / "campaigns" / "addon_gapfill"


def merged_root(data_root: Path) -> Path:
    return paper2_root(data_root) / "campaigns" / "merged"


def run_logs_root(data_root: Path) -> Path:
    return paper2_root(data_root) / "logs" / "RUN_LOGS"


def env_for_subprocess(root: Path, extra: dict | None = None) -> dict:
    env = os.environ.copy()
    legacy_dir = str(legacy_snapgen_dir(root))
    src_dir = str(root / "src")
    old_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join([src_dir, legacy_dir, old_pp]) if old_pp else os.pathsep.join([src_dir, legacy_dir])
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("PYTHONUNBUFFERED", "1")
    if extra:
        env.update({str(k): str(v) for k, v in extra.items()})
    return env


def run_legacy(root: Path, script_name: str, args: Sequence[str]) -> int:
    legacy_dir = legacy_snapgen_dir(root)
    script_path = legacy_dir / script_name
    if not script_path.exists():
        raise FileNotFoundError(f"Legacy SNAPGEN script not found: {script_path}")
    cmd = [sys.executable, str(script_path), *map(str, args)]
    print("[snapgen] cwd:", legacy_dir)
    print("[snapgen] cmd:", " ".join(str(x) for x in cmd))
    return subprocess.call(cmd, cwd=str(legacy_dir), env=env_for_subprocess(root))


def ensure_data_layout(data_root: Path) -> None:
    dirs = [
        data_root,
        data_root / "paper1_mechanical" / "snapshots" / "raw",
        data_root / "paper1_mechanical" / "snapshots" / "processed",
        data_root / "paper1_mechanical" / "snapshots" / "archives",
        data_root / "paper1_mechanical" / "snapshots" / "metadata",
        data_root / "paper1_mechanical" / "rom_training" / "pod_bases",
        data_root / "paper1_mechanical" / "rom_training" / "trained_models",
        data_root / "paper1_mechanical" / "rom_training" / "scalers",
        data_root / "paper1_mechanical" / "rom_training" / "metrics",
        data_root / "paper1_mechanical" / "results" / "figures",
        data_root / "paper1_mechanical" / "results" / "tables",
        data_root / "paper1_mechanical" / "results" / "reports",
        data_root / "paper1_mechanical" / "logs",
        paper2_root(data_root) / "snapshots" / "raw",
        paper2_root(data_root) / "snapshots" / "processed",
        paper2_root(data_root) / "snapshots" / "archives",
        paper2_root(data_root) / "snapshots" / "metadata",
        baseline_root(data_root),
        addon_root(data_root),
        merged_root(data_root),
        paper2_root(data_root) / "rom_training" / "pod_bases",
        paper2_root(data_root) / "rom_training" / "podi",
        paper2_root(data_root) / "rom_training" / "pod_gpr",
        paper2_root(data_root) / "rom_training" / "pod_nn",
        paper2_root(data_root) / "rom_training" / "pod_ae",
        paper2_root(data_root) / "rom_training" / "scalers",
        paper2_root(data_root) / "rom_training" / "metrics",
        paper2_root(data_root) / "convergence" / "csv",
        paper2_root(data_root) / "convergence" / "caches",
        paper2_root(data_root) / "convergence" / "figures",
        paper2_root(data_root) / "results" / "figures",
        paper2_root(data_root) / "results" / "tables",
        paper2_root(data_root) / "results" / "reports",
        run_logs_root(data_root),
        data_root / "shared" / "manifests",
        data_root / "shared" / "checksums",
        data_root / "shared" / "sample_archives",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def safe_symlink(target: Path, link: Path) -> None:
    target = target.resolve()
    if link.exists() or link.is_symlink():
        if link.is_symlink() and link.resolve() == target:
            return
        raise FileExistsError(f"Refusing to overwrite existing path: {link}")
    link.symlink_to(target, target_is_directory=True)


def link_legacy_output_roots(root: Path, data_root: Path) -> None:
    legacy_dir = legacy_snapgen_dir(root)
    legacy_dir.mkdir(parents=True, exist_ok=True)
    ensure_data_layout(data_root)
    links = {
        "THERMO_SNAPSHOTS": baseline_root(data_root),
        "THERMO_SNAPSHOTS_ADDON_GAPFILL_DOUBLE": addon_root(data_root),
        "THERMO_SNAPSHOTS_MERGED_DOUBLE": merged_root(data_root),
        "RUN_LOGS": run_logs_root(data_root),
    }
    for name, target in links.items():
        safe_symlink(target, legacy_dir / name)


def parse_unknown_flags(flags: Sequence[str]) -> List[str]:
    return [str(x) for x in flags]


def command_setup(args: argparse.Namespace) -> int:
    root = repo_root()
    data_root = Path(args.data_root).expanduser().resolve() if args.data_root else default_data_root(root)
    ensure_data_layout(data_root)
    if args.link_legacy:
        link_legacy_output_roots(root, data_root)
    print(f"Data root prepared: {data_root}")
    if args.link_legacy:
        print(f"Legacy output symlinks prepared in: {legacy_snapgen_dir(root)}")
    return 0


def command_list_cases(args: argparse.Namespace) -> int:
    root = repo_root()
    return run_legacy(root, "THERMOMECHANICAL_SNAPGEN_CASES.py", ["--list-cases"])


def command_compile_check(args: argparse.Namespace) -> int:
    root = repo_root()
    scripts = [
        "THERMOMECHANICAL_FOM_ROM.py",
        "THERMOMECHANICAL_SNAPGEN_COMMON.py",
        "THERMOMECHANICAL_SNAPGEN_CASES.py",
        "RUN_SNAPGEN_MONOLITHIC_FREE_EDGE_PATCH_MPTP_COUNTS.py",
        "THERMO_SNAPGEN_BASELINE_AWARE_ADDON.py",
        "SNAPGEN_STATUS_ANY.py",
        "PAPER2_BYPASS_BEFORE_SNAPSHOT_PROJECTION.py",
    ]
    for s in scripts:
        rc = run_legacy(root, s, ["--help"] if s == "THERMOMECHANICAL_SNAPGEN_CASES.py" else [])
        # Most legacy scripts have main side effects; for a true syntax check use py_compile.
        break
    cmd = [sys.executable, "-m", "py_compile", *[str(legacy_snapgen_dir(root) / s) for s in scripts]]
    print("[snapgen] cmd:", " ".join(cmd))
    return subprocess.call(cmd, cwd=str(legacy_snapgen_dir(root)), env=env_for_subprocess(root))


def command_run_case(args: argparse.Namespace, unknown: Sequence[str]) -> int:
    root = repo_root()
    data_root = Path(args.data_root).expanduser().resolve() if args.data_root else default_data_root(root)
    output_root = Path(args.output_root).expanduser().resolve() if args.output_root else baseline_root(data_root)
    cmd = [
        "--case", str(args.case),
        "--panel-type", args.panel_type,
        "--bc-type", args.bc_type,
        "--load-type", args.load_type,
        "--n-snapshots", str(args.n_snapshots),
        "--solve-mode", args.solve_mode,
        "--output-root", str(output_root),
        "--solver-module", "THERMOMECHANICAL_FOM_ROM",
    ]
    if args.resume:
        cmd.append("--resume")
    if args.overwrite:
        cmd.append("--overwrite")
    if args.make_plots:
        cmd.append("--make-plots")
    if args.status:
        cmd.append("--status")
    cmd.extend(parse_unknown_flags(unknown))
    return run_legacy(root, "THERMOMECHANICAL_SNAPGEN_CASES.py", cmd)


def command_run_campaign(args: argparse.Namespace, unknown: Sequence[str]) -> int:
    root = repo_root()
    data_root = Path(args.data_root).expanduser().resolve() if args.data_root else default_data_root(root)
    output_root = Path(args.output_root).expanduser().resolve() if args.output_root else baseline_root(data_root)
    log_root = Path(args.log_root).expanduser().resolve() if args.log_root else run_logs_root(data_root)
    cmd = [
        "--cases", args.cases,
        "--panel-type", args.panel_type,
        "--bc-type", args.bc_type,
        "--load-type", args.load_type,
        "--run-flag", "overwrite" if args.overwrite else "resume",
        "--output-root", str(output_root),
        "--log-root", str(log_root),
        "--solver-module", "THERMOMECHANICAL_FOM_ROM",
    ]
    if args.dry_run:
        cmd.append("--dry-run")
    if not args.make_plots:
        cmd.append("--no-make-plots")
    cmd.extend(parse_unknown_flags(unknown))
    return run_legacy(root, "RUN_SNAPGEN_MONOLITHIC_FREE_EDGE_PATCH_MPTP_COUNTS.py", cmd)


def command_addon(args: argparse.Namespace, unknown: Sequence[str]) -> int:
    root = repo_root()
    data_root = Path(args.data_root).expanduser().resolve() if args.data_root else default_data_root(root)
    cmd = [
        "--mode", args.mode,
        "--cases", args.cases,
        "--baseline-root", str(Path(args.baseline_root).expanduser().resolve() if args.baseline_root else baseline_root(data_root)),
        "--addon-root", str(Path(args.addon_root).expanduser().resolve() if args.addon_root else addon_root(data_root)),
        "--merged-root", str(Path(args.merged_root).expanduser().resolve() if args.merged_root else merged_root(data_root)),
        "--panel-type", args.panel_type,
        "--bc-type", args.bc_type,
        "--load-type", args.load_type,
        "--run-flag", "overwrite" if args.overwrite else "resume",
        "--log-root", str(run_logs_root(data_root)),
        "--solver-module", "THERMOMECHANICAL_FOM_ROM",
    ]
    if args.dry_run:
        cmd.append("--dry-run")
    cmd.extend(parse_unknown_flags(unknown))
    return run_legacy(root, "THERMO_SNAPGEN_BASELINE_AWARE_ADDON.py", cmd)


def command_status(args: argparse.Namespace) -> int:
    root = repo_root()
    # The legacy status script discovers THERMO_SNAPSHOTS* and RUN_LOGS in its CWD.
    # Run setup-data-layout --link-legacy first so those names point outside Git.
    return run_legacy(root, "SNAPGEN_STATUS_ANY.py", [])


def scan_npz(path: Path) -> dict:
    info = {
        "archive_path": str(path),
        "archive_name": path.name,
        "size_bytes": path.stat().st_size,
        "modified_utc": datetime.utcfromtimestamp(path.stat().st_mtime).isoformat(timespec="seconds") + "Z",
    }
    try:
        import numpy as np
        with np.load(path, allow_pickle=True) as z:
            info["keys"] = ",".join(z.files)
            for key in ["parameters", "snapshots", "theta_snapshots", "mech_parameters", "thermal_parameters", "fom_snapshots", "mus"]:
                if key in z.files:
                    arr = z[key]
                    info[f"{key}_shape"] = "x".join(map(str, arr.shape))
    except Exception as exc:
        info["read_error"] = repr(exc)
    return info


def command_manifest(args: argparse.Namespace) -> int:
    root = repo_root()
    data_root = Path(args.data_root).expanduser().resolve() if args.data_root else default_data_root(root)
    ensure_data_layout(data_root)
    patterns = [
        data_root / "paper1_mechanical" / "snapshots",
        data_root / "paper2_thermomechanical",
    ]
    rows = []
    for base in patterns:
        if base.exists():
            for p in sorted(base.rglob("*.npz")):
                if p.name.startswith("sample_") and not args.include_tmp:
                    continue
                rows.append(scan_npz(p))
    out_csv = data_root / "shared" / "manifests" / "snapshot_manifest.csv"
    out_json = data_root / "shared" / "manifests" / "snapshot_manifest.json"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for r in rows for k in r.keys()})
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    out_json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Manifest rows: {len(rows)}")
    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_json}")
    return 0


def command_paper1_compat(args: argparse.Namespace) -> int:
    root = repo_root()
    print("Paper-1 compatibility bypass is preserved here:")
    print(legacy_snapgen_dir(root) / "PAPER2_BYPASS_BEFORE_SNAPSHOT_PROJECTION.py")
    print()
    print("Use it only when running old Paper-1 ROM code against Project-2 archives.")
    print("For the new repo, prefer direct Project-2 archive readers instead of this compatibility path.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unified Project-2 thermomechanical SNAPGEN wrapper.")
    sub = p.add_subparsers(dest="command", required=True)

    setup = sub.add_parser("setup-data-layout", help="Create external data directories and optional legacy symlinks.")
    setup.add_argument("--data-root", default="")
    setup.add_argument("--link-legacy", action="store_true")
    setup.set_defaults(func=command_setup)

    lc = sub.add_parser("list-cases", help="List the 10 Project-2 physical case definitions.")
    lc.set_defaults(func=command_list_cases)

    cc = sub.add_parser("compile-check", help="Syntax-check preserved legacy SNAPGEN scripts.")
    cc.set_defaults(func=command_compile_check)

    rc = sub.add_parser("run-case", help="Run one Project-2 snapshot case.")
    rc.add_argument("--case", type=int, required=True)
    rc.add_argument("--n-snapshots", type=int, default=2)
    rc.add_argument("--panel-type", default="monolithic", choices=["monolithic", "contact1", "contact2", "auto"])
    rc.add_argument("--bc-type", default="free_edge", choices=["free_edge", "simply_supported"])
    rc.add_argument("--load-type", default="patch", choices=["patch", "linear", "multi_patch", "uniform"])
    rc.add_argument("--solve-mode", default="coupled", choices=["coupled", "one_way", "mechanical"])
    rc.add_argument("--data-root", default="")
    rc.add_argument("--output-root", default="")
    rc.add_argument("--resume", action="store_true")
    rc.add_argument("--overwrite", action="store_true")
    rc.add_argument("--make-plots", action="store_true")
    rc.add_argument("--status", action="store_true")
    rc.set_defaults(func=command_run_case)

    camp = sub.add_parser("run-campaign", help="Run sequential official/selected Project-2 cases.")
    camp.add_argument("--cases", default="1 2 3 4 5 6 7 8 9 10")
    camp.add_argument("--panel-type", default="monolithic", choices=["monolithic", "contact1", "contact2", "auto"])
    camp.add_argument("--bc-type", default="free_edge", choices=["free_edge", "simply_supported"])
    camp.add_argument("--load-type", default="patch", choices=["patch", "linear", "multi_patch", "uniform"])
    camp.add_argument("--data-root", default="")
    camp.add_argument("--output-root", default="")
    camp.add_argument("--log-root", default="")
    camp.add_argument("--resume", action="store_true")
    camp.add_argument("--overwrite", action="store_true")
    camp.add_argument("--make-plots", action="store_true", default=True)
    camp.add_argument("--no-make-plots", action="store_false", dest="make_plots")
    camp.add_argument("--dry-run", action="store_true")
    camp.set_defaults(func=command_run_campaign)

    addon = sub.add_parser("addon", help="Plan/run/merge baseline-aware addon snapshots.")
    addon.add_argument("--mode", default="plan", choices=["plan", "generate-parameters", "run-addon", "merge", "all"])
    addon.add_argument("--cases", default="1 2 3 4 5 6 7 8 9 10")
    addon.add_argument("--panel-type", default="monolithic", choices=["monolithic", "contact1", "contact2", "auto"])
    addon.add_argument("--bc-type", default="free_edge", choices=["free_edge", "simply_supported"])
    addon.add_argument("--load-type", default="patch", choices=["patch", "linear", "multi_patch", "uniform"])
    addon.add_argument("--data-root", default="")
    addon.add_argument("--baseline-root", default="")
    addon.add_argument("--addon-root", default="")
    addon.add_argument("--merged-root", default="")
    addon.add_argument("--resume", action="store_true")
    addon.add_argument("--overwrite", action="store_true")
    addon.add_argument("--dry-run", action="store_true")
    addon.set_defaults(func=command_addon)

    st = sub.add_parser("status", help="Run the preserved universal status checker.")
    st.set_defaults(func=command_status)

    mf = sub.add_parser("manifest", help="Create a manifest of external Paper-1/Paper-2 snapshot archives.")
    mf.add_argument("--data-root", default="")
    mf.add_argument("--include-tmp", action="store_true")
    mf.set_defaults(func=command_manifest)

    pc = sub.add_parser("paper1-compat", help="Locate the preserved Paper-1 compatibility bypass.")
    pc.set_defaults(func=command_paper1_compat)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args, unknown = parser.parse_known_args(argv)
    if args.command in {"run-case", "run-campaign", "addon"}:
        return args.func(args, unknown)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
