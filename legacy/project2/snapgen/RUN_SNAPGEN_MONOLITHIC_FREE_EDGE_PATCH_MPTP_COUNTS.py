#!/usr/bin/env python3
"""
Sequential log-based thermomechanical snapshot campaign runner.

Purpose
-------
Run the 10 thermomechanical MP/TP cases with fixed campaign settings:

    panel type  : monolithic
    n_vert      : 0
    n_horiz     : 0
    BC          : free_edge
    load        : patch
    snapshots   : MP/TP-dependent counts
                  0MP1TP=100, 1MP1TP=150, 1MP2TP=150, 2MP2TP=200, 2MP3TP=200
    solve mode  : coupled

The runner writes one campaign log and one log per case.  It is intended to be
launched with nohup so the terminal can be closed/free while the campaign keeps
running.

Recommended first clean official run after a 1-snapshot test:

    nohup python -u RUN_SNAPGEN_MONOLITHIC_FREE_EDGE_PATCH_MPTP_COUNTS.py \
      --run-flag overwrite \
      > RUN_LOGS/NOHUP_MONOLITHIC_FREE_EDGE_PATCH_MPTP_COUNTS.out 2>&1 &

If the run is interrupted, resume with:

    nohup python -u RUN_SNAPGEN_MONOLITHIC_FREE_EDGE_PATCH_MPTP_COUNTS.py \
      --run-flag resume \
      > RUN_LOGS/NOHUP_MONOLITHIC_FREE_EDGE_PATCH_MPTP_COUNTS_RESUME.out 2>&1 &
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, List


# -----------------------------------------------------------------------------
# Default campaign settings
# -----------------------------------------------------------------------------
DEFAULT_CASES = list(range(1, 11))
# MP/TP-dependent official snapshot counts.
# Case map:
#   CASE 01,06 : 0MP1TP -> 100
#   CASE 02,07 : 1MP1TP -> 150
#   CASE 03,08 : 1MP2TP -> 150
#   CASE 04,09 : 2MP2TP -> 200
#   CASE 05,10 : 2MP3TP -> 200
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
DEFAULT_N_SNAPSHOTS = 0  # 0 means: use DEFAULT_N_SNAPSHOTS_BY_CASE.

DEFAULT_PANEL_TYPE = "monolithic"
DEFAULT_N_VERT = 0
DEFAULT_N_HORIZ = 0
DEFAULT_BC_TYPE = "free_edge"
DEFAULT_LOAD_TYPE = "patch"

DEFAULT_SEED = 100
DEFAULT_PLATE_RESOLUTION = 64
DEFAULT_PLATE_DEGREE = 2
DEFAULT_HEAT_NX = 64
DEFAULT_HEAT_NY = 32
DEFAULT_HEAT_NZ = 16
DEFAULT_HEAT_DEGREE = 1
DEFAULT_THETA_CG_DEGREE = 1
DEFAULT_THETA_QUADRATURE = 20

DEFAULT_SOLVE_MODE = "coupled"
DEFAULT_OMEGA = 0.7
DEFAULT_TOL_W = "1e-5"
DEFAULT_TOL_THETA = "1e-5"
DEFAULT_MAX_COUPLING_ITERS = 25

DEFAULT_OUTPUT_ROOT = "THERMO_SNAPSHOTS"
DEFAULT_SOLVER_MODULE = "THERMOMECHANICAL_FOM_ROM"
DEFAULT_SAMPLING = "lhs"

SNAPGEN_SCRIPT = "THERMOMECHANICAL_SNAPGEN_CASES.py"
FOM_FILE = "THERMOMECHANICAL_FOM_ROM.py"
COMMON_FILE = "THERMOMECHANICAL_SNAPGEN_COMMON.py"


# -----------------------------------------------------------------------------
# Logging helpers
# -----------------------------------------------------------------------------

def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def write_line(path: Path, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", buffering=1) as f:
        f.write(text + "\n")


def banner(title: str) -> str:
    line = "=" * 96
    return f"{line}\n{title}\n{line}"


def run_logged(cmd: List[str], *, cwd: Path, case_log: Path, campaign_log: Path, env: dict) -> int:
    """Run a command and stream stdout/stderr into both logs and current stdout."""
    pretty = " ".join(shlex.quote(x) for x in cmd)
    header = f"\n[{now()}] COMMAND:\n{pretty}\n"
    write_line(case_log, header)
    write_line(campaign_log, header)
    print(header, flush=True)

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
        write_line(case_log, line)
        write_line(campaign_log, line)
        print(line, flush=True)

    return proc.wait()


def run_quiet_check(cmd: List[str], *, cwd: Path, campaign_log: Path, env: dict) -> int:
    pretty = " ".join(shlex.quote(x) for x in cmd)
    write_line(campaign_log, f"\n[{now()}] CHECK COMMAND: {pretty}")
    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if proc.stdout:
        for line in proc.stdout.rstrip("\n").splitlines():
            write_line(campaign_log, line)
    return int(proc.returncode)


# -----------------------------------------------------------------------------
# CLI and command construction
# -----------------------------------------------------------------------------

def parse_cases(text: str) -> List[int]:
    raw = str(text).replace(",", " ").split()
    cases = [int(x) for x in raw]
    for c in cases:
        if c < 1 or c > 10:
            raise ValueError(f"Invalid case {c}. Use cases 1..10.")
    return cases


def snapshots_for_case(args: argparse.Namespace, case_id: int) -> int:
    """Return the number of snapshots for this case.

    Default behavior uses the official MP/TP-dependent snapshot counts:
        0MP1TP -> 100, 1MP1TP -> 150, 1MP2TP -> 150,
        2MP2TP -> 200, 2MP3TP -> 200.

    If --n-snapshots is provided with a positive value, that value overrides
    the MP/TP schedule for all selected cases.
    """
    n_user = int(getattr(args, "n_snapshots", 0))
    if n_user > 0:
        return n_user
    return int(DEFAULT_N_SNAPSHOTS_BY_CASE[int(case_id)])


def snapshot_schedule_text(cases: Iterable[int]) -> str:
    parts = []
    for c in cases:
        parts.append(f"CASE {int(c)} -> {DEFAULT_N_SNAPSHOTS_BY_CASE[int(c)]}")
    return ", ".join(parts)


def build_case_command(args: argparse.Namespace, case_id: int) -> List[str]:
    run_flag = "--" + args.run_flag.strip().lstrip("-")
    if run_flag not in ("--resume", "--overwrite"):
        raise ValueError("--run-flag must be either 'resume' or 'overwrite'.")

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
        "--n-snapshots", str(snapshots_for_case(args, case_id)),
        "--seed", str(args.seed),
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
        "--output-root", args.output_root,
        "--solver-module", args.solver_module,
        "--sampling", args.sampling,
        run_flag,
    ]

    if args.make_plots:
        cmd.append("--make-plots")
    if args.save_field_plots:
        cmd.append("--save-field-plots")
    if args.strict:
        cmd.append("--strict")

    return cmd


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run the monolithic/free-edge/patch 10-case thermomechanical snapshot campaign with logs."
    )
    p.add_argument("--cases", type=str, default="1 2 3 4 5 6 7 8 9 10", help="Case list, e.g. '1 2 3' or '1,2,3'.")
    p.add_argument(
        "--n-snapshots",
        type=int,
        default=DEFAULT_N_SNAPSHOTS,
        help=(
            "Snapshot override for every selected case. "
            "Default 0 uses the MP/TP schedule: "
            "0MP1TP=100, 1MP1TP=150, 1MP2TP=150, 2MP2TP=200, 2MP3TP=200."
        ),
    )

    p.add_argument("--panel-type", type=str, default=DEFAULT_PANEL_TYPE, choices=["monolithic", "contact1", "contact2", "auto"])
    p.add_argument("--n-vert", type=int, default=DEFAULT_N_VERT)
    p.add_argument("--n-horiz", type=int, default=DEFAULT_N_HORIZ)
    p.add_argument("--bc-type", type=str, default=DEFAULT_BC_TYPE, choices=["free_edge", "simply_supported"])
    p.add_argument("--load-type", type=str, default=DEFAULT_LOAD_TYPE, choices=["patch", "linear", "multi_patch", "uniform"])

    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--plate-resolution", type=int, default=DEFAULT_PLATE_RESOLUTION)
    p.add_argument("--plate-degree", type=int, default=DEFAULT_PLATE_DEGREE)
    p.add_argument("--heat-nx", type=int, default=DEFAULT_HEAT_NX)
    p.add_argument("--heat-ny", type=int, default=DEFAULT_HEAT_NY)
    p.add_argument("--heat-nz", type=int, default=DEFAULT_HEAT_NZ)
    p.add_argument("--heat-degree", type=int, default=DEFAULT_HEAT_DEGREE)
    p.add_argument("--theta-cg-degree", type=int, default=DEFAULT_THETA_CG_DEGREE)
    p.add_argument("--theta-quadrature", type=int, default=DEFAULT_THETA_QUADRATURE)

    p.add_argument("--solve-mode", type=str, default=DEFAULT_SOLVE_MODE, choices=["coupled", "one_way", "mechanical"])
    p.add_argument("--omega", type=float, default=DEFAULT_OMEGA)
    p.add_argument("--tol-w", type=str, default=DEFAULT_TOL_W)
    p.add_argument("--tol-theta", type=str, default=DEFAULT_TOL_THETA)
    p.add_argument("--max-coupling-iters", type=int, default=DEFAULT_MAX_COUPLING_ITERS)

    p.add_argument("--output-root", type=str, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--solver-module", type=str, default=DEFAULT_SOLVER_MODULE)
    p.add_argument("--sampling", type=str, default=DEFAULT_SAMPLING, choices=["lhs", "random"])

    p.add_argument("--run-flag", type=str, default="resume", choices=["resume", "overwrite"], help="Use overwrite for first clean official run; use resume after interruption.")
    p.add_argument("--log-root", type=str, default="RUN_LOGS")
    p.add_argument("--campaign-name", type=str, default="MONOLITHIC_FREE_EDGE_PATCH_MPTP_COUNTS")
    p.add_argument("--make-plots", action="store_true", default=True)
    p.add_argument("--no-make-plots", action="store_false", dest="make_plots")
    p.add_argument("--save-field-plots", action="store_true", help="Usually keep off to avoid extra files/time.")
    p.add_argument("--strict", action="store_true")
    p.add_argument("--continue-on-error", action="store_true", help="Continue to next case even if one case fails.")
    p.add_argument("--skip-precheck", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    cases = parse_cases(args.cases)

    cwd = Path.cwd().resolve()
    log_root = Path(args.log_root).resolve()
    log_root.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    campaign_log = log_root / f"CAMPAIGN_{args.campaign_name}_{timestamp}.log"

    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("NUMEXPR_NUM_THREADS", "1")
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("PYTHONUNBUFFERED", "1")

    header = banner("SNAPGEN LOG-BASED CAMPAIGN RUNNER START")
    print(header, flush=True)
    write_line(campaign_log, header)
    write_line(campaign_log, f"Started local : {now()}")
    write_line(campaign_log, f"Started UTC   : {utc_now()}")
    write_line(campaign_log, f"Working dir   : {cwd}")
    write_line(campaign_log, f"Python        : {sys.executable}")
    write_line(campaign_log, f"Conda env     : {env.get('CONDA_DEFAULT_ENV', 'unknown')}")
    write_line(campaign_log, f"Cases         : {cases}")
    write_line(campaign_log, f"Panel/BC/load : {args.panel_type} / {args.bc_type} / {args.load_type}")
    if int(args.n_snapshots) > 0:
        write_line(campaign_log, f"n_snapshots   : {args.n_snapshots} for every selected case")
    else:
        write_line(campaign_log, "n_snapshots   : MP/TP-dependent official schedule")
        write_line(campaign_log, f"schedule      : {snapshot_schedule_text(cases)}")
    write_line(campaign_log, f"run_flag      : --{args.run_flag}")
    write_line(campaign_log, f"Thread env    : OMP={env.get('OMP_NUM_THREADS')}, OPENBLAS={env.get('OPENBLAS_NUM_THREADS')}, MKL={env.get('MKL_NUM_THREADS')}, NUMEXPR={env.get('NUMEXPR_NUM_THREADS')}")

    required = [FOM_FILE, COMMON_FILE, SNAPGEN_SCRIPT]
    missing = [f for f in required if not (cwd / f).exists()]
    if missing:
        msg = f"ERROR: missing required files in {cwd}: {missing}"
        print(msg, flush=True)
        write_line(campaign_log, msg)
        return 2

    if not args.skip_precheck:
        checks = [
            [sys.executable, "-m", "py_compile", FOM_FILE],
            [sys.executable, "-m", "py_compile", COMMON_FILE],
            [sys.executable, "-m", "py_compile", SNAPGEN_SCRIPT],
            [sys.executable, SNAPGEN_SCRIPT, "--list-cases"],
        ]
        for chk in checks:
            rc = run_quiet_check(chk, cwd=cwd, campaign_log=campaign_log, env=env)
            if rc != 0:
                msg = f"ERROR: precheck failed with exit code {rc}: {' '.join(chk)}"
                print(msg, flush=True)
                write_line(campaign_log, msg)
                return rc
        write_line(campaign_log, "Prechecks completed successfully.")
        print("Prechecks completed successfully.", flush=True)

    if args.dry_run:
        print("\nDRY RUN COMMANDS:", flush=True)
        write_line(campaign_log, "\nDRY RUN COMMANDS:")
        for case_id in cases:
            cmd = build_case_command(args, case_id)
            txt = " ".join(shlex.quote(x) for x in cmd)
            print(txt, flush=True)
            write_line(campaign_log, txt)
        return 0

    total_start = time.time()
    failed_cases = []

    for case_id in cases:
        n_case = snapshots_for_case(args, case_id)
        case_log = log_root / f"CASE_{case_id}_{args.panel_type}_{args.bc_type}_{args.load_type}_{n_case}SNAPS.log"
        title = banner(f"RUNNING CASE {case_id}")
        print(title, flush=True)
        write_line(campaign_log, title)
        write_line(case_log, title)
        write_line(case_log, f"Started: {now()}")
        write_line(case_log, f"Campaign log: {campaign_log}")
        write_line(case_log, f"Snapshots for this case: {n_case}")

        cmd = build_case_command(args, case_id)
        case_start = time.time()
        rc = run_logged(cmd, cwd=cwd, case_log=case_log, campaign_log=campaign_log, env=env)
        dt = time.time() - case_start

        finish_msg = f"CASE {case_id} finished at {now()} with exit status {rc}; wall_time={dt/3600:.3f} h"
        print(finish_msg, flush=True)
        write_line(case_log, finish_msg)
        write_line(campaign_log, finish_msg)

        if rc != 0:
            failed_cases.append(case_id)
            if not args.continue_on_error:
                msg = f"ERROR: stopping campaign because CASE {case_id} failed. Failed cases: {failed_cases}"
                print(msg, flush=True)
                write_line(campaign_log, msg)
                return rc

    total_dt = time.time() - total_start
    footer = banner("SNAPGEN CAMPAIGN FINISHED")
    print(footer, flush=True)
    write_line(campaign_log, footer)
    write_line(campaign_log, f"Finished local : {now()}")
    write_line(campaign_log, f"Finished UTC   : {utc_now()}")
    write_line(campaign_log, f"Total wall time: {total_dt/3600:.3f} h")
    write_line(campaign_log, f"Failed cases   : {failed_cases}")
    print(f"Campaign log: {campaign_log}", flush=True)
    print(f"Failed cases: {failed_cases}", flush=True)

    return 1 if failed_cases else 0


if __name__ == "__main__":
    raise SystemExit(main())
