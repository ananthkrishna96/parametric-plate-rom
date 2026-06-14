#!/usr/bin/env python3
from __future__ import annotations

import os
import re
import csv
import json
import shlex
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

SNAPGEN_DIR = Path.cwd()

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

PANEL_TO_NVNH = {
    "monolithic": (0, 0),
    "contact1": (1, 0),
    "contact2": (2, 0),
}

def line(ch="=", n=100):
    print(ch * n)

def title(text: str):
    print()
    line("=")
    print(text)
    line("=")

def section(text: str):
    print()
    print(text)
    print("-" * 100)

def fmt_time(seconds: Optional[float]) -> str:
    if seconds is None:
        return "unknown"
    try:
        s = float(seconds)
    except Exception:
        return "unknown"
    if s < 0 or s != s:
        return "unknown"
    if s < 60:
        return f"{s:.1f}s"
    m = s / 60.0
    if m < 60:
        return f"{m:.1f}min"
    h = m / 60.0
    if h < 48:
        return f"{h:.2f}h"
    d = h / 24.0
    return f"{d:.2f}d"

def run_cmd(cmd: List[str]) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return ""

def split_cmd(cmd: str) -> List[str]:
    try:
        return shlex.split(cmd)
    except Exception:
        return cmd.split()

def opt(tokens: List[str], name: str, default: Optional[str] = None) -> Optional[str]:
    flag = f"--{name}"
    for i, t in enumerate(tokens):
        if t == flag and i + 1 < len(tokens):
            return tokens[i + 1]
        if t.startswith(flag + "="):
            return t.split("=", 1)[1]
    return default

def has_flag(tokens: List[str], name: str) -> bool:
    return f"--{name}" in tokens

def parse_cases(text: Optional[str]) -> List[int]:
    if not text:
        return list(range(1, 11))
    out = []
    for x in re.split(r"[,\s]+", text.strip()):
        if not x:
            continue
        try:
            c = int(x)
            if 1 <= c <= 10:
                out.append(c)
        except Exception:
            pass
    return out or list(range(1, 11))

def safe_token(value: str) -> str:
    txt = str(value).strip().upper()
    out = []
    for ch in txt:
        if ch.isalnum():
            out.append(ch)
        elif ch in "_-":
            out.append("_")
        else:
            out.append("_")
    token = "".join(out)
    while "__" in token:
        token = token.replace("__", "_")
    return token.strip("_")

def panel_label(nv: int, nh: int) -> str:
    nv, nh = int(nv), int(nh)
    if nv == 0 and nh == 0:
        return "MONOLITHIC_NV0_NH0"
    if nv == 1 and nh == 0:
        return "CONTACT_NV1_NH0"
    if nv == 2 and nh == 0:
        return "CONTACT_NV2_NH0"
    return f"CUSTOM_NV{nv}_NH{nh}"

def case_dir_name(case_id: int, panel_type: str, nv: int, nh: int, bc: str, load: str) -> str:
    base = CASE_NAMES[int(case_id)]
    return "__".join([
        safe_token(base),
        "PANEL_" + panel_label(nv, nh),
        "BC_" + safe_token(bc),
        "LOAD_" + safe_token(load),
    ])

def count_tmp(outdir: Path) -> Tuple[int, int]:
    tmp = outdir / "tmp"
    if not tmp.exists():
        return 0, 0
    done = len(list(tmp.glob("sample_*.npz")))
    fail = len(list(tmp.glob("sample_*_FAILED.json")))
    return done, fail

def final_archive_exists(outdir: Path) -> bool:
    return bool(list(outdir.glob("SNAPSHOTS__*.npz"))) or (outdir / "snapshots.npz").exists()

def read_status_json(outdir: Path) -> Dict[str, Any]:
    path = outdir / "_STATUS.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}

def rows_from_summary(outdir: Path) -> List[Dict[str, str]]:
    path = outdir / "summary.csv"
    if not path.exists():
        return []
    try:
        with path.open("r", newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []

def estimate_avg_time(outdir: Path) -> Optional[float]:
    st = read_status_json(outdir)
    if st.get("avg_sample_time_sec") is not None:
        try:
            return float(st["avg_sample_time_sec"])
        except Exception:
            pass

    rows = rows_from_summary(outdir)
    vals = []
    for r in rows:
        try:
            v = float(r.get("wall_time_sec", "nan"))
            if v == v and v > 0:
                vals.append(v)
        except Exception:
            pass
    if vals:
        return sum(vals) / len(vals)

    tmp = outdir / "tmp"
    vals = []
    if tmp.exists():
        for fp in sorted(tmp.glob("sample_*.npz"))[-20:]:
            try:
                import numpy as np
                data = np.load(fp, allow_pickle=True)
                if "metadata_json" in data.files:
                    meta = json.loads(str(data["metadata_json"].item()))
                    v = float(meta.get("wall_time_sec", 0.0))
                    if v > 0:
                        vals.append(v)
            except Exception:
                pass
    if vals:
        return sum(vals) / len(vals)

    return None

def detect_processes() -> List[Dict[str, Any]]:
    ps = run_cmd(["ps", "-u", os.environ.get("USER", ""), "-o", "pid=,ppid=,etime=,%cpu=,%mem=,command="])
    procs = []
    for line0 in ps.splitlines():
        line = line0.strip()
        if not line:
            continue
        if "grep" in line:
            continue
        if not any(k in line for k in [
            "RUN_SNAPGEN",
            "THERMOMECHANICAL_SNAPGEN_CASES.py",
            "THERMO_SNAPGEN_BASELINE_AWARE_ADDON.py",
        ]):
            continue
        parts = line.split(None, 5)
        if len(parts) < 6:
            continue
        pid, ppid, etime, cpu, mem, cmd = parts
        procs.append(dict(pid=pid, ppid=ppid, etime=etime, cpu=cpu, mem=mem, cmd=cmd, tokens=split_cmd(cmd)))
    return procs

def infer_run_from_process(proc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    tokens = proc["tokens"]
    cmd = proc["cmd"]

    if "THERMOMECHANICAL_SNAPGEN_CASES.py" in cmd:
        case = opt(tokens, "case")
        if not case:
            return None

        panel = opt(tokens, "panel-type", "auto") or "auto"
        nv = int(opt(tokens, "n-vert", "0") or 0)
        nh = int(opt(tokens, "n-horiz", "0") or 0)
        if panel in PANEL_TO_NVNH:
            nv, nh = PANEL_TO_NVNH[panel]
        elif panel == "auto":
            if nv == 0 and nh == 0:
                panel = "monolithic"
            elif nv == 1 and nh == 0:
                panel = "contact1"
            elif nv == 2 and nh == 0:
                panel = "contact2"

        return dict(
            source="direct_child_case",
            pid=proc["pid"],
            ppid=proc["ppid"],
            etime=proc["etime"],
            cases=[int(case)],
            current_case=int(case),
            n_override=int(opt(tokens, "n-snapshots", "0") or 0),
            panel_type=panel,
            n_vert=nv,
            n_horiz=nh,
            bc_type=opt(tokens, "bc-type", "free_edge") or "free_edge",
            load_type=opt(tokens, "load-type", "patch") or "patch",
            output_root=opt(tokens, "output-root", "THERMO_SNAPSHOTS") or "THERMO_SNAPSHOTS",
            run_flag="overwrite" if has_flag(tokens, "overwrite") else "resume",
        )

    if "RUN_SNAPGEN" in cmd:
        panel = opt(tokens, "panel-type", "monolithic") or "monolithic"
        nv = int(opt(tokens, "n-vert", "0") or 0)
        nh = int(opt(tokens, "n-horiz", "0") or 0)
        if panel in PANEL_TO_NVNH:
            nv, nh = PANEL_TO_NVNH[panel]

        n_override = int(opt(tokens, "n-snapshots", "0") or 0)

        return dict(
            source="campaign_runner",
            pid=proc["pid"],
            ppid=proc["ppid"],
            etime=proc["etime"],
            cases=parse_cases(opt(tokens, "cases", None)),
            current_case=None,
            n_override=n_override,
            panel_type=panel,
            n_vert=nv,
            n_horiz=nh,
            bc_type=opt(tokens, "bc-type", "free_edge") or "free_edge",
            load_type=opt(tokens, "load-type", "patch") or "patch",
            output_root=opt(tokens, "output-root", "THERMO_SNAPSHOTS") or "THERMO_SNAPSHOTS",
            run_flag=opt(tokens, "run-flag", "resume") or "resume",
        )

    if "THERMO_SNAPGEN_BASELINE_AWARE_ADDON.py" in cmd:
        panel = opt(tokens, "panel-type", "monolithic") or "monolithic"
        nv = int(opt(tokens, "n-vert", "0") or 0)
        nh = int(opt(tokens, "n-horiz", "0") or 0)
        if panel in PANEL_TO_NVNH:
            nv, nh = PANEL_TO_NVNH[panel]

        return dict(
            source="baseline_aware_addon_runner",
            pid=proc["pid"],
            ppid=proc["ppid"],
            etime=proc["etime"],
            cases=parse_cases(opt(tokens, "cases", None)),
            current_case=None,
            n_override=0,
            panel_type=panel,
            n_vert=nv,
            n_horiz=nh,
            bc_type=opt(tokens, "bc-type", "free_edge") or "free_edge",
            load_type=opt(tokens, "load-type", "patch") or "patch",
            output_root=opt(tokens, "addon-root", "THERMO_SNAPSHOTS_ADDON_GAPFILL_DOUBLE") or "THERMO_SNAPSHOTS_ADDON_GAPFILL_DOUBLE",
            baseline_root=opt(tokens, "baseline-root", "THERMO_SNAPSHOTS") or "THERMO_SNAPSHOTS",
            merged_root=opt(tokens, "merged-root", "THERMO_SNAPSHOTS_MERGED_DOUBLE") or "THERMO_SNAPSHOTS_MERGED_DOUBLE",
            addon_factor=float(opt(tokens, "addon-factor", "1.0") or 1.0),
            addon_counts=opt(tokens, "addon-counts", "") or "",
            addon_n_snapshots=int(opt(tokens, "addon-n-snapshots", "-1") or -1),
            run_flag=opt(tokens, "run-flag", "resume") or "resume",
        )

    return None

def addon_count_for_case(run: Dict[str, Any], c: int) -> int:
    text = run.get("addon_counts", "")
    if text:
        for part in re.split(r"[,\s]+", text.strip()):
            if not part or ":" not in part:
                continue
            k, v = part.split(":", 1)
            try:
                if int(k) == c:
                    return int(v)
            except Exception:
                pass
    if int(run.get("addon_n_snapshots", -1)) >= 0:
        return int(run["addon_n_snapshots"])
    return int(round(float(run.get("addon_factor", 1.0)) * DEFAULT_COUNTS[c]))

def target_for_case(run: Dict[str, Any], c: int) -> int:
    if run["source"] == "baseline_aware_addon_runner":
        return addon_count_for_case(run, c)
    n_override = int(run.get("n_override", 0) or 0)
    if n_override > 0:
        return n_override
    return DEFAULT_COUNTS[c]

def case_progress(run: Dict[str, Any], c: int) -> Dict[str, Any]:
    target = target_for_case(run, c)
    name = case_dir_name(
        c,
        str(run["panel_type"]),
        int(run["n_vert"]),
        int(run["n_horiz"]),
        str(run["bc_type"]),
        str(run["load_type"]),
    )
    outdir = Path(str(run["output_root"])) / name
    done, fail = count_tmp(outdir)
    archive = final_archive_exists(outdir)
    avg = estimate_avg_time(outdir)
    remaining = max(0, target - done - fail)
    eta = None if avg is None else remaining * avg

    if archive:
        state = "COMPLETED"
    elif done > 0:
        state = "RUNNING/PARTIAL"
    elif outdir.exists():
        state = "CREATED"
    else:
        state = "NOT_STARTED"

    return dict(
        case=c,
        name=CASE_NAMES[c],
        target=target,
        done=done,
        fail=fail,
        remaining=remaining,
        archive=archive,
        state=state,
        avg=avg,
        eta=eta,
        outdir=outdir,
        dir_name=name,
    )

def print_active_processes(procs: List[Dict[str, Any]]) -> None:
    section("1) ACTIVE SNAPGEN PROCESSES")
    if not procs:
        print("No active SNAPGEN process found.")
        return
    print(f"{'PID':>8} {'PPID':>8} {'ELAPSED':>12} {'CPU%':>7} {'MEM%':>7}  COMMAND")
    print("-" * 100)
    for p in procs:
        cmd_short = p["cmd"]
        if len(cmd_short) > 150:
            cmd_short = cmd_short[:147] + "..."
        print(f"{p['pid']:>8} {p['ppid']:>8} {p['etime']:>12} {p['cpu']:>7} {p['mem']:>7}  {cmd_short}")

def choose_main_run(runs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not runs:
        return None
    for src in ["campaign_runner", "baseline_aware_addon_runner", "direct_child_case"]:
        for r in runs:
            if r["source"] == src:
                return r
    return runs[0]

def print_current_child_detail(runs: List[Dict[str, Any]]) -> None:
    section("2) CURRENT RUNNING CASE")
    child = None
    for r in runs:
        if r["source"] == "direct_child_case":
            child = r
            break
    if not child:
        print("No active THERMOMECHANICAL_SNAPGEN_CASES.py child process detected.")
        return

    c = int(child["current_case"])
    info = case_progress(child, c)

    print(f"Detected active case : CASE {c:02d} | {info['name']}")
    print(f"Panel / BC / load    : {child['panel_type']}  /  {child['bc_type']}  /  {child['load_type']}")
    print(f"Output root          : {child['output_root']}")
    print(f"Output folder        : {info['outdir']}")
    print()
    print(f"Progress             : {info['done']}/{info['target']} done | failed={info['fail']} | remaining={info['remaining']}")
    print(f"Average per sample   : {fmt_time(info['avg'])}")
    print(f"ETA current case     : {fmt_time(info['eta'])}")
    print(f"Final archive        : {'YES' if info['archive'] else 'NO'}")
    print(f"State                : {info['state']}")

    st = read_status_json(info["outdir"])
    latest = st.get("latest_completed")
    if latest:
        print()
        print("Latest completed:")
        print(f"  sample index        : {int(latest.get('index', -1)) + 1}")
        print(f"  converged           : {latest.get('converged')}")
        print(f"  coupled iterations  : {latest.get('coupled_iterations')}")
        print(f"  err_w               : {latest.get('err_w')}")
        print(f"  err_theta           : {latest.get('err_theta')}")
        print(f"  wall time           : {fmt_time(latest.get('wall_time_sec'))}")

def print_run_summary(run: Optional[Dict[str, Any]]) -> None:
    section("3) DETECTED RUN SUMMARY AND ETA")
    if not run:
        print("No active run could be inferred.")
        return

    print(f"Run type             : {run['source']}")
    print(f"Runner PID           : {run['pid']}")
    print(f"Elapsed              : {run['etime']}")
    print(f"Cases requested      : {' '.join(str(c) for c in run['cases'])}")
    print(f"Panel / BC / load    : {run['panel_type']}  /  {run['bc_type']}  /  {run['load_type']}")
    print(f"Output root          : {run['output_root']}")
    print(f"Run flag             : {run.get('run_flag', 'unknown')}")

    rows = [case_progress(run, c) for c in run["cases"]]

    known_avgs = [r["avg"] for r in rows if r["avg"] is not None]
    global_avg = sum(known_avgs) / len(known_avgs) if known_avgs else None

    total_target = sum(r["target"] for r in rows)
    total_done = sum(r["done"] for r in rows)
    total_fail = sum(r["fail"] for r in rows)
    total_remaining = sum(r["remaining"] for r in rows)

    eta_total = 0.0
    eta_known = False
    for r in rows:
        avg = r["avg"] if r["avg"] is not None else global_avg
        if avg is not None:
            eta_total += r["remaining"] * avg
            eta_known = True

    print()
    print(f"Total progress       : {total_done}/{total_target} done | failed={total_fail} | remaining={total_remaining}")
    print(f"Estimated run ETA    : {fmt_time(eta_total if eta_known else None)}")
    print(f"Average basis        : {'case-specific + global fallback' if eta_known else 'unknown'}")

    print()
    print(f"{'CASE':>6} {'TARGET':>8} {'DONE':>8} {'FAIL':>6} {'REM':>8} {'AVG/SAMPLE':>14} {'ETA':>12} {'ARCHIVE':>9}  STATE")
    print("-" * 100)
    for r in rows:
        avg = r["avg"] if r["avg"] is not None else global_avg
        eta = r["remaining"] * avg if avg is not None else None
        print(
            f"{r['case']:>6} {r['target']:>8} {r['done']:>8} {r['fail']:>6} {r['remaining']:>8} "
            f"{fmt_time(avg):>14} {fmt_time(eta):>12} {('YES' if r['archive'] else 'NO'):>9}  {r['state']}"
        )

def print_all_partial_roots() -> None:
    section("4) ALL SNAPSHOT OUTPUT FOLDERS FOUND")
    roots = sorted([p for p in Path(".").glob("THERMO_SNAPSHOTS*") if p.is_dir()])
    if not roots:
        print("No THERMO_SNAPSHOTS* roots found.")
        return

    all_dirs = []
    for root in roots:
        for d in sorted(root.iterdir()):
            if d.is_dir() and d.name.startswith("CASE_"):
                all_dirs.append(d)

    if not all_dirs:
        print("No case folders found under THERMO_SNAPSHOTS* roots.")
        return

    print(f"{'ROOT':<36} {'CASE FOLDER':<68} {'DONE':>6} {'FAIL':>5} {'ARCH':>5}")
    print("-" * 125)
    for d in all_dirs:
        done, fail = count_tmp(d)
        arch = "YES" if final_archive_exists(d) else "NO"
        root = str(d.parent)
        folder = d.name
        if len(root) > 35:
            root = root[:32] + "..."
        if len(folder) > 67:
            folder = folder[:64] + "..."
        print(f"{root:<36} {folder:<68} {done:>6} {fail:>5} {arch:>5}")

def print_latest_logs() -> None:
    section("5) LATEST LOGS")
    logs = []
    logroot = Path("RUN_LOGS")
    if logroot.exists():
        logs = sorted(logroot.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
        nohuplogs = sorted(logroot.glob("NOHUP*.out"), key=lambda p: p.stat().st_mtime, reverse=True)
        logs = logs[:3] + nohuplogs[:2]

    if not logs:
        print("No RUN_LOGS/*.log or RUN_LOGS/NOHUP*.out found.")
        return

    for fp in logs[:5]:
        print()
        print(f"--- {fp} ---")
        try:
            lines = fp.read_text(errors="replace").splitlines()[-8:]
            for l in lines:
                print(l)
        except Exception as exc:
            print(f"Could not read log: {exc}")

def main():
    title("SNAPGEN UNIVERSAL STATUS CHECK")
    print(f"Time          : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Working dir   : {SNAPGEN_DIR}")
    print(f"User          : {os.environ.get('USER', 'unknown')}")

    procs = detect_processes()
    runs = []
    for p in procs:
        r = infer_run_from_process(p)
        if r:
            runs.append(r)

    print_active_processes(procs)
    print_current_child_detail(runs)
    main_run = choose_main_run(runs)
    print_run_summary(main_run)
    print_all_partial_roots()
    print_latest_logs()

    print()
    line("=")
    print("STATUS CHECK FINISHED")
    line("=")

if __name__ == "__main__":
    main()
