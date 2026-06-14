#!/usr/bin/env python3
"""
Single-entry thermomechanical snapshot generator for Project 2.

This version keeps 10 physical MP/TP case types, but lets the user run each case
with the required panel/load/BC variants from the terminal.

Active campaign structure
-------------------------
Physical case types:
    10 cases: 5 summer + 5 winter, with MP/TP parameter ranges.

Panel variants of interest:
    monolithic : n_vert=0, n_horiz=0
    contact1  : n_vert=1, n_horiz=0
    contact2  : n_vert=2, n_horiz=0

Load variants of interest:
    patch
    linear
    multi_patch

Boundary-condition variants:
    free_edge
    simply_supported

The common generator automatically writes the panel type, BC type, and load type
into the output folder and primary final archive filename, for example:

    CASE_01_SUMMER_0MP1TP__PANEL_MONOLITHIC_NV0_NH0__BC_FREE_EDGE__LOAD_PATCH/
        SNAPSHOTS__CASE_01_SUMMER_0MP1TP__PANEL_MONOLITHIC_NV0_NH0__BC_FREE_EDGE__LOAD_PATCH.npz
        snapshots.npz   # compatibility hardlink/copy

Examples
--------
List all physical case types:
    python THERMOMECHANICAL_SNAPGEN_CASES.py --list-cases

Small monolithic patch test:
    python THERMOMECHANICAL_SNAPGEN_CASES.py --case 1 --panel-type monolithic --load-type patch --bc-type free_edge --n-snapshots 2 --solve-mode coupled --make-plots

Contact case with one vertical interface and linear load:
    python THERMOMECHANICAL_SNAPGEN_CASES.py --case 4 --panel-type contact1 --load-type linear --bc-type free_edge --n-snapshots 100 --resume --make-plots

Contact case with two vertical interfaces and multi-patch load:
    python THERMOMECHANICAL_SNAPGEN_CASES.py --case 10 --panel-type contact2 --load-type multi_patch --bc-type simply_supported --n-snapshots 120 --resume --make-plots
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict

from THERMOMECHANICAL_SNAPGEN_COMMON import CaseConfig, ParamRange, run_case_from_cli

THIS_FILE = "THERMOMECHANICAL_SNAPGEN_CASES.py"


# -----------------------------------------------------------------------------
# Nominal physical parameter blocks
# -----------------------------------------------------------------------------

def _base_mech() -> dict:
    """Nominal White-HDPE-like mechanical parameter block."""
    return dict(Dx=6.0e3, Dy=6.0e3, Dxy=2.5e3, Ds=1.8e3, ks=5.0e5, f=-3.0e3)


def _summer_thermal() -> dict:
    """Nominal summer thermal/environmental block."""
    return dict(
        T_amb=305.15,
        T_sub=306.15,
        h_con=10.0,
        eps_r=0.90,
        q_s=255.0,
        h_c_cont=150.0,
        h_c_gap=7.0,
        eta_c=20.0,
        kx=0.40,
        ky=0.40,
        kz=0.40,
        alpha1=1.3e-4,
        alpha2=1.3e-4,
        rho=950.0,
    )


def _winter_thermal() -> dict:
    """Nominal winter thermal/environmental block."""
    return dict(
        T_amb=279.15,
        T_sub=283.15,
        h_con=12.0,
        eps_r=0.90,
        q_s=50.0,
        h_c_cont=150.0,
        h_c_gap=7.0,
        eta_c=20.0,
        kx=0.40,
        ky=0.40,
        kz=0.40,
        alpha1=1.3e-4,
        alpha2=1.3e-4,
        rho=950.0,
    )


def _base_case_kwargs() -> dict:
    """Default geometry/load/solver settings shared by all physical case types.

    These defaults are intentionally neutral.  For the official campaign, choose
    the actual panel/load/BC variant from the terminal using --panel-type,
    --load-type, and --bc-type.
    """
    return dict(
        bc_type="free_edge",
        load_type="patch",
        n_vert=0,
        n_horiz=0,
        plate_resolution=64,
        plate_degree=2,
        heat_nx=64,
        heat_ny=32,
        heat_nz=16,
        heat_degree=1,
        theta_cg_degree=1,
        theta_quadrature=20,
        solve_mode="coupled",
        omega=0.7,
        tol_w=1.0e-5,
        tol_theta=1.0e-5,
        max_coupling_iters=25,
    )


# -----------------------------------------------------------------------------
# Ten physical MP/TP case types
# -----------------------------------------------------------------------------

def build_case_configs() -> Dict[int, CaseConfig]:
    """Return the 10 base thermomechanical snapshot case configurations.

    These are physical parameter-case types only.  The panel/BC/load variant is
    selected at runtime and is written into the output folder/archive name by the
    common generator.
    """
    common = _base_case_kwargs()
    cases = {
        1: CaseConfig(
            case_id=1,
            case_name="CASE_01_SUMMER_0MP1TP",
            filename=THIS_FILE,
            season="summer",
            mp_tp_label="0MP1TP",
            physical_meaning="Summer solar-flux sweep at fixed panel mechanics; isolates absorbed solar heating effect.",
            recommended_n_snapshots=50,
            expected_solver_difficulty="low-moderate",
            nominal_mech=_base_mech(),
            nominal_thermal=_summer_thermal(),
            parameter_ranges=[
                ParamRange("q_s", 100.0, 850.0, unit="W/m^2", scale="linear", role="TP", description="absorbed solar heat flux"),
            ],
            **common,
        ),
        2: CaseConfig(
            case_id=2,
            case_name="CASE_02_SUMMER_1MP1TP",
            filename=THIS_FILE,
            season="summer",
            mp_tp_label="1MP1TP",
            physical_meaning="Summer mechanical load amplitude plus solar input; loaded panel under sun exposure.",
            recommended_n_snapshots=75,
            expected_solver_difficulty="moderate",
            nominal_mech=_base_mech(),
            nominal_thermal=_summer_thermal(),
            parameter_ranges=[
                ParamRange("f", -6500.0, -1500.0, unit="N/m^2", scale="linear", role="MP", description="downward mechanical load amplitude"),
                ParamRange("q_s", 150.0, 850.0, unit="W/m^2", scale="linear", role="TP", description="absorbed solar heat flux"),
            ],
            **common,
        ),
        3: CaseConfig(
            case_id=3,
            case_name="CASE_03_SUMMER_1MP2TP",
            filename=THIS_FILE,
            season="summer",
            mp_tp_label="1MP2TP",
            physical_meaning="Summer support stiffness with warm air/substrate variability.",
            recommended_n_snapshots=75,
            expected_solver_difficulty="moderate",
            nominal_mech=_base_mech(),
            nominal_thermal=_summer_thermal(),
            parameter_ranges=[
                ParamRange("ks", 100000.0, 2000000.0, unit="N/m^3", scale="log", role="MP", description="unilateral Winkler foundation stiffness"),
                ParamRange("T_amb", 298.15, 313.15, unit="K", scale="linear", role="TP", description="ambient air temperature"),
                ParamRange("T_sub", 295.15, 310.15, unit="K", scale="linear", role="TP", description="substrate temperature"),
            ],
            **common,
        ),
        4: CaseConfig(
            case_id=4,
            case_name="CASE_04_SUMMER_2MP2TP",
            filename=THIS_FILE,
            season="summer",
            mp_tp_label="2MP2TP",
            physical_meaning="Summer orthotropy/load interaction with solar and convective exchange.",
            recommended_n_snapshots=100,
            expected_solver_difficulty="moderate-high",
            nominal_mech=_base_mech(),
            nominal_thermal=_summer_thermal(),
            parameter_ranges=[
                ParamRange("D_ratio", 0.5, 2.0, unit="-", scale="linear", role="MP", description="orthotropic stiffness ratio Dx/Dy at fixed D_eff"),
                ParamRange("f", -7000.0, -2000.0, unit="N/m^2", scale="linear", role="MP", description="downward mechanical load amplitude"),
                ParamRange("q_s", 150.0, 850.0, unit="W/m^2", scale="linear", role="TP", description="absorbed solar heat flux"),
                ParamRange("h_con", 5.0, 20.0, unit="W/(m^2 K)", scale="linear", role="TP", description="top convection coefficient"),
            ],
            **common,
        ),
        5: CaseConfig(
            case_id=5,
            case_name="CASE_05_SUMMER_2MP3TP",
            filename=THIS_FILE,
            season="summer",
            mp_tp_label="2MP3TP",
            physical_meaning="Hardest summer coupled contact-gap case: support/load plus solar/contact conductance variations.",
            recommended_n_snapshots=120,
            expected_solver_difficulty="high",
            nominal_mech=_base_mech(),
            nominal_thermal=_summer_thermal(),
            parameter_ranges=[
                ParamRange("ks", 100000.0, 3000000.0, unit="N/m^3", scale="log", role="MP", description="unilateral Winkler foundation stiffness"),
                ParamRange("f", -9000.0, -2000.0, unit="N/m^2", scale="linear", role="MP", description="downward mechanical load amplitude"),
                ParamRange("q_s", 150.0, 850.0, unit="W/m^2", scale="linear", role="TP", description="absorbed solar heat flux"),
                ParamRange("h_c_cont", 75.0, 300.0, unit="W/(m^2 K)", scale="log", role="TP", description="bottom contact conductance"),
                ParamRange("h_c_gap", 3.0, 15.0, unit="W/(m^2 K)", scale="log", role="TP", description="bottom gap conductance"),
            ],
            **common,
        ),
        6: CaseConfig(
            case_id=6,
            case_name="CASE_06_WINTER_0MP1TP",
            filename=THIS_FILE,
            season="winter",
            mp_tp_label="0MP1TP",
            physical_meaning="Winter ambient-temperature sweep at fixed mechanics; isolates cold-air thermal bending effect.",
            recommended_n_snapshots=50,
            expected_solver_difficulty="low-moderate",
            nominal_mech=_base_mech(),
            nominal_thermal=_winter_thermal(),
            parameter_ranges=[
                ParamRange("T_amb", 268.15, 285.15, unit="K", scale="linear", role="TP", description="cold ambient air temperature"),
            ],
            **common,
        ),
        7: CaseConfig(
            case_id=7,
            case_name="CASE_07_WINTER_1MP1TP",
            filename=THIS_FILE,
            season="winter",
            mp_tp_label="1MP1TP",
            physical_meaning="Winter load amplitude with substrate temperature variability; cold mechanically loaded flooring.",
            recommended_n_snapshots=75,
            expected_solver_difficulty="moderate",
            nominal_mech=_base_mech(),
            nominal_thermal=_winter_thermal(),
            parameter_ranges=[
                ParamRange("f", -7000.0, -1500.0, unit="N/m^2", scale="linear", role="MP", description="downward mechanical load amplitude"),
                ParamRange("T_sub", 273.15, 288.15, unit="K", scale="linear", role="TP", description="cold substrate temperature"),
            ],
            **common,
        ),
        8: CaseConfig(
            case_id=8,
            case_name="CASE_08_WINTER_1MP2TP",
            filename=THIS_FILE,
            season="winter",
            mp_tp_label="1MP2TP",
            physical_meaning="Winter support stiffness with ambient/substrate contrast.",
            recommended_n_snapshots=75,
            expected_solver_difficulty="moderate",
            nominal_mech=_base_mech(),
            nominal_thermal=_winter_thermal(),
            parameter_ranges=[
                ParamRange("ks", 100000.0, 2000000.0, unit="N/m^3", scale="log", role="MP", description="unilateral Winkler foundation stiffness"),
                ParamRange("T_amb", 268.15, 285.15, unit="K", scale="linear", role="TP", description="cold ambient air temperature"),
                ParamRange("T_sub", 273.15, 288.15, unit="K", scale="linear", role="TP", description="substrate temperature"),
            ],
            **common,
        ),
        9: CaseConfig(
            case_id=9,
            case_name="CASE_09_WINTER_2MP2TP",
            filename=THIS_FILE,
            season="winter",
            mp_tp_label="2MP2TP",
            physical_meaning="Winter orthotropy/load with cold air and convection variability.",
            recommended_n_snapshots=100,
            expected_solver_difficulty="moderate-high",
            nominal_mech=_base_mech(),
            nominal_thermal=_winter_thermal(),
            parameter_ranges=[
                ParamRange("D_ratio", 0.5, 2.0, unit="-", scale="linear", role="MP", description="orthotropic stiffness ratio Dx/Dy at fixed D_eff"),
                ParamRange("f", -7500.0, -2000.0, unit="N/m^2", scale="linear", role="MP", description="downward mechanical load amplitude"),
                ParamRange("T_amb", 268.15, 285.15, unit="K", scale="linear", role="TP", description="cold ambient air temperature"),
                ParamRange("h_con", 5.0, 25.0, unit="W/(m^2 K)", scale="linear", role="TP", description="top convection coefficient"),
            ],
            **common,
        ),
        10: CaseConfig(
            case_id=10,
            case_name="CASE_10_WINTER_2MP3TP",
            filename=THIS_FILE,
            season="winter",
            mp_tp_label="2MP3TP",
            physical_meaning="Hardest winter case: support/effective stiffness plus cold air, substrate, and weak winter solar flux.",
            recommended_n_snapshots=120,
            expected_solver_difficulty="high",
            nominal_mech=_base_mech(),
            nominal_thermal=_winter_thermal(),
            parameter_ranges=[
                ParamRange("ks", 100000.0, 3000000.0, unit="N/m^3", scale="log", role="MP", description="unilateral Winkler foundation stiffness"),
                ParamRange("D_eff", 4000.0, 12000.0, unit="N m", scale="linear", role="MP", description="geometric-mean bending rigidity"),
                ParamRange("T_amb", 268.15, 285.15, unit="K", scale="linear", role="TP", description="cold ambient air temperature"),
                ParamRange("T_sub", 273.15, 288.15, unit="K", scale="linear", role="TP", description="substrate temperature"),
                ParamRange("q_s", 0.0, 250.0, unit="W/m^2", scale="linear", role="TP", description="weak winter absorbed solar heat flux"),
            ],
            **common,
        ),
    }
    return cases


# -----------------------------------------------------------------------------
# Case selector utilities
# -----------------------------------------------------------------------------

def print_case_table(cases: Dict[int, CaseConfig]) -> None:
    print("\nAvailable thermomechanical physical case types:\n")
    print(f"{'Case':>4}  {'Base case type':<26} {'Season':<7} {'MP/TP':<8} {'Default n':>9}  Meaning")
    print("-" * 124)
    for cid in sorted(cases):
        c = cases[cid]
        print(
            f"{cid:>4}  {c.case_name:<26} {c.season:<7} {c.mp_tp_label:<8} "
            f"{c.recommended_n_snapshots:>9}  {c.physical_meaning}"
        )
    print("\nPanel variants of interest:")
    print("  --panel-type monolithic   -> n_vert=0, n_horiz=0")
    print("  --panel-type contact1     -> n_vert=1, n_horiz=0")
    print("  --panel-type contact2     -> n_vert=2, n_horiz=0")
    print("\nLoad variants of interest:")
    print("  --load-type patch")
    print("  --load-type linear")
    print("  --load-type multi_patch")
    print("\nBoundary-condition variants:")
    print("  --bc-type free_edge")
    print("  --bc-type simply_supported")
    print("\nChoose one with:  --case 1   or   --case CASE_01_SUMMER_0MP1TP\n")


def parse_case_token(token: str, cases: Dict[int, CaseConfig]) -> int:
    raw = str(token).strip()
    if not raw:
        raise ValueError("Empty --case value.")

    if raw.isdigit():
        cid = int(raw)
        if cid in cases:
            return cid

    key = raw.upper().replace("-", "_")
    aliases = {}
    for cid, cfg in cases.items():
        aliases[str(cid)] = cid
        aliases[f"{cid:02d}"] = cid
        aliases[cfg.case_name.upper()] = cid
        aliases[cfg.case_name.upper().replace("CASE_", "")] = cid
        aliases[f"CASE_{cid:02d}"] = cid

    if key in aliases:
        return aliases[key]

    valid = ", ".join(str(k) for k in sorted(cases))
    raise ValueError(f"Unknown case '{token}'. Valid numeric cases are: {valid}")


def main() -> None:
    cases = build_case_configs()

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--case", "-c", type=str, default=None)
    pre.add_argument("--list-cases", action="store_true")
    pre.add_argument("--show-cases", action="store_true")
    known, remaining = pre.parse_known_args()

    if known.list_cases or known.show_cases:
        print_case_table(cases)
        return

    if known.case is None:
        print_case_table(cases)
        print("ERROR: Please choose one base case type using --case. Example: --case 1")
        sys.exit(2)

    try:
        cid = parse_case_token(known.case, cases)
    except Exception as exc:
        print_case_table(cases)
        print(f"ERROR: {exc}")
        sys.exit(2)

    selected = cases[cid]
    print(f"Selected base case type {cid}: {selected.case_name}")

    # Remove --case/--list-cases from argv before calling the common parser.
    sys.argv = [sys.argv[0]] + remaining
    run_case_from_cli(selected)


if __name__ == "__main__":
    main()
