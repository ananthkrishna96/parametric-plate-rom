# ============================================================================
# PAPER 2 / PROJECT 2 SNAPSHOT COMPATIBILITY BYPASS
# Paste this block immediately before:
#     ### Snapshot Projection tool
#
# Purpose:
#   Convert Project-2 thermomechanical SNAPGEN archives
#       keys: parameters, snapshots, theta_snapshots, ...
#   into the Paper-1 ROM code's expected archive interface
#       keys: mus, fom_snapshots
#
# It also redirects:
#     os.path.expanduser("~/Documents/PAPER_1/SNAPSHOTS")
# to a safe compatibility folder inside:
#     ~/Documents/PAPER_2/SNAPGEN/ROM_COMPAT_PAPER1_STYLE_ACTIVE
#
# This avoids editing every hardcoded Paper-1 base/path dictionary later.
# ============================================================================

from pathlib import Path
import os
import sys
import json
import shutil
import numpy as np

# ----------------------------------------------------------------------------
# USER CONFIGURATION — change only these lines for each Project-2 archive.
# ----------------------------------------------------------------------------
P2_SNAPGEN_ROOT = Path("~/Documents/PAPER_2/SNAPGEN").expanduser()

# Choose the actual Project-2 snapshot archive to adapt.
# Available from your copied dataset:
#   monolithic : cases 1,2,3,4,5,6,7,8,9,10
#   contact1  : cases 3,5
P2_TARGET_CASE  = 5              # 1..10
P2_TARGET_PANEL = "contact1"     # "monolithic" or "contact1" for current copied dataset
P2_TARGET_BC    = "free_edge"
P2_TARGET_LOAD  = "patch"

# Keep True. It creates the projected and reprojected files immediately,
# so later sections can run even if you skip the old projection cell.
P2_CREATE_PROJECTED_AND_REPROJECTED = True

# Keep True. It redirects only the old Paper-1 snapshot base path in this Python session.
P2_REDIRECT_PAPER1_BASE = True

# ----------------------------------------------------------------------------
# Internal helpers.
# ----------------------------------------------------------------------------
_CASE_NAMES = {
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

_PANEL = {
    "monolithic": dict(n_vert=0, n_horiz=0, label="MONOLITHIC_NV0_NH0", old_case=1, old_variant="monolithic"),
    "contact1":   dict(n_vert=1, n_horiz=0, label="CONTACT_NV1_NH0",    old_case=3, old_variant="contact_1_0"),
    "contact2":   dict(n_vert=2, n_horiz=0, label="CONTACT_NV2_NH0",    old_case=3, old_variant="contact_2_0"),
}

def _safe_token(x):
    txt = str(x).strip().upper()
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

def _p2_run_name(case_id, panel, bc, load):
    return "__".join([
        _safe_token(_CASE_NAMES[int(case_id)]),
        "PANEL_" + _PANEL[panel]["label"],
        "BC_" + _safe_token(bc),
        "LOAD_" + _safe_token(load),
    ])

def _old_proxy_relative_path(old_case, old_variant, n_snaps):
    """
    Return the exact path pattern expected by the old Paper-1 dictionaries.
    """
    if (old_case, old_variant) == (1, "monolithic"):
        return Path(f"CASE_1/nv0_nh0_free_edge_patch/nonlinear_snapshots_{n_snaps}.npz")
    if (old_case, old_variant) == (2, "monolithic"):
        return Path(f"CASE_2/nonlinear_snapshots_{n_snaps}.npz")
    if (old_case, old_variant) == (2, "contact_1_0"):
        return Path(f"CASE_2/nonlinear_snapshots_{n_snaps}.npz")
    if (old_case, old_variant) == (3, "contact_1_0"):
        return Path(f"CASE_3/nv1_nh0_free_edge_patch/nonlinear_snapshots_{n_snaps}.npz")
    if (old_case, old_variant) == (3, "contact_2_0"):
        return Path(f"CASE_3/nv2_nh0_free_edge_linear/nonlinear_snapshots_{n_snaps}.npz")
    if (old_case, old_variant) == (3, "contact_1_1"):
        return Path(f"CASE_3/nv1_nh1_simply_supported_multi_patch/nonlinear_snapshots_{n_snaps}.npz")
    raise ValueError(f"Unsupported old proxy selector: {(old_case, old_variant)}")

def _as_mixed_solution_snapshots_from_cg(cg_snapshots, *, n_vert, n_horiz, bc_type, load_type):
    """
    Project Project-2 global CG snapshots into the Paper-1-style Solution/mixed space.

    For monolithic runs, W and V_CG have the same scalar CG layout, so this returns
    the original CG snapshots.

    For contact runs, this builds a mixed Solution vector by projecting the same
    global CG field onto each subdomain component. This is exactly compatible with
    the old Paper-1 projection/checking logic that later reconstructs the physical
    global field by multiplying each component with chi_subdomain and summing.
    """
    # Import Project-2 solver explicitly from the copied SNAPGEN folder.
    if str(P2_SNAPGEN_ROOT) not in sys.path:
        sys.path.insert(0, str(P2_SNAPGEN_ROOT))

    from THERMOMECHANICAL_FOM_ROM import GeneralMultiphysicsSolver as P2Solver
    from dolfin import Function, project

    solver_tmp = P2Solver(study_case=1)
    solver_tmp.bc_type = str(bc_type)
    solver_tmp.load_type = str(load_type)
    solver_tmp.size = 64
    solver_tmp.degree = 2
    solver_tmp.define_domain(int(n_vert), int(n_horiz), plot_subdomains=False)
    solver_tmp.setup_kirchhoff_problem()

    cg_snapshots = np.asarray(cg_snapshots, dtype=float)

    # Monolithic: W is scalar CG, same physical layout.
    if solver_tmp.N_subdomains == 1:
        if solver_tmp.W.dim() == cg_snapshots.shape[1]:
            return cg_snapshots.copy(), solver_tmp
        out = []
        u = Function(solver_tmp.V_CG)
        for row in cg_snapshots:
            u.vector()[:] = row
            w = project(u, solver_tmp.W)
            out.append(w.vector().get_local().copy())
        return np.asarray(out), solver_tmp

    # Contact/multi-panel: expand global CG to mixed component space.
    out = []
    u_cg = Function(solver_tmp.V_CG)

    for row in cg_snapshots:
        u_cg.vector()[:] = row
        w_mixed = Function(solver_tmp.W)
        w_mixed.vector().zero()

        for j, _sid in enumerate(solver_tmp.subdomain_ids):
            V_sub = solver_tmp.W.sub(j).collapse()
            comp = project(u_cg, V_sub)
            dofmap_mixed = solver_tmp.W.sub(j).dofmap()
            dofmap_sub = V_sub.dofmap()

            for dof_mixed, dof_sub in zip(dofmap_mixed.dofs(), dofmap_sub.dofs()):
                w_mixed.vector()[dof_mixed] = comp.vector()[dof_sub]

        try:
            w_mixed.vector().apply("insert")
        except Exception:
            pass

        out.append(w_mixed.vector().get_local().copy())

    return np.asarray(out), solver_tmp

# ----------------------------------------------------------------------------
# Locate and load Project-2 archive.
# ----------------------------------------------------------------------------
if P2_TARGET_PANEL not in _PANEL:
    raise ValueError(f"P2_TARGET_PANEL must be one of {list(_PANEL)}.")

_p2_run = _p2_run_name(P2_TARGET_CASE, P2_TARGET_PANEL, P2_TARGET_BC, P2_TARGET_LOAD)
_p2_dir = P2_SNAPGEN_ROOT / "THERMO_SNAPSHOTS" / _p2_run
_p2_archive = _p2_dir / f"SNAPSHOTS__{_p2_run}.npz"

if not _p2_archive.exists():
    raise FileNotFoundError(
        f"Project-2 archive not found:\n{_p2_archive}\n\n"
        "Check P2_TARGET_CASE / P2_TARGET_PANEL / P2_TARGET_BC / P2_TARGET_LOAD."
    )

_p2_data = np.load(_p2_archive, allow_pickle=True)
_p2_keys = list(_p2_data.files)

if "parameters" not in _p2_keys or "snapshots" not in _p2_keys:
    raise KeyError(
        f"Project-2 archive must contain keys 'parameters' and 'snapshots'.\n"
        f"Archive: {_p2_archive}\nKeys: {_p2_keys}"
    )

_p2_mu = np.asarray(_p2_data["parameters"], dtype=float)
_p2_cg = np.asarray(_p2_data["snapshots"], dtype=float)
len_Snaps = int(_p2_mu.shape[0])     # old Paper-1 code uses this global variable

# ----------------------------------------------------------------------------
# Create compatibility archive tree.
# ----------------------------------------------------------------------------
_proxy = _PANEL[P2_TARGET_PANEL]
P2_OLD_PROXY_CASE = int(_proxy["old_case"])
P2_OLD_PROXY_VARIANT = str(_proxy["old_variant"])

P2_COMPAT_ROOT = P2_SNAPGEN_ROOT / "ROM_COMPAT_PAPER1_STYLE_ACTIVE"
_rel_solution = _old_proxy_relative_path(P2_OLD_PROXY_CASE, P2_OLD_PROXY_VARIANT, len_Snaps)
_solution_file = P2_COMPAT_ROOT / _rel_solution
_projected_file = Path(str(_solution_file).replace(".npz", "_projected.npz"))
_reprojected_file = Path(str(_solution_file).replace(".npz", "_reprojected_Solution.npz"))

_solution_file.parent.mkdir(parents=True, exist_ok=True)

print("=" * 100)
print("PROJECT-2 → PAPER-1 ROM COMPATIBILITY BYPASS")
print("=" * 100)
print(f"Project-2 source archive : {_p2_archive}")
print(f"Project-2 archive keys   : {_p2_keys}")
print(f"Project-2 parameters     : {_p2_mu.shape}")
print(f"Project-2 CG snapshots   : {_p2_cg.shape}")
print(f"Compatibility root       : {P2_COMPAT_ROOT}")
print(f"Old proxy selector       : CASE={P2_OLD_PROXY_CASE}, VARIANT='{P2_OLD_PROXY_VARIANT}'")
print(f"Old len_Snaps            : {len_Snaps}")
print()

_solution_snaps, _compat_solver = _as_mixed_solution_snapshots_from_cg(
    _p2_cg,
    n_vert=_proxy["n_vert"],
    n_horiz=_proxy["n_horiz"],
    bc_type=P2_TARGET_BC,
    load_type=P2_TARGET_LOAD,
)

np.savez_compressed(
    _solution_file,
    mus=_p2_mu,
    fom_snapshots=_solution_snaps,
    source_project2_archive=np.asarray(str(_p2_archive)),
    source_project2_keys=np.asarray(_p2_keys, dtype=str),
    source_project2_case=np.asarray(int(P2_TARGET_CASE)),
    source_project2_panel=np.asarray(str(P2_TARGET_PANEL)),
)

print(f"Written old-style Solution archive:")
print(f"  {_solution_file}")
print(f"  mus shape           : {_p2_mu.shape}")
print(f"  fom_snapshots shape : {_solution_snaps.shape}")

if P2_CREATE_PROJECTED_AND_REPROJECTED:
    # Projected archive is directly the global CG Project-2 displacement snapshot matrix.
    np.savez_compressed(
        _projected_file,
        mus=_p2_mu,
        fom_snapshots=_p2_cg,
        source_project2_archive=np.asarray(str(_p2_archive)),
        note=np.asarray("Project-2 snapshots are already global CG displacement snapshots."),
    )

    # Reprojected Solution archive is the same mixed/Solution representation created above.
    np.savez_compressed(
        _reprojected_file,
        mus=_p2_mu,
        fom_snapshots=_solution_snaps,
        source_project2_archive=np.asarray(str(_p2_archive)),
        note=np.asarray("CG-to-Solution compatibility representation for Paper-1 checking code."),
    )

    print("Written projected/reprojected old-style archives:")
    print(f"  projected   : {_projected_file}")
    print(f"  reprojected : {_reprojected_file}")

# ----------------------------------------------------------------------------
# Patch old Paper-1 dictionaries/labels in the current session.
# ----------------------------------------------------------------------------
try:
    # Get exact Project-2 parameter labels from the case configuration.
    if str(P2_SNAPGEN_ROOT) not in sys.path:
        sys.path.insert(0, str(P2_SNAPGEN_ROOT))
    from THERMOMECHANICAL_SNAPGEN_CASES import build_case_configs
    _cfg = build_case_configs()[int(P2_TARGET_CASE)]
    _p2_names = [p.name for p in _cfg.parameter_ranges]
    _p2_units = [p.unit for p in _cfg.parameter_ranges]
    _p2_ranges = [(float(p.lower), float(p.upper)) for p in _cfg.parameter_ranges]

    if "parameter_ranges" not in globals() or not isinstance(globals().get("parameter_ranges"), dict):
        parameter_ranges = {}
    if "parameter_labels" not in globals() or not isinstance(globals().get("parameter_labels"), dict):
        parameter_labels = {}

    # Override only the old proxy case key used by the downstream Paper-1 code.
    parameter_ranges[P2_OLD_PROXY_CASE] = _p2_ranges
    parameter_labels[P2_OLD_PROXY_CASE] = {
        "names": _p2_names,
        "units": _p2_units,
    }

    print()
    print("Patched Paper-1-style parameter labels/ranges for old proxy case:")
    print(f"  parameter_ranges[{P2_OLD_PROXY_CASE}] = {_p2_ranges}")
    print(f"  parameter_labels[{P2_OLD_PROXY_CASE}] = {parameter_labels[P2_OLD_PROXY_CASE]}")
except Exception as exc:
    print()
    print(f"WARNING: Could not patch parameter labels/ranges automatically: {exc}")

# Use the Project-2 solver class for downstream space construction.
try:
    if str(P2_SNAPGEN_ROOT) not in sys.path:
        sys.path.insert(0, str(P2_SNAPGEN_ROOT))
    from THERMOMECHANICAL_FOM_ROM import GeneralMultiphysicsSolver as _P2GeneralMultiphysicsSolver
    GeneralMultiphysicsSolver = _P2GeneralMultiphysicsSolver
    print()
    print("Patched GeneralMultiphysicsSolver to Project-2 THERMOMECHANICAL_FOM_ROM.GeneralMultiphysicsSolver.")
except Exception as exc:
    print()
    print(f"WARNING: Could not patch GeneralMultiphysicsSolver automatically: {exc}")

# ----------------------------------------------------------------------------
# Redirect old Paper-1 snapshot base to the compatibility root in this Python session.
# ----------------------------------------------------------------------------
if P2_REDIRECT_PAPER1_BASE:
    _old_expanduser = getattr(os.path, "_paper2_original_expanduser", os.path.expanduser)
    os.path._paper2_original_expanduser = _old_expanduser

    def _paper2_expanduser_redirect(path):
        p = str(path)
        if p.rstrip("/") == "~/Documents/PAPER_1/SNAPSHOTS":
            return str(P2_COMPAT_ROOT)
        return _old_expanduser(path)

    os.path.expanduser = _paper2_expanduser_redirect
    print()
    print("Activated path redirect inside this Python session:")
    print("  os.path.expanduser('~/Documents/PAPER_1/SNAPSHOTS')")
    print(f"  -> {P2_COMPAT_ROOT}")

# ----------------------------------------------------------------------------
# Write manifest.
# ----------------------------------------------------------------------------
_manifest = {
    "p2_snapgen_root": str(P2_SNAPGEN_ROOT),
    "p2_archive": str(_p2_archive),
    "p2_run_name": _p2_run,
    "p2_target_case": int(P2_TARGET_CASE),
    "p2_target_panel": str(P2_TARGET_PANEL),
    "p2_target_bc": str(P2_TARGET_BC),
    "p2_target_load": str(P2_TARGET_LOAD),
    "p2_archive_keys": _p2_keys,
    "p2_parameters_shape": list(_p2_mu.shape),
    "p2_cg_snapshots_shape": list(_p2_cg.shape),
    "compat_root": str(P2_COMPAT_ROOT),
    "old_proxy_case": int(P2_OLD_PROXY_CASE),
    "old_proxy_variant": str(P2_OLD_PROXY_VARIANT),
    "old_solution_file": str(_solution_file),
    "old_projected_file": str(_projected_file),
    "old_reprojected_file": str(_reprojected_file),
    "len_Snaps": int(len_Snaps),
}
(P2_COMPAT_ROOT / "ACTIVE_P2_COMPAT_MANIFEST.json").write_text(json.dumps(_manifest, indent=2))

print()
print("IMPORTANT FOR THE OLD PAPER-1 CELLS:")
print(f"  Use old selector CASE={P2_OLD_PROXY_CASE}, VARIANT='{P2_OLD_PROXY_VARIANT}'")
print(f"  Use old selector case={P2_OLD_PROXY_CASE}, variant='{P2_OLD_PROXY_VARIANT}'")
print(f"  len_Snaps is already set to {len_Snaps}.")
print()
print("Compatibility preparation complete.")
print("=" * 100)
