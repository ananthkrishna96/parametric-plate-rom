#!/usr/bin/env python3
"""Build POD preprocessing artifacts for one Project-2 archive.

Example
-------
python scripts/project2/build_pod_dataset.py \
  --case CASE_02_SUMMER_1MP1TP__PANEL_MONOLITHIC_NV0_NH0__BC_FREE_EDGE__LOAD_PATCH \
  --train-fraction 0.8 \
  --energy-tol 0.9999 \
  --write
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Allow running from a source checkout without requiring installation first.
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from paramplate.io.data_locations import load_data_locations
from paramplate.rom.datasets import load_project2_dataset_by_name
from paramplate.rom.preprocessing import build_pod_preprocessing, save_pod_preprocessing


DEFAULT_CASE = "CASE_02_SUMMER_1MP1TP__PANEL_MONOLITHIC_NV0_NH0__BC_FREE_EDGE__LOAD_PATCH"


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case", default=DEFAULT_CASE, help="Project-2 case folder name or unique substring.")
    p.add_argument("--data-root", default=None, help="External data root. Overrides local YAML/env config.")
    p.add_argument("--output-root", default=None, help="POD output root. Defaults to paper2 rom_training/pod_bases.")
    p.add_argument("--train-fraction", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--energy-tol", type=float, default=0.9999)
    p.add_argument("--max-rank", type=int, default=None)
    p.add_argument("--no-center", action="store_true", help="Disable mean-centering before SVD.")
    p.add_argument("--no-theta", action="store_true", help="Skip theta/thermal POD even when theta snapshots are present.")
    p.add_argument("--write", action="store_true", help="Write POD artifacts to external data root.")
    p.add_argument("--overwrite", action="store_true", help="Overwrite an existing non-empty POD output folder.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    loc = load_data_locations(data_root=args.data_root)

    ds = load_project2_dataset_by_name(
        args.case,
        locations=loc,
        load_theta=not args.no_theta,
    )

    print("data_root:", loc.data_root)
    print("source_archive:", ds.source_path)
    print("case_name:", ds.source_path.parent.name)
    print("project:", ds.project)
    print("parameters:", "x".join(map(str, ds.parameters.shape)))
    print("snapshots :", "x".join(map(str, ds.snapshots.shape)))
    print("theta     :", "none" if ds.theta_snapshots is None else "x".join(map(str, ds.theta_snapshots.shape)))

    result = build_pod_preprocessing(
        ds,
        train_fraction=args.train_fraction,
        seed=args.seed,
        energy_tol=args.energy_tol,
        max_rank=args.max_rank,
        center=not args.no_center,
        include_theta=not args.no_theta,
    )

    print("")
    print("Train/test:")
    print(f"  train: {result.split.n_train}")
    print(f"  test : {result.split.n_test}")

    print("")
    print("POD displacement:")
    print(f"  rank selected        : {result.pod_w.rank}")
    print(f"  retained energy      : {result.pod_w.retained_energy:.12f}")
    print(f"  rel. train error     : {result.relerr_w_train:.6e}")
    print(f"  rel. test error      : {result.relerr_w_test:.6e}")

    if result.pod_theta is not None:
        print("")
        print("POD theta:")
        print(f"  rank selected        : {result.pod_theta.rank}")
        print(f"  retained energy      : {result.pod_theta.retained_energy:.12f}")
        print(f"  rel. train error     : {result.relerr_theta_train:.6e}")
        print(f"  rel. test error      : {result.relerr_theta_test:.6e}")

    output_root = Path(args.output_root).expanduser() if args.output_root else loc.paper2_rom_root / "pod_bases"
    output_dir = output_root / ds.source_path.parent.name

    if args.write:
        written = save_pod_preprocessing(result, output_dir, overwrite=args.overwrite)
        print("")
        print("Wrote POD artifacts to:")
        print(" ", written)
    else:
        print("")
        print("Dry run only. Add --write to save artifacts to:")
        print(" ", output_dir)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
