"""Public capability inventory for the three thesis studies."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CapabilityStatus(str, Enum):
    VERIFIED = "IMPLEMENTED AND VERIFIED IN THIS REBUILD"
    ENVIRONMENT_DEPENDENT = "IMPLEMENTED, ENVIRONMENT-DEPENDENT TEST NOT RUN"
    EXTERNAL_DATA = "DOCUMENTED EXTERNAL-DATA WORKFLOW"
    NOT_SUPPORTED = "NOT SUPPORTED BY CURRENT AUTHORITATIVE FILES"


@dataclass(frozen=True)
class Capability:
    study: str
    item: str
    status: CapabilityStatus
    implementation: str
    note: str = ""


CAPABILITIES: tuple[Capability, ...] = (
    # Static mechanics
    Capability("static mechanics", "FOM", CapabilityStatus.ENVIRONMENT_DEPENDENT, "paramplate.mechanical.fem.MechanicalPlateFOM"),
    Capability("static mechanics", "verification/smoke path", CapabilityStatus.ENVIRONMENT_DEPENDENT, "paramplate.mechanical.verification"),
    Capability("static mechanics", "snapshot/data pipeline", CapabilityStatus.EXTERNAL_DATA, "paramplate.mechanical.snapshots"),
    Capability("static mechanics", "POD", CapabilityStatus.VERIFIED, "paramplate.rom.pod"),
    Capability("static mechanics", "intrusive POD--Galerkin", CapabilityStatus.ENVIRONMENT_DEPENDENT, "paramplate.mechanical.intrusive"),
    Capability("static mechanics", "POD--Proj diagnostic", CapabilityStatus.VERIFIED, "paramplate.rom.pod"),
    Capability("static mechanics", "PODI--RBF", CapabilityStatus.VERIFIED, "paramplate.rom.predictors"),
    Capability("static mechanics", "PODI--Linear", CapabilityStatus.VERIFIED, "paramplate.rom.predictors"),
    Capability("static mechanics", "POD--GPR", CapabilityStatus.VERIFIED, "paramplate.rom.predictors"),
    Capability("static mechanics", "POD--NN", CapabilityStatus.VERIFIED, "paramplate.rom.neural"),
    Capability("static mechanics", "post-processing", CapabilityStatus.VERIFIED, "paramplate.postprocessing"),
    Capability("static mechanics", "configuration", CapabilityStatus.VERIFIED, "configs/mechanical"),
    Capability("static mechanics", "documentation", CapabilityStatus.VERIFIED, "docs/workflows/mechanical.md"),
    Capability("static mechanics", "tests", CapabilityStatus.VERIFIED, "tests"),
    # Thermomechanics
    Capability("steady thermomechanics", "coupled FOM", CapabilityStatus.ENVIRONMENT_DEPENDENT, "paramplate.thermomechanical.fem.ThermomechanicalPlateFOM"),
    Capability("steady thermomechanics", "thermal-to-plate reduction", CapabilityStatus.ENVIRONMENT_DEPENDENT, "paramplate.thermomechanical.thermal_driver"),
    Capability("steady thermomechanics", "partitioned solver", CapabilityStatus.VERIFIED, "paramplate.thermomechanical.coupling"),
    Capability("steady thermomechanics", "snapshot/data pipeline", CapabilityStatus.EXTERNAL_DATA, "paramplate.thermomechanical.snapshots"),
    Capability("steady thermomechanics", "displacement POD", CapabilityStatus.VERIFIED, "paramplate.rom.pod"),
    Capability("steady thermomechanics", "thermal-driver POD", CapabilityStatus.VERIFIED, "paramplate.rom.pod"),
    Capability("steady thermomechanics", "hybrid mechanical POD--Galerkin", CapabilityStatus.ENVIRONMENT_DEPENDENT, "paramplate.thermomechanical.hybrid"),
    Capability("steady thermomechanics", "POD--Proj diagnostic", CapabilityStatus.VERIFIED, "paramplate.rom.pod"),
    Capability("steady thermomechanics", "PODI--RBF", CapabilityStatus.VERIFIED, "paramplate.rom.predictors"),
    Capability("steady thermomechanics", "PODI--Linear", CapabilityStatus.VERIFIED, "paramplate.rom.predictors"),
    Capability("steady thermomechanics", "POD--GPR", CapabilityStatus.VERIFIED, "paramplate.rom.predictors"),
    Capability("steady thermomechanics", "POD--NN", CapabilityStatus.VERIFIED, "paramplate.rom.neural"),
    Capability("steady thermomechanics", "POD--DL-ROM", CapabilityStatus.VERIFIED, "paramplate.rom.deep"),
    Capability("steady thermomechanics", "DL-ROM", CapabilityStatus.VERIFIED, "paramplate.rom.deep"),
    Capability("steady thermomechanics", "post-processing", CapabilityStatus.VERIFIED, "paramplate.postprocessing"),
    Capability("steady thermomechanics", "configuration", CapabilityStatus.VERIFIED, "configs/thermomechanical"),
    Capability("steady thermomechanics", "documentation", CapabilityStatus.VERIFIED, "docs/workflows/thermomechanical.md"),
    Capability("steady thermomechanics", "tests", CapabilityStatus.VERIFIED, "tests"),
    # Transient dynamics
    Capability("transient dynamics", "FOM", CapabilityStatus.ENVIRONMENT_DEPENDENT, "paramplate.dynamics.fem.TransientPlateFOM"),
    Capability("transient dynamics", "active dynamic DOFs", CapabilityStatus.VERIFIED, "paramplate.dynamics.active_dofs"),
    Capability("transient dynamics", "Newmark integration", CapabilityStatus.VERIFIED, "paramplate.dynamics.newmark"),
    Capability("transient dynamics", "verification/smoke path", CapabilityStatus.ENVIRONMENT_DEPENDENT, "paramplate.dynamics.verification"),
    Capability("transient dynamics", "complete-trajectory snapshot pipeline", CapabilityStatus.EXTERNAL_DATA, "paramplate.dynamics.trajectories"),
    Capability("transient dynamics", "trajectory-level split", CapabilityStatus.VERIFIED, "paramplate.rom.splits"),
    Capability("transient dynamics", "POD", CapabilityStatus.VERIFIED, "paramplate.rom.pod"),
    Capability("transient dynamics", "POD--Proj diagnostic", CapabilityStatus.VERIFIED, "paramplate.rom.pod"),
    Capability("transient dynamics", "PODI--RBF", CapabilityStatus.VERIFIED, "paramplate.rom.predictors"),
    Capability("transient dynamics", "PODI--Linear", CapabilityStatus.VERIFIED, "paramplate.rom.predictors"),
    Capability("transient dynamics", "POD--GPR", CapabilityStatus.VERIFIED, "paramplate.rom.predictors"),
    Capability("transient dynamics", "POD--NN", CapabilityStatus.VERIFIED, "paramplate.rom.neural"),
    Capability("transient dynamics", "POD--DL-ROM", CapabilityStatus.VERIFIED, "paramplate.rom.deep"),
    Capability("transient dynamics", "parameter--time prediction", CapabilityStatus.VERIFIED, "paramplate.rom.features"),
    Capability("transient dynamics", "post-processing", CapabilityStatus.VERIFIED, "paramplate.postprocessing"),
    Capability("transient dynamics", "configuration", CapabilityStatus.VERIFIED, "configs/dynamics"),
    Capability("transient dynamics", "documentation", CapabilityStatus.VERIFIED, "docs/workflows/transient.md"),
    Capability("transient dynamics", "tests", CapabilityStatus.VERIFIED, "tests"),
    Capability("transient dynamics", "direct DL-ROM", CapabilityStatus.NOT_SUPPORTED, "", "The current thesis does not report direct DL-ROM for transient displacement."),
)


def capabilities_for_study(study: str) -> tuple[Capability, ...]:
    key = study.strip().lower()
    aliases = {
        "mechanical": "static mechanics",
        "static": "static mechanics",
        "thermomechanical": "steady thermomechanics",
        "thermo": "steady thermomechanics",
        "dynamics": "transient dynamics",
        "transient": "transient dynamics",
    }
    key = aliases.get(key, key)
    return tuple(item for item in CAPABILITIES if item.study == key)
