from __future__ import annotations

from paramplate.capabilities import CAPABILITIES, CapabilityStatus, capabilities_for_study


def test_acceptance_matrix_covers_three_studies_without_false_transient_direct_dl() -> None:
    studies = {item.study for item in CAPABILITIES}
    assert studies == {"static mechanics", "steady thermomechanics", "transient dynamics"}
    static = capabilities_for_study("mechanical")
    thermo = capabilities_for_study("thermo")
    transient = capabilities_for_study("transient")
    assert any(item.item == "intrusive POD--Galerkin" for item in static)
    assert any(item.item == "DL-ROM" and item.status == CapabilityStatus.VERIFIED for item in thermo)
    direct = next(item for item in transient if item.item == "direct DL-ROM")
    assert direct.status == CapabilityStatus.NOT_SUPPORTED
