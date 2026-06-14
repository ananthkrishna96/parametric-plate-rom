from pathlib import Path


def test_legacy_file_is_preserved():
    path = Path("legacy/project2/THERMOMECHANICAL_FOM_ROM_legacy.py")
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "class GeneralMultiphysicsSolver" in text
    assert "def heat_solve" in text
    assert "class PODNNReducedOrderModel" in text


def test_project2_compatibility_layer_exists():
    path = Path("src/paramplate/project2/legacy_definitions.py")
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "class GeneralMultiphysicsSolver" in text
    assert "class PODAutoencoderROM" in text
