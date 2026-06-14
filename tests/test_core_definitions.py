from paramplate.core import (
    BoundaryConditionType,
    CaseDefinition,
    LoadType,
    ParameterRange,
    ParameterRole,
    PlateGeometry,
)


def test_plate_geometry_validation():
    geometry = PlateGeometry(length=2.0, width=1.0, thickness=0.05)
    geometry.validate()


def test_parameter_range_validation():
    parameter_range = ParameterRange(
        name="Dx",
        lower=1.0,
        upper=10.0,
        role=ParameterRole.MECHANICAL,
    )
    parameter_range.validate()


def test_case_definition_validation():
    case = CaseDefinition(
        case_id=1,
        name="demo_case",
        description="Small test case.",
        parameter_ranges=(
            ParameterRange(name="q_s", lower=100.0, upper=500.0, role=ParameterRole.THERMAL),
        ),
        boundary_condition=BoundaryConditionType.FREE_EDGE,
        load_type=LoadType.PATCH,
    )
    case.validate()
