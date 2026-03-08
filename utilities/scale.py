from typing import TypedDict
import opensim as osim
import numpy as np
import os
from pathlib import Path
import ezc3d

# Drawn from Johnson's opensim rat model (in millimeters to match recorded values in .c3d files)
base_femur_length = float(np.linalg.norm([-0.0035, -0.0312, -0.005]) * 1000)
base_tibia_length = float(np.linalg.norm([0.0016, 0.039, -0.0037]) * 1000)


# Equations from Hicks
def thigh_mass(mass: float):
    return (8.3313 * mass + 3.6883) / 1000


def thigh_com(
    side: str, femur_length: float, mass: float
) -> tuple[float, float, float]:
    side = side[0].capitalize()
    if side not in ["L", "R"]:
        raise ValueError("Side must be 'L' or 'R'")
    z_sign = -1 if side == "L" else 1
    return (
        femur_length * (-8.7844332 / 100000),
        mass * 0.148741316 * (-42.118041 / 100),
        mass * 0.098448042 * (2.00427791 / 100) * z_sign,
    )


def thigh_moi(
    side: str, femur_length: float, mass: float
) -> tuple[float, float, float]:
    side = side[0].capitalize()
    if side not in ["L", "R"]:
        raise ValueError("Side must be 'L' or 'R'")
    return (
        (0.00189568) * (mass) * (femur_length / 1000) ** 2,
        (0.00143871) * (mass) * (femur_length / 1000) ** 2,
        (0.00248006) * (mass) * (femur_length / 1000) ** 2,
    )


def shank_mass(mass: float):
    return (3.2096 * mass + 3.0047) / 1000


def shank_com(
    side: str, tibia_length: float, mass: float
) -> tuple[float, float, float]:
    side = side[0].capitalize()
    if side not in ["L", "R"]:
        raise ValueError("Side must be 'L' or 'R'")
    z_sign = -1 if side == "L" else 1
    return (
        (mass) * 0.09004923 * (-2.43352222 / 100),
        tibia_length * (67.363643 / 100000),
        (mass * 0.07731125) * (1.71207065 / 100) * z_sign,
    )


def shank_moi(
    side: str, tibia_length: float, mass: float
) -> tuple[float, float, float]:
    side = side[0].capitalize()
    if side not in ["L", "R"]:
        raise ValueError("Side must be 'L' or 'R'")
    return (
        (0.00104229) * (mass) * (tibia_length / 1000) ** 2,
        (0.00029337) * (mass) * (tibia_length / 1000) ** 2,
        (0.00104734) * (mass) * (tibia_length / 1000) ** 2,
    )


def foot_mass(mass: float):
    return (2.2061 * mass + 0.87788) / 1000


def foot_com(side: str, foot_length: float, mass: float) -> tuple[float, float, float]:
    side = side[0].capitalize()
    if side not in ["L", "R"]:
        raise ValueError("Side must be 'L' or 'R'")
    z_sign = -1 if side == "L" else 1
    return (
        (mass * 0.04627387) * (-4.294993 / 100),
        foot_length * (-42.78009 / 100000),
        (mass * 0.07246637) * (0.6265934 / 100) * z_sign,
    )  # TODO: Still need to check the weird thing Brody does with this in the old code


def foot_moi(side: str, foot_length: float, mass: float) -> tuple[float, float, float]:
    side = side[0].capitalize()
    if side not in ["L", "R"]:
        raise ValueError("Side must be 'L' or 'R'")
    return (
        (0.000384786) * (mass) * (foot_length / 1000) ** 2,
        (0.0000518802) * (mass) * (foot_length / 1000) ** 2,
        (0.000364591) * (mass) * (foot_length / 1000) ** 2,
    )


class RatScalingParameters(TypedDict):
    Mass: float
    RFemurLength: float
    RTibiaLength: float
    LFemurLength: float
    LTibiaLength: float
    RFootLength: float
    LFootLength: float


# Define required markers and parameters for validation
required_markers = [
    "TAIL",
    "SPL6",
    "LASI",
    "RASI",  # Torso
    "LHIP",
    "LKNE",
    "LANK",
    "LTOE",  # Left leg
    "RHIP",
    "RKNE",
    "RANK",
    "RTOE",  # Right leg
]


def scaling_parameters_from_c3d(file_path: str) -> RatScalingParameters:
    c3d = ezc3d.c3d(file_path)
    if "PROCESSING" not in c3d.parameters:
        raise ValueError("C3D file does not contain PROCESSING parameters.")
    params = {}
    for key in RatScalingParameters.__annotations__.keys():
        if key not in c3d.parameters["PROCESSING"]:
            raise ValueError(f"Marker {key} not found in C3D file.")
        params[key] = c3d.parameters["PROCESSING"][key]["value"][0]
    return RatScalingParameters(**params)


model_path = Path("models")
unscaled_model_path = model_path / "rat_hindlimb_bilateral.osim"
xml_path = model_path / "xml"
marker_set_path = xml_path / "rat_hindlimb_bilateral_markers.xml"
generic_setup_path = xml_path / "rat_hindlimb_bilateral_scale_setup.xml"


# TODO: Make this just use the generic setup file and fill in the gaps
def scale_opensim_model(
    name: str,
    trc_file_name: str,
    parameters: RatScalingParameters,
    output_dir: str = ".",
    time_start: float | None = None,
    time_end: float | None = None,
    scaled_model_name: str | None = None,
    marker_model_name: str | None = None,
    scale_factors_name: str | None = None,
    setup_name: str | None = None,
):
    """
    Create scaled OpenSim model from a static rat trial.

    Note: OpenSim's path handling is trash and inconsistent
    """
    output_path = Path(output_dir)
    scale_tool = osim.ScaleTool(str(generic_setup_path))
    scale_tool.setName(name)

    model_scaler: osim.ModelScaler = scale_tool.getModelScaler()
    if scaled_model_name is None:
        scaled_model_name = f"{name}_scaled.osim"
    model_scaler.setOutputModelFileName(scaled_model_name)

    if scale_factors_name is None:
        scale_factors_name = f"{name}_scale.xml"
    model_scaler.setOutputScaleFileName(scale_factors_name)

    model_scaler.setMarkerFileName(trc_file_name)

    time_range = osim.ArrayDouble()
    if time_start is None or time_end is None:
        trc = osim.MarkerData(str(trc_file_name))
        time_start = time_start or trc.getStartFrameTime()
        time_end = time_end or trc.getLastFrameTime()
    time_range.set(0, time_start)
    time_range.set(1, time_end)
    model_scaler.setTimeRange(time_range)

    subject_mass = parameters.get("Mass", None)
    if subject_mass is None:
        raise ValueError("parameters must include 'Mass' for scaling.")
    scale_tool.setSubjectMass(subject_mass)

    # Manual scaling factors
    scale_set: osim.ScaleSet = model_scaler.getScaleSet()
    scale_set.get(0).setScaleFactors(
        osim.Vec3(parameters["RFemurLength"] / base_femur_length)
    )
    scale_set.get(1).setScaleFactors(
        osim.Vec3(parameters["RTibiaLength"] / base_tibia_length)
    )
    scale_set.get(2).setScaleFactors(
        osim.Vec3(parameters["LFemurLength"] / base_femur_length)
    )
    scale_set.get(3).setScaleFactors(
        osim.Vec3(parameters["LTibiaLength"] / base_tibia_length)
    )

    marker_placer: osim.MarkerPlacer = scale_tool.getMarkerPlacer()
    if marker_model_name is None:
        marker_model_name = f"{name}_marker.osim"
    marker_placer.setOutputModelFileName(marker_model_name)
    marker_placer.setMarkerFileName(trc_file_name)
    marker_placer.setTimeRange(time_range)

    if setup_name is None:
        setup_name = f"{name}_scale_setup.xml"
    setup_path = output_path / setup_name
    scale_tool.printToXML(setup_path)

    scale_tool = osim.ScaleTool(str(setup_path))
    scale_tool.run()

    scaled_model = osim.Model()
    scaled_model.setName("")

    marker_model_path = os.path.join(output_dir, marker_model_name)
    marker_model = osim.Model(marker_model_path)
    marker_model.setName(marker_model_name.replace(".osim", ""))

    for model in [scaled_model, marker_model]:
        for side in ["L", "R"]:
            side_short = side[0].lower()
            model_body_set: osim.BodySet = model.getBodySet()

            femur_length = parameters[f"{side}FemurLength"]
            thigh: osim.Body = model_body_set.get(f"femur_{side_short}")
            thigh.set_mass(thigh_mass(subject_mass))
            thigh.set_mass_center(
                osim.Vec3(*thigh_com(side, femur_length, subject_mass))
            )
            thigh.set_inertia(
                osim.Vec6(*thigh_moi(side, femur_length, subject_mass), 0, 0, 0)
            )

            tibia_length = parameters[f"{side}TibiaLength"]
            shank: osim.Body = model_body_set.get(f"tibia_{side_short}")
            shank.set_mass(shank_mass(subject_mass))
            shank.set_mass_center(
                osim.Vec3(*shank_com(side, tibia_length, subject_mass))
            )
            shank.set_inertia(
                osim.Vec6(*shank_moi(side, tibia_length, subject_mass), 0, 0, 0)
            )

            foot_length = parameters[f"{side}FootLength"]
            foot: osim.Body = model_body_set.get(f"foot_{side_short}")
            foot.set_mass(foot_mass(subject_mass))
            foot.set_mass_center(osim.Vec3(*foot_com(side, foot_length, subject_mass)))
            foot.set_inertia(
                osim.Vec6(*foot_moi(side, foot_length, subject_mass), 0, 0, 0)
            )
        out_path = os.path.join(output_dir, model.getName() + ".osim")
        model.printToXML(out_path)

    return
