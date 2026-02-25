from typing import TypedDict
import opensim as osim
import numpy as np
import os
from pathlib import Path

# Drawn from Johnson's opensim rat model
base_femur_length: float = float(np.linalg.norm([-0.0035, -0.0312, -0.005]) * 1000)
base_tibia_length: float = float(np.linalg.norm([0.0016, 0.039, -0.0037]) * 1000)


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


def _normalized_trc_for_scaling(marker_file_name: str, output_dir: str) -> str:
    """Return marker file basename suitable for OpenSim scaling (meters).

    If input TRC is in millimeters, create a converted sibling TRC in meters and
    return that filename. Otherwise, return original filename unchanged.
    """
    marker_path = Path(output_dir) / marker_file_name
    if not marker_path.exists() or marker_path.suffix.lower() != ".trc":
        return marker_file_name

    lines = marker_path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 6:
        return marker_file_name

    meta_header_tokens = lines[1].split()
    meta_value_tokens = lines[2].split()
    if not meta_header_tokens or not meta_value_tokens:
        return marker_file_name

    try:
        units_idx = [token.lower() for token in meta_header_tokens].index("units")
    except ValueError:
        return marker_file_name
    if units_idx >= len(meta_value_tokens):
        return marker_file_name

    units = meta_value_tokens[units_idx].strip().lower()
    if units not in {"mm", "millimeter", "millimeters"}:
        return marker_file_name

    converted_name = f"{marker_path.stem}_meters{marker_path.suffix}"
    converted_path = marker_path.with_name(converted_name)
    if converted_path.exists():
        return converted_name

    meta_value_tokens[units_idx] = "m"
    lines[2] = "\t".join(meta_value_tokens)

    converted_lines = lines[:5]
    for row in lines[5:]:
        tokens = row.split()
        if len(tokens) < 2:
            converted_lines.append(row)
            continue

        converted_tokens = tokens[:2]
        for value in tokens[2:]:
            try:
                converted_tokens.append(f"{float(value) / 1000.0:.9f}")
            except ValueError:
                converted_tokens.append(value)
        converted_lines.append("\t".join(converted_tokens))

    converted_path.write_text("\n".join(converted_lines) + "\n", encoding="utf-8")
    return converted_name


def scaling_parameters_from_c3d(file_path: str) -> RatScalingParameters:
    import ezc3d

    c3d = ezc3d.c3d(file_path)
    if "PROCESSING" not in c3d.parameters:
        raise ValueError("C3D file does not contain PROCESSING parameters.")
    params = {}
    for key in RatScalingParameters.__annotations__.keys():
        if key not in c3d.parameters["PROCESSING"]:
            raise ValueError(f"Marker {key} not found in C3D file.")
        params[key] = c3d.parameters["PROCESSING"][key]["value"][0]
    return RatScalingParameters(**params)


def scale_opensim_model(
    name: str,
    unscaled_model_path: str,
    marker_set_path: str,
    marker_file_name: str,
    parameters: RatScalingParameters,
    output_dir: str = ".",
    scale_setup_path: str | None = None,
    time_start: float | None = None,
    time_end: float | None = None,
):
    """
    Create scaled OpenSim models (one with markers moved, one without) from a static rat trial.

    Note: OpenSim's path handling is trash and inconsistent
    """
    unscaled_model_path = os.path.abspath(unscaled_model_path)
    marker_set_path = os.path.abspath(marker_set_path)
    output_dir = os.path.abspath(output_dir)
    marker_file_name = os.path.basename(
        marker_file_name
    )  # Ensure we only use the file name, not the path
    marker_file_name = _normalized_trc_for_scaling(marker_file_name, output_dir)

    if scale_setup_path is not None and os.path.exists(scale_setup_path):
        scale_tool = osim.ScaleTool(os.path.abspath(scale_setup_path))
    else:
        scale_tool = osim.ScaleTool()
    scale_tool.setName(name)

    model_scaler: osim.ModelScaler = scale_tool.getModelScaler()
    model_scaler.setApply(True)
    scaling_order = osim.ArrayStr()
    scaling_order.append("manualScale")
    model_scaler.setScalingOrder(scaling_order)
    scaled_model_path = os.path.join(output_dir, f"{name}_scaled.osim")
    model_scaler.setOutputModelFileName(scaled_model_path)
    scale_factors_path = os.path.join(output_dir, f"{name}_scale.xml")
    model_scaler.setOutputScaleFileName(scale_factors_path)
    model_scaler.setMarkerFileName(marker_file_name)

    time_range = osim.ArrayDouble()
    # TODO: handle None
    time_range.set(0, time_start)
    time_range.set(1, time_end)
    model_scaler.setTimeRange(time_range)

    subject_mass = parameters.get("Mass", None)
    if subject_mass is None:
        raise ValueError("parameters must include 'Mass' for scaling.")
    scale_tool.setSubjectMass(subject_mass)

    # Manual scaling factors - This is probably the only thing before run that cannot be abstracted out
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
    marker_placer.setApply(True)
    marker_model_name = f"{name}_marker.osim"
    marker_placer.setOutputModelFileName(marker_model_name)
    marker_placer.setMarkerFileName(marker_file_name)
    marker_placer.setTimeRange(time_range)

    generic_model_maker: osim.GenericModelMaker = scale_tool.getGenericModelMaker()
    generic_model_maker.setModelFileName(unscaled_model_path)
    generic_model_maker.setMarkerSetFileName(marker_set_path)

    new_scale_setup_path = os.path.join(output_dir, f"{name}_scale_setup.xml")
    scale_tool.printToXML(new_scale_setup_path)

    scale_tool = osim.ScaleTool(
        new_scale_setup_path
    )  # I don't think this is necessary, but it seems to be MAMP convention

    scale_tool.run()

    scaled_model = osim.Model(scaled_model_path)
    scaled_model.setName(scaled_model_path.replace(".osim", ""))

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

    return (
        scaled_model_path,
        marker_model_path,
        new_scale_setup_path,
        scale_factors_path,
    )
