import re
from pathlib import Path
import opensim as osim
import ezc3d
from loguru import logger

from rathindlimb.scale_utils import (
    RatScalingParameters,
    scale_opensim_model,
    scaling_parameters_from_c3d,
)

# Model and setup file paths
project_root = Path(__file__).resolve().parents[2]
models_dir = project_root / "models" / "osim"
unscaled_model_path = models_dir / "rat_hindlimb_bilateral.osim"
marker_set_path = models_dir / "rat_hindlimb_bilateral_markerset.xml"
generic_scale_setup_path = models_dir / "rat_hindlimb_bilateral_scale_setup.xml"
generic_ik_setup_path = models_dir / "rat_hindlimb_bilateral_ik_setup.xml"

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

required_parameters = [
    "Mass",
    "Length",
    "RFemurLength",
    "RTibiaLength",
    "RFootLength",
    "LFemurLength",
    "LTibiaLength",
    "LFootLength",
]


def remove_muscles(model: osim.Model) -> osim.Model:
    """Remove all muscles from a model in-place and return the same model."""
    force_set: osim.ForceSet = model.upd_ForceSet()
    indices_to_remove = []
    for i in range(force_set.getSize()):
        if osim.Muscle.safeDownCast(force_set.get(i)) is not None:
            indices_to_remove.append(i)

    for i in indices_to_remove[::-1]:
        force_set.remove(i)

    return model


def update_model(
    model: osim.Model,
    save_path: str | Path,
) -> osim.Model:
    """
    Helper function to update and save the OpenSim model.

    Returns the updated model.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.finalizeFromProperties()
    model.finalizeConnections()
    model.printToXML(str(save_path))

    return osim.Model(str(save_path))


def _c3d_to_trc(c3d_path: Path, trc_path: Path) -> Path:
    adapter = osim.C3DFileAdapter()
    tables = adapter.read(str(c3d_path))
    marker_table = adapter.getMarkersTable(tables)
    osim.TRCFileAdapter.write(marker_table, str(trc_path))
    return trc_path


def run_ik(
    model_file: Path,
    marker_file: Path,
    output_motion_file: Path,
    results_directory: Path,
    use_existing: bool = True,
) -> Path:
    if use_existing and output_motion_file.exists():
        logger.info(f"Using existing IK output: {output_motion_file.name}")
        return output_motion_file

    target_output_file = output_motion_file
    if output_motion_file.exists() and not use_existing:
        target_output_file = output_motion_file.with_name(
            f"{output_motion_file.stem}_new{output_motion_file.suffix}"
        )

    settings = IKSettings(
        model_file=str(model_file),
        marker_file=str(marker_file),
        output_motion_file=str(target_output_file),
        results_directory=str(results_directory),
    )
    result = settings.run()
    return Path(result.motion_file)


def run_id(
    model_file: Path,
    coordinates_file: Path,
    external_loads_file: Path,
    output_forces_file: Path,
    results_directory: Path,
    use_existing: bool = True,
) -> Path:
    if use_existing and output_forces_file.exists():
        logger.info(f"Using existing ID output: {output_forces_file.name}")
        return output_forces_file

    target_output_file = output_forces_file
    if output_forces_file.exists() and not use_existing:
        target_output_file = output_forces_file.with_name(
            f"{output_forces_file.stem}_new{output_forces_file.suffix}"
        )

    settings = IDSettings(
        model_file=str(model_file),
        coordinates_file=str(coordinates_file),
        external_loads_file=str(external_loads_file),
        output_forces_file=str(target_output_file),
        results_directory=str(results_directory),
        lowpass_cutoff_frequency=15.0,
        excluded_forces=["Muscles"],
    )
    result = settings.run()
    return Path(result.forces_file)


def _get_time_range(motion_file: Path) -> tuple[float, float]:
    storage = osim.Storage(str(motion_file))
    return storage.getFirstTime(), storage.getLastTime()


def _sanitize_model_name(model_path: Path) -> None:
    expected_name = model_path.stem
    try:
        model = osim.Model(str(model_path))
        if str(model.getName()) != expected_name:
            model.setName(expected_name)
            model.printToXML(str(model_path))
        return
    except Exception:
        content = model_path.read_text(encoding="utf-8")
        updated_content = re.sub(
            r"<Model name=\"[^\"]+\">",
            f'<Model name="{expected_name}">',
            content,
            count=1,
        )
        model_path.write_text(updated_content, encoding="utf-8")


def run_mocap_session(
    session_directory: str | Path,
    trial_stems: list[str] | None = None,
) -> dict[str, Path]:
    """Run C3D to Scale+IK+ID+CMC workflow for a mocap session.

    Args:
        session_directory: Directory containing static and walking C3D trials.

    Returns:
        Dictionary with key output paths.
    """
    session_path = Path(session_directory).resolve()
    if not session_path.exists():
        raise FileNotFoundError(f"Session directory not found: {session_path}")

    static_candidates = sorted(session_path.glob("Static*.c3d"))
    if len(static_candidates) == 0:
        raise FileNotFoundError(f"No static C3D files found in {session_path}")
    static_c3d = static_candidates[-1]

    walk_candidates = sorted(session_path.glob("Walk*.c3d"))
    if trial_stems:
        stem_set = {stem.strip() for stem in trial_stems if stem.strip()}
        walk_candidates = [trial for trial in walk_candidates if trial.stem in stem_set]
    if len(walk_candidates) == 0:
        raise FileNotFoundError(f"No walking C3D files found in {session_path}")

    logger.info(f"Using static trial: {static_c3d.name}")
    static_trc = session_path / f"{static_c3d.stem}.trc"
    if not static_trc.exists():
        _c3d_to_trc(static_c3d, static_trc)

    scale_params: RatScalingParameters = scaling_parameters_from_c3d(str(static_c3d))
    static_c3d_data = ezc3d.c3d(str(static_c3d), extract_forceplat_data=False)
    point_rate = float(static_c3d_data.parameters["POINT"]["RATE"]["value"][0])
    first_frame = int(static_c3d_data.header["points"]["first_frame"])
    last_frame = int(static_c3d_data.header["points"]["last_frame"])
    time_start = first_frame / point_rate
    time_end = last_frame / point_rate

    scaled_model_path_str, marker_model_path_str, _, _ = scale_opensim_model(
        name=static_c3d.stem,
        unscaled_model_path=str(unscaled_model_path),
        marker_set_path=str(marker_set_path),
        marker_file_name=static_trc.name,
        parameters=scale_params,
        output_dir=str(session_path),
        scale_setup_path=str(generic_scale_setup_path),
        time_start=time_start,
        time_end=time_end,
    )

    marker_model_path = Path(marker_model_path_str)
    scaled_model_path = Path(scaled_model_path_str)
    _sanitize_model_name(marker_model_path)
    _sanitize_model_name(scaled_model_path)

    completed_trials: list[Path] = []
    for walk_c3d in walk_candidates:
        trial_name = walk_c3d.stem
        trial_enf_path = session_path / f"{trial_name}.Trial.enf"
        if not trial_enf_path.exists():
            logger.warning(f"Skipping {trial_name}: missing {trial_enf_path.name}")
            continue

        logger.info(f"Processing trial: {trial_name}")
        trc_path = session_path / f"{trial_name}.trc"
        if not trc_path.exists():
            _c3d_to_trc(walk_c3d, trc_path)

        fp_mot_path = session_path / f"{trial_name}_fp.mot"
        external_loads_xml = session_path / f"{trial_name}_fp_setup.xml"
        _c3d_to_external_loads(
            c3d_path=walk_c3d,
            trial_enf_path=trial_enf_path,
            force_mot_path=fp_mot_path,
            external_loads_xml_path=external_loads_xml,
        )

        ik_motion_file = session_path / f"{trial_name}_ik.mot"
        run_ik(
            model_file=marker_model_path,
            marker_file=trc_path,
            output_motion_file=ik_motion_file,
            results_directory=session_path,
        )

        id_forces_file = session_path / f"{trial_name}_id.sto"
        run_id(
            model_file=marker_model_path,
            coordinates_file=ik_motion_file,
            external_loads_file=external_loads_xml,
            output_forces_file=id_forces_file,
            results_directory=session_path,
        )

        initial_time, final_time = _get_time_range(ik_motion_file)
        try:
            run_cmc(
                model_file=scaled_model_path,
                initial_time=initial_time,
                final_time=final_time,
                external_loads_file=external_loads_xml,
                desired_kinematics_file=ik_motion_file,
                results_directory=session_path,
            )
        except RuntimeError as cmc_error:
            logger.warning(f"CMC failed for {trial_name}: {cmc_error}")
        completed_trials.append(walk_c3d)

    if len(completed_trials) == 0:
        raise RuntimeError(f"No trials processed in {session_path}")

    return {
        "session_directory": session_path,
        "scaled_model": scaled_model_path,
        "marker_model": marker_model_path,
        "last_trial": completed_trials[-1],
    }
