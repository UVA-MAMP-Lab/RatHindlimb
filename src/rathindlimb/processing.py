from dataclasses import dataclass
import argparse
import re
from pathlib import Path
import polars as pl
import opensim as osim
import ezc3d
from loguru import logger

from osimpy import (
    IKSettings,
    IDSettings,
    CMCSettings,
    OpenSimExternalForce,
    export_external_loads,
    export_mot,
    get_forceplate_body_mapping_from_enf,
)

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


@dataclass
class ModelBranchPaths:
    """Branch-aware paths for model artifacts.

    Intermediates are written to ``model_dir/.build/<branch>/`` so the top-level
    models directory remains uncluttered.
    """

    model_dir: str | Path
    branch: str = "with_muscles"
    keep_intermediates: bool = False

    def __post_init__(self):
        self.model_dir = Path(self.model_dir)
        self.intermediate_dir = self.model_dir / ".build" / self.branch
        self.intermediate_dir.mkdir(parents=True, exist_ok=True)

    def checkpoint_path(self, stage: str, suffix: str = ".osim") -> Path:
        stage_slug = stage.strip().replace(" ", "_").replace("/", "_")
        return self.intermediate_dir / f"{stage_slug}{suffix}"

    def temp_path(self, filename: str) -> Path:
        return self.intermediate_dir / filename

    def final_path(self, model_name: str) -> Path:
        if model_name.endswith(".osim"):
            return self.model_dir / model_name
        return self.model_dir / f"{model_name}.osim"

    def cleanup(self):
        if self.keep_intermediates or not self.intermediate_dir.exists():
            return
        for path in self.intermediate_dir.glob("*"):
            if path.is_file():
                path.unlink()


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
    save_path: str | Path | None = None,
    *,
    branch_paths: ModelBranchPaths | None = None,
    stage: str | None = None,
    final_name: str | None = None,
) -> osim.Model:
    """
    Helper function to update and save the OpenSim model.

    Returns the updated model.
    """
    if save_path is None:
        if branch_paths is not None and stage is not None:
            save_path = branch_paths.checkpoint_path(stage)
        elif branch_paths is not None and final_name is not None:
            save_path = branch_paths.final_path(final_name)
        else:
            raise ValueError(
                "Provide save_path, or provide branch_paths with either stage or final_name."
            )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    model.finalizeFromProperties()
    model.finalizeConnections()
    model.printToXML(str(save_path))

    return osim.Model(str(save_path))


def _extract_forceplate_names(c3d: ezc3d.c3d) -> list[str]:
    analog_descriptions = c3d.parameters.get("ANALOG", {}).get("DESCRIPTIONS", {}).get(
        "value", []
    )
    channel_mapping = c3d.parameters.get("FORCE_PLATFORM", {}).get("CHANNEL", {}).get(
        "value", None
    )

    if channel_mapping is None or len(analog_descriptions) == 0:
        return []

    names: list[str] = []
    for platform_index in range(channel_mapping.shape[1]):
        first_channel_index = int(channel_mapping[0, platform_index]) - 1
        if 0 <= first_channel_index < len(analog_descriptions):
            names.append(str(analog_descriptions[first_channel_index]).strip())
    return names


def _extract_forceplate_number(forceplate_name: str, fallback_index: int) -> int:
    bracket_match = re.search(r"\[(\d+)\]", forceplate_name)
    if bracket_match:
        return int(bracket_match.group(1))

    suffix_match = re.search(r"(\d+)$", forceplate_name)
    if suffix_match:
        return int(suffix_match.group(1))

    return fallback_index


def _c3d_to_trc(c3d_path: Path, trc_path: Path) -> Path:
    adapter = osim.C3DFileAdapter()
    tables = adapter.read(str(c3d_path))
    marker_table = adapter.getMarkersTable(tables)
    osim.TRCFileAdapter.write(marker_table, str(trc_path))
    return trc_path


def _c3d_to_external_loads(
    c3d_path: Path,
    trial_enf_path: Path,
    force_mot_path: Path,
    external_loads_xml_path: Path,
) -> tuple[Path, Path]:
    c3d = ezc3d.c3d(str(c3d_path), extract_forceplat_data=True)
    platforms = c3d.data.get("platform", [])
    if not platforms:
        raise ValueError(f"No force platform data found in {c3d_path}")

    forceplate_names = _extract_forceplate_names(c3d)
    if len(forceplate_names) != len(platforms):
        forceplate_names = [f"Force Plate [{index + 1}]" for index in range(len(platforms))]

    fp_to_body = get_forceplate_body_mapping_from_enf(str(trial_enf_path))
    if not fp_to_body:
        raise ValueError(f"No forceplate mapping found in ENF file: {trial_enf_path}")

    analog_rate = float(c3d.parameters["ANALOG"]["RATE"]["value"][0])
    n_force_frames = platforms[0]["force"].shape[1]
    time = [frame_index / analog_rate for frame_index in range(n_force_frames)]

    mot_dict: dict[str, list[float]] = {"time": time}
    external_forces: list[OpenSimExternalForce] = []

    for platform_index, platform_data in enumerate(platforms):
        forceplate_number = _extract_forceplate_number(
            forceplate_names[platform_index], platform_index + 1
        )

        if forceplate_number not in fp_to_body:
            continue

        force = platform_data["force"]
        cop = platform_data["center_of_pressure"]
        moment = platform_data["moment"]

        prefix = f"force{forceplate_number}"
        mot_dict[f"{prefix}_vx"] = force[0, :].tolist()
        mot_dict[f"{prefix}_vy"] = force[1, :].tolist()
        mot_dict[f"{prefix}_vz"] = force[2, :].tolist()
        mot_dict[f"{prefix}_px"] = cop[0, :].tolist()
        mot_dict[f"{prefix}_py"] = cop[1, :].tolist()
        mot_dict[f"{prefix}_pz"] = cop[2, :].tolist()
        mot_dict[f"moment{forceplate_number}_x"] = moment[0, :].tolist()
        mot_dict[f"moment{forceplate_number}_y"] = moment[1, :].tolist()
        mot_dict[f"moment{forceplate_number}_z"] = moment[2, :].tolist()

        external_forces.append(
            OpenSimExternalForce(
                name=f"FP{forceplate_number}",
                applied_to_body=fp_to_body[forceplate_number],
                force_expressed_in_body="ground",
                point_expressed_in_body="ground",
                force_identifier=f"{prefix}_v",
                point_identifier=f"{prefix}_p",
                torque_identifier=f"moment{forceplate_number}_",
            )
        )

    if len(external_forces) == 0:
        raise ValueError(
            f"No matching force platforms found between {c3d_path.name} and {trial_enf_path.name}"
        )

    force_dataframe = pl.DataFrame(mot_dict)
    export_mot(str(force_mot_path), force_dataframe)
    export_external_loads(
        str(external_loads_xml_path),
        external_forces=external_forces,
        datafile_name=force_mot_path.name,
    )

    return force_mot_path, external_loads_xml_path


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


def run_cmc(
    model_file: Path,
    initial_time: float,
    final_time: float,
    external_loads_file: Path,
    desired_kinematics_file: Path,
    results_directory: Path,
) -> CMCSettings:
    task_set_file = models_dir / "rat_hindlimb_bilateral_taskSet.xml"
    constraints_file = models_dir / "rat_hindlimb_bilateral_controlconstraints.xml"
    force_set_file = models_dir / "rat_hindlimb_bilateral_actuators.xml"

    attempts = [
        {
            "label": "baseline",
            "kwargs": {
                "constraints_file": str(constraints_file),
                "use_fast_optimization_target": True,
                "lowpass_cutoff_frequency": -1.0,
                "optimizer_max_iterations": 500,
            },
        },
        {
            "label": "relaxed_no_constraints",
            "kwargs": {
                "constraints_file": None,
                "use_fast_optimization_target": False,
                "lowpass_cutoff_frequency": 6.0,
                "optimizer_max_iterations": 1000,
            },
        },
        {
            "label": "relaxed_window",
            "kwargs": {
                "constraints_file": None,
                "use_fast_optimization_target": False,
                "lowpass_cutoff_frequency": 4.0,
                "optimizer_max_iterations": 1200,
                "initial_time": initial_time + 0.01,
                "final_time": final_time - 0.01,
            },
        },
    ]

    last_error: RuntimeError | None = None
    for attempt in attempts:
        kwargs = attempt["kwargs"]
        attempt_initial_time = kwargs.get("initial_time", initial_time)
        attempt_final_time = kwargs.get("final_time", final_time)
        logger.info(
            "Running CMC attempt '{label}' ({start:.4f}s to {end:.4f}s)",
            label=attempt["label"],
            start=attempt_initial_time,
            end=attempt_final_time,
        )

        settings = CMCSettings(
            model_file=str(model_file),
            initial_time=attempt_initial_time,
            final_time=attempt_final_time,
            external_loads_file=str(external_loads_file),
            desired_kinematics_file=str(desired_kinematics_file),
            task_set_file=str(task_set_file),
            constraints_file=kwargs.get("constraints_file"),
            force_set_files=[str(force_set_file)],
            results_directory=str(results_directory),
            use_fast_optimization_target=kwargs["use_fast_optimization_target"],
            lowpass_cutoff_frequency=kwargs["lowpass_cutoff_frequency"],
            optimizer_max_iterations=kwargs["optimizer_max_iterations"],
        )
        try:
            settings.run()
            logger.success("CMC succeeded with attempt '{label}'", label=attempt["label"])
            return settings
        except RuntimeError as error:
            last_error = error
            logger.warning(
                "CMC attempt '{label}' failed: {error}",
                label=attempt["label"],
                error=error,
            )

    raise RuntimeError(f"All CMC attempts failed. Last error: {last_error}")


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
            f"<Model name=\"{expected_name}\">",
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


def run_moco_inverse():
    raise NotImplementedError("MOCO inverse workflow is not implemented yet.")


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run rat hindlimb OpenSim workflow for a mocap session directory."
    )
    parser.add_argument(
        "session_directory",
        type=str,
        help="Path to session directory (e.g., data/mocap/BAA01/Baseline)",
    )
    parser.add_argument(
        "--trial",
        action="append",
        default=None,
        help="Specific trial stem to process (e.g., --trial Walk05). Use multiple flags for multiple trials.",
    )
    return parser


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    outputs = run_mocap_session(args.session_directory, trial_stems=args.trial)
    logger.success(
        "Workflow completed for {session}. Scaled model: {scaled_model}",
        session=outputs["session_directory"],
        scaled_model=outputs["scaled_model"],
    )
