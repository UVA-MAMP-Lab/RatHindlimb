import os
import glob
import numpy as np
import polars as pl
from .scale_utils import scale_opensim_model
from movedb.core import Trial
from movedb.file_io import (
    parse_enf_file, 
    export_trc, 
    opensim_ik, 
    opensim_id, 
    OpenSimExternalForce,
    export_external_loads,
    export_mot,
)
from .filters import marker_filter, force_plate_filter

import warnings

# Model and setup file paths
unscaled_model_path = os.path.join('models', 'osim', 'FinalBilateral_millard_tsl_gait.osim')
marker_set_path = os.path.join('models', 'osim', 'rat_hindlimb_bilateral_markerset.xml')
generic_scale_setup_path = os.path.join('models', 'osim', 'rat_hindlimb_bilateral_scale_setup.xml')
generic_ik_setup_path = os.path.join('models', 'osim', 'rat_hindlimb_bilateral_ik_setup.xml')

# Define required markers and parameters for validation
required_markers = [
    "TAIL", "SPL6", "LASI", "RASI",  # Torso
    "LHIP", "LKNE", "LANK", "LTOE",  # Left leg
    "RHIP", "RKNE", "RANK", "RTOE"   # Right leg
]

required_parameters = [
    "Mass", "Length", "RFemurLength", "RTibiaLength", "RFootLength",
    "LFemurLength", "LTibiaLength", "LFootLength"
]

def valid_static(trial: Trial, 
                 required_markers: list[str],
                 required_parameters: list[str]
                 ) -> bool:
    """
    For rat trials, a valid static trial will have:
        - At least one frame with all required markers
        - OpenSim takes the average of all frames to calculate marker distances for scaling
        - Parameters needed for scaling
    """
    # There must be at least one frame with all required markers
    if not trial.find_full_frames(required_markers):
        warnings.warn(f"Trial {trial.name} has no frames with all required markers")
        return False
    for parameter in required_parameters:
        if parameter not in trial.parameters:
            warnings.warn(f"Trial {trial.name} is missing required parameter {parameter}")
            return False
    return True

def valid_walk(trial: Trial, required_markers: list[str]) -> bool:
    """
    For rat trials, a valid walk trial will have:
        - 7 events in frame / time order:
        - Lead Foot Strike
        - Opposite Foot Off
        - Opposite Foot Strike
        - Lead Foot Off
        - Lead Foot Strike
        - Opposite Foot Off
        - Opposite Foot Strike
        - Markers for every frame in events
        - Force plate contexts labeled for left and right
        - If > 1 force plate labeled for either left or right, must be consecutive (?)
    """
    # Check order of events
    if len(trial.events) != 7:
        warnings.warn(f"Trial {trial.name} has {len(trial.events)} events, expected 7")
        return False
    # Check for correct context + label order (Events should already be in order by frame/time)
    lead_context = trial.events[0].context
    opposite_context = "Right" if lead_context == "Left" else "Left"
    expected_order = [
        (lead_context, "Foot Strike"),
        (opposite_context, "Foot Off"),
        (opposite_context, "Foot Strike"),
        (lead_context, "Foot Off"),
        (lead_context, "Foot Strike"),
        (opposite_context, "Foot Off"),
        (opposite_context, "Foot Strike"),
    ]        
    for event, expected in zip(trial.events, expected_order):
        if (event.context, event.label) != expected:
            warnings.warn(f"Trial {trial.name} has events out of order or with incorrect context/label")
            return False
    # Check for required markers for every frame between first and last events
    first_event_frame = trial.events[0].get_frame(trial.points.rate)
    last_event_frame = trial.events[-1].get_frame(trial.points.rate)
    
    # Ensure we have valid frame numbers
    if first_event_frame is None:
        warnings.warn(f"Trial {trial.name} first event has no valid frame number")
        return False
    if last_event_frame is None:
        warnings.warn(f"Trial {trial.name} last event has no valid frame number")
        return False
    
    # Check for gaps in required markers between events
    gaps = trial.check_point_gaps(required_markers, regions=[(first_event_frame, last_event_frame)])
    if gaps:
        # Check if any marker has gaps
        has_gaps = any(gap_list for gap_list in gaps.values())
        if has_gaps:
            warnings.warn(f"Trial {trial.name} has gaps in required markers between events: {gaps}")
            return False
    
    return True

def process_static(trial: Trial,
                   trc_output_path: str,
                   output_dir: str = '',
                   rotation: np.ndarray = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]),
                   position_units: str = 'm',
                   ):
    if not valid_static(trial, required_markers, required_parameters):
        raise ValueError(f"Not a valid static trial: {trial.name}")
    
    trc_directory = os.path.dirname(trc_output_path)
    if not os.path.exists(trc_directory):
        os.makedirs(trc_directory)
        
    export_trc(
        filepath=trc_output_path,
        markers=trial.points.to_dict(include_residual=False),
        time=trial.points.time,
        rate=trial.points.rate,
        units=trial.points.units,
        output_units=position_units,
        rotation=rotation,
    )
    
    if not output_dir:
        output_dir = trc_directory
        
    scaled_model_path, marker_model_path, scale_setup_path, scale_factors_path = scale_opensim_model(
        trial,
        unscaled_model_path=unscaled_model_path,
        marker_set_path=marker_set_path,
        marker_file_name=os.path.basename(trc_output_path),
        output_dir=output_dir,
        scale_setup_path=generic_scale_setup_path,
    )
    # TODO: Find a better way to output paths
    return scaled_model_path, marker_model_path, scale_setup_path, scale_factors_path

def get_applied_bodies(enf_path: str,
                       mapping: dict[str, str] = {'Left': 'foot_l', 'Right': 'foot_r'},
                       ) -> dict[int, str]:
    """
    Parse the ENF file to determine which force platforms are applied to which bodies.
    This function assumes the ENF file contains keys like 'FP1', 'FP2', etc., mapping to context names as in Vicon Nexus.

    Args:
        enf_path (str): Path to the ENF file.
        mapping (dict[str, str]): Mapping of force platform context names to body names.
    Returns:
        dict[int, str]: Dictionary mapping force platform indices to body names.
    """
    # Read .enf file to determine applied bodies
    enf_data = parse_enf_file(enf_path)
    # Find keys that match the pattern 'fp%d' where %d is a digit and extract the digit and value
    contacted_fp = {}
    for key, value in enf_data.items():
        if key.lower().startswith('fp') and key[2:].isdigit():
            body = mapping.get(str(value), None)
            if body is None:
                continue
            contacted_fp[int(key[2:])] = body
    return contacted_fp

def process_walking(trial: Trial,
                    model_path: str,
                    enf_path: str,
                    trc_output_path: str,
                    output_dir: str = '',
                    rotation: np.ndarray = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]),
                    position_units: str = 'm',
                    force_units: str = 'N',
                    moment_units: str = 'Nm',
                    ):
    
    if not valid_walk(trial, required_markers):
        raise ValueError(f"Not a valid walking trial: {trial.name}")
    
    trc_directory = os.path.dirname(trc_output_path)
    if not os.path.exists(trc_directory):
        os.makedirs(trc_directory)
    
    # Filter and export marker data
    markers = trial.points.to_dict(include_residual=False)
    f = marker_filter(trial.points.rate)
    filtered_markers = {
        marker: np.asarray(f(data))
        for marker, data in markers.items() if marker in required_markers
    }

    export_trc(
        filepath=trc_output_path,
        markers=filtered_markers,
        time=trial.points.time,
        rate=trial.points.rate,
        units=trial.points.units,
        output_units=position_units,
        rotation=rotation,
    )
    
    start_event = trial.events[0]
    end_event = trial.events[-1]
    start_time = start_event.get_time(trial.points.rate)
    end_time = end_event.get_time(trial.points.rate)

    if not output_dir:
        output_dir = trc_directory

    ik_results_path, ik_setup_path = opensim_ik(
        name=trial.name,
        model_path=model_path,
        trc_path=trc_output_path,
        output_dir=output_dir,
        start_time=start_time,
        end_time=end_time,
    )
    
    # Get applied bodies from the ENF file
    contacted_fp = get_applied_bodies(enf_path)
    
    mot_path = os.path.join(output_dir, f"{trial.name}_fp.mot")
    mot_data = pl.DataFrame()
    ext_forces = []
    for i, fp in enumerate(trial.force_platforms):
        force_identifier = f'force{i+1}_v'
        point_identifier = f'force{i+1}_p'
        torque_identifier = f'moment{i+1}_'
        
        if i + 1 not in contacted_fp:
            warnings.warn(f"Force platform {i+1} not found in ENF file, skipping.")
            continue
        ext_forces.append(OpenSimExternalForce(
            name=f"FP{i+1}",
            applied_to_body=contacted_fp[i + 1],
            force_identifier=force_identifier,
            point_identifier=point_identifier,
            torque_identifier=torque_identifier,
            data_source_name=mot_path
        ))

        # Apply rotation to the force platform data        
        fp = fp.apply_rotation(rotation)  
        
        force = -fp.get_force(force_units)
        position = fp.get_center_of_pressure(position_units)
        moment = -fp.get_free_moment(moment_units)

        mot_data = mot_data.with_columns(
            pl.Series(force_identifier + 'x', force[:, 0], dtype=pl.Float64),
            pl.Series(force_identifier + 'y', force[:, 1], dtype=pl.Float64),
            pl.Series(force_identifier + 'z', force[:, 2], dtype=pl.Float64),
            pl.Series(point_identifier + 'x', position[:, 0], dtype=pl.Float64),
            pl.Series(point_identifier + 'y', position[:, 1], dtype=pl.Float64),
            pl.Series(point_identifier + 'z', position[:, 2], dtype=pl.Float64),
            pl.Series(torque_identifier + 'x', moment[:, 0], dtype=pl.Float64),
            pl.Series(torque_identifier + 'y', moment[:, 1], dtype=pl.Float64),
            pl.Series(torque_identifier + 'z', moment[:, 2], dtype=pl.Float64),
        )

    # Filter
    f = force_plate_filter(trial.analogs.rate)
    mot_data = mot_data.with_columns(
        pl.all()
        .map_batches(f)
    )
        
    mot_data = mot_data.with_columns(
        pl.Series("time", trial.analogs.time, dtype=pl.Float64)
    )    
        
    export_mot(
        filepath=mot_path,
        data=mot_data,
    )
    external_loads_path = os.path.join(output_dir, f"{trial.name}_fp_setup.xml")
    export_external_loads(
        filepath=external_loads_path,
        external_forces=ext_forces,
        datafile_name=mot_path
    )
    
    id_results_path, id_setup_path = opensim_id(
        name=trial.name,
        model_path=model_path,
        ik_path=ik_results_path,
        output_dir=output_dir,
        filter_cutoff=15.0,
        external_loads_file=external_loads_path,
        excluded_forces=['Muscles']
    )

def process_session(session_path: str,
                    rotation: np.ndarray = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]),
                    position_units: str = 'm',
                    force_units: str = 'N',
                    moment_units: str = 'Nm',
):
    session_path = os.path.normpath(session_path)
    if not os.path.isdir(session_path):
        raise ValueError(f"Session path is not a directory: {session_path}")
    
    
    session = os.path.basename(session_path)
    classification = os.path.basename(os.path.dirname(session_path))
    
    results = {
        'static_trial': None, # .pkl for static trial
        'scaled_model': None, # .osim for scaled model
        'walking_trials': {} # {<trial_name>.pkl: {'ik': <ik_path>, 'id': <id_path>, 'grf': <grf_path>}}
    }
    # Look for static trials (files with "Static" in the name)
    static_pattern = os.path.join(session_path, "*Static*.c3d")
    static_files = sorted(glob.glob(static_pattern), reverse=True)  # Reverse alphabetical
    for static_file in static_files:
        try:
            trial_name=os.path.basename(static_file).replace('.c3d', '')
            trial = Trial.from_c3d_file(static_file, trial_name=trial_name, session_name=session, classification=classification)
            scaled_model_path, marker_model_path, scale_setup_path, scale_factors_path = process_static(
                trial,
                trc_output_path=static_file.replace('.c3d', '.trc'),
                output_dir=session_path,
                rotation=rotation,
                position_units=position_units
            )
            print(f"Successfully processed static trial: {os.path.basename(static_file)}")
            pkl_path = static_file.replace('.c3d', '.pkl')
            trial.to_pkl(pkl_path)
            results['static_trial'] = pkl_path
            results['scaled_model'] = marker_model_path
            break
        except Exception as e:
            print(f"Error processing static trial {static_file}: {e}")
            continue
    else:
        # Handle case where no valid static trial was found
        print(f"No valid static trial found in {session_path}")
        return
    walk_pattern = os.path.join(session_path, "*Walk*.c3d")
    walking_trials = sorted(glob.glob(walk_pattern))
    for walk_file in walking_trials:
        try:
            trial_name = os.path.basename(walk_file).replace('.c3d', '')
            trial = Trial.from_c3d_file(walk_file, trial_name=trial_name, session_name=session, classification=classification)
            enf_file = walk_file.replace('.c3d', '.Trial.enf')
            process_walking(
                trial,
                model_path=marker_model_path,
                enf_path=enf_file,
                trc_output_path=walk_file.replace('.c3d', '.trc'),
                output_dir=session_path,
                rotation=rotation,
                position_units=position_units,
                force_units=force_units,
                moment_units=moment_units,
            )
            print(f"Successfully processed walking trial: {os.path.basename(walk_file)}")
            pkl_path = walk_file.replace('.c3d', '.pkl')
            trial.to_pkl(pkl_path)
            walk_results = {
                'ik': walk_file.replace('.c3d', '_ik.mot'),
                'id': walk_file.replace('.c3d', '_id.sto'),
                'grf': walk_file.replace('.c3d', '_fp_setup.xml')
            }
            results['walking_trials'][pkl_path] = walk_results
        except Exception as e:
            print(f"Error processing walking trial {walk_file}: {e}")
            continue
    return results

def create_session_globs(spec: dict) -> list[str]:
    """
    Create glob patterns for session directories based on the control group specification.
    
    Specification format:
    {
        "Classification": {
            "SubjectPattern": ["Session1", "Session2", ...]
        }
    }
    """
    session_globs = []
    for classification, subjects in spec.items():
        for subject_pattern, sessions in subjects.items():
            for session in sessions:
                if subject_pattern == "*":
                    # Use glob wildcard for all subjects
                    pattern = os.path.join(classification, "*", session)
                else:
                    # Use specific subject name
                    pattern = os.path.join(classification, subject_pattern, session)
                session_globs.append(pattern)
                print(f"Created pattern: {pattern}")  # Debug output
    return session_globs

def process_spec(root_dir: str, spec: dict) -> dict:
    """
    Process the control group specification to create a structured dictionary of session globs.
    """
    results = {}
    for session_glob in create_session_globs(spec):
        # Find all matching session directories
        full_pattern = os.path.join(root_dir, session_glob)
        print(f"Searching with pattern: {full_pattern}")
        session_dirs = glob.glob(full_pattern)
        print(f"Found {len(session_dirs)} matching directories")
        
        for session_dir in session_dirs:
            # Process each session directory
            print(f"Processing session: {session_dir}")
            try:
                session_results = process_session(session_dir)
                classification = os.path.basename(os.path.dirname(os.path.dirname(session_dir)))
                subject = os.path.basename(os.path.dirname(session_dir))
                session_name = os.path.basename(session_dir)

                if classification not in results:
                    results[classification] = {}
                if subject not in results[classification]:
                    results[classification][subject] = {}
                results[classification][subject][session_name] = session_results

            except Exception as e:
                print(f"Error processing session {session_dir}: {e}")
    return results