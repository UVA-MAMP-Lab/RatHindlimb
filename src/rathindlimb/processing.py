import os
import glob
import numpy as np
import polars as pl
from .scale_utils import scale_opensim_model, RatScalingParameters
from movedb.ingest import C3DAdapter, parse_enf_file
from movedb.models import Trial
from movedb.osim import export_trc, export_mot, export_external_loads, opensim_id, opensim_ik, OpenSimExternalForce
from .filters import marker_filter, force_plate_filter

from loguru import logger

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
    if not trial.points.find_full_frames(required_markers):
        logger.warning(f"Trial {trial.name} has no frames with all required markers")
        return False
        
    # Check if we have enough data frames (at least one frame with all markers)
    # This is a simplified check - in the old API find_full_frames did more sophisticated validation
    for marker in trial.markers:
        if marker.name in required_markers:
            marker_df = marker.to_polars
            if marker_df.is_empty():
                print(f"Trial {trial.name} has no data for marker {marker.name}")
                return False
    
    for parameter in required_parameters:
        if parameter not in trial.parameters:
            logger.warning(f"Trial {trial.name} is missing required parameter {parameter}")
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
    # Check for stance swing phase events
    for side in ['Left', 'Right']:
        lead_context = side
        opposite_context = 'Right' if side == 'Left' else 'Left'
        stance_swing_events = [
            (lead_context, "Foot Strike"),
            (opposite_context, "Foot Off"),
            (opposite_context, "Foot Strike"),
            (lead_context, "Foot Off"),
            (lead_context, "Foot Strike"),
            (opposite_context, "Foot Off"),
            (opposite_context, "Foot Strike")
        ]
        phases = trial.get_event_sequences(stance_swing_events, strict=True)
        if len(phases) >= 1:
            print(f"Found {len(phases)} stance-swing phases for {side} side in trial {trial.name}")
            break
    else:
        logger.warning(f"Trial {trial.name} does not have valid stance-swing phases for Left or Right side")
        return False
    
    # Check if all required markers are present
    marker_names = {marker.name for marker in trial.markers}
    missing_markers = [m for m in required_markers if m not in marker_names]
    if missing_markers:
        print(f"Trial {trial.name} is missing required markers: {missing_markers}")
        return False
    
    # Check for gaps in required markers between events
    gaps = trial.points.get_gaps(required_markers, regions=[(first_event_frame, last_event_frame)])
    has_gaps = any(gap_list for gap_list in gaps.values())
    if has_gaps:
        logger.warning(f"Trial {trial.name} has gaps in required markers between events: {gaps}")
        return False
    
    # Check for force plate contexts labeled for left and right from ENF file
    if not trial.force_platforms:
        logger.warning(f"Trial {trial.name} has no force platforms")
        return False
    if len(trial.force_platforms) < 2:
        logger.warning(f"Trial {trial.name} has less than 2 force platforms, cannot determine left/right")
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
        
    # Convert trial markers to the old format for backward compatibility
    markers = {}
    rate = None
    units = None
    time = None
    
    for marker in trial.markers:
        if marker.name in required_markers:
            # Get data as arrays
            marker_df = marker.to_polars
            if not marker_df.is_empty():
                # Extract xyz coordinates 
                x_coords = marker_df.get_column("x").to_numpy()
                y_coords = marker_df.get_column("y").to_numpy()
                z_coords = marker_df.get_column("z").to_numpy()
                markers[marker.name] = np.column_stack([x_coords, y_coords, z_coords])
                
                # Get rate, units, time from first marker (assuming all markers have same properties)
                if rate is None:
                    rate = marker.rate
                    units = marker.units
                    # Convert timedelta to seconds for time array
                    timestamps = marker_df.get_column("timestamp").to_numpy()
                    time = np.array([t.total_seconds() for t in timestamps])
    
    if rate is None or time is None or units is None:
        raise ValueError("No valid markers found with required names")
        
    export_trc(
        filepath=trc_output_path,
        markers=markers,
        time=time,
        rate=rate,
        units=units,
        output_units=position_units,
        rotation=rotation,
    )
    
    if not output_dir:
        output_dir = trc_directory
    
    # Convert trial parameters to typed dict for scaling
    scaling_params = RatScalingParameters(
        Mass=trial.parameters["Mass"],
        RFemurLength=trial.parameters["RFemurLength"],
        RTibiaLength=trial.parameters["RTibiaLength"],
        LFemurLength=trial.parameters["LFemurLength"],
        LTibiaLength=trial.parameters["LTibiaLength"],
        RFootLength=trial.parameters["RFootLength"],
        LFootLength=trial.parameters["LFootLength"],
    )
        
    scaled_model_path, marker_model_path, scale_setup_path, scale_factors_path = scale_opensim_model(
        name=trial.name,
        unscaled_model_path=unscaled_model_path,
        marker_set_path=marker_set_path,
        marker_file_name=os.path.basename(trc_output_path),
        parameters=scaling_params,
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
                    ) -> bool:
    # Get applied bodies from the ENF file
    contacted_fp = get_applied_bodies(enf_path)
    if 'foot_l' not in contacted_fp.values() or 'foot_r' not in contacted_fp.values():
        logger.warning(f"Trial {trial.name} does not have left and right foot force platforms labeled in ENF file")
        return False
    if not valid_walk(trial, required_markers):
        raise ValueError(f"Not a valid walking trial: {trial.name}")
    
    trc_directory = os.path.dirname(trc_output_path)
    if not os.path.exists(trc_directory):
        os.makedirs(trc_directory)
    
    # Filter and export marker data
    # Convert new format to old format for backward compatibility
    markers = {}
    rate = None
    units = None
    time = None
    
    for marker in trial.markers:
        if marker.name in required_markers:
            # Get data as arrays
            marker_df = marker.to_polars
            if not marker_df.is_empty():
                # Extract xyz coordinates 
                x_coords = marker_df.get_column("x").to_numpy()
                y_coords = marker_df.get_column("y").to_numpy()
                z_coords = marker_df.get_column("z").to_numpy()
                markers[marker.name] = np.column_stack([x_coords, y_coords, z_coords])
                
                # Get rate, units, time from first marker (assuming all markers have same properties)
                if rate is None:
                    rate = marker.rate
                    units = marker.units
                    # Convert timedelta to seconds for time array
                    timestamps = marker_df.get_column("timestamp").to_numpy()
                    time = np.array([t.total_seconds() for t in timestamps])
    
    if rate is None or time is None or units is None:
        raise ValueError("No valid markers found with required names")
        
    f = marker_filter(rate)
    filtered_markers = {
        marker: np.asarray(f(data))
        for marker, data in markers.items() if marker in required_markers
    }

    export_trc(
        filepath=trc_output_path,
        markers=filtered_markers,
        time=time,
        rate=rate,
        units=units,
        output_units=position_units,
        rotation=rotation,
    )
    
    start_event = trial.events[0]
    end_event = trial.events[-1]
    start_time = start_event.time.total_seconds() if start_event.time else 0.0
    end_time = end_event.time.total_seconds() if end_event.time else 0.0

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
    
    
    
    mot_path = os.path.join(output_dir, f"{trial.name}_fp.mot")
    mot_data = pl.DataFrame()
    ext_forces = []
    for i, fp in enumerate(trial.forceplates):
        force_identifier = f'force{i+1}_v'
        point_identifier = f'force{i+1}_p'
        torque_identifier = f'moment{i+1}_'
        
        if i + 1 not in contacted_fp:
            logger.warning(f"Force platform {i+1} not found in ENF file, skipping.")
            continue
        ext_forces.append(OpenSimExternalForce(
            name=f"FP{i+1}",
            applied_to_body=contacted_fp[i + 1],
            force_identifier=force_identifier,
            point_identifier=point_identifier,
            torque_identifier=torque_identifier,
            data_source_name=os.path.basename(mot_path),
        ))

        # Apply rotation to the force platform data        
        fp_rotated = fp.rotate(rotation)  
        
        force = -fp_rotated.force
        position = fp_rotated.cop
        moment = -fp_rotated.freemoment

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

    # Get analog rate from first analog channel (assuming they're all the same)
    analog_rate = trial.analogs[0].rate if trial.analogs else None
    if analog_rate is None:
        raise ValueError("No analog data found for filtering")
        
    # Filter
    f = force_plate_filter(analog_rate)
    mot_data = mot_data.with_columns(
        pl.all()
        .map_batches(f)
    )
    
    # Get time from first force platform data
    if trial.forceplates:
        fp_timestamps = trial.forceplates[0].timestamp
        time_seconds = np.array([t.total_seconds() for t in fp_timestamps])
    else:
        # Fallback to creating time array based on data length and rate
        n_frames = len(mot_data)
        time_seconds = np.arange(n_frames) / analog_rate
        
    mot_data = mot_data.with_columns(
        pl.Series("time", time_seconds, dtype=pl.Float64)
    )    
        
    export_mot(
        filepath=mot_path,
        data=mot_data,
    )
    external_loads_path = os.path.join(output_dir, f"{trial.name}_fp_setup.xml")
    export_external_loads(
        filepath=external_loads_path,
        external_forces=ext_forces,
        datafile_name=os.path.basename(mot_path)
    )
    # TODO: Deal with paths
    id_results_path, id_setup_path = opensim_id(
        name=trial.name,
        model_path=model_path,
        ik_path=ik_results_path,
        output_dir=output_dir,
        filter_cutoff=15.0,
        external_loads_file=external_loads_path,
        excluded_forces=['Muscles']
    )
    
    return True

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
            adapter = C3DAdapter.from_file(static_file)
            trial = adapter.to_trial(name=trial_name)
            scaled_model_path, marker_model_path, scale_setup_path, scale_factors_path = process_static(
                trial,
                trc_output_path=static_file.replace('.c3d', '.trc'),
                output_dir=session_path,
                rotation=rotation,
                position_units=position_units
            )
            print(f"Successfully processed static trial: {os.path.basename(static_file)}")
            pkl_path = static_file.replace('.c3d', '.pkl')
            # TODO: Implement trial serialization for new API
            # trial.to_pkl(pkl_path)
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
            adapter = C3DAdapter.from_file(walk_file)
            trial = adapter.to_trial(name=trial_name)
            enf_file = walk_file.replace('.c3d', '.Trial.enf')
            
            success = process_walking(
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
            if not success:
                print(f"Skipping walking trial {walk_file} due to validation failure")
                continue
            print(f"Successfully processed walking trial: {os.path.basename(walk_file)}")
            walk_results = {
                'ik': walk_file.replace('.c3d', '_ik.mot'),
                'id': walk_file.replace('.c3d', '_id.sto'),
                'fp_setup': walk_file.replace('.c3d', '_fp_setup.xml'),
                'fp_mot': walk_file.replace('.c3d', '_fp.mot'),
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
            "SubjectPattern": ["Session1", "Session2",Let me first examine the current movedb models to understand the new structure, and then analyze the differences with the old version used in the processing functions.

 ...]
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