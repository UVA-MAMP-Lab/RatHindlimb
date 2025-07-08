from movedb.core import Trial
from loguru import logger

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
        logger.info(f"Trial {trial.name} has no frames with all required markers")
        return False
    for parameter in required_parameters:
        if parameter not in trial.parameters:
            logger.info(f"Trial {trial.name} is missing required parameter {parameter}")
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
        logger.info(f"Trial {trial.name} has {len(trial.events)} events, expected 7")
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
            logger.info(f"Trial {trial.name} has events out of order or with incorrect context/label")
            return False
    # Check for required markers for every frame between first and last events
    first_event = trial.events[0].get_frame(trial.points.rate) or 0
    last_event = trial.events[-1].get_frame(trial.points.rate) or trial.points.total_frames
    if trial.check_point_gaps(required_markers, regions = [(first_event, last_event)]):
        logger.info(f"Trial {trial.name} has gaps in required markers between events")
        return False
    
    # TODO: Check for force plate contexts labeled for left and right
    
    return True

