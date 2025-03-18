import opensim as osim
import numpy as np

from src.musculoskeletal_graph import MusculoskeletalGraph
from src.manal_optimization import TSLOptimization

def thelen_to_millard(thelen : osim.Thelen2003Muscle) -> osim.Millard2012EquilibriumMuscle:
    """Convert Thelen2003Muscle to Millard2012EquilibriumMuscle."""
    try:
        thelen = osim.Thelen2003Muscle.safeDownCast(thelen)
    except:
        print(f"Input muscle {thelen.getName()} is not a Thelen2003Muscle object.")
        return None

    millard = osim.Millard2012EquilibriumMuscle()
    millard.setName(thelen.getName())

    # Copy geometry path
    millard.set_GeometryPath(thelen.get_GeometryPath())

    # Max isometric force
    millard.set_max_isometric_force(thelen.get_max_isometric_force())

    # Optimal fiber length
    millard.set_optimal_fiber_length(thelen.get_optimal_fiber_length())

    # Tendon slack length    
    millard.set_tendon_slack_length(thelen.get_tendon_slack_length())
    
    # Pennation angle at optimal
    millard.set_pennation_angle_at_optimal(thelen.get_pennation_angle_at_optimal())

    # Ignore tendon compliance
    millard.set_ignore_tendon_compliance(thelen.get_ignore_tendon_compliance())

    # Fiber damping
    millard.set_fiber_damping(0.1)

    # Default activation
    millard.set_default_activation(thelen.get_default_activation())

    # Minimum activation
    millard.set_minimum_activation(thelen.get_minimum_activation())

    # Active Force Length Curve
    millard.set_ActiveForceLengthCurve(osim.ActiveForceLengthCurve())

    # Force Velocity Curve
    millard.set_ForceVelocityCurve(osim.ForceVelocityCurve())

    # Fiber Force Length Curve
    millard.set_FiberForceLengthCurve(osim.FiberForceLengthCurve())

    # Tendon Force Length Curve
    millard.set_TendonForceLengthCurve(osim.TendonForceLengthCurve())

    return millard 


def model_thelen_to_millard(model : osim.Model) -> osim.Model:
    """Convert all Thelen2003Muscle to Millard2012EquilibriumMuscle in the model."""
    force_set : osim.ForceSet = model.upd_ForceSet()
    indices_to_remove = []
    for i in range(force_set.getSize()):
        try:
            muscle = osim.Thelen2003Muscle.safeDownCast(force_set.get(i))
            if muscle is None:
                continue
        except:
            continue
        millard = thelen_to_millard(muscle)
        if millard:
            # force_set.set(i, millard) # This does not work
            force_set.append(millard)
            indices_to_remove.append(i)
            # print(f"Converted muscle {muscle.getName()} to Millard2012EquilibriumMuscle.")
    for i in indices_to_remove[::-1]:
        force_set.remove(i)
    return model

def optimize_model_tsl(model: osim.Model) -> osim.Model:
    """Calculate tendon slack lengths using range of motion for each muscle in the model."""
    
    new_model = model.clone()
    graph = MusculoskeletalGraph(new_model)
    
    muscle_lengths = graph.get_all_muscle_lengths_rom(min_points=10)
    
    for muscle_name, muscle_data in muscle_lengths.items():
        muscle = graph.get_muscle(muscle_name)
        opt = TSLOptimization.from_osim_muscle(muscle, lm_norm_range=(0.5, 1.6))
        lmt_raw = muscle_data[muscle_name]
        lmt = lmt_raw.drop_duplicates().sort_values().to_numpy()
        lts = opt.optimize(lmt, method='SLSQP', objective='ssdp')
        # print(f"Muscle: {muscle_name}, Tendon slack lengths: {lts}")
        lts = lts[lts > np.finfo(float).eps]  # Remove any zero values
        lts = lts[lts < 1.5]
        if len(lts) == 0:
            print(f"Warning: No valid tendon slack lengths found for muscle {muscle_name}. Setting it to rigid.")
            new_model.getMuscles().get(muscle_name).set_ignore_tendon_compliance(True)
            continue
        # Set the tendon slack length in the new model
        new_model.getMuscles().get(muscle_name).set_tendon_slack_length(np.mean(lts))
    return new_model