import opensim as osim
import numpy as np
import pandas as pd

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

def optimize_model_tsl(model: osim.Model, 
                       muscle_lengths: dict[str, pd.DataFrame], 
                       lm_norm_range: tuple[float, float] = (0.5, 1.6),
                       method: str = 'SLSQP',
                       objective: str = 'ssdp',
                       max_tsl: float = 1.5
                       ) -> osim.Model:
    for muscle_name, muscle_data in muscle_lengths.items():
        muscle = model.getMuscles().get(muscle_name)
        if muscle is None:
            print(f"Warning: Muscle {muscle_name} not found in the model.")
            continue
        opt = TSLOptimization.from_osim_muscle(muscle, lm_norm_range)
        lmt_raw = muscle_data['length']
        lmt = lmt_raw.drop_duplicates().sort_values().to_numpy()
        lts = opt.optimize(lmt, method=method, objective=objective)
        # print(f"Muscle: {muscle_name}, Tendon slack lengths: {lts}")
        lts = lts[lts > np.finfo(float).eps]  # Remove any zero values
        lts = lts[lts < max_tsl]
        if len(lts) == 0:
            print(f"Warning: No valid tendon slack lengths found for muscle {muscle_name}. Setting it to rigid.")
            model.getMuscles().get(muscle_name).set_ignore_tendon_compliance(True)
            continue
        # Set the tendon slack length in the model
        model.getMuscles().get(muscle_name).set_tendon_slack_length(np.mean(lts))
    return model

def attachments_to_csv(model: osim.Model, filename: str) -> bool:
    """
    Write the muscle attachment points to a CSV file.
    
    Columns will be:
    - muscle_name
    - frame_name
    - x
    - y
    - z
    """
    try:
        # Initialize the system 
        model.initSystem()
        with open(filename, 'w') as f:
            f.write("muscle_name,frame_name,x,y,z\n")
            muscles : osim.SetMuscles = model.getMuscles()
            for i in range(muscles.getSize()):
                muscle : osim.Muscle = muscles.get(i)
                geo_path : osim.GeometryPath = muscle.getGeometryPath()
                if geo_path is None:
                    continue
                path_points : osim.PathPointSet = geo_path.getPathPointSet()
                for j in range(path_points.getSize()):
                    path_point : osim.PathPoint = osim.PathPoint.safeDownCast(path_points.get(j))
                    frame : osim.PhysicalFrame = path_point.getParentFrame()
                    loc : osim.Vec3 = path_point.get_location()
                    x = loc.get(0)
                    y = loc.get(1)
                    z = loc.get(2)
                    f.write(f"{muscle.getName()},{frame.getName()},{x},{y},{z}\n")   
    except Exception as e:
        print(f"Error writing attachment points to {filename}: {e}")
        return False
    
    print(f"Attachment points written to {filename}.")
    return True