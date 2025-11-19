from pyopensim.simulation import (
    Thelen2003Muscle, 
    Millard2012EquilibriumMuscle,
    ActiveForceLengthCurve,
    ForceVelocityCurve,
    FiberForceLengthCurve,
    TendonForceLengthCurve,
    Model,
    ForceSet,
    Muscle,
    SetMuscles,
    PathPoint,
    PathPointSet,
    GeometryPath,
    PhysicalFrame
)   
from pyopensim.simbody import Vec3
def thelen_to_millard(thelen : Thelen2003Muscle) -> Millard2012EquilibriumMuscle:
    """Convert Thelen2003Muscle to Millard2012EquilibriumMuscle."""
    try:
        thelen = Thelen2003Muscle.safeDownCast(thelen)
    except:
        raise TypeError(f"Input muscle {thelen.getName()} is not a Thelen2003Muscle object.")
    millard = Millard2012EquilibriumMuscle()
    millard.setName(thelen.getName())

    # Copy geometry path
    millard.set_path(thelen.getGeometryPath())

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
    millard.set_ActiveForceLengthCurve(ActiveForceLengthCurve())

    # Force Velocity Curve
    millard.set_ForceVelocityCurve(ForceVelocityCurve())

    # Fiber Force Length Curve
    millard.set_FiberForceLengthCurve(FiberForceLengthCurve())

    # Tendon Force Length Curve
    millard.set_TendonForceLengthCurve(TendonForceLengthCurve())

    return millard 

def model_thelen_to_millard(model : Model) -> Model:
    """Convert all Thelen2003Muscle to Millard2012EquilibriumMuscle in the model."""
    force_set : ForceSet = model.upd_ForceSet()
    indices_to_remove = []
    for i in range(force_set.getSize()):
        try:
            muscle = Thelen2003Muscle.safeDownCast(force_set.get(i))
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

def attachments_to_csv(model: Model, filename: str) -> bool:
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
            muscles : SetMuscles = model.getMuscles()
            for i in range(muscles.getSize()):
                muscle : Muscle = muscles.get(i)
                geo_path : GeometryPath = muscle.getGeometryPath()
                if geo_path is None:
                    continue
                path_points : PathPointSet = geo_path.getPathPointSet()
                for j in range(path_points.getSize()):
                    path_point : PathPoint = PathPoint.safeDownCast(path_points.get(j))
                    frame : PhysicalFrame = path_point.getParentFrame()
                    loc : Vec3 = path_point.get_location()
                    x = loc.get(0)
                    y = loc.get(1)
                    z = loc.get(2)
                    f.write(f"{muscle.getName()},{frame.getName()},{x},{y},{z}\n")   
    except Exception as e:
        print(f"Error writing attachment points to {filename}: {e}")
        return False
    
    print(f"Attachment points written to {filename}.")
    return True