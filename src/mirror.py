import opensim as osim
from src.musculoskeletal_graph import MusculoskeletalGraph
import re

def mirror_body(body : osim.Body, axes : list) -> osim.Body:
    """Mirror a body across specified axes."""
    # Validate axes
    if not all(axis in [0, 1, 2] for axis in axes):
        raise ValueError("Axes must be a list of 0, 1, or 2.")
    mirrored_body : osim.Body = body.clone()
    
    # Center of mass
    com = mirrored_body.getMassCenter()
    com[axes] *= -1 
    mirrored_body.setMassCenter(com)
    
    # Inertia
    inertia = mirrored_body.getInertia()
    mirrored_body.setInertia(mirror_inertia(inertia, axes))
    
    # Frame geometry
    frame_geometry = mirrored_body.get_frame_geometry()
    if frame_geometry is not None:
        # Mirror the geometry
        mirrored_body.set_frame_geometry(mirror_geometry(frame_geometry, axes))

    # Attached geometry
    i = 0
    while True:
        try:
            attached_geometry : osim.Geometry = mirrored_body.get_attached_geometry(i)
            if attached_geometry is None:
                break
        except Exception:
            break
        mirrored_body.set_attached_geometry(i, mirror_geometry(attached_geometry, axes))
        i += 1
        

    return mirrored_body


def mirror_geometry(geometry : osim.Geometry, axes: list) -> osim.Geometry:
    """Mirror a geometry across specified axes."""
    # Validate axes
    if not all(axis in [0, 1, 2] for axis in axes):
        raise ValueError("Axes must be a list of 0, 1, or 2.")
    mirrored_geometry : osim.Geometry = geometry.clone()
    
    # Scale factors
    scale_factors = mirrored_geometry.get_scale_factors()
    scale_factors[axes] *= -1
    mirrored_geometry.set_scale_factors(scale_factors)
    return mirrored_geometry
    

def mirror_joint(joint: osim.Joint, axes: list) -> osim.Joint:
    """Mirror a joint across specified axes."""
    if not all(axis in [0, 1, 2] for axis in axes):
        raise ValueError("Axes must be a list of 0, 1, or 2.")
    mirrored_joint : osim.Joint = joint.clone()
    
    pass


def mirror_inertia(inertia: osim.Inertia, axes: list) -> osim.Inertia:
    if not all(axis in [0, 1, 2] for axis in axes):
        raise ValueError("Axes must be a list of 0, 1, or 2.")
    
    # Get moments and products
    moments: osim.Vec3 = inertia.getMoments()  # xx, yy, zz
    products: osim.Vec3 = inertia.getProducts()  # xy, xz, yz

    # Negate the appropriate products based on the axes
    if 0 in axes:  # Mirroring across x-axis
        products[0] *= -1  # Negate xy
        products[1] *= -1  # Negate xz
    if 1 in axes:  # Mirroring across y-axis
        products[0] *= -1  # Negate xy
        products[2] *= -1  # Negate yz
    if 2 in axes:  # Mirroring across z-axis
        products[1] *= -1  # Negate xz
        products[2] *= -1  # Negate yz

    # Return the mirrored inertia
    return osim.Inertia(
        moments[0], moments[1], moments[2],
        products[0], products[1], products[2]
    )

def mirror_model(input_model_path, output_model_path, 
                 axes: list[int]=[0, 1, 2],
                 ground_name='ground', 
                 exclude_bodies: list[str]=None,
                 joint_name_mapping: dict[str, str]={r'(.+)_r': r'\1_l'},
                 frame_name_mapping: dict[str, str]={r'(.+)_r': r'\1_l'},
                 
                 body_name_mapping: dict[str, str]={r'(.+)_r': r'\1_l'},
                 muscle_name_mapping: dict[str, str]={r'R_(.+)': r'L_\1'}) -> osim.Model:
    """
    Mirrors an OpenSim model across a specified plane.

    Parameters:
        input_model_path (str): Path to the input .osim model file.
        output_model_path (str): Path to save the mirrored .osim model file.
        axes (list): List of axes to mirror across (0, 1, 2).
        ground_name (str): Name of the ground body to use for mirroring.
        exclude_bodies (list): List of body names to exclude from mirroring.
        body_name_mapping (dict): Dictionary to replace string in body names.
        muscle_name_mapping (dict): Dictionary to replace string in muscle names.
    """
    # Validate axes
    if not all(axis in [0, 1, 2] for axis in axes):
        raise ValueError("Axes must be a list of 0, 1, or 2.")

    model = MusculoskeletalGraph(input_model_path)
    mirrored_model : osim.Model = model.clone()
    
    mirrored_bodies = {}
    
    # Traverse joints
    for joint_name, body_names in model.joint_bodies.items():
        joint: osim.Joint = model.get_joint(joint_name)
        
        ## Frames
        # Rename and mirror translation (what about rotation?)
        while True:
            try:
                frame: osim.PhysicalOffsetFrame = joint.get_frames(i)
                if frame is None:
                    break
            except Exception:
                break
            # Rename
            frame_name = frame.getName()
            # Change parent
            # Mirror translation         

        ## Coordinates
        # Rename
        
        
        ## Spatial Transform
        try:
            custom_joint: osim.CustomJoint = osim.CustomJoint.safeDownCast(joint)
            if custom_joint is None:
                raise ValueError("Joint is not a CustomJoint.")
            # Get the spatial transform
            spatial_transform: osim.SpatialTransform = custom_joint.getSpatialTransform()
            # Mirror the spatial transform
            transform_axes : list[osim.Vec3] = spatial_transform.getAxes()
            for i, transform_axis in enumerate(transform_axes):
                if any(transform_axis[axes]):
                    get_trans_func = 'get_translation' + str(i+1)
                    trans: osim.TransformAxis = getattr(spatial_transform, get_trans_func)()
                    func: osim.Function = trans.getFunction()
                    concreteClass = func.getConcreteClassName()
                    match concreteClass:
                        case 'SimmSpline':
                            # Mirror the function
                            func: osim.SimmSpline = osim.SimmSpline.safeDownCast(func)
                            if func is None:
                                raise ValueError("Function is not a SimmSpline.")
                            for i in range(func.getSize()):
                                y: float = func.getY(i)
                                func.setY(i, -y)
                        case 'LinearFunction':
                            pass
                    
        except Exception as e:
            print(f"Error processing joint {joint_name}: {e}")
            continue

        
    mirrored_model.finalizeConnections()
    # Save the mirrored model
    mirrored_model.printToXML(output_model_path)
# Example usage
if __name__ == "__main__":
    input_model = "models/rat_hindlimb_millard_y2j.osim"
    output_model = "models/rat_hindlimb_millard_y2j_bilateral.osim"
    mirror_model(input_model, output_model, 
                 ground_name='ground', 
                 axes=[2], 
                 exclude_bodies=['ground', 'spine'], 
                 joint_name_mapping={'(.+)_r': r'\1_l'}, 
                 body_name_mapping={'(.+)_r': r'\1_l'}, 
                 muscle_name_mapping={'R_(.+)': r'L_\1'}
    )


