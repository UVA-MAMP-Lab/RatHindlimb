import opensim as osim
from src.musculoskeletal_graph import MusculoskeletalGraph

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
    mirrored_body.setInertia(mirror_inertia(inertia))
    
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
    
    # Wrap objects
    wrap_objects : osim.WrapObjectSet= mirrored_body.getWrapObjectSet()
    for i in range(wrap_objects.getSize()):
        wrap_object : osim.WrapObject = wrap_objects.get(i)
        # TODO
        

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
    mirrored_joint : osim.Joint = joint.clone()
    
    joint.get

    pass


def mirror_inertia(inertia : osim.Inertia) -> osim.Inertia:
    moments = inertia.getMoments()
    products = inertia.getProducts()
    return osim.Inertia(
            moments[0], moments[1], moments[2],
            -products[0], -products[1], -products[2]
        )

def mirror_model(input_model_path, output_model_path, 
                 plane='xy', 
                 ground_name='ground', 
                 exclude_bodies=None,
                 replace_strings={'r': 'l'}):
    """
    Mirrors an OpenSim model across a specified plane.

    Parameters:
        input_model_path (str): Path to the input .osim model file.
        output_model_path (str): Path to save the mirrored .osim model file.
        plane (str): Plane across which to mirror ('xy', 'yz', or 'xz').
        ground_name (str): Name of the ground body to use for mirroring.
        exclude_bodies (list): List of body names to exclude from mirroring.
        replace_strings (dict): Dictionary to replace string in body names.
    """

    # Determine the mirroring axis based on the plane
    if plane == 'xy':
        mirror_axis = 2  # z-axis
    elif plane == 'yz':
        mirror_axis = 0  # x-axis
    elif plane == 'xz':
        mirror_axis = 1  # y-axis
    else:
        raise ValueError("Invalid plane. Choose from 'xy', 'yz', or 'xz'.")

    model = MusculoskeletalGraph(input_model_path)
    mirrored_model : osim.Model = model.clone()
    
    # Traverse joints and bodies to mirror them
    
    
    
    mirrored_model.finalizeConnections()
    # Save the mirrored model
    mirrored_model.printToXML(output_model_path)

# Example usage
if __name__ == "__main__":
    input_model = "models/rat_hindlimb_millard_y2j.osim"
    output_model = "models/rat_hindlimb_millard_y2j_bilateral.osim"
    mirror_model(input_model, output_model, root_body="pelvis_r", plane="xy")


