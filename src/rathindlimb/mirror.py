import pyopensim as osim
import re

def mirror_geometry(geometry : osim.Geometry, axes: list) -> osim.Geometry:
    """Mirror a geometry across specified axes."""
    # Validate axes
    if not all(axis in [0, 1, 2] for axis in axes):
        raise ValueError("Axes must be a list of 0, 1, or 2.")
    original_scale_factors = geometry.get_scale_factors() 
    # Create a new list with mirrored values
    mirrored_values = [
        -original_scale_factors.get(i) if i in axes else original_scale_factors.get(i)
        for i in range(3)
    ]
    new_scale_factors = osim.Vec3(mirrored_values[0], mirrored_values[1], mirrored_values[2])
    # Set the new Vec3 back onto the geometry
    geometry.set_scale_factors(new_scale_factors)
    return geometry # Return the modified geometry (modification was in place)


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


def mirror_body(body : osim.Body, axes : list) -> osim.Body:
    """Mirror a body across specified axes."""
    # Validate axes
    if not all(axis in [0, 1, 2] for axis in axes):
        raise ValueError("Axes must be a list of 0, 1, or 2.")    
    # Center of mass
    com = body.getMassCenter()
    com = osim.Vec3([-com[i] if i in axes else com[i] for i in range(3)])
    body.setMassCenter(com)
    
    # Inertia
    inertia = body.getInertia()
    body.setInertia(mirror_inertia(inertia, axes))
    
    # Frame geometry - This changes the body axes, not the mesh
    # frame_geometry = body.get_frame_geometry()
    # if frame_geometry is not None:
    #     # Mirror the geometry
    #     body.set_frame_geometry(mirror_geometry(frame_geometry, axes))

    # Attached geometry - Assuming only one geometry at index 0
    try:
        # Get a reference to the geometry object held by the body at index 0
        attached_geometry_ref : osim.Geometry = body.upd_attached_geometry(0)

        if attached_geometry_ref:
            # Mirror the geometry object directly using the reference
            mirror_geometry(attached_geometry_ref, axes)
        else:
            print(f"    No geometry found at index 0 for body {body.getName()}.")

    except RuntimeError as e:
        # Catching RuntimeError specifically, as this is common for SWIG index errors
        print(f"    Caught RuntimeError accessing index 0. Assuming no geometry attached or error. Details: {e}")
    except Exception as e:
        # Catch any other Python-level exceptions that might occur
        print(f"    Caught unexpected Python exception accessing index 0: {type(e).__name__} - {e}")


    # Wrap objects TODO: update names of wrap objects
    wrap_objects : osim.WrapObjectSet = body.upd_WrapObjectSet()
    for i in range(wrap_objects.getSize()):
        wrap_object : osim.WrapObject = wrap_objects.get(i)
        class_name = wrap_object.getConcreteClassName()
        # TODO: Do any mirroring for specific wrap objects here
        # Mirror xyz_body_rotation
        xyz_body_rotation: osim.Vec3 = wrap_object.get_xyz_body_rotation()
        # Negate the appropriate axes based on the axes
        xyz_body_rotation = osim.Vec3([-xyz_body_rotation[i] if i not in axes else xyz_body_rotation[i] for i in range(3)])
        wrap_object.set_xyz_body_rotation(xyz_body_rotation)
        # Mirror translation
        translation: osim.Vec3 = wrap_object.get_translation()
        # Negate the appropriate axes based on the axes
        translation = osim.Vec3([-translation[i] if i in axes else translation[i] for i in range(3)])
        wrap_object.set_translation(translation)
        # TODO: Deal with quadrants   
    print(f"Successfully mirrored body {body.getName()}")
    return body
    
def regex_mapping(mapping: dict[str, str], name: str) -> str:
    """Apply regex mapping to a name."""
    for pattern, replacement in mapping.items():
        if re.match(pattern, name):
            return re.sub(pattern, replacement, name)
    return name

def mirror_joint(joint: osim.Joint, axes: list) -> osim.Joint:
    parent_frame: osim.PhysicalFrame = joint.getParentFrame()
    child_frame: osim.PhysicalFrame = joint.getChildFrame()
        
    ## Coordinates
    num_coords = joint.numCoordinates()
    # Rename
    for i in range(num_coords):
        coord: osim.Coordinate = joint.upd_coordinates(i)
        coord_name = coord.getName()
    
    ## Joint type specific mirroring
    joint_type = joint.getConcreteClassName()
    match joint_type:
        case 'PinJoint':
            raise NotImplementedError("PinJoint mirroring not implemented.")
        case 'SliderJoint':
            raise NotImplementedError("SliderJoint mirroring not implemented.")
        case 'WeldJoint':
            raise NotImplementedError("WeldJoint mirroring not implemented.")
        case 'CustomJoint':
            joint: osim.CustomJoint = osim.CustomJoint.safeDownCast(joint)
            if joint is None:
                raise ValueError("Joint is not a CustomJoint.")
            # Get the spatial transform
            spatial_transform: osim.SpatialTransform = joint.updSpatialTransform()
            # Mirror the spatial transform
            transform_axes : list[osim.Vec3] = spatial_transform.getAxes()
            for i, vec in enumerate(transform_axes):
                # Assume that order is always rotation1-3, translation1-3
                get_transform_function = ('get_rotation' + str(i+1)) if i < 3 else ('get_translation' + str(i-2))
                set_transform_function = ('set_rotation' + str(i+1)) if i < 3 else ('set_translation' + str(i-2))
                transform_axis: osim.TransformAxis = getattr(spatial_transform, get_transform_function)()
                transform_coords: osim.ArrayStr = transform_axis.getCoordinateNamesInArray()
                for j in range(transform_coords.getSize()):
                    coord_name = transform_coords.get(j)
                    transform_axis.set_coordinates(j, new_name)
                if any(vec[axes]):            
                    axis_function: osim.Function = transform_axis.getFunction()
                    concreteClass = axis_function.getConcreteClassName()
                    match concreteClass:
                        case 'SimmSpline':
                            # Mirror the function
                            simm_spline: osim.SimmSpline = osim.SimmSpline.safeDownCast(axis_function)
                            if simm_spline is None:
                                raise ValueError("Function is not a SimmSpline.")
                            for k in range(simm_spline.getSize()):
                                y: float = simm_spline.getY(k)
                                simm_spline.setY(k, -y)
                        case 'LinearFunction':
                            linear_func: osim.LinearFunction = osim.LinearFunction.safeDownCast(axis_function)
                            if linear_func is None:
                                raise ValueError("Function is not a LinearFunction.")
                            linear_func.setSlope(-linear_func.getSlope())
                            linear_func.setIntercept(-linear_func.getIntercept())
                        case 'Constant':
                            constant_func: osim.Constant = osim.Constant.safeDownCast(axis_function)
                            if constant_func is None:
                                raise ValueError("Function is not a Constant.")
                            constant_func.setValue(-constant_func.getValue())
                        case 'MultiplierFunction':
                            mult_func: osim.MultiplierFunction = osim.MultiplierFunction.safeDownCast(axis_function)
                            if mult_func is None:
                                raise ValueError("Function is not a MultiplierFunction.")
                            # For MultiplierFunction, negate the scale
                            mult_func.setScale(-mult_func.getScale())
                        case _:
                            print(f"Unsupported function type: {concreteClass}, not mirroring")
                    # Update the function in the transform axis
                    transform_axis.setFunction(axis_function)
                # Update the transform axis in the spatial transform
                getattr(spatial_transform, set_transform_function)(transform_axis)
    # Rename and mirror translation
    i = 0  # Initialize counter
    while True:
        try:
            frame: osim.PhysicalOffsetFrame = joint.upd_frames(i)
            if frame is None:
                break
        except Exception:
            break
        # Rename
        frame_name = frame.getName()
        # frame.setName(new_name)
        # Mirror translation   
        translation = frame.get_translation()
        # Negate the appropriate axes based on the axes
        translation = osim.Vec3([-translation[i] if i in axes else translation[i] for i in range(3)])
        frame.set_translation(translation)      
        # Mirror orientation? 
        orientation = frame.get_orientation()
        orientation = osim.Vec3([-orientation[i] if i not in axes else orientation[i] for i in range(3)])
        frame.set_orientation(orientation)
        i += 1

def mirror_model(input_model_path, output_model_path, 
                 axes: list[int]=[0, 1, 2],
                 ground_name='ground', 
                 exclude_bodies: list[str]=None,
                 joint_name_mapping: dict[str, str]={r'(.+)_r': r'\1_l'},
                 frame_name_mapping: dict[str, str]={r'(.+)_r_(.+)': r'\1_l_\2'},
                 coord_name_mapping: dict[str, str]={r'(.+)_r_(.+)': r'\1_l_\2'},
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

    model = MusculoskeletalGraph(osim.Model(input_model_path).clone())
    
    # Dictionary to cache original names to mirrored names
    mirror_map = {}

    # Bodies
    for body_name in model.body_graph.keys():
        if exclude_bodies and (body_name in exclude_bodies or body_name == ground_name):
            print(f"Skipping body {body_name}.")
            continue
        body: osim.Body = model.get_body(body_name)
        print(f"Mirroring body {body_name}.")
        # Check if the body is in the exclude list
        # Mirror the body
        mirrored_body = mirror_body(body.clone(), axes)
        # Rename the body
        # Check if the body is in the mirror map
        if body_name in mirror_map:
            # Replace the body name with the mirrored name
            new_name = mirror_map[body_name]
        else:
            # Create a new name for the mirrored body
            new_name = regex_mapping(body_name_mapping, body_name)
            mirror_map[body_name] = new_name
        mirrored_body.setName(new_name)
        model.addBody(mirrored_body)
    # Joints
    for joint_name, bodies in model.joint_bodies.items():
        if exclude_bodies and (bodies[0] in exclude_bodies and bodies[1] in exclude_bodies):
            print(f"Skipping joint {joint_name}.")
            continue
        joint: osim.Joint = model.get_joint(joint_name)
        print(f"Mirroring joint {joint_name}.")
        mirrorred_joint = mirror_joint(joint.clone(), axes)
        # Add the mirrored joint to the model
        model.addJoint(joint)
    
    # Muscles TODO
    for muscle_name, body_names in model.muscle_attachments.items():
        muscle: osim.Muscle = model.get_muscle(muscle_name)
        print(f"Mirroring muscle {muscle_name}.")
        # Check if the muscle is in the exclude list
        # Mirror the muscle
        mirrored_muscle = mirror_muscle(muscle.clone(), axes)
        # Rename the muscle
        # Check if the muscle is in the mirror map
        if muscle_name in mirror_map:
            # Replace the muscle name with the mirrored name
            new_name = mirror_map[muscle_name]
        else:
            # Create a new name for the mirrored muscle
            new_name = regex_mapping(muscle_name_mapping, muscle_name)
            mirror_map[muscle_name] = new_name
        mirrored_muscle.setName(new_name)
        model.updMuscles().append(mirrored_muscle)
    model.finalizeConnections()
    # Save the mirrored model
    model.printToXML(output_model_path)
    print(f"Mirrored model saved to {output_model_path}.")  
    
# Example usage
if __name__ == "__main__":
    input_model = "models/rat_hindlimb_millard_y2j_tsl_r.osim"
    output_model = "models/rat_hindlimb_millard_y2j_tsl_bilateral.osim"
    mirror_model(input_model, output_model, 
                 ground_name='ground', 
                 axes=[2], 
                 exclude_bodies=['ground', 'spine']
                )
