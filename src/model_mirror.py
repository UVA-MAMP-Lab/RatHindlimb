import opensim as osim
import re
from typing import List, Dict, Optional, Tuple, Set, Union, Any # Added Set, Union, Any
from musculoskeletal_graph import MusculoskeletalGraph # Assuming this is a custom module

class ModelMirror:
    """
    Handles mirroring of an OpenSim model across specified axes.

    Encapsulates the logic for mirroring bodies, joints, muscles,
    and their associated properties and geometries. Uses composition,
    operating on a clone of the input model.
    """

    def __init__(self,
                 input_model_path_or_obj: Union[str, osim.Model], # Accept path or model object
                 mirror_axes: List[int],
                 ground_name: str = 'ground',
                 exclude_bodies: Optional[List[str]] = None,
                 name_mappings: Optional[Dict[str, Dict[str, str]]] = None,
                 graph_helper: Optional[MusculoskeletalGraph] = None): # Optional graph helper
        """
        Initializes the ModelMirror.

        Args:
            input_model_path_or_obj: Path to the input .osim file or an existing osim.Model object.
                                     If an object is passed, it will be cloned.
            mirror_axes: List of axes indices (0=X, 1=Y, 2=Z) to mirror across.
                         Typically a single axis for planar symmetry (e.g., [0] for YZ).
            ground_name: Name of the ground body.
            exclude_bodies: List of body names to exclude from mirroring (ground is always excluded).
            name_mappings: Dictionary defining regex patterns for renaming components.
                           Keys should match component types ('body', 'joint', etc.).
                           Values: Dict[pattern_str, replacement_str].
            graph_helper: An optional instance of MusculoskeletalGraph (or similar)
                          for potentially faster lookups (currently not used directly in
                          mirroring logic but could be integrated).
        """
        if not all(axis in [0, 1, 2] for axis in mirror_axes):
            raise ValueError("mirror_axes must be a list containing 0, 1, or 2.")
        if not mirror_axes:
             raise ValueError("mirror_axes cannot be empty.")

        # --- Model Handling ---
        if isinstance(input_model_path_or_obj, str):
            self.original_model: osim.Model = osim.Model(input_model_path_or_obj)
        elif isinstance(input_model_path_or_obj, osim.Model):
            # Clone the input model to avoid modifying the original object directly
            self.original_model = input_model_path_or_obj.clone()
        else:
            raise TypeError("input_model_path_or_obj must be a file path (str) or an osim.Model instance.")

        # Work on a separate clone for mirroring
        self.mirrored_model: osim.Model = self.original_model.clone()
        self.mirrored_model.setName(self.original_model.getName() + "_mirrored")

        # --- Configuration ---
        self.mirror_axes: List[int] = mirror_axes
        self.ground_name: str = ground_name
        # Ensure ground is always excluded, use a set for efficient lookup
        _exclude_list = exclude_bodies if exclude_bodies else []
        self.exclude_bodies: Set[str] = set(_exclude_list) | {self.ground_name}

        # Default name mappings (adjust regex as needed for your convention)
        default_mappings: Dict[str, Dict[str, str]] = {
            'body': {r'(.+)_r$': r'\1_l', r'(.+)_R$': r'\1_L'},
            'joint': {r'(.+)_r$': r'\1_l', r'(.+)_R$': r'\1_L'},
            'frame': {r'(.+)_r$': r'\1_l', r'(.+)_R$': r'\1_L'},
            'coord': {r'(.+)_r$': r'\1_l', r'(.+)_R$': r'\1_L'},
            'muscle': {r'(.+)_r$': r'\1_l', r'(.+)_R$': r'\1_L'},
            'misc': {r'(.+)_r$': r'\1_l', r'(.+)_R$': r'\1_L'}, # For wrap objects, markers etc.
        }
        # Merge user-provided mappings with defaults (user mappings take precedence)
        self.name_mappings: Dict[str, Dict[str, str]] = default_mappings
        if name_mappings:
             self.name_mappings.update(name_mappings) # Overwrite defaults with user specifics

        # Cache for original -> mirrored names to ensure consistency during a single mirror() call
        self.component_rename_map: Dict[str, str] = {}

        # Store the graph helper if provided (currently unused in core logic)
        self.graph_helper: Optional[MusculoskeletalGraph] = graph_helper

        print(f"ModelMirror initialized for '{self.original_model.getName()}'. Mirroring across axes {self.mirror_axes}. Excluding: {self.exclude_bodies}")

    def _mirror_vec3(self, vec: osim.Vec3) -> osim.Vec3:
        """Mirrors a Vec3 by negating components along the mirror axes."""
        return osim.Vec3(
            -vec[0] if 0 in self.mirror_axes else vec[0],
            -vec[1] if 1 in self.mirror_axes else vec[1],
            -vec[2] if 2 in self.mirror_axes else vec[2]
        )

    def _mirror_inertia(self, inertia: osim.Inertia) -> osim.Inertia:
        """Mirrors inertia properties (products of inertia)."""
        moments: osim.Vec3 = inertia.getMoments()  # xx, yy, zz (diagonal terms don't change sign)
        products: osim.Vec3 = inertia.getProducts()  # xy, xz, yz (off-diagonal terms)

        # Products of inertia p_ij change sign if the mirror transformation changes the sign
        # of exactly *one* of the coordinates i or j.
        # Flip p_xy if mirroring involves x OR y but not both (XOR)
        # Flip p_xz if mirroring involves x OR z but not both (XOR)
        # Flip p_yz if mirroring involves y OR z but not both (XOR)
        flip_xy: bool = (0 in self.mirror_axes) ^ (1 in self.mirror_axes)
        flip_xz: bool = (0 in self.mirror_axes) ^ (2 in self.mirror_axes)
        flip_yz: bool = (1 in self.mirror_axes) ^ (2 in self.mirror_axes)

        mirrored_products = osim.Vec3(
            -products[0] if flip_xy else products[0],
            -products[1] if flip_xz else products[1],
            -products[2] if flip_yz else products[2]
        )

        # Mass is scalar, doesn't change. Center of mass handled separately.
        # Create new Inertia object with original moments and mirrored products.
        return osim.Inertia(
            moments[0], moments[1], moments[2],
            mirrored_products[0], mirrored_products[1], mirrored_products[2]
        )

    def _get_mirrored_name(self, original_name: str, component_type: str) -> str:
        """
        Gets the mirrored name for a component based on mapping rules.

        Uses a cache (self.component_rename_map) for consistency within a single
        mirror() operation. Falls back to 'misc' mapping if specific type not found.

        Args:
            original_name: The current name of the component.
            component_type: The type of the component (e.g., 'body', 'joint', 'coord').

        Returns:
            The potentially modified name after applying regex rules.
        """
        # Check cache first
        if original_name in self.component_rename_map:
            return self.component_rename_map[original_name]

        # Get the relevant mapping rules, fallback to 'misc'
        mapping: Dict[str, str] = self.name_mappings.get(component_type, self.name_mappings.get('misc', {}))

        mirrored_name: str = original_name # Start with original name
        applied_rule: bool = False
        for pattern, replacement in mapping.items():
            # Use re.sub which handles non-matches gracefully (returns original string)
            new_name = re.sub(pattern, replacement, original_name)
            if new_name != original_name: # Check if substitution occurred
                 mirrored_name = new_name
                 applied_rule = True
                 # print(f"    Name mapping: '{original_name}' -> '{mirrored_name}' (Rule: '{pattern}' -> '{replacement}')")
                 break # Apply only the first matching rule for this component type

        # Optional: Warning if no rule was applied for a component type expected to change
        # if not applied_rule and component_type in ['body', 'joint', 'muscle', 'coord']:
        #     print(f"    Warning: No name mapping rule applied to '{original_name}' (type: {component_type}).")

        # Store in cache for this mirroring run
        # Important: Check if the *target* mirrored name is already mapped from *another* source
        # This detects potential collisions where two different originals map to the same mirrored name.
        # Note: This check doesn't prevent overwriting if the *same* original is processed twice.
        if mirrored_name != original_name:
             existing_sources = [src for src, dest in self.component_rename_map.items() if dest == mirrored_name]
             if existing_sources:
                  print(f"    Warning: Name collision detected! Mirrored name '{mirrored_name}' was already generated from '{existing_sources[0]}'. Overwriting mapping for '{original_name}'.")

        self.component_rename_map[original_name] = mirrored_name
        return mirrored_name

    def _mirror_geometry(self, geometry: osim.Geometry) -> None:
        """Mirrors Geometry scale factors in place based on self.mirror_axes."""
        if not isinstance(geometry, osim.Mesh) and not isinstance(geometry, osim.Brick) and \
           not isinstance(geometry, osim.Sphere) and not isinstance(geometry, osim.Ellipsoid) and \
           not isinstance(geometry, osim.Cylinder) and not isinstance(geometry, osim.Cone):
            # Only mirror types known to have scale factors or simple dimensions
            # print(f"    Skipping geometry mirroring for type: {geometry.getConcreteClassName()}")
            return

        try:
             # Primarily affects scale factors
             if hasattr(geometry, 'get_scale_factors') and hasattr(geometry, 'set_scale_factors'):
                  original_scale_factors: osim.Vec3 = geometry.get_scale_factors()
                  mirrored_sf: osim.Vec3 = self._mirror_vec3(original_scale_factors)
                  geometry.set_scale_factors(mirrored_sf)
                  # print(f"    Mirrored scale factors for geometry")

             # Handle dimensions for simple shapes (optional, depends on desired effect)
             # Example: Mirroring Brick dimensions across YZ (axis 0) would swap Y and Z half-lengths?
             # This requires careful thought - usually scale factors are sufficient.
             # if isinstance(geometry, osim.Brick):
             #     dims = geometry.get_half_lengths()
             #     mirrored_dims = self._mirror_vec3(dims) # Or specific logic
             #     geometry.set_half_lengths(mirrored_dims)

        except Exception as e:
             print(f"    Warning: Failed to mirror properties for geometry '{geometry.getName()}': {e}")


        # Note: Mesh file itself is NOT mirrored. The transformation of the frame
        # it's attached to handles the spatial mirroring. Scale factors adjust asymmetry.

    def _mirror_wrap_object(self, wrap_object: osim.WrapObject) -> None:
        """Mirrors properties of a WrapObject in place."""
        original_name = wrap_object.getName()
        # print(f"      Mirroring wrap object '{original_name}'...")

        try:
            # Mirror translation property
            if hasattr(wrap_object, 'get_translation') and hasattr(wrap_object, 'set_translation'):
                translation: osim.Vec3 = wrap_object.get_translation()
                mirrored_translation = self._mirror_vec3(translation)
                wrap_object.set_translation(mirrored_translation)

            # Mirror rotation property (xyz_body_rotation) - COMPLEX
            if hasattr(wrap_object, 'get_xyz_body_rotation') and hasattr(wrap_object, 'set_xyz_body_rotation'):
                xyz_body_rotation: osim.Vec3 = wrap_object.get_xyz_body_rotation()
                # Mirroring rotation depends heavily on the mirror plane and rotation sequence.
                # Heuristic: Negate rotations about axes *in* the mirror plane.
                # Example: Mirror across YZ plane (axis 0). Plane contains Y, Z axes.
                #          Negate Ry and Rz. Keep Rx.
                mirrored_rot = list(xyz_body_rotation) # Convert to list for modification
                if 0 in self.mirror_axes: # Mirror YZ
                     mirrored_rot[1] *= -1 # Negate Ry
                     mirrored_rot[2] *= -1 # Negate Rz
                if 1 in self.mirror_axes: # Mirror XZ
                     mirrored_rot[0] *= -1 # Negate Rx
                     mirrored_rot[2] *= -1 # Negate Rz
                if 2 in self.mirror_axes: # Mirror XY
                     mirrored_rot[0] *= -1 # Negate Rx
                     mirrored_rot[1] *= -1 # Negate Ry

                # Handle double negation if mirroring across multiple axes (e.g., origin inversion)
                # If mirrored across 2 axes, the axis *not* mirrored keeps its sign.
                # If mirrored across 3 axes, all rotations negate? (Test this)

                # TODO: Thoroughly VALIDATE this rotation logic for your specific use case!
                wrap_object.set_xyz_body_rotation(osim.Vec3(mirrored_rot[0], mirrored_rot[1], mirrored_rot[2]))
                # print(f"      (Warning: Wrap object rotation mirroring needs validation: {xyz_body_rotation} -> {mirrored_rot})")


            # Mirror other properties based on WrapObject type (dimensions, quadrants)
            if isinstance(wrap_object, osim.WrapCylinder):
                 # Radius, length usually don't need mirroring unless they define asymmetry
                 pass
            elif isinstance(wrap_object, osim.WrapSphere):
                 # Radius usually doesn't need mirroring
                 pass
            elif isinstance(wrap_object, osim.WrapEllipsoid):
                 # Dimensions might need mirroring similar to rotation logic if axes aren't aligned with body
                 # Quadrant might need flipping
                 if hasattr(wrap_object, 'get_quadrant') and hasattr(wrap_object, 'set_quadrant'):
                      quadrant: str = wrap_object.get_quadrant()
                      new_quadrant: str = quadrant
                      # Example: Mirroring across YZ (axis 0), flip between +x / -x
                      if 0 in self.mirror_axes:
                           if quadrant == "+x": new_quadrant = "-x"
                           elif quadrant == "-x": new_quadrant = "+x"
                      # Add similar logic for Y (axis 1) and Z (axis 2) mirroring if needed
                      if 1 in self.mirror_axes:
                           if quadrant == "+y": new_quadrant = "-y"
                           elif quadrant == "-y": new_quadrant = "+y"
                      if 2 in self.mirror_axes:
                           if quadrant == "+z": new_quadrant = "-z"
                           elif quadrant == "-z": new_quadrant = "+z"

                      if new_quadrant != quadrant:
                           wrap_object.set_quadrant(new_quadrant)
                           # print(f"      Flipped quadrant from '{quadrant}' to '{new_quadrant}'")
            # Add other wrap types if necessary

            # Rename the wrap object
            mirrored_name = self._get_mirrored_name(original_name, 'misc') # Use 'misc' or specific type
            if mirrored_name != original_name:
                 wrap_object.setName(mirrored_name)
                 # print(f"      Renamed wrap object to '{mirrored_name}'")

        except Exception as e:
             print(f"    Warning: Failed to mirror wrap object '{original_name}': {e}")


    def _mirror_body(self, body: osim.Body) -> None:
        """Mirrors a Body's properties in place (Mass center, Inertia, Geometry, Wraps)."""
        original_name = body.getName()
        print(f"  Mirroring body '{original_name}'...")

        try:
            # Mirror Mass Center
            body.setMassCenter(self._mirror_vec3(body.getMassCenter()))

            # Mirror Inertia
            body.setInertia(self._mirror_inertia(body.getInertia()))

            # Mirror Attached Geometry (in place)
            geom_set = body.upd_attached_geometry() # Get mutable set proxy
            num_geom = geom_set.getSize()
            for i in range(num_geom):
                try:
                    geom = geom_set.get(i) # Get geom reference
                    self._mirror_geometry(geom) # Modify geom in place
                except Exception as e_geom:
                    # Attempt to get geometry name for better error message
                    geom_name = "unknown"
                    try: geom_name = geom_set.get(i).getName()
                    except: pass
                    print(f"    Warning: Could not mirror geometry {i} ('{geom_name}') for body '{original_name}': {e_geom}")

            # Mirror Wrap Objects (in place)
            wrap_set = body.upd_WrapObjectSet() # Get mutable set proxy
            num_wraps = wrap_set.getSize()
            for i in range(num_wraps):
                 try:
                     wrap_obj = wrap_set.get(i) # Get wrap object reference
                     self._mirror_wrap_object(wrap_obj) # Modifies wrap object in place
                 except Exception as e_wrap:
                     # Attempt to get wrap object name for better error message
                     wrap_name = "unknown"
                     try: wrap_name = wrap_set.get(i).getName()
                     except: pass
                     print(f"    Warning: Could not mirror wrap object {i} ('{wrap_name}') for body '{original_name}': {e_wrap}")

            # Rename the body itself
            mirrored_name = self._get_mirrored_name(original_name, 'body')
            if mirrored_name != original_name:
                 body.setName(mirrored_name)
                 print(f"    Renamed body to '{mirrored_name}'")

        except Exception as e:
             print(f"  Error during mirroring of body '{original_name}': {e}")


    def _mirror_transform_axis_function(self, axis_function: osim.Function) -> None:
        """
        Mirrors (negates) the output of common OpenSim Function types.
        Modifies the function object in place.
        """
        if not axis_function:
             print("    Warning: Attempted to mirror a null function.")
             return

        concrete_class: str = axis_function.getConcreteClassName()
        # print(f"        Mirroring function of type: {concrete_class}")

        # Use safeDownCast for type checking and getting specific methods
        if spline := osim.SimmSpline.safeDownCast(axis_function):
            # Negate Y values
            x_values = spline.getX() # Get X values (Vector)
            y_values = spline.getY() # Get Y values (Vector)
            new_y_values = osim.Vector(y_values.size(), 0.0)
            for k in range(y_values.size()):
                new_y_values[k] = -y_values[k]
            # Create a new spline with negated Y values
            # Note: SimmSpline constructor might not exist or take Vecs directly.
            # Alternative: Modify in place if possible, or replace with a new function.
            # Direct modification:
            # for k in range(spline.getSize()): spline.setY(k, -spline.getY(k)) # Check if setY(index, value) exists
            # If not, might need to replace the function entirely in the TransformAxis
            print(f"        Warning: SimmSpline mirroring might require function replacement, not just Y negation.")

        elif lin_func := osim.LinearFunction.safeDownCast(axis_function):
            # Negate slope and intercept
            lin_func.setSlope(-lin_func.getSlope())
            lin_func.setIntercept(-lin_func.getIntercept())

        elif const_func := osim.Constant.safeDownCast(axis_function):
            # Negate the constant value
            const_func.setValue(-const_func.getValue())

        elif mult_func := osim.MultiplierFunction.safeDownCast(axis_function):
            # Negate the scale factor
            mult_func.setScale(-mult_func.getScale())
            # The embedded function within MultiplierFunction usually shouldn't be negated itself.

        elif poly_func := osim.PolynomialFunction.safeDownCast(axis_function):
            # Negate all coefficients
            coeffs: osim.Vector = poly_func.getCoefficients()
            new_coeffs = osim.Vector(coeffs.size())
            for k in range(coeffs.size()):
                new_coeffs[k] = -coeffs[k]
            poly_func.setCoefficients(new_coeffs)

        elif step_func := osim.StepFunction.safeDownCast(axis_function):
             # Negate start, end values. Keep times the same.
             step_func.setStartValue(-step_func.getStartValue())
             step_func.setEndValue(-step_func.getEndValue())

        # Add other function types as needed (e.g., PiecewiseLinearFunction)
        else:
            print(f"    Warning: Unsupported function type '{concrete_class}' in TransformAxis. Cannot automatically mirror function values.")


    def _mirror_joint(self, joint: osim.Joint) -> None:
        """Mirrors a Joint's properties, frames, and kinematics in place."""
        original_name = joint.getName()
        print(f"  Mirroring joint '{original_name}'...")

        try:
            # --- 1. Rename Coordinates associated directly with the joint ---
            coord_set = joint.upd_CoordinateSet() # Get mutable coordinate set
            num_coords = coord_set.getSize()
            for i in range(num_coords):
                coord = coord_set.get(i)
                original_coord_name = coord.getName()
                mirrored_coord_name = self._get_mirrored_name(original_coord_name, 'coord')
                if original_coord_name != mirrored_coord_name:
                    print(f"    Renaming coordinate '{original_coord_name}' to '{mirrored_coord_name}'")
                    coord.setName(mirrored_coord_name)

                    # Mirroring coordinate defaults/ranges is complex and depends on definition.
                    # Example: If 'r_knee_angle' becomes 'l_knee_angle', the range might stay the same,
                    # but if 'r_hip_adduction' becomes 'l_hip_adduction', the signs might flip.
                    # This requires semantic understanding beyond simple mirroring.
                    # Add specific logic here if needed based on coordinate conventions.
                    # try:
                    #     coord.setDefaultValue(-coord.getDefaultValue()) # Example: Flip default
                    #     min_r, max_r = coord.getRangeMin(), coord.getRangeMax()
                    #     coord.setRangeMin(-max_r) # Example: Flip range
                    #     coord.setRangeMax(-min_r)
                    #     print(f"      (Warning: Coordinate '{mirrored_coord_name}' default/range mirroring applied - VERIFY)")
                    # except Exception as e_coord_mirror:
                    #     print(f"      Warning: Could not mirror default/range for {mirrored_coord_name}: {e_coord_mirror}")


            # --- 2. Mirror Frames owned by the Joint (Parent and Child Offset Frames) ---
            # Access frames via the generic Component interface if needed, or specific API
            # Note: joint.updPropertyByName("frames") gives Object, need casting.
            # Easier to use getParentFrame/getChildFrame if they are the primary frames to mirror.
            # If the joint *owns* the offset frames (common), they need mirroring.
            # Let's assume we need to mirror the PhysicalOffsetFrames listed in the 'frames' property.

            frame_prop = joint.getPropertyByName("frames")
            mirrored_offset_frame_names: Dict[str, str] = {} # Store original->mirrored names for owned frames

            for i in range(frame_prop.size()):
                 # Property value is an AbstractProperty, get value as Object, then downcast
                 frame_obj = frame_prop.getValueAsObject(i)
                 frame = osim.PhysicalOffsetFrame.safeDownCast(frame_obj)

                 if frame:
                     original_frame_name = frame.getName()
                     print(f"    Mirroring owned frame '{original_frame_name}'...")

                     # Mirror translation
                     frame.set_translation(self._mirror_vec3(frame.get_translation()))

                     # Mirror orientation - Use heuristic, add warning
                     # (Same logic as wrap object rotation, needs validation)
                     vec3_orient: osim.Vec3 = frame.get_orientation() # Euler angles
                     mirrored_orient_list = list(vec3_orient)
                     if 0 in self.mirror_axes: mirrored_orient_list[1] *= -1; mirrored_orient_list[2] *= -1
                     if 1 in self.mirror_axes: mirrored_orient_list[0] *= -1; mirrored_orient_list[2] *= -1
                     if 2 in self.mirror_axes: mirrored_orient_list[0] *= -1; mirrored_orient_list[1] *= -1
                     mirrored_orient_vec3 = osim.Vec3(mirrored_orient_list[0], mirrored_orient_list[1], mirrored_orient_list[2])
                     frame.set_orientation(mirrored_orient_vec3)
                     print(f"      (Warning: Frame '{original_frame_name}' orientation mirroring needs validation: {vec3_orient} -> {mirrored_orient_vec3})")


                     # Rename frame
                     mirrored_frame_name = self._get_mirrored_name(original_frame_name, 'frame')
                     if mirrored_frame_name != original_frame_name:
                          frame.setName(mirrored_frame_name)
                          print(f"      Renamed owned frame to '{mirrored_frame_name}'")
                     mirrored_offset_frame_names[original_frame_name] = mirrored_frame_name # Store mapping

                 else:
                     # Handle other frame types if necessary
                     print(f"    Skipping non-PhysicalOffsetFrame in joint's 'frames' property: {frame_obj.getConcreteClassName()}")


            # --- 3. Update Socket Connections for Parent/Child Frames ---
            # Sockets connect by *name*. Update connectee name based on cached renames.
            parent_socket = joint.updSocket("parent_frame")
            orig_parent_connectee = parent_socket.getConnecteeName()
            # Check if it was an owned offset frame that got renamed
            mirrored_parent_connectee = mirrored_offset_frame_names.get(orig_parent_connectee,
                                                                     self._get_mirrored_name(orig_parent_connectee, 'body')) # Default to body rename if not owned frame
            if mirrored_parent_connectee != orig_parent_connectee:
                print(f"    Updating parent socket connectee: '{orig_parent_connectee}' -> '{mirrored_parent_connectee}'")
                parent_socket.setConnecteeName(mirrored_parent_connectee)

            child_socket = joint.updSocket("child_frame")
            orig_child_connectee = child_socket.getConnecteeName()
            mirrored_child_connectee = mirrored_offset_frame_names.get(orig_child_connectee,
                                                                    self._get_mirrored_name(orig_child_connectee, 'body'))
            if mirrored_child_connectee != orig_child_connectee:
                print(f"    Updating child socket connectee: '{orig_child_connectee}' -> '{mirrored_child_connectee}'")
                child_socket.setConnecteeName(mirrored_child_connectee)


            # --- 4. Mirror Joint Kinematics (SpatialTransform for CustomJoints) ---
            if custom_joint := osim.CustomJoint.safeDownCast(joint):
                print("    Mirroring CustomJoint SpatialTransform...")
                spatial_transform = custom_joint.updSpatialTransform()

                for i in range(6): # Iterate through 6 TransformAxes (3 rot, 3 trans)
                    transform_axis = spatial_transform.updTransformAxis(i) # Get mutable axis
                    axis_vec: osim.Vec3 = transform_axis.getAxis() # Direction/axis of motion

                    # Rename coordinates used by this axis's function
                    coord_names_prop = transform_axis.getPropertyByName("coordinates")
                    coord_names_list: List[str] = []
                    changed_names: bool = False
                    for j in range(coord_names_prop.size()):
                        orig_coord_name = coord_names_prop.getValueAsString(j)
                        mirrored_coord_name = self._get_mirrored_name(orig_coord_name, 'coord')
                        coord_names_list.append(mirrored_coord_name)
                        if orig_coord_name != mirrored_coord_name:
                            changed_names = True

                    if changed_names:
                        # Update the coordinate names property
                        # This might require clearing and appending, or specific API method
                        # transform_axis.setCoordinateNames(coord_names_list) # Check OpenSim API
                        # Alternative (if property is simple list of strings):
                        coord_names_prop.clear()
                        for name in coord_names_list:
                             coord_names_prop.appendValue(name)
                        print(f"      Updated coordinate names for TransformAxis {i} to: {coord_names_list}")


                    # Determine if the function associated with this axis needs negation
                    negate_function: bool = False
                    is_rotation: bool = i < 3
                    is_translation: bool = i >= 3
                    axis_vec_list = [axis_vec[0], axis_vec[1], axis_vec[2]] # For easier indexing checks

                    # Logic: Negate function if the motion it produces is flipped by the mirror.
                    # - Translation along a mirror axis is flipped.
                    # - Rotation *about* an axis *perpendicular* to the mirror plane is flipped.
                    #   (Equivalent to: rotation *about* an axis *in* the mirror plane is NOT flipped)

                    for mirror_axis_idx in self.mirror_axes: # 0, 1, or 2
                        if is_translation and abs(axis_vec_list[mirror_axis_idx]) > 1e-6:
                            # Translation component along the mirror axis exists -> flip translation sense
                            negate_function = not negate_function # Toggle (handles multiple mirror axes)
                        elif is_rotation:
                            # Rotation is flipped if the rotation axis itself IS the mirror axis normal
                            # (i.e., rotation axis is perpendicular to the mirror plane)
                            rotation_axis_is_normal = abs(axis_vec_list[mirror_axis_idx]) > 1e-6 and \
                                                      abs(axis_vec_list[(mirror_axis_idx + 1) % 3]) < 1e-6 and \
                                                      abs(axis_vec_list[(mirror_axis_idx + 2) % 3]) < 1e-6
                            if rotation_axis_is_normal:
                                negate_function = not negate_function # Toggle

                    # Apply function negation if needed
                    if negate_function:
                        print(f"      Negating function for TransformAxis {i} (Axis: {axis_vec})")
                        # Get a mutable reference to the function and mirror it in place
                        func = transform_axis.updFunction()
                        self._mirror_transform_axis_function(func)
                        # No need to setFunction if updFunction modified it in place

            # Add mirroring logic for other joint types if necessary (Pin, Slider, Weld, Ball)
            # Often involves mirroring default coordinate values or frame transforms.
            elif isinstance(joint, osim.PinJoint):
                 print("    (PinJoint specific mirroring not implemented - check coordinate defaults)")
            elif isinstance(joint, osim.BallJoint):
                 print("    (BallJoint specific mirroring not implemented - check coordinate defaults)")
            # ... other joint types ...


            # --- 5. Rename the Joint ---
            mirrored_joint_name = self._get_mirrored_name(original_name, 'joint')
            if mirrored_joint_name != original_name:
                 joint.setName(mirrored_joint_name)
                 print(f"    Renamed joint to '{mirrored_joint_name}'")

        except Exception as e:
             print(f"  Error during mirroring of joint '{original_name}': {e}")


    def _mirror_muscle(self, muscle: osim.Muscle) -> None:
        """Mirrors a Muscle's geometry path points and properties in place."""
        original_name = muscle.getName()
        print(f"  Mirroring muscle '{original_name}'...")

        try:
            # Mirror GeometryPath points
            geom_path = muscle.updGeometryPath()
            path_points = geom_path.updPathPointSet() # Get mutable set
            num_points = path_points.getSize()
            for i in range(num_points):
                point = path_points.get(i) # Get mutable point
                # Mirror location relative to its parent frame
                point.set_location(self._mirror_vec3(point.getLocation()))

                # Update the socket connectee name for the frame this point attaches to
                socket = point.updSocket("parent_frame") # Socket is named 'parent_frame' in PathPoint
                orig_connectee = socket.getConnecteeName()
                # Assume points attach to frames on bodies, try body mapping first
                mirrored_connectee = self._get_mirrored_name(orig_connectee, 'body')
                # If body mapping didn't change it, try frame mapping (in case it attaches to an offset frame)
                if mirrored_connectee == orig_connectee:
                     mirrored_connectee = self._get_mirrored_name(orig_connectee, 'frame')

                if orig_connectee != mirrored_connectee:
                    # print(f"    Updating muscle point {i} socket: '{orig_connectee}' -> '{mirrored_connectee}'")
                    socket.setConnecteeName(mirrored_connectee)

            # Mirror Muscle properties if needed (e.g., pennation angle?)
            # OptimalFiberLength, TendonSlackLength usually symmetric.
            # MaxIsometricForce usually symmetric.
            # Pennation angle might flip sign depending on definition and mirror plane.
            if hasattr(muscle, 'get_pennation_angle_at_optimal') and hasattr(muscle, 'set_pennation_angle_at_optimal'):
                 # Example: Mirroring across YZ (axis 0). If pennation is in XY plane, angle might flip.
                 # This requires careful consideration of the muscle's geometry.
                 # print(f"    (Warning: Muscle '{original_name}' pennation angle mirroring not implemented - requires validation)")
                 pass

            # Rename Muscle
            mirrored_name = self._get_mirrored_name(original_name, 'muscle')
            if mirrored_name != original_name:
                 muscle.setName(mirrored_name)
                 print(f"    Renamed muscle to '{mirrored_name}'")

        except Exception as e:
             print(f"  Error during mirroring of muscle '{original_name}': {e}")

        # TODO: Mirror other forces/actuators as needed (Ligaments, BushingForces, etc.)


    def _mirror_marker(self, marker: osim.Marker) -> None:
        """Mirrors a Marker's location and renames it in place."""
        original_name = marker.getName()
        # print(f"  Mirroring marker '{original_name}'...")
        try:
            # Mirror location relative to its parent frame
            marker.set_location(self._mirror_vec3(marker.get_location()))

            # Update socket connectee name
            socket = marker.updSocket("parent_frame")
            orig_connectee = socket.getConnecteeName()
            # Assume markers attach to frames on bodies, try body mapping first
            mirrored_connectee = self._get_mirrored_name(orig_connectee, 'body')
            if mirrored_connectee == orig_connectee: # Fallback to frame mapping
                 mirrored_connectee = self._get_mirrored_name(orig_connectee, 'frame')

            if orig_connectee != mirrored_connectee:
                 # print(f"    Updating marker socket: '{orig_connectee}' -> '{mirrored_connectee}'")
                 socket.setConnecteeName(mirrored_connectee)

            # Rename marker
            mirrored_marker_name = self._get_mirrored_name(original_name, 'misc')
            if mirrored_marker_name != original_name:
                 marker.setName(mirrored_marker_name)
                 # print(f"    Renamed marker to '{mirrored_marker_name}'")

        except Exception as e:
             print(f"  Error during mirroring of marker '{original_name}': {e}")


    def mirror(self) -> osim.Model:
        """
        Performs the mirroring operation on the internal `mirrored_model`.

        Iterates through bodies, joints, forces (muscles), and markers,
        applying mirroring transformations and renaming based on configuration.

        Returns:
            The modified `osim.Model` object (`self.mirrored_model`).
        """
        print(f"\nStarting mirroring process for model '{self.mirrored_model.getName()}'...")
        self.component_rename_map.clear() # Reset name map for this run

        # --- Pre-populate rename map for excluded bodies ---
        # This ensures components attached to excluded bodies keep their original connections
        print(f"Excluded components (will not be mirrored, names kept): {self.exclude_bodies}")
        for name in self.exclude_bodies:
             self.component_rename_map[name] = name

        # --- Mirror Bodies ---
        body_set = self.mirrored_model.updBodySet()
        # Iterate using index as names might change during iteration
        num_bodies = body_set.getSize()
        print(f"\nProcessing {num_bodies} Bodies...")
        for i in range(num_bodies):
            body = body_set.get(i)
            body_name = body.getName() # Get name before potential modification
            if body_name in self.exclude_bodies:
                # print(f"  Skipping excluded body: {body_name}")
                continue
            self._mirror_body(body) # Mirrors in place


        # --- Mirror Joints ---
        joint_set = self.mirrored_model.updJointSet()
        num_joints = joint_set.getSize()
        print(f"\nProcessing {num_joints} Joints...")
        for i in range(num_joints):
             joint = joint_set.get(i)
             joint_name = joint.getName() # Get name before potential modification

             # Check if the joint connects bodies that are *both* excluded
             try:
                 parent_body_name = joint.getParentFrame().findBaseFrame().getName()
                 child_body_name = joint.getChildFrame().findBaseFrame().getName()

                 # Use the rename map to see if the *original* names were excluded
                 if parent_body_name in self.exclude_bodies and child_body_name in self.exclude_bodies:
                     print(f"  Skipping joint '{joint_name}' connecting excluded bodies '{parent_body_name}' and '{child_body_name}'.")
                     # Ensure joint name and its components are mapped to themselves
                     self.component_rename_map[joint_name] = joint_name
                     coord_set = joint.getCoordinateSet()
                     for j in range(coord_set.getSize()):
                          coord = coord_set.get(j)
                          self.component_rename_map[coord.getName()] = coord.getName()
                     # Add similar logic for owned frames if needed
                     continue # Skip mirroring this joint
             except Exception as e_joint_check:
                  print(f"  Warning: Could not check exclusion status for joint '{joint_name}': {e_joint_check}")
                  # Proceed with mirroring cautiously

             # Mirror the joint IN PLACE
             self._mirror_joint(joint)

        # --- Mirror Forces (Muscles, Ligaments, etc.) ---
        force_set = self.mirrored_model.updForceSet()
        num_forces = force_set.getSize()
        print(f"\nProcessing {num_forces} Forces...")
        for i in range(num_forces):
             force = force_set.get(i)
             force_name = force.getName() # Get name before potential modification

             if muscle := osim.Muscle.safeDownCast(force):
                 # Check if muscle attaches to any excluded body (simplified check)
                 attaches_to_excluded: bool = False
                 try:
                     geom_path = muscle.getGeometryPath()
                     path_points = geom_path.getPathPointSet()
                     for j in range(path_points.getSize()):
                          body_name = path_points.get(j).getParentFrame().findBaseFrame().getName()
                          if body_name in self.exclude_bodies:
                               attaches_to_excluded = True
                               break
                 except Exception as e_muscle_check:
                      print(f"  Warning: Could not check attachments for muscle '{force_name}': {e_muscle_check}")

                 if attaches_to_excluded:
                      print(f"  Skipping muscle '{force_name}' attached to an excluded body.")
                      self.component_rename_map[force_name] = force_name # Keep name
                      continue # Skip mirroring

                 self._mirror_muscle(muscle) # Mirrors in place

             elif isinstance(force, osim.Ligament):
                  print(f"  Skipping Ligament '{force_name}' (mirroring not implemented).")
                  self.component_rename_map[force_name] = force_name # Keep name
             # Add elif for other Force types (BushingForce, SpringGeneralizedForce, etc.) if needed
             else:
                  print(f"  Skipping Force '{force_name}' of type {force.getConcreteClassName()} (mirroring not implemented).")
                  self.component_rename_map[force_name] = force_name # Keep name


        # --- Mirror Markers ---
        marker_set = self.mirrored_model.updMarkerSet()
        num_markers = marker_set.getSize()
        print(f"\nProcessing {num_markers} Markers...")
        for i in range(num_markers):
             marker = marker_set.get(i)
             marker_name = marker.getName() # Get name before potential modification
             try:
                 parent_frame_name = marker.getParentFrame().findBaseFrame().getName()
                 if parent_frame_name in self.exclude_bodies:
                      # print(f"  Skipping marker '{marker_name}' attached to excluded frame '{parent_frame_name}'.")
                      self.component_rename_map[marker_name] = marker_name # Keep name
                      continue # Skip mirroring
             except Exception as e_marker_check:
                  print(f"  Warning: Could not check parent frame for marker '{marker_name}': {e_marker_check}")

             self._mirror_marker(marker) # Mirrors in place

        # --- Mirror Other Components ---
        # TODO: Add ContactGeometry mirroring (location, orientation, frame sockets, renaming)
        # TODO: Add Constraint mirroring (Constraint-specific properties, frame sockets, renaming)
        # TODO: Add Controller mirroring (often requires semantic changes, not just geometric)

        # --- Finalize ---
        print("\nFinalizing model connections...")
        # This step ensures that sockets connect correctly after potential renaming.
        # It rebuilds internal data structures based on the current component names and socket connectee names.
        try:
             self.mirrored_model.finalizeConnections()
             print("Connections finalized successfully.")
        except Exception as e:
             print(f"Warning: Exception during finalizeConnections: {e}. Model integrity might be compromised. Manual inspection recommended.")

        print("\nMirroring process complete.")
        return self.mirrored_model

    def save_model(self, output_model_path: str) -> None:
        """Saves the mirrored model to an .osim file."""
        print(f"Saving mirrored model to: {output_model_path}")
        save_success = self.mirrored_model.printToXML(output_model_path)
        if save_success:
             print("Model saved successfully.")
        else:
             print("Error: Failed to save the model.")


# --- Example Usage ---
if __name__ == "__main__":
    # Define parameters carefully
    # Make sure this path is correct
    input_model = "models/rat_hindlimb_millard_y2j_tsl_r.osim"
    output_model = "models/rat_hindlimb_millard_y2j_tsl_l_oop_typed.osim"

    # Example: Mirror across the YZ plane (negating X-coordinates, axis index 0)
    # Check your model's coordinate system convention!
    mirror_axis_index = 0 # Index for X-axis

    # Naming convention: Assumes things ending in _r need mirroring to _l
    # Use start/end anchors (^) / ($) for more precise matching if needed.
    custom_mappings: Dict[str, Dict[str, str]] = {
        'body': {r'_r$': r'_l'},
        'joint': {r'_r$': r'_l'},
        'frame': {r'_r$': r'_l'},
        'coord': {r'_r$': r'_l'},
        'muscle': {r'_r$': r'_l'},
        'misc': {r'_r$': r'_l'} # Catch-all for markers, wrap objects etc.
    }
    # Bodies to NOT mirror (e.g., shared central bodies like pelvis, ground)
    exclude: List[str] = ['ground', 'spine_base', 'pelvis'] # Adjust to your model's names

    try:
        # Create the mirror tool instance
        mirror_tool = ModelMirror(
            input_model_path_or_obj=input_model,
            mirror_axes=[mirror_axis_index], # Mirror across YZ plane
            ground_name='ground',
            exclude_bodies=exclude,
            name_mappings=custom_mappings
        )

        # Perform the mirroring (modifies the internal mirrored_model)
        mirrored_osim_model: osim.Model = mirror_tool.mirror()

        # Save the result
        mirror_tool.save_model(output_model)

    except FileNotFoundError:
         print(f"Error: Input model file not found at '{input_model}'")
    except ValueError as ve:
         print(f"Configuration Error: {ve}")
    except Exception as e:
         print(f"An unexpected error occurred during mirroring: {e}")
         import traceback
         traceback.print_exc()

