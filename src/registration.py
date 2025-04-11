import numpy as np
import open3d as o3d
import copy
import os
import random

def load_mesh(file_path):
    """Load an STL file and return as an Open3D mesh"""
    mesh = o3d.io.read_triangle_mesh(file_path)
    mesh.compute_vertex_normals()
    return mesh

def compute_fpfh_features(mesh, radius_normal=0.1, radius_feature=0.3):
    """Compute FPFH features for the mesh"""
    # Use fixed number of points with deterministic sampling instead of poisson disk
    pcd = mesh.sample_points_uniformly(number_of_points=5000)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    
    # Compute FPFH features
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    
    return pcd, pcd_fpfh

def global_registration(source_pcd, target_pcd, source_fpfh, target_fpfh, distance_threshold=0.05):
    """Perform global registration using RANSAC"""
    
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_pcd, target_pcd, source_fpfh, target_fpfh,
        True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
        ],
        o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
    )
    
    return result

def refine_registration(source, target, transformation, distance_threshold=0.02):
    """Refine the registration using ICP"""
    
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    
    return result

def try_visualize_registration(source, target, transformation=None):
    """Try to visualize the registration results with fallback options"""
    try:
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        
        # Set different colors
        source_temp.paint_uniform_color([1, 0.706, 0])    # Orange
        target_temp.paint_uniform_color([0, 0.651, 0.929])  # Blue
        
        if transformation is not None:
            source_temp.transform(transformation)
        
        # Try Open3D visualization
        try:
            o3d.visualization.draw_geometries([source_temp, target_temp])
            print("Visualization successful with Open3D")
        except Exception as e:
            print(f"Open3D visualization failed: {e}")
            print("Saving image instead...")
            
            # Try off-screen rendering
            try:
                vis = o3d.visualization.Visualizer()
                vis.create_window(visible=False)
                vis.add_geometry(source_temp)
                vis.add_geometry(target_temp)
                vis.poll_events()
                vis.update_renderer()
                image = vis.capture_screen_float_buffer(True)
                o3d.io.write_image("registration_result.png", image)
                vis.destroy_window()
                print("Registration image saved as 'registration_result.png'")
            except Exception as e2:
                print(f"Off-screen rendering failed: {e2}")
                print("Visualization not available on this system.")
    
    except Exception as e:
        print(f"Visualization preparation failed: {e}")
        print("Skipping visualization, but registration was performed.")

def determine_registration_parameters(source_mesh, target_mesh):
    """Determine optimal registration parameters based on mesh characteristics"""
    # Calculate bounding box to estimate feature size
    source_bbox = source_mesh.get_axis_aligned_bounding_box()
    target_bbox = target_mesh.get_axis_aligned_bounding_box()
    
    # Calculate average edge length as another feature size indicator
    source_vertices = np.asarray(source_mesh.vertices)
    target_vertices = np.asarray(target_mesh.vertices)
    
    # Measure mesh density and feature size
    source_extent = np.linalg.norm(source_bbox.get_extent())
    target_extent = np.linalg.norm(target_bbox.get_extent())
    
    # Scale parameters based on normalized mesh size (both meshes are normalized)
    radius_normal = min(0.05, 2.0 / max(len(source_vertices), len(target_vertices)) ** (1/3))
    radius_feature = min(0.15, 5.0 / max(len(source_vertices), len(target_vertices)) ** (1/3))
    distance_threshold = min(0.05, 2.5 / max(len(source_vertices), len(target_vertices)) ** (1/3))
    
    print(f"Automatically determined parameters:")
    print(f"  - Normal estimation radius: {radius_normal:.4f}")
    print(f"  - Feature radius: {radius_feature:.4f}")  
    print(f"  - Distance threshold: {distance_threshold:.4f}")
    
    return radius_normal, radius_feature, distance_threshold

def ensure_deterministic_mesh(mesh):
    """Ensure mesh has deterministic vertex order for consistent point sampling"""
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    # Sort vertices by their coordinates
    sorted_indices = np.lexsort((vertices[:, 2], vertices[:, 1], vertices[:, 0]))
    sorted_vertices = vertices[sorted_indices]
    
    # Create a mapping from old to new vertex indices
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(sorted_indices)}
    
    # Remap triangle indices
    remapped_triangles = np.array([[index_map[v] for v in tri] for tri in triangles])
    
    # Create new mesh with sorted vertices
    new_mesh = o3d.geometry.TriangleMesh()
    new_mesh.vertices = o3d.utility.Vector3dVector(sorted_vertices)
    new_mesh.triangles = o3d.utility.Vector3iVector(remapped_triangles)
    
    # Copy other properties if available
    if mesh.has_vertex_normals():
        normals = np.asarray(mesh.vertex_normals)
        new_mesh.vertex_normals = o3d.utility.Vector3dVector(normals[sorted_indices])
    
    if mesh.has_vertex_colors():
        colors = np.asarray(mesh.vertex_colors)
        new_mesh.vertex_colors = o3d.utility.Vector3dVector(colors[sorted_indices])
    
    if mesh.has_triangle_normals():
        new_mesh.compute_triangle_normals()
    
    return new_mesh

def register_meshes(source_path, target_path, output_path=None, debug_path=None, seed=42):
    """
    Register two meshes and return transformation information
    
    Args:
        source_path: Path to the source STL file
        target_path: Path to the target STL file
        output_path: Path to save the registered mesh
        debug_path: Optional directory to save intermediate meshes for debugging
        seed: Random seed for reproducibility
        
    Returns:
        transform_info: Dictionary containing transformation information
    """
    # Set seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    
    # Also set Open3D's random seed if available
    try:
        o3d.utility.random.seed(seed)
    except:
        print("Warning: Could not set Open3D random seed directly.")
    
    # Create debug directory if specified
    if debug_path and not os.path.exists(debug_path):
        os.makedirs(debug_path)
    
    # 1. Load original meshes
    print("Loading meshes...")
    source_mesh_orig = load_mesh(source_path)
    target_mesh_orig = load_mesh(target_path)
    
    # Make vertex order deterministic by sorting vertices
    # This ensures consistent behavior in point sampling
    source_mesh_orig = ensure_deterministic_mesh(source_mesh_orig)
    target_mesh_orig = ensure_deterministic_mesh(target_mesh_orig)
    
    if debug_path:
        o3d.io.write_triangle_mesh(os.path.join(debug_path, "1_source_orig.ply"), source_mesh_orig)
        o3d.io.write_triangle_mesh(os.path.join(debug_path, "1_target_orig.ply"), target_mesh_orig)
    
    # 2. Make working copies
    source_mesh = copy.deepcopy(source_mesh_orig)
    target_mesh = copy.deepcopy(target_mesh_orig)
    
    # 3. Get centers and scales before normalization
    source_center_orig = source_mesh.get_center()
    target_center_orig = target_mesh.get_center()
    
    source_vertices = np.asarray(source_mesh.vertices)
    source_scale_orig = np.max(np.linalg.norm(source_vertices - source_center_orig, axis=1))
    
    target_vertices = np.asarray(target_mesh.vertices)
    target_scale_orig = np.max(np.linalg.norm(target_vertices - target_center_orig, axis=1))
    
    print(f"Source center: {source_center_orig}, scale: {source_scale_orig}")
    print(f"Target center: {target_center_orig}, scale: {target_scale_orig}")
    
    # 4. Normalize both meshes for registration (center at origin, scale to unit size)
    source_mesh.translate(-source_center_orig)
    source_mesh.scale(1.0/source_scale_orig, center=np.array([0, 0, 0]))
    
    target_mesh.translate(-target_center_orig)
    target_mesh.scale(1.0/target_scale_orig, center=np.array([0, 0, 0]))
    
    if debug_path:
        o3d.io.write_triangle_mesh(os.path.join(debug_path, "2_source_normalized.ply"), source_mesh)
        o3d.io.write_triangle_mesh(os.path.join(debug_path, "2_target_normalized.ply"), target_mesh)
    
    # 5. Determine optimal registration parameters
    radius_normal, radius_feature, distance_threshold = determine_registration_parameters(source_mesh, target_mesh)
    
    # 6. Compute features for registration
    print("Computing features...")
    source_pcd, source_fpfh = compute_fpfh_features(source_mesh, radius_normal, radius_feature)
    target_pcd, target_fpfh = compute_fpfh_features(target_mesh, radius_normal, radius_feature)
    
    # 7. Perform global registration
    print("Performing global registration...")
    global_result = global_registration(source_pcd, target_pcd, source_fpfh, target_fpfh, distance_threshold)
    
    # 8. Refine registration
    print("Refining registration...")
    icp_result = refine_registration(source_pcd, target_pcd, global_result.transformation, distance_threshold)
    
    # Get the final transformation in normalized space
    norm_transformation = icp_result.transformation
    
    # 9. Create a debug visualization of normalized registration if requested
    if debug_path:
        debug_source = copy.deepcopy(source_mesh)
        debug_source.transform(norm_transformation)
        debug_source.paint_uniform_color([1, 0, 0])  # Red
        target_mesh.paint_uniform_color([0, 0, 1])   # Blue
        o3d.io.write_triangle_mesh(os.path.join(debug_path, "3_source_registered_normalized.ply"), debug_source)
        
        combined = copy.deepcopy(target_mesh)
        combined += debug_source
        o3d.io.write_triangle_mesh(os.path.join(debug_path, "3_combined_normalized.ply"), combined)
    
    # 10. Create and save final registered mesh that matches target coordinate system
    if output_path:
        # Start with fresh copy of source
        registered_mesh = copy.deepcopy(source_mesh_orig)
        
        # CRITICAL FIX: Build complete transformation pipeline properly
        # Step 1: Create matrices for normalization transformations
        source_norm_matrix = np.eye(4)
        source_norm_matrix[:3, :3] *= 1.0/source_scale_orig
        source_norm_matrix[:3, 3] = -source_center_orig / source_scale_orig
        
        target_denorm_matrix = np.eye(4)
        target_denorm_matrix[:3, :3] *= target_scale_orig
        target_denorm_matrix[:3, 3] = target_center_orig
        
        # Step 2: Combine matrices: denormalize_target * registration * normalize_source
        # This ensures that the source mesh will be aligned with the target mesh in scale and origin
        complete_transform = np.matmul(target_denorm_matrix, 
                                      np.matmul(norm_transformation, source_norm_matrix))
        
        # Step 3: Apply combined transformation directly to original source mesh
        registered_mesh.transform(complete_transform)
        
        # Save the registered mesh
        o3d.io.write_triangle_mesh(output_path, registered_mesh)
        print(f"Registered mesh saved to {output_path}")
        
        if debug_path:
            # Also save target for comparison
            o3d.io.write_triangle_mesh(os.path.join(debug_path, "4_target_orig.ply"), target_mesh_orig)
            registered_mesh.paint_uniform_color([1, 0, 0])  # Red
            combined_final = copy.deepcopy(target_mesh_orig)
            combined_final += registered_mesh
            o3d.io.write_triangle_mesh(os.path.join(debug_path, "4_combined_final.ply"), combined_final)
        
        # Return transformation info
        transform_info = {
            'complete_transform': complete_transform,
            'fitness': icp_result.fitness,
            'inlier_rmse': icp_result.inlier_rmse,
            'source_center_orig': source_center_orig,
            'source_scale_orig': source_scale_orig,
            'target_center_orig': target_center_orig,
            'target_scale_orig': target_scale_orig,
            'parameters': {
                'radius_normal': radius_normal,
                'radius_feature': radius_feature,
                'distance_threshold': distance_threshold,
                'seed': seed
            }
        }
    else:
        # Just return the transformation info without saving
        transform_info = {
            'norm_transformation': norm_transformation,
            'source_center_orig': source_center_orig,
            'source_scale_orig': source_scale_orig,
            'target_center_orig': target_center_orig,
            'target_scale_orig': target_scale_orig,
            'fitness': icp_result.fitness,
            'inlier_rmse': icp_result.inlier_rmse,
            'parameters': {
                'radius_normal': radius_normal,
                'radius_feature': radius_feature,
                'distance_threshold': distance_threshold,
                'seed': seed
            }
        }
    
    return transform_info

def convert_points_between_meshes(points, transform_info, reverse=False):
    """
    Convert points between source and target coordinate systems
    
    Args:
        points: numpy array of shape (N, 3) containing the points to transform
        transform_info: dictionary containing transformation information
        reverse: if True, convert from target to source coordinate system
                 if False, convert from source to target coordinate system
    
    Returns:
        transformed_points: numpy array of shape (N, 3) containing the transformed points
    """
    # Get the complete transformation
    if 'complete_transform' in transform_info:
        transformation = transform_info['complete_transform']
        
        # Create homogeneous coordinates
        homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))
        
        if reverse:
            # Invert the transformation
            inv_transformation = np.linalg.inv(transformation)
            transformed_homogeneous = np.dot(homogeneous_points, inv_transformation.T)
        else:
            transformed_homogeneous = np.dot(homogeneous_points, transformation.T)
        
        # Return the 3D points (discard homogeneous coordinate)
        return transformed_homogeneous[:, :3]
    else:
        # We need to build the complete transformation first
        source_center = transform_info['source_center_orig']
        source_scale = transform_info['source_scale_orig']
        target_center = transform_info['target_center_orig']
        target_scale = transform_info['target_scale_orig']
        norm_transformation = transform_info['norm_transformation']
        
        # Build matrices
        source_norm_matrix = np.eye(4)
        source_norm_matrix[:3, :3] *= 1.0/source_scale
        source_norm_matrix[:3, 3] = -source_center / source_scale
        
        target_denorm_matrix = np.eye(4)
        target_denorm_matrix[:3, :3] *= target_scale
        target_denorm_matrix[:3, 3] = target_center
        
        # Combine
        if reverse:
            source_denorm_matrix = np.eye(4)
            source_denorm_matrix[:3, :3] *= source_scale
            source_denorm_matrix[:3, 3] = source_center
            
            target_norm_matrix = np.eye(4)
            target_norm_matrix[:3, :3] *= 1.0/target_scale
            target_norm_matrix[:3, 3] = -target_center / target_scale
            
            inv_norm_transformation = np.linalg.inv(norm_transformation)
            
            # target -> normalize target -> inverse registration -> denormalize source
            complete_transform = np.matmul(source_denorm_matrix, 
                                           np.matmul(inv_norm_transformation, target_norm_matrix))
        else:
            # source -> normalize source -> registration -> denormalize target
            complete_transform = np.matmul(target_denorm_matrix, 
                                           np.matmul(norm_transformation, source_norm_matrix))
        
        # Create homogeneous coordinates
        homogeneous_points = np.hstack((points, np.ones((points.shape[0], 1))))
        transformed_homogeneous = np.dot(homogeneous_points, complete_transform.T)
        
        # Return the 3D points
        return transformed_homogeneous[:, :3]

# Example usage
if __name__ == "__main__":
    # Replace with your actual file paths
    source_path = "source.stl"
    target_path = "target.stl"
    output_path = "output.stl"
    
    # Optional debug directory - uncomment to save debug meshes
    # debug_path = "debug_meshes"
    
    # Register meshes and save the result
    # Use seed=42 for reproducibility
    transform_info = register_meshes(
        source_path, 
        target_path, 
        output_path, 
        # debug_path=debug_path,
        seed=42
    )
    
    print("\nRegistration quality:")
    print(f"Fitness: {transform_info['fitness']}")
    print(f"Inlier RMSE: {transform_info['inlier_rmse']}")
    print(f"Registration parameters: {transform_info['parameters']}")
    
    # Example: Transform a point from source to target
    example_point = np.array([[10., 20., 30.]])
    transformed_point = convert_points_between_meshes(example_point, transform_info)
    
    print("\nExample point transformation:")
    print(f"Original point (source space): {example_point[0]}")
    print(f"Transformed point (target space): {transformed_point[0]}")
    
    # Transform back to verify
    back_transformed = convert_points_between_meshes(transformed_point, transform_info, reverse=True)
    print(f"Back-transformed point (source space): {back_transformed[0]}")
    print(f"Transformation error: {np.linalg.norm(example_point - back_transformed)}")