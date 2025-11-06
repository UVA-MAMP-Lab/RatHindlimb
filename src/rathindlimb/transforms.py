import numpy as np

def find_homogeneous_transform(points_a, points_b):
    """
    Find the homogeneous transformation that maps points from coordinate system A to B.
    
    Parameters:
    points_a: numpy array of shape (n, 3) - Points in coordinate system A
    points_b: numpy array of shape (n, 3) - Same points in coordinate system B
    
    Returns:
    transform: numpy array of shape (4, 4) - Homogeneous transformation matrix
    error: float - Mean squared error of the transformation
    """
    if points_a.shape[0] < 3 or points_b.shape[0] < 3:
        raise ValueError("At least 3 point correspondences are required")
    
    # Calculate centroids
    centroid_a = np.mean(points_a, axis=0)
    centroid_b = np.mean(points_b, axis=0)
    
    # Center the points
    centered_a = points_a - centroid_a
    centered_b = points_b - centroid_b
    
    # Calculate the covariance matrix
    H = centered_a.T @ centered_b
    
    # Use SVD to find the rotation
    U, _, Vt = np.linalg.svd(H)
    
    # Ensure proper rotation (handle reflections)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Calculate the translation
    t = centroid_b - R @ centroid_a
    
    # Create the homogeneous transformation matrix
    transform = np.eye(4)
    transform[:3, :3] = R
    transform[:3, 3] = t
    
    # Calculate the error
    points_a_homog = np.hstack((points_a, np.ones((points_a.shape[0], 1))))
    transformed_points = (transform @ points_a_homog.T).T[:, :3]
    error = np.mean(np.sum((transformed_points - points_b) ** 2, axis=1))
    
    return transform, error

def find_transform_linear_system(points_a, points_b):
    """
    Find the homogeneous transformation that maps points from coordinate system A to B
    using a system of linear equations.
    
    Parameters:
    points_a: numpy array of shape (n, 3) - Points in coordinate system A
    points_b: numpy array of shape (n, 3) - Same points in coordinate system B
    
    Returns:
    transform: numpy array of shape (4, 4) - Homogeneous transformation matrix
    error: float - Mean squared error of the transformation
    """
    if points_a.shape[0] < 4:
        raise ValueError("At least 4 non-coplanar point correspondences are required")
    
    n_points = points_a.shape[0]
    
    # Set up the linear system
    # For each point, we have equations:
    # r11*xa + r12*ya + r13*za + tx = xb
    # r21*xa + r22*ya + r23*za + ty = yb
    # r31*xa + r32*ya + r33*za + tz = zb
    
    # Create the coefficient matrix A and the right-hand side b
    A = np.zeros((3 * n_points, 12))
    b = np.zeros(3 * n_points)
    
    for i in range(n_points):
        xa, ya, za = points_a[i]
        xb, yb, zb = points_b[i]
        
        # Equations for x coordinate
        A[3*i, 0] = xa
        A[3*i, 1] = ya
        A[3*i, 2] = za
        A[3*i, 3] = 1
        b[3*i] = xb
        
        # Equations for y coordinate
        A[3*i+1, 4] = xa
        A[3*i+1, 5] = ya
        A[3*i+1, 6] = za
        A[3*i+1, 7] = 1
        b[3*i+1] = yb
        
        # Equations for z coordinate
        A[3*i+2, 8] = xa
        A[3*i+2, 9] = ya
        A[3*i+2, 10] = za
        A[3*i+2, 11] = 1
        b[3*i+2] = zb
    
    # Solve the system using least squares
    x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
    
    # Extract the components of the transformation matrix
    r11, r12, r13, tx = x[0:4]
    r21, r22, r23, ty = x[4:8]
    r31, r32, r33, tz = x[8:12]
    
    # Construct the transformation matrix
    transform = np.array([
        [r11, r12, r13, tx],
        [r21, r22, r23, ty],
        [r31, r32, r33, tz],
        [0, 0, 0, 1]
    ])
    
    # Calculate the error
    points_a_homog = np.hstack((points_a, np.ones((points_a.shape[0], 1))))
    transformed_points = (transform @ points_a_homog.T).T[:, :3]
    error = np.mean(np.sum((transformed_points - points_b) ** 2, axis=1))
    
    # Ensure the rotation matrix is orthogonal (optional refinement)
    # This step ensures we have a proper rotation matrix
    U, _, Vt = np.linalg.svd(transform[:3, :3])
    refined_rotation = U @ Vt
    
    refined_transform = transform.copy()
    refined_transform[:3, :3] = refined_rotation
    
    return transform, refined_transform, error