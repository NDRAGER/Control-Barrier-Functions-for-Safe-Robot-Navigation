import cv2
import numpy as np
from pupil_apriltags import Detector

def extract_apriltag_corners(image_path):
    """Extract AprilTag corners from image using pupil-apriltags"""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    detector = Detector(
        families='tag36h11',
        nthreads=1,
        quad_decimate=1.0,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0
    )
    
    detections = detector.detect(gray)
    
    tag_0 = None
    for detection in detections:
        if detection.tag_id == 0:
            tag_0 = detection
            break
    
    if tag_0 is None:
        raise ValueError("AprilTag with ID 0 not found in image")
    
    corners_2d = tag_0.corners
    
    print(f"Detected AprilTag ID {tag_0.tag_id}")
    for i, corner in enumerate(corners_2d):
        print(f"Corner {i}: ({corner[0]:.2f}, {corner[1]:.2f})")
    
    return corners_2d

def setup_apriltag_3d_corners(tag_size=0.01):
    """Define 3D positions of AprilTag corners in tag's body-centric frame"""
    half_size = tag_size / 2.0
    
    corners_3d = np.array([
        [-half_size, -half_size, 0],  # Corner 0: lower-left
        [ half_size, -half_size, 0],  # Corner 1: lower-right  
        [ half_size,  half_size, 0],  # Corner 2: upper-right
        [-half_size,  half_size, 0],  # Corner 3: upper-left
    ], dtype=np.float32)
    
    return corners_3d

def solve_pnp_opencv(corners_2d, corners_3d, K_matrix):
    """Solve PnP problem using OpenCV's solvePnP"""
    image_points = np.array(corners_2d, dtype=np.float32)
    object_points = corners_3d
    dist_coeffs = np.zeros((4,1))
    
    success, rvec, tvec = cv2.solvePnP(
        object_points, 
        image_points, 
        K_matrix, 
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if not success:
        raise RuntimeError("solvePnP failed to find a solution")
    
    R, _ = cv2.Rodrigues(rvec)
    
    return R, tvec, rvec

def rotation_matrix_to_euler(R):
    """Convert rotation matrix to Euler angles (roll, pitch, yaw)"""
    sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
    singular = sy < 1e-6
    
    if not singular:
        x = np.arctan2(R[2,1], R[2,2])  # Roll
        y = np.arctan2(-R[2,0], sy)     # Pitch
        z = np.arctan2(R[1,0], R[0,0])  # Yaw
    else:
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0
    
    return np.array([x, y, z])

def compute_reprojection_errors(corners_2d, corners_3d, K_matrix, rvec, tvec):
    """Compute reprojection errors"""
    projected_points, _ = cv2.projectPoints(
        corners_3d, 
        rvec, 
        tvec, 
        K_matrix, 
        np.zeros((4,1))
    )
    
    projected_points = projected_points.reshape(-1, 2)
    
    errors = []
    for i in range(len(corners_2d)):
        error = np.sqrt(
            (projected_points[i,0] - corners_2d[i][0])**2 + 
            (projected_points[i,1] - corners_2d[i][1])**2
        )
        errors.append(error)
    
    return errors, projected_points

def main():
    # Image path
    image_path = '/Users/satviktajne/Desktop/Sem 1/Mobile robotics /CODES/hw3/frame_0.jpg'
    
    # Camera matrix K from Problem 1 - REPLACE WITH YOUR ACTUAL VALUES
    K = np.array([
        [800.0, 0.0, 640.0],  # fx, skew, cx
        [0.0, 800.0, 480.0],  # 0, fy, cy
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)
    
    print("Part (a): PnP Problem Formulation")
    print("min_X Σ(i=1 to N) ||ũᵢ - π(K, X, Pᵢ)||²")
    print("where X ∈ SE(3) is the camera pose to estimate\n")
    
    print("Part (b): Implementation")
    
    # Extract AprilTag corners
    corners_2d = extract_apriltag_corners(image_path)
    
    # Setup 3D corners
    corners_3d = setup_apriltag_3d_corners(tag_size=0.01)
    
    print("\n3D Corner positions in tag frame (meters):")
    for i, corner in enumerate(corners_3d):
        print(f"Corner {i}: ({corner[0]:.3f}, {corner[1]:.3f}, {corner[2]:.3f})")
    
    # Solve PnP using OpenCV
    R, tvec, rvec = solve_pnp_opencv(corners_2d, corners_3d, K)
    
    # Convert to Euler angles
    euler = rotation_matrix_to_euler(R)
    
    # Print results
    print("\nEstimated Camera Pose in AprilTag 0's Body-Centric Frame:")
    print(f"Translation (x, y, z) [meters]:")
    print(f"  x: {tvec[0,0]:.6f}")
    print(f"  y: {tvec[1,0]:.6f}")
    print(f"  z: {tvec[2,0]:.6f}")
    print(f"  Distance from tag: {np.linalg.norm(tvec):.6f} m")
    
    print(f"\nRotation (Roll, Pitch, Yaw) [radians]:")
    print(f"  Roll:  {euler[0]:.6f}")
    print(f"  Pitch: {euler[1]:.6f}")
    print(f"  Yaw:   {euler[2]:.6f}")
    
    print(f"\nRotation (Roll, Pitch, Yaw) [degrees]:")
    print(f"  Roll:  {np.degrees(euler[0]):.2f}°")
    print(f"  Pitch: {np.degrees(euler[1]):.2f}°")
    print(f"  Yaw:   {np.degrees(euler[2]):.2f}°")
    
    print(f"\nRotation Matrix:")
    for i in range(3):
        print(f"  [{R[i,0]:.6f}, {R[i,1]:.6f}, {R[i,2]:.6f}]")
    
    # Compute reprojection errors for validation
    errors, projected_points = compute_reprojection_errors(
        corners_2d, corners_3d, K, rvec, tvec
    )
    
    print("\nReprojection Errors:")
    for i in range(4):
        print(f"Corner {i}: {errors[i]:.3f} pixels")
    
    print(f"\nMean reprojection error: {np.mean(errors):.3f} pixels")
    
    # Final answer for submission
    print("\nFINAL ANSWER:")
    print(f"Camera Position: x={tvec[0,0]:.4f}m, y={tvec[1,0]:.4f}m, z={tvec[2,0]:.4f}m")
    print(f"Camera Rotation: roll={np.degrees(euler[0]):.1f}°, pitch={np.degrees(euler[1]):.1f}°, yaw={np.degrees(euler[2]):.1f}°")

if __name__ == "__main__":
    main()