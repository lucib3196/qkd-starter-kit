# --- Standard Library ---
from typing import Sequence, Tuple

# --- Third-Party Packages ---
import cv2
import numpy as np
import cv2.aruco as aruco
from cv2.typing import MatLike
from cv2.aruco import DetectorParameters, Dictionary
from .drawing_utils import draw_square_frame, draw_id, display_distance_marker


def detect_markers(
    frame: MatLike, aruco_dict: Dictionary, parameters: DetectorParameters
) -> Sequence[Tuple[Sequence[MatLike], int]] | None:
    """
    Detects ArUco markers in a given frame.

    Parameters:
        frame (numpy.ndarray): The input frame (grayscale or color image) in which to detect ArUco markers.
        aruco_dict (cv2.aruco.Dictionary): The ArUco dictionary to use for marker detection.
        parameters (cv2.aruco.DetectorParameters): Detection parameters for the ArUco detector.

    Returns:
        list: A list of tuples, where each tuple contains the detected marker's corners and its ID.
    Example: [((corner1, corner2, corner3, corner4), id), ...]
    """
    detector = aruco.ArucoDetector(aruco_dict, parameters)
    markers, ids, _ = detector.detectMarkers(frame)
    if markers and ids is not None:
        detected = []
        for corners, marker_id in zip(markers, ids):
            detected.append((corners, int(marker_id[0])))
        return detected
    return None


def get_marker_corners(marker: Sequence[MatLike]) -> Sequence[Tuple[int, int]]:
    """
    Extract the 4 corners of an ArUco marker in (x, y) order.
    Follows a clockwise order so first coord is top left then second is top right etc

    marker: array shaped (1,4,2) or (4,2)
    """
    # Flatten if shape is (1,4,2)

    m = marker[0]

    # marker is now shape (4, 2)
    m = m.astype(int)

    return [
        tuple(m[0]),
        tuple(m[1]),
        tuple(m[2]),
        tuple(m[3]),
    ]


def get_marker_center(marker: Sequence[MatLike]) -> Tuple[int, int]:
    """
    Compute the center of an ArUco marker by averaging its corners.
    """

    m = marker[0]

    # marker is now shape (4, 2)
    m = m.astype(int)

    # marker shape is (4,2)
    x_coords = m[:, 0]
    y_coords = m[:, 1]

    center_x = int(np.sum(x_coords) / 4)
    center_y = int(np.sum(y_coords) / 4)

    return (center_x, center_y)


def estimate_pose_and_transformation_matrix(
    marker: Sequence[MatLike],
    camera_matrix: np.ndarray,
    distortion_coeff: np.ndarray,
    marker_length=0.1,
):
    """
    Estimates the pose of an ArUco marker and returns the transformation matrix along with its distance,
    rotation vector, and translation vector.

    Parameters:
        marker (numpy.ndarray): Marker corners.
        camera_matrix (numpy.ndarray): Camera intrinsic matrix.
        distortion_coeff (numpy.ndarray): Camera distortion coefficients.
        marker_length (float): The physical length of the marker in meters (default is 0.1).

    Returns:
        tuple: (transformation_matrix, distance, rvec, tvec)
            - transformation_matrix (numpy.ndarray): 4x4 transformation matrix.
            - distance (float): Distance to the marker in centimeters.
            - rvec (numpy.ndarray): Rotation vector.
            - tvec (numpy.ndarray): Translation vector.
    """
    # Estimate pose of the marker and get the transformation matrix
    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
        marker, marker_length, camera_matrix, distortion_coeff
    )
    R, _ = cv2.Rodrigues(rvec)
    # print(f"This is the marker length{marker_length}")
    transformation_matrix = np.hstack((R, tvec[0].T))
    transformation_matrix = np.vstack((transformation_matrix, np.array([0, 0, 0, 1])))

    # Calculate the norm distance of the marker in centimeters
    distance = np.linalg.norm(tvec) * 100
    return transformation_matrix, distance, rvec, tvec


def track_and_render_marker(
    frame, marker, marker_id, camera_matrix, distortion_coefficient, marker_length=0.1
):
    """
    Tracks the marker and renders its square frame, axis, and ID on the frame.

    This function estimates the pose of the marker, draws a square frame around it,
    annotates its ID, and renders the coordinate axes based on the estimated pose.
    It serves as the main function for marker tracking in the project.

    Parameters:
        frame (numpy.ndarray): The frame to draw on.
        marker (numpy.ndarray): The detected marker's corners.
        marker_id (int): The ID of the marker.
        camera_matrix (numpy.ndarray): The camera's intrinsic matrix.
        distortion_coefficient (numpy.ndarray): The camera's distortion coefficients.
        marker_length (float): The physical length of the marker in meters.
    """
    marker_center = get_marker_center(marker)
    marker_coord = get_marker_corners(marker)
    transformation_matrix, distance, rvec, tvec = (
        estimate_pose_and_transformation_matrix(
            marker, camera_matrix, distortion_coefficient, marker_length
        )
    )

    draw_square_frame(frame, marker_coord)
    draw_id(frame, marker_coord, marker_id)
    display_distance_marker(frame, marker_id, distance)
    cv2.drawFrameAxes(
        frame, camera_matrix, distortion_coefficient, rvec, tvec, marker_length
    )
    return transformation_matrix
