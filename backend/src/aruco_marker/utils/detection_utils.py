import cv2.aruco as aruco
from cv2.typing import MatLike
import numpy as np
from cv2.aruco import DetectorParameters, Dictionary
from typing import Sequence, Tuple


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
