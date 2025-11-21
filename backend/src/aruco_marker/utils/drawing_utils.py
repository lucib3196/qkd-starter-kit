import cv2
from cv2.typing import MatLike
from typing import Tuple, Sequence


def draw_corners_circ(frame: MatLike, center: Tuple[int | float, int | float]):
    """Draw a small filled circle at the given marker corner center."""

    cx, cy = int(center[0]), int(center[1])

    cv2.circle(
        frame,
        (cx, cy),  # center point
        10,  # radius
        (0, 255, 0),  # color (green)
        thickness=-1,  # filled circle
    )


def draw_square_frame(
    frame, coordinates: Sequence[Tuple[int | float, int | float]]
) -> None:
    """
    Draw a square frame around a marker using its 4 corner coordinates.

    Parameters:
        frame (numpy.ndarray): The frame on which to draw.
        coordinates (Sequence): Expected format:
            [(top_left), (top_right), (bottom_right), (bottom_left), center]
            Only the first four points are used.
    """

    if len(coordinates) < 4:
        print("draw_square_frame: Not enough coordinates provided:", coordinates)
        return

    # Unpack only the four corners
    top_left, top_right, bottom_right, bottom_left = coordinates[:4]

    # Ensure integer (x, y) tuples for OpenCV
    corners = [
        (int(top_left[0]), int(top_left[1])),
        (int(top_right[0]), int(top_right[1])),
        (int(bottom_right[0]), int(bottom_right[1])),
        (int(bottom_left[0]), int(bottom_left[1])),
    ]

    # Draw edges of the square in order
    for i in range(4):
        start = corners[i]
        end = corners[(i + 1) % 4]  # wraps back to first corner
        cv2.line(frame, start, end, (0, 255, 0), 2)


def draw_id(frame, coordinates: Sequence[Tuple[int | float, int | float]], marker_id=0):
    """
    Draws the marker ID at the top-right corner of the marker on the frame.

    Parameters:
        frame (numpy.ndarray): The frame/image to draw on.
        coordinates (list): A list of corner coordinates, where the second element is used as the position.
        marker_id (int): The ID to draw.

    Returns:
        numpy.ndarray: The frame with the marker ID drawn.
    """
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 1
    color = (0, 255, 0)
    thickness = 2

    top_right = coordinates[1]
    top_right = (int(top_right[0]), int(top_right[1]))
    cv2.putText(frame, str(marker_id), top_right, font, font_scale, color, thickness)
    return frame


def display_distance_marker(frame, marker_id, distance):
    """
    Displays the marker ID and its distance on the frame.

    Parameters:
        frame (numpy.ndarray): The image frame to display the text on.
        marker_id (int): The ID of the marker.
        distance (float): The distance to the marker in centimeters.
    """
    cv2.putText(
        frame,
        f"Marker: {marker_id}, Distance: {distance:.2f} cm",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        3,
    )
