import cv2
import cv2.aruco as aruco
import time
import numpy as np
from . import WebcamVideoStreamThreaded, FPS, putIterationsPerSec, VideoShow
import traceback
import numpy
# Constants
aruco_dict_type = cv2.aruco.DICT_6X6_250

def find_marker(frame, aruco_dict, parameters):
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

    marker_arr = []
    if ids is not None:
        for marker_corner, marker_id in zip(markers, ids):
            marker_arr.append((marker_corner, marker_id))

    return marker_arr

def get_corner_and_center(marker_corner):
    """
    Calculates the corners and center of an ArUco marker.

    Parameters:
        marker_corner (numpy.ndarray): A 4x2 array containing the coordinates of the marker's corners.
                                       Example: [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]

    Returns:
        list: A list containing the four corners (top-left, top-right, bottom-right, bottom-left) as tuples of integers,
              and the center point as a tuple.
              Example: [(top_left), (top_right), (bottom_right), (bottom_left), (center)]
    """
    corners_abcd = marker_corner.reshape((4, 2))
    top_left, top_right, bottom_right, bottom_left = corners_abcd

    top_left = (int(top_left[0]), int(top_left[1]))
    top_right = (int(top_right[0]), int(top_right[1]))
    bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
    bottom_left = (int(bottom_left[0]), int(bottom_left[1]))

    center_x = (top_left[0] + top_right[0] + bottom_right[0] + bottom_left[0]) // 4
    center_y = (top_left[1] + top_right[1] + bottom_right[1] + bottom_left[1]) // 4
    center = (center_x, center_y)

    return [top_left, top_right, bottom_right, bottom_left, center]

def draw_id(frame, coor, marker_id):
    """
    Draws the marker ID at the top-left corner of the marker on the frame.

    Parameters:
        frame (numpy.ndarray): The frame/image to draw on.
        coor (list): A list of corner coordinates, where coor[0] is the top-left corner.
        marker_id (int): The ID to draw.

    Returns:
        numpy.ndarray: The frame with the marker ID drawn.
    """
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 1
    color = (0, 255, 0)
    thickness = 2

    top_right = coor[1]
    cv2.putText(frame, str(marker_id), top_right, font, font_scale, color, thickness)
    return frame

def draw_square_frame(image_frame, coordinates):
    [top_left, top_right, bottom_right, bottom_left, center] = coordinates
    # Draw lines connecting the marker corners
    cv2.line(image_frame, top_left, top_right, (0, 255, 0), 2)
    cv2.line(image_frame, top_right, bottom_right, (0, 255, 0), 2)
    cv2.line(image_frame, bottom_right, bottom_left, (0, 255, 0), 2)
    cv2.line(image_frame, bottom_left, top_left, (0, 255, 0), 2)
    
    cv2.circle(image_frame, center, 5, (255, 0, 255), cv2.FILLED) # Center dot 
    
    
def estimate_marker_pose_single(frame, marker, camera_matrix, distortion_coeff, marker_id, marker_length=0.1):
    """Estimate the pose of an ArUco marker and display its distance on the frame.

    Args:
        frame (numpy.ndarray): The image frame where the marker is detected.
        marker (numpy.ndarray): The detected marker corners.
        camera_matrix (numpy.ndarray): Camera matrix for intrinsic parameters.
        distortion_coeff (numpy.ndarray): Distortion coefficients of the camera.
        marker_id (int): ID of the marker being processed.
        marker_length (float, optional): Length of the marker's side in meters. Defaults to 0.1.

    Returns:
        None: The function modifies the input frame to display marker ID and distance.
    """

    # Estimate the pose of the marker
    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(marker, marker_length, camera_matrix, distortion_coeff)

    # Calculate the distance to the marker in centimeters
    distance = np.linalg.norm(tvec) * 100  # Convert to cm

    # Display the marker ID and distance on the frame
    cv2.putText(
        frame,
        f"Marker: {marker_id}, Distance: {distance:.2f} cm",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        3
    )
    return distance
    
def estimate_marker_pose_multiple(frame, marker_array, camera_matrix, distortion_coeff, marker_lenght = 0.1):
    all_text = ""
    for marker, marker_id in marker_array:
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(marker, marker_lenght, camera_matrix, distortion_coeff)
        distance = np.linalg.norm(tvec) * 100  # Convert to cm
        all_text += f"Marker: {marker_id}, Distance: {distance:.2f} cm \n"
    print(all_text)    
     # Display text on the frame
    y_off = 200  # Starting vertical position for the text
    for line in all_text.split('\n'):
        if line.strip():  # Skip empty lines
            cv2.putText(
                frame,
                text=line,
                org=(10, y_off),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(0, 0, 0),
                thickness=1,
                lineType=cv2.LINE_AA
            )
            y_off += 20  # Add spacing between lines

def draw_field(img, markers, ids, default_order=[1, 5, 10, 42]):
    """
    Draws a quadrilateral on the image based on marker positions.

    Parameters:
        img (numpy.ndarray): The input image.
        markers (list): List of marker corner coordinates.
        ids (list): List of marker IDs corresponding to the markers.
        default_order (list): List of IDs defining the desired order of corners.

    Returns:
        tuple: (img_new, squarefound)
            img_new (numpy.ndarray): Image with the quadrilateral drawn.
            squarefound (bool): Whether a quadrilateral was successfully drawn.
    """
    if len(markers) == 4:
        markers_sorted = [None] * 4
        try:
            for idx, sorted_corner_id in enumerate(default_order):
                if sorted_corner_id in ids:
                    index = ids.index(sorted_corner_id)
                    markers_sorted[idx] = markers[index]
                else:
                    raise ValueError(f"ID {sorted_corner_id} not found in detected IDs.")

            contours = np.array(markers_sorted)
            overlay = img.copy()
            cv2.fillPoly(overlay, pts=[contours], color=(255, 215, 0))
            alpha = 0.4
            img_new = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
            squarefound = True
        except ValueError as e:
            print(f"Error: {e}")
            img_new = img
            squarefound = False
    else:
        img_new = img
        squarefound = False

    return img_new, squarefound

def draw_rounder_corner(frame, coor):
    """
    Draws a circle at the top-left corner of the marker.

    Parameters:
        frame (numpy.ndarray): The input frame.
        coor (list): A list of corner coordinates, where coor[0] is the top-left corner.

    Returns:
        numpy.ndarray: The frame with the circle drawn.
    """
    top_left = coor[0]
    radius = 5
    color = (255, 0, 0)
    thickness = -1

    cv2.circle(frame, top_left, radius, color, thickness)
    return frame

def display_spec(img, marker_array):
    """
    Displays the number of markers found on the given image.

    Parameters:
        img (numpy.ndarray): The image where the text will be displayed.
        marker_array (list): A list of detected markers.

    Returns:
        numpy.ndarray: The image with the text displayed.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    font_scale = 0.5
    color = (0, 0, 0)

    amount_marker = len(marker_array)
    spec = f"{amount_marker} markers found."

    x, y = 15, 30
    text_size = cv2.getTextSize(spec, font, font_scale, thickness)[0]
    cv2.rectangle(img, (x - 5, y - text_size[1] - 5), (x + text_size[0] + 5, y + 5), (255, 255, 255), -1)
    cv2.putText(img, spec, (x, y), font, font_scale, color, thickness)

    print(spec)
    return img


def draw_bounded_area(frame, marker_array):
    ordered_id = [1,5,10,42]
    
    all_corners = [(marker_id,get_corner_and_center(marker)[0]) for marker,marker_id in marker_array]
    sorted_corners = sorted(
        all_corners,
        key = lambda x: ordered_id.index(x[0]) if x[0] in ordered_id else float('inf')
    )
    boundary = np.array([corner[1] for corner in sorted_corners])
    reshaped_boundary = boundary.reshape((-1, 1, 2)) # reshape for opencv
    
    cv2.polylines(frame,[reshaped_boundary], isClosed=True, color = (255,0,255), thickness=3)
    cv2.fillPoly(frame, [reshaped_boundary], color=(0, 255, 0))  # Fill with green color
    
    overlay = frame.copy()
    alpha = 0.75  # Transparency factor.
        # Following line overlays transparent rectangle over the image
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)# 
    
    # Draw a cross line for each corner
    cv2.line(frame, boundary[0], boundary[2],(255,255,255), thickness=1)
    cv2.line(frame, boundary[1], boundary[3],(255,255,255), thickness=1)
    
    
    # Determine center point
    x, y = determine_intersection_point(x=boundary[0], y=boundary[2], u=boundary[1], v=boundary[3])
    x, y = int(x), int(y)
    
    print(x,y)

    # Draw the circle
    cv2.circle(frame, center=(x, y), radius=3, color=(255, 255, 255), thickness=1)
    return frame


def determine_intersection_point(x, y, u, v):
    """
    Determine the intersection point of two lines defined by points (x, y) and (u, v).
    """
    # Fit lines to points x,y and u,v
    coefficients1 = np.polyfit(x, y, 1)  # Line 1: y = m1*x + c1
    coefficients2 = np.polyfit(u, v, 1)  # Line 2: y = m2*x + c2

    polynomial1 = np.poly1d(coefficients1)
    polynomial2 = np.poly1d(coefficients2)

    # Difference between the two polynomials
    diff = polynomial1 - polynomial2

    # Find roots (intersection points)
    intersection = np.roots(diff)

    # Check for no intersection or invalid result
    if len(intersection) == 0:
        raise ValueError("The lines do not intersect or are parallel.")

    # Evaluate y-value at the intersection point
    y_val = polynomial1(intersection)

    # Return the first intersection point as a tuple
    return float(intersection[0]), float(y_val[0])

    

def main(src=1):
    """
    Main function to process video stream and detect ArUco markers.

    Parameters:
        src (int): The video source (default is 1).

    Returns:
        None
    """
    
    marker_hold = True # This just says to hold the marker position
    
    try:
        video_stream = WebcamVideoStreamThreaded(src).start()
        video_display = VideoShow(video_stream.frame).start()
        fps_counter = FPS().start()

        while True:
            if video_stream.stopped or video_display.stopped:
                video_stream.stop()
                video_display.stop()
                break

            frame = video_stream.frame
            if frame is not None:
                # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # # Try to increase the contrast of the image requires more work 
                # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                # gray = clahe.apply(gray)
                gray=frame
                
                aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
                parameters = cv2.aruco.DetectorParameters()

                marker_array = find_marker(frame, aruco_dict, parameters)
                display_spec(gray, marker_array)
                

                if marker_array:
                    estimate_marker_pose_multiple(frame,marker_array,camera_matrix=video_stream.camera_matrix, distortion_coeff=video_stream.camera_dist)

                    if len(marker_array)==4:
                        draw_bounded_area(gray,marker_array)
                        
                    # Draw stuff on the marker for detection 
                    for marker, id in marker_array:
                        coord = get_corner_and_center(marker)
                        draw_id(frame, coord, id)
                        draw_rounder_corner(frame, coord)
                        draw_square_frame(frame,coordinates=coord)
                        
                        

                frame_with_fps = putIterationsPerSec(gray, fps_counter.fps())
                video_display.frame = frame_with_fps

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            fps_counter.update()

    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
        video_stream.stop()
        video_display.stop()
        exit()

if __name__ == '__main__':
    main()
