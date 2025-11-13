import cv2
import cv2.aruco as aruco
import numpy as np
import pickle
import os
from threading import Thread
from . import WebcamVideoStreamThreaded, FPS, putIterationsPerSec,VideoShow



def detect_aruco_marker(image_frame, camera_matrix, distortion_coeff, aruco_dict_type=cv2.aruco.DICT_ARUCO_ORIGINAL):
    """
    Detects ArUco markers and overlays their corner coordinates and 3D pose axes.

    Parameters:
    - image_frame: The frame in which to detect markers.
    - camera_matrix: Camera calibration matrix.
    - distortion_coeff: Distortion coefficients from camera calibration.
    - aruco_dict_type: Type of ArUco dictionary to use.

    Returns:
    - Annotated frame with detected markers and 3D pose axes.
    """
    # Convert the image to grayscale for marker detection
    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)
    
    # Load the predefined ArUco dictionary and detection parameters
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters()
    
    
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    # Detect markers in the grayscale frame
    corners, ids, rejected = detector.detectMarkers(gray)
    print("Detected Marks:", ids)

    if ids is not None:
        # Process each detected marker
        for marker_corner, marker_id in zip(corners, ids):
            print(f"[INFO] Marker ID: {marker_id}")

            # Extract and reshape marker corners
            corners_abcd = marker_corner.reshape((4, 2))
            (top_left, top_right, bottom_right, bottom_left) = corners_abcd

            # Convert corner coordinates to integers for drawing
            top_left = (int(top_left[0]), int(top_left[1]))
            top_right = (int(top_right[0]), int(top_right[1]))
            bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
            bottom_left = (int(bottom_left[0]), int(bottom_left[1]))
            
            # Calculate the center of the marker
            center_x = (top_left[0] + top_right[0] + bottom_right[0] + bottom_left[0]) // 4
            center_y = (top_left[1] + top_right[1] + bottom_right[1] + bottom_left[1]) // 4
            center = (center_x, center_y)


            # Draw lines connecting the marker corners
            cv2.line(image_frame, top_left, top_right, (0, 255, 0), 2)
            cv2.line(image_frame, top_right, bottom_right, (0, 255, 0), 2)
            cv2.line(image_frame, bottom_right, bottom_left, (0, 255, 0), 2)
            cv2.line(image_frame, bottom_left, top_left, (0, 255, 0), 2)
            
            cv2.circle(image_frame, center, 5, (255, 0, 255), cv2.FILLED) # Center dot 

            # Estimate the marker's pose
            marker_length = 0.038  # Marker size in meters
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(marker_corner, marker_length, camera_matrix, distortion_coeff)

            # Calculate and display the marker's distance
            distance = np.linalg.norm(tvec) * 100  # Convert to cm
            cv2.drawFrameAxes(image_frame, camera_matrix, distortion_coeff, rvec, tvec, marker_length)
            cv2.putText(image_frame, f"Distance: {distance:.2f} cm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

    return image_frame

def thread_video_stream(source=0):
    """
    Runs video capturing and displaying in separate threads.

    Parameters:
    - source: Video source index or file path.
    """
    # Initialize the video stream and display threads
    video_stream = WebcamVideoStreamThreaded(source).start()
    video_display = VideoShow(video_stream.frame).start()
    fps_counter = FPS().start()

    while True:
        # Stop threads if either thread signals to stop
        if video_stream.stopped or video_display.stopped:
            video_stream.stop()
            video_display.stop()
            break

        # Process the current frame
        frame = video_stream.frame
        if frame is not None:
            frame_with_markers = detect_aruco_marker(
                image_frame=frame,
                camera_matrix=video_stream.camera_matrix,
                distortion_coeff=video_stream.camera_dist
            )
            # Overlay FPS information on the frame
            frame_with_fps = putIterationsPerSec(frame_with_markers, fps_counter.fps())
            video_display.frame = frame_with_fps

        # Update FPS counter
        fps_counter.update()

if __name__ == "__main__":
    """
    Main function to start the threaded video stream with ArUco marker detection.
    """
    thread_video_stream(0)
