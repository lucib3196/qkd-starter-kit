import cv2
from gpiozero import AngularServo
import numpy as np
import time
from . import WebcamVideoStreamThreaded, FPS, putIterationsPerSec, VideoShow

from gpiozero.pins.pigpio import PiGPIOFactory
from gpiozero import Device
Device.pin_factory = PiGPIOFactory()

BASE_MIN = 0
BASE_MAX = 270
# Initialize servos
base_servo = AngularServo(17, min_angle=BASE_MIN, max_angle=BASE_MAX)
upper_servo = AngularServo(27, min_angle=-90, max_angle=90)

def detect_aruco_marker(image_frame, camera_matrix, distortion_coeff, marker_length=0.038, aruco_dict_type=cv2.aruco.DICT_6X6_250, last_center=(0, 0)):
    # Convert the image to grayscale for marker detection
    gray = cv2.cvtColor(image_frame, cv2.COLOR_BGR2GRAY)

    # Load the predefined ArUco dictionary and detection parameters
    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    # Detect markers in the grayscale frame
    corners, ids, rejected = detector.detectMarkers(gray)
    print("Detected Marks:", ids)

    center = last_center  # Default to the last known center when no markers are detected

    if ids is not None and len(ids) > 0:
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

            cv2.circle(image_frame, center, 5, (255, 0, 255), cv2.FILLED)  # Center dot

            # Estimate the marker's pose
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(marker_corner, marker_length, camera_matrix, distortion_coeff)

            # Calculate and display the marker's distance
            distance = np.linalg.norm(tvec) * 100  # Convert to cm
            cv2.drawFrameAxes(image_frame, camera_matrix, distortion_coeff, rvec, tvec, marker_length)
            cv2.putText(image_frame, f"Distance: {distance:.2f} cm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)
    else:
        print("[INFO] No markers detected.")
        # Maintain the last center to prevent jittering

    return image_frame, center

def main(src=0):
    # Define the camera
    # Initialize the video stream and display threads
    video_stream = WebcamVideoStreamThreaded(src).start()
    video_display = VideoShow(video_stream.frame).start()
    fps_counter = FPS().start()

    # Keep track of the last center position
    last_center = (0, 0)

    while True:
        # Stop threads if either thread signals to stop
        if video_stream.stopped or video_display.stopped:
            video_stream.stop()
            video_display.stop()
            break
        
        frame = video_stream.frame
        if frame is not None:
            frame_with_markers, center = detect_aruco_marker(
                image_frame=frame,
                camera_matrix=video_stream.camera_matrix,
                distortion_coeff=video_stream.camera_dist,
                last_center=last_center
            )

            # Update last center to the current center
            last_center = center

            # Overlay FPS information on the frame
            frame_with_fps = putIterationsPerSec(frame_with_markers, fps_counter.fps())

            # Map the center coordinates to servo angles
            servo_x = np.interp(center[0], [0, video_stream.frame_width], [BASE_MIN, BASE_MAX])
            servo_y = np.interp(center[1], [0, video_stream.frame_height], [-90, 90])

            # Update servo positions
            base_servo.angle = servo_x
            upper_servo.angle = servo_y

            # Display the servo angles on the frame
            cv2.putText(frame_with_fps, f'Servo X: {servo_x:.2f} deg', (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
            cv2.putText(frame_with_fps, f'Servo Y: {servo_y:.2f} deg', (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

            video_display.frame = frame_with_fps

        # Update FPS counter
        fps_counter.update()

if __name__ == '__main__':
    main()
