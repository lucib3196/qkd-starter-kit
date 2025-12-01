import cv2
import cvzone
from cvzone.FaceDetectionModule import FaceDetector
from . import WebcamVideoStreamThreaded, FPS, putIterationsPerSec, VideoShow
from gpiozero.pins.pigpio import PiGPIOFactory
from gpiozero import Device, AngularServo
import numpy as np
import time

# Set up PiGPIOFactory for GPIO control
Device.pin_factory = PiGPIOFactory()

# Initialize servos
base_servo = AngularServo(17, min_angle=-135, max_angle=135)
upper_servo = AngularServo(27, min_angle=-90, max_angle=90)

# Set initial servo angles
servo_x = 0
servo_y = 0
base_servo.angle = servo_x
# upper_servo.angle = servo_y

def main(src=0):
    """
    Main function for face detection and servo control.
    Optimized for reduced CPU usage.
    
    Parameters:
    - src (int or str): Video source (default: 0 for webcam).
    """
    # Initialize the video stream and display threads
    video_stream = WebcamVideoStreamThreaded(src).start()
    video_display = VideoShow(video_stream.frame).start()
    fps_counter = FPS().start()

    # Initialize the FaceDetector object
    face_detector = FaceDetector(minDetectionCon=0.5, modelSelection=0)

    # Keep track of the last center position and timing for servo updates
    last_center = (0, 0)
    last_update_time = time.time()
    update_interval = 0.5  # Update servos every 0.5 seconds

    frame_skip = 5  # Process every 2nd frame
    frame_count = 0

    servo_x = 0  # Ensure servo_x is defined within the function
    servo_y = 0  # Ensure servo_y is defined within the function

    while True:
        frame_count += 1

        # Stop threads if either thread signals to stop
        if video_stream.stopped or video_display.stopped:
            video_stream.stop()
            video_display.stop()
            break

        frame = video_stream.frame

        # Skip frames to reduce processing load
        if frame_count % frame_skip != 0:
            continue

        # Detect faces in the frame
        frame, face_bboxes = face_detector.findFaces(frame, draw=False)

        if face_bboxes:
            for bbox in face_bboxes:
                # Extract bounding box details
                x, y, w, h = bbox['bbox']
                center = bbox["center"]
                score = int(bbox['score'][0] * 100)

                # Draw face annotations
                cv2.circle(frame, center, 5, (255, 0, 255), cv2.FILLED)  # Mark the center of the face
                cvzone.putTextRect(frame, f'{score}%', (x, y - 10))  # Display confidence score
                cvzone.cornerRect(frame, (x, y, w, h))  # Draw bounding box with corners

                # Servo and Position Mapping (Update only when necessary)
                if time.time() - last_update_time >= update_interval:
                    servo_x = np.interp(center[0], [0, video_stream.frame_width], [-135, 135])
                    servo_y = np.interp(center[1], [0, video_stream.frame_height], [90, -90])
                    base_servo.angle = servo_x
                    # upper_servo.angle = servo_y
                    last_update_time = time.time()

                # Display x and y position below the bounding box
                position_text = f"x: {center[0]}, y: {center[1]}"
                cv2.putText(
                    frame,
                    position_text,
                    (x, y + h + 30),  # Position below the bounding box
                    cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (0, 255, 0),
                    2
                )

        # Display servo positions
        cv2.putText(frame, f'Servo X: {servo_x:.1f} deg', (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        cv2.putText(frame, f'Servo Y: {servo_y:.1f} deg', (50, 80), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        # Update the display thread
        video_display.frame = frame

    # Clean up resources
    cv2.destroyAllWindows()

if __name__ == "__main__":
    """
    Main entry point for the script.
    Initializes the video stream and starts face detection and servo control.
    """
    main(0)
