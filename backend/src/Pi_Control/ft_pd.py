import io
import os
import pickle
import signal
import logging
import random
import threading
from time import sleep, time
import datetime

import cv2
import numpy as np
from flask import Flask, Response
import cvzone
from cvzone.FaceDetectionModule import FaceDetector
from picamera2 import Picamera2
from gpiozero import Device, AngularServo, Servo
from gpiozero.pins.pigpio import PiGPIOFactory

# Custom modules (adjust relative import paths as needed)
from ..Camera.pi_camera_streamer import CalibratedCamera, PiVideo
from ..ArUcoMarker.utils import find_marker, track_and_render_marker, draw_center_frame
from ..PID import PIDController

# =======================
# Global Configuration
# =======================

# Set up run paths for face tracking (separate from marker tracking)
base_dir = os.path.dirname(__file__)
face_run_path = os.path.join(base_dir, "face_runs")
os.makedirs(face_run_path, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S").replace(":", "-")
current_run = os.path.join(face_run_path, f"run_{timestamp}")
os.makedirs(current_run, exist_ok=True)

# Configure logging (log file stored in the face_runs directory)
log_file = os.path.join(face_run_path, f"Face_Motion_{timestamp}.log")
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.info("Face tracking logging system initialized.")

# =======================
# Hardware and Detector Setup
# =======================

# Initialize the face detector with minimum detection confidence.
face_detector = FaceDetector(minDetectionCon=0.5, modelSelection=0)
# Servo angle limits and GPIO pin assignments
PAN_MIN, PAN_MAX = -135, 135
TILT_MIN, TILT_MAX = -90, 90
PAN_SERVO_PIN, TILT_SERVO_PIN = 17, 27


# Configure GPIO and initialize pan servo for face tracking.
# PID Controller parameters
kp_pan, ki_pan, kd_pan = 0.2, 0.0, 0.2
kp_tilt, ki_tilt, kd_tilt = 0.1, 0.0, 0.2

# ======================
# Hardware Initialization
# ======================

# Set GPIO pin factory for consistent pin handling
Device.pin_factory = PiGPIOFactory()

# Initialize Pan and Tilt servos with the appropriate pulse widths
pan_servo = AngularServo(
    PAN_SERVO_PIN, min_angle=PAN_MIN, max_angle=PAN_MAX,
    min_pulse_width=0.5 / 1000, max_pulse_width=2.5 / 1000
)
tilt_servo = AngularServo(
    TILT_SERVO_PIN, min_angle=TILT_MIN, max_angle=TILT_MAX,
    min_pulse_width=0.5 / 1000, max_pulse_width=2.5 / 1000
)

pan_servo.angle = 0
tilt_servo.angle = 0



# Initialize PID controllers for pan and tilt
pan_controller = PIDController(Kp=kp_pan, Ki=ki_pan, Kd=kd_pan)
tilt_controller = PIDController(Kp=kp_tilt, Ki=ki_tilt, Kd=kd_tilt)

# Store PID parameters in a configuration dictionary
config = {
    "kp_pan": kp_pan, "ki_pan": ki_pan, "kd_pan": kd_pan,
    "kp_tilt": kp_tilt, "ki_tilt": ki_tilt, "kd_tilt": kd_tilt
}


# =======================
# Flask Application Setup
# =======================

def create_app(stream_instance):
    """
    Create and configure the Flask application for video streaming.
    
    Args:
        stream_instance (PiVideo): The video stream instance.
        
    Returns:
        Flask: Configured Flask application.
    """
    app = Flask(__name__)
    start_time = time()

    # Initialize an empty data array to store run data.
    # Columns: Time, Pan_Error, Pan_Angle, Face_Confidence
    data_array = np.empty((0, 4))

    def generate_frames():
        """Generator that yields JPEG frames as multipart responses."""
        with stream_instance.clients_lock:
            stream_instance.clients += 1
            logging.info(f"Client connected. Total clients: {stream_instance.clients}")

        try:
            while True:
                frame_data = None
                with stream_instance.lock:
                    if stream_instance.frame_buffer is not None:
                        frame_data = stream_instance.frame_buffer

                elapsed_time = time() - start_time

                if frame_data is not None:
                    width = stream_instance.width
                    height = stream_instance.height
                    frame_center = (width // 2, height // 2)

                    # Decode the image from the frame buffer.
                    nparr = np.frombuffer(frame_data, dtype=np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    # Draw a center marker.
                    draw_center_frame(frame, frame_center)

                    # Detect faces in the frame.
                    frame, face_bboxes = face_detector.findFaces(frame, draw=False)
                    face_confidence = 0  # Default confidence in case no face is found.
                    if face_bboxes:
                        for bbox in face_bboxes:
                            # Extract bounding box details.
                            x, y, w, h = bbox['bbox']
                            center = bbox["center"]
                            face_confidence = int(bbox['score'][0] * 100)

                            # Draw face annotations.
                            cv2.circle(frame, center, 5, (255, 0, 255), cv2.FILLED)
                            cvzone.putTextRect(frame, f'{face_confidence}%', (x, y - 10))
                            cvzone.cornerRect(frame, (x, y, w, h))
                            position_text = f"x: {x}, y: {y}"
                            cv2.putText(
                                frame, position_text, (x, y + h + 30),
                                cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2
                            )
                            
                            # For face tracking, we use the center of the face.
                            x_f = center[0]
                            x_r = frame_center[0]
                            pan_error = (x_f - x_r)/x_r*100
                            
                            y_f = center[1]
                            y_r = frame_center[1]
                            
                            tilt_error = (y_f - y_r)/y_r*100
                            
                            
                            logging.info(f"Elapsed Time: {elapsed_time:.2f}s | Pan Error: {pan_error:.2f} | Face Confidence: {face_confidence}%")
                            
                            current_pan = pan_servo.angle if pan_servo.angle is not None else 0
                            pan_correction = pan_controller.compute(pan_error, elapsed_time)
                            new_pan = np.clip(current_pan - pan_correction, PAN_MIN, PAN_MAX)
                            pan_servo.angle = new_pan
                            
                            # Compute tilt error and apply PID correction
                            current_tilt = tilt_servo.angle if tilt_servo.angle is not None else 0
                            tilt_correction = tilt_controller.compute(tilt_error, elapsed_time)
                            new_tilt = np.clip(current_tilt + tilt_correction, TILT_MIN, TILT_MAX)
                            tilt_servo.angle = new_tilt
                            
                            # Log and store data (only one face is used for control).
                            data_entry = np.array([[elapsed_time, pan_error, new_pan, face_confidence]])
                            data_array[:] = np.vstack((data_array, data_entry))
                            break  # Process only the first detected face.
                            
                    # Re-encode the processed frame to JPEG bytes.
                    ret, jpeg = cv2.imencode('.jpg', frame)
                    if ret:
                        frame_data = jpeg.tobytes()
                    else:
                        logging.error("Failed to encode frame to JPEG")
                        continue

                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n\r\n')
                sleep(1 / stream_instance.framerate)
        except KeyboardInterrupt:
            logging.info("Ctrl-C detected. Exiting gracefully...")
        finally:
            with stream_instance.clients_lock:
                stream_instance.clients -= 1
                logging.info(f"Client disconnected. Remaining clients: {stream_instance.clients}")
            # Save the run data to a CSV file.
            fname = os.path.join(current_run, 'data_face.csv')
            np.savetxt(fname, data_array, delimiter=",",
                       header="Time,Pan_Error,Pan_Angle,Face_Confidence", comments="", fmt="%.5f")
            logging.info(f"Saved data at: {fname}")
            # Optionally, call a plotting function here if desired.
            # plot_data(fname, config, face_run_path)

    @app.route('/video_feed')
    def video_feed():
        """HTTP route for the video stream."""
        return Response(
            generate_frames(),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )

    @app.route('/')
    def index():
        """HTTP route for the main page displaying the video stream."""
        return """
        <html>
            <head>
                <title>Face Tracking Stream</title>
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <style>
                    body { margin: 0; padding: 0; background: #000; }
                    .container { display: flex; justify-content: center; align-items: center; min-height: 100vh; }
                    img { max-width: 100%; height: auto; }
                </style>
            </head>
            <body>
                <h1>Face Tracking</h1>
                <div class="container">
                    <img src="/video_feed" alt="Camera Stream" />
                </div>
            </body>
        </html>
        """

    return app

# =======================
# Main Execution Block
# =======================

if __name__ == "__main__":
    # Paths for the Pi camera calibration data.
    CAMERA_MATRIX_PATH = 'src/Camera/calibration/pical1280720new/cameraMatrix.pkl'
    CAMERA_DISTORTION_PATH = "src/Camera/calibration/pical1280720new/dist.pkl"

    # Initialize the calibrated camera.
    calibrated_camera = CalibratedCamera(
        cam_mat_path=CAMERA_MATRIX_PATH,
        cam_dist_path=CAMERA_DISTORTION_PATH,
        frame_width=1280,
        frame_height=720
    )
    # Start the video stream.
    stream = PiVideo(
        calibrated_camera=calibrated_camera,
        framerate=30,
        brightness=0.0,
        saturation=1,
        contrast=1
    ).start()
    # Create and run the Flask application.
    app = create_app(stream)
    app.run(host='0.0.0.0', port=5000)
