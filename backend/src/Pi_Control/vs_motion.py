
# Logging Configuration
# This sets the logging level, determining the lowest level of messages that will be logged.
# 
# Levels (from lowest to highest severity):
# - DEBUG: Detailed information, typically for diagnosing problems.
# - INFO: General information about program execution.
# - WARNING: Indicates something unexpected or potential issues.
# - ERROR: A serious issue that prevents a part of the program from functioning.
# - CRITICAL: A very severe error indicating the program may not continue running.
#
# If set to DEBUG, all messages (DEBUG, INFO, WARNING, ERROR, CRITICAL) will be shown.
# If set to INFO, only INFO and more severe messages (INFO, WARNING, ERROR, CRITICAL) will be shown.
# If set to WARNING, only WARNING and more severe messages (WARNING, ERROR, CRITICAL) will be shown, and so on.
import logging
import os
import datetime
import random
import threading
import signal
import io
import time
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import sleep, time
from flask import Flask, Response
from gpiozero import Device, AngularServo, Servo
from gpiozero.pins.pigpio import PiGPIOFactory
from picamera2 import Picamera2
from ..Camera.fps import FPS, putIterationsPerSec
# Import custom modules
from ..ArUcoMarker.utils import find_marker, track_and_render_marker, draw_center_frame
from ..PID import PIDController
from ..Camera.pi_camera_streamer import CalibratedCamera, PiVideo
import time



# ========================
# Constants and Parameters
# ========================

# Servo angle limits and GPIO pin assignments
PAN_MIN, PAN_MAX = -135, 135
TILT_MIN, TILT_MAX = -90, 90
PAN_SERVO_PIN, TILT_SERVO_PIN = 17, 27

# ArUco marker parameters
ARUCO_DICT_TYPE = cv2.aruco.DICT_5X5_1000
MARKER_LENGTH = 0.033 # meters
MARKER_ID = 5     # ID of the marker to track
TRACK_POINT = False # Whether to track an offset point
marker_point = np.array([0.1, 0, 0, 1]).reshape(4, 1)  # Offset of the point to track in meters

# PID Controller parameters
kp_pan, ki_pan, kd_pan = 0.5, 0.0, 0.5
kp_tilt, ki_tilt, kd_tilt = 0.5, 0.0, 0.5

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

# ========================
# Directory and Logging Setup
# ========================

# Determine base directory and create a "runs" folder if it doesn't exist
base_dir = os.path.dirname(__file__)
run_path = os.path.join(base_dir, "runs")
os.makedirs(run_path, exist_ok=True)

# Generate a timestamp for the current run (ensuring a valid folder name)
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S").replace(":", "-")
current_run = os.path.join(run_path, f"run_{timestamp}")
os.makedirs(current_run,exist_ok=True)

# Set up log file path and configure logging
log_file = os.path.join(current_run, f"Vs_Motion_{timestamp}.log")
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.info("Logging system initialized.")

# Log the running parameters for reference
msg_n = 25
message_str = f"""
Pan and Tilt Mechanism: 
Current Date: {timestamp}

{'-' * msg_n} Running Parameters {'-' * msg_n}
Pan Servo: Pin {PAN_SERVO_PIN}, PID Values: {pan_controller.get_specs()}, Angle Range {PAN_MIN}-{PAN_MAX}, Current Angle {tilt_servo.angle}
Tilt Servo: Pin {TILT_SERVO_PIN}, PID Values: {tilt_controller.get_specs()}, Angle Range {TILT_MIN}-{TILT_MAX}, Current Angle {pan_servo.angle}

Tracking Point: {TRACK_POINT}
{'-' * (msg_n * 2)}
"""
logging.info(message_str)


# ========================
# Data Plotting Function
# ========================

def plot_data(fname, config, path_to_save=current_run):
    """
    Plots the pan and tilt error over time and saves the figures.
    
    Parameters:
      - fname (str): Path to the CSV file containing data.
      - config (dict): Configuration dictionary containing PID parameters.
      - path_to_save (str): Directory path to save the plots.
    """
    # Load the CSV data file
    df = pd.read_csv(fname)

    # ---------------------
    # Plot Pan Error
    # ---------------------
    plt.figure(figsize=(15, 8))
    plt.plot(df["Time"], df["Pan_Error"], linestyle="-", label="Pan Error")
    plt.xlabel("Time (s)")
    plt.ylabel("Error (deg)")
    plt.title(f"Tracking Error Over Time - Pan Error\n"
              f"PID K_p: {config.get('kp_pan', '')}, "
              f"K_i: {config.get('ki_pan', '')}, "
              f"K_d: {config.get('kd_pan', '')}")
    plt.legend()
    plt.tight_layout()

    # Ensure the save directory exists and save the pan error plot
    os.makedirs(path_to_save, exist_ok=True)
    pan_path = os.path.join(path_to_save, "pan_response.png")
    plt.savefig(pan_path)
    plt.close()
    logging.info(f"Pan plot saved at: {pan_path}")

    # ---------------------
    # Plot Tilt Error
    # ---------------------
    plt.figure(figsize=(15, 8))
    plt.plot(df["Time"], df["Tilt_Error"], linestyle="-", label="Tilt Error")
    plt.xlabel("Time (s)")
    plt.ylabel("Error (deg)")
    plt.title(f"Tracking Error Over Time - Tilt Error\n"
              f"PID K_p: {config.get('kp_tilt', '')}, "
              f"K_i: {config.get('ki_tilt', '')}, "
              f"K_d: {config.get('kd_tilt', '')}")
    plt.legend()
    plt.tight_layout()

    tilt_path = os.path.join(path_to_save, "tilt_response.png")
    plt.savefig(tilt_path)
    plt.close()
    logging.info(f"Tilt plot saved at: {tilt_path}")

# ========================
# Main Application Function
# ========================

def main(stream_instance):
    """
    Main function to start the Flask app and process the video stream.
    
    Parameters:
      - stream_instance: Instance of the video stream.
    
    Returns:
      - Flask app instance.
    """
    print("Starting Full Pan–Tilt Tracking")
    app = Flask(__name__)

    def motion_track():
        """
        Generator function that processes the video stream, performs marker detection, 
        applies PID control to adjust servo angles, and streams the output.
        """
        start_time = time.time()

        # Start the video stream and log the client connection
        stream_instace = stream_instance.start()
        with stream_instance.clients_lock:
            stream_instance.clients += 1
            logging.info(f"Client connected. Total clients: {stream_instance.clients}")

        # Initialize an empty data array to store run data
        data_array = np.empty((0, 5))

        try:
            while True:
                frame_data = None

                # Safely retrieve the latest frame from the stream
                with stream_instance.lock:
                    if stream_instance.frame_buffer is not None:
                        frame_data = stream_instance.frame_buffer

                if frame_data is not None:
                    # Convert byte data to a NumPy array and decode the image
                    nparr = np.frombuffer(frame_data, dtype=np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    # Undistort the frame using the calibrated camera settings
                    frame = stream_instace.calibrated_camera.undistort_frame(frame)

                    if frame is not None:
                        elapsed_time = time.time() - start_time
                        width = stream_instance.width
                        height = stream_instance.height
                        center = (width // 2, height // 2)

                        # Draw a marker at the center of the frame
                        draw_center_frame(frame, center)

                        # Marker detection using the predefined ArUco dictionary
                        aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
                        parameters = cv2.aruco.DetectorParameters()
                        marker_array = find_marker(frame, aruco_dict, parameters)

                        if marker_array:
                            for marker, marker_id in marker_array:
                                # Track and render the marker, obtaining the transformation matrix
                                transformation_matrix = track_and_render_marker(
                                    frame,
                                    marker,
                                    marker_id,
                                    camera_matrix=stream_instace.camera_matrix,
                                    distortion_coefficient=stream_instace.camera_dist,
                                    marker_length=MARKER_LENGTH
                                )

                                # Optionally compute an offset point transformation if tracking a point
                                if TRACK_POINT:
                                    T0point = transformation_matrix @ marker_point
                                    tvec = T0point[:-1, -1]
                                else:
                                    tvec = transformation_matrix[:-1, -1]

                                x, y, z = tvec

                                # Compute pan error and apply PID correction
                                pan_error = np.degrees(np.arctan2(x, z))
                                current_pan = pan_servo.angle if pan_servo.angle is not None else 0
                                pan_correction = pan_controller.compute(pan_error, elapsed_time)
                                new_pan = np.clip(current_pan - pan_correction, PAN_MIN, PAN_MAX)
                                pan_servo.angle = new_pan

                                # Compute tilt error and apply PID correction
                                tilt_error = np.degrees(np.arctan2(y, z))
                                current_tilt = tilt_servo.angle if tilt_servo.angle is not None else 0
                                tilt_correction = tilt_controller.compute(tilt_error, elapsed_time)
                                new_tilt = np.clip(current_tilt + tilt_correction, TILT_MIN, TILT_MAX)
                                tilt_servo.angle = new_tilt

                                logging.info(f"Time: {time.time():.2f}s | Pan Error: {pan_error:.2f}, Tilt Error: {tilt_error:.2f}")
                                logging.info(f"Pan Angle: {pan_servo.angle}, Tilt Angle: {tilt_servo.angle}")

                                # Log the data: elapsed time, pan error, new pan angle, tilt error, new tilt angle
                                data_entry = np.array([[elapsed_time, pan_error, new_pan, tilt_error, new_tilt]])
                                data_array = np.vstack((data_array, data_entry))

                    # Re-encode the processed frame to JPEG bytes
                    ret, jpeg = cv2.imencode('.jpg', frame)
                    if ret:
                        frame_data = jpeg.tobytes()
                    else:
                        logging.error("Failed to encode frame to JPEG")
                        continue

                    # Yield the frame data formatted for streaming
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n\r\n')

                # Maintain the desired frame rate
                sleep(1 / stream_instance.framerate)
        except KeyboardInterrupt:
            logging.info("Ctrl-C detected. Exiting gracefully...")
        finally:
            # On exit, decrement the client count and save run data
            with stream_instance.clients_lock:
                stream_instance.clients -= 1
                logging.info(f"Client disconnected. Remaining clients: {stream_instance.clients}")

            fname = os.path.join(current_run, 'data_pan.csv')
            np.savetxt(fname, data_array, delimiter=",",
                    header="Time,Pan_Error,Pan_Angle,Tilt_Error,Tilt_Angle", comments="", fmt="%.5f")
            logging.info(f"Saved data at: {fname}")
            plot_data(fname, config, current_run)

    # ---------------------
    # Flask Routes
    # ---------------------

    @app.route('/video_feed')
    def video_feed():
        """Route to access the video stream."""
        return Response(
            motion_track(),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )

    @app.route('/')
    def index():
        """Route for the main page."""
        return """
        <html>
            <head>
                <title>Pi Camera Stream</title>
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <style>
                    body { margin: 0; padding: 0; background: #000; }
                    .container { 
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        min-height: 100vh;
                    }
                    img { max-width: 100%; height: auto; }
                </style>
            </head>
            <body>
                <h1>Motion Tracking aRUco</h1>
                <div class="container">
                    <img src="/video_feed" alt="Camera Stream" />
                </div>
            </body>
        </html>
        """

    return app

# ========================
# Main Execution Block
# ========================

if __name__ == "__main__":
    # Paths for the camera calibration data
    CAMERA_MATRIX_PATH = 'src/Camera/calibration/pical1280720new/cameraMatrix.pkl'
    CAMERA_DISTORTION_PATH = "src/Camera/calibration/pical1280720new/dist.pkl"

    # Initialize the calibrated camera
    calibrated_camera = CalibratedCamera(
        cam_mat_path=CAMERA_MATRIX_PATH,
        cam_dist_path=CAMERA_DISTORTION_PATH,
        frame_width=1280,
        frame_height=720
    )

    # Start the video stream with specified camera settings
    stream = PiVideo(
        calibrated_camera=calibrated_camera,
        framerate=60,
        brightness=0.0,
        contrast=1.0
    ).start()

    # Create and run the Flask application
    app = main(stream)
    app.run(host='0.0.0.0', port=5000)
