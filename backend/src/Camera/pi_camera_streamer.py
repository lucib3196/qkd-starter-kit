"""
camera_streamer.py

This module handles video streaming using a calibrated PiCamera with Picamera2 and Flask.
It loads calibration parameters for the camera, performs image undistortion, captures frames,
and serves them via an HTTP endpoint. This is a simple test to verify that the camera is
working and that streaming functions correctly; it does not integrate the full pi_camera_streamer functionality.

Dependencies:
    - OpenCV (cv2)
    - NumPy
    - Flask
    - Picamera2
    - Standard libraries: io, os, pickle, signal, threading, time, logging
    - Utility functions from .utils: define_camera_settings, load_camera_calibration
"""

import io
import os
import pickle
import signal
from time import sleep, time
import logging
import threading
from threading import Thread

import cv2
import numpy as np
from flask import Flask, Response
from picamera2 import Picamera2
from dataclasses import dataclass

from .utils import define_camera_settings, load_camera_calibration

# Get the base directory for the camera folder
base_path = os.path.dirname(os.path.abspath(__file__))


@dataclass
class CalibratedCamera:
    cam_mat_path: str
    cam_dist_path: str
    frame_height: int
    frame_width: int

    def __post_init__(self):
        """Load calibration parameters and calibrate the camera."""
        self.load_calibration_parameters()
        self.calibrate_camera()

    def load_calibration_parameters(self):
        """
        Load the camera matrix and distortion coefficients from disk.
        The file paths are constructed relative to the current working directory.
        """
        camera_matrix_path = os.path.join(os.getcwd(), self.cam_mat_path)
        camera_distortion_path = os.path.join(os.getcwd(), self.cam_dist_path)

        try:
            with open(camera_matrix_path, "rb") as file:
                self.camera_matrix = pickle.load(file)
            with open(camera_distortion_path, "rb") as file:
                self.camera_dist = pickle.load(file)
        except Exception as e:
            # print(f"Error Loading Camera Calibration: {e}")
            logging.error(f"Error Loading Camera Calibration: {e}")

    def calibrate_camera(self):
        """
        Compute an optimal new camera matrix and region of interest for undistortion.
        Uses the stored camera matrix and distortion coefficients.
        """
        self.new_cam_mtx, self.roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.camera_dist,
            (self.frame_width, self.frame_height), 1,
            (self.frame_width, self.frame_height)
        )

    def undistort_frame(self, frame):
        """
        Undistort the provided frame using the calibrated camera parameters.

        Args:
            frame (numpy.ndarray): The distorted input image.

        Returns:
            numpy.ndarray: The undistorted and cropped image.
        """
        undistorted_frame = cv2.undistort(
            frame, self.camera_matrix, self.camera_dist, None, self.new_cam_mtx
        )
        x, y, w, h = self.roi
        return undistorted_frame[y: y + h, x: x + w]


class PiVideo:
    def __init__(self, calibrated_camera: CalibratedCamera, framerate=60, format="XBGR8888",
                 brightness=0.0, contrast=0.8, saturation=1.3):
        """
        Initialize the PiVideo stream with a calibrated camera and camera settings.

        Args:
            calibrated_camera (CalibratedCamera): Instance containing calibration data.
            framerate (int): Desired frame rate.
            format (str): Image format.
            brightness (float): Camera brightness.
            contrast (float): Camera contrast.
            saturation (float): Camera saturation.
        """
        self.width = calibrated_camera.frame_width
        self.height = calibrated_camera.frame_height
        self.framerate = framerate
        self.calibrated_camera = calibrated_camera
        self.camera_matrix = self.calibrated_camera.new_cam_mtx
        self.camera_dist = self.calibrated_camera.camera_dist

        # Threading lock and event for safe frame access and stopping the stream
        self.lock = threading.Lock()
        self.stop_event = threading.Event()

        # Frame counter and client tracking
        self.frame_count = 0
        self.clients = 0
        self.clients_lock = threading.Lock()

        # Initialize the PiCamera2 and configure it
        self.picam2 = Picamera2(0)
        config = self.picam2.create_video_configuration(
            main={"size": (self.width, self.height), "format": "YUV420"},  # Valid format
            controls={
                "Brightness": float(brightness),  
                "Contrast": float(contrast),  
                "Saturation": float(saturation),  
                "AnalogueGain": 4.0,  # Ensure it's a float
                "AeEnable": False,  # Ensure it's a bool
                "ExposureTime": int(1000000 / framerate),  # Ensure it's an integer
            },
            buffer_count=4
        )

        self.picam2.configure(config)
        # Initialize the frame buffer
        self.buffer = io.BytesIO()
        self.frame_buffer = None

    def start(self):
        """Start capturing video frames in a separate thread."""
        self.picam2.start()
        self.capture_thread = threading.Thread(
            target=self._capture_frames,
            daemon=True,
            name='CaptureThread'
        )
        self.capture_thread.start()
        return self

    def stop(self):
        """Stop the video streaming."""
        self.stop_event.set()
        if hasattr(self, 'picam2'):
            self.picam2.stop()

    def _capture_frames(self):
        """Continuously capture frames from the camera, undistort, encode, and store them."""
        frame_interval = 1 / self.framerate
        retries = 0
        max_retries = 3

        while not self.stop_event.is_set():
            try:
                start_time = time()

                # Capture if there are clients or if no frame is in the buffer
                if self.clients > 0 or self.frame_buffer is None:
                    self.buffer.seek(0)
                    self.buffer.truncate()

                    (buffers, metadata) = self.picam2.capture_buffers(["main"])
                    buffer = buffers[0]
                    image_array = self.picam2.helpers.make_array(
                        buffer, self.picam2.camera_configuration()["main"]
                    )
                    
                    
                    # Undistort the captured frame before JPEG encoding
                    ret, jpeg = cv2.imencode('.jpg',image_array)

                    with self.lock:
                        if ret:
                            # print(jpeg)
                            self.frame_buffer = jpeg.tobytes()

                    if self.frame_count % 300 == 0:
                        logging.info(f"Stream stats - Frame: {self.frame_count}, "
                                     f"Size: {len(self.frame_buffer) if self.frame_buffer else 0} bytes, "
                                     f"Clients: {self.clients}")
                    self.frame_count += 1
                    retries = 0  # Reset retries on success

                # Maintain frame rate
                elapsed = time() - start_time
                sleep_time = max(0, frame_interval - elapsed)
                if sleep_time > 0:
                    sleep(sleep_time)
            except RuntimeError as e:
                logging.error(f"Runtime error during capture: {e}")
                retries += 1
                if retries >= max_retries:
                    logging.error("Max retries exceeded. Restarting camera...")
                    self.picam2.stop()
                    sleep(1)  # Wait before restarting
                    self.picam2.start()
                    retries = 0
            except Exception as e:
                logging.error(f"Unexpected error during capture: {e}")
                sleep(0.1)


def create_app(stream_instance):
    """
    Create and configure the Flask application for video streaming.

    Args:
        stream_instance (PiVideo): The video stream instance.

    Returns:
        Flask: Configured Flask application.
    """
    app = Flask(__name__)

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

                if frame_data is not None:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_data + b'\r\n\r\n')
                sleep(1 / stream_instance.framerate)
        finally:
            with stream_instance.clients_lock:
                stream_instance.clients -= 1
                logging.info(f"Client disconnected. Remaining clients: {stream_instance.clients}")

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
                <title>Pi Camera Stream</title>
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <style>
                    body { margin: 0; padding: 0; background: #000; }
                    .container { display: flex; justify-content: center; align-items: center; min-height: 100vh; }
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


class WebCamVideoStream:
    def __init__(self, src=0, calibrated_camera: CalibratedCamera = None):
        """
        Initialize the USB webcam video stream.

        Args:
            src (int): The camera source index.
            calibrated_camera (CalibratedCamera): Instance for camera calibration data.
        """
        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            logging.error("Error: Unable to access the camera.")
            exit()
        self.width = calibrated_camera.frame_width
        self.height = calibrated_camera.frame_height


if __name__ == "__main__":
    # Current settings for the Pi camera calibration
    CAMERA_MATRIX_PATH = "src/Camera/calibration/pi_cal/cameraMatrix.pkl"
    CAMERA_DISTORTION_PATH = "src/Camera/calibration/pi_cal/dist.pkl"

    calibrated_camera = CalibratedCamera(
        cam_mat_path=CAMERA_MATRIX_PATH,
        cam_dist_path=CAMERA_DISTORTION_PATH,
        frame_width=1080,
        frame_height=1080
    )
    stream = PiVideo(
        calibrated_camera=calibrated_camera,
        framerate=60,
        brightness=0.0,
        contrast=1.0
    ).start()
    app = create_app(stream)
    app.run(host='0.0.0.0', port=5000)
