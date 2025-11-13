"""
face_tracking_streamer.py

Overview:
This script sets up a Flask web server to stream live video from a PiCamera. It processes each frame with cvzone's FaceDetector to detect and annotate faces (drawing bounding boxes, confidence scores, and positional information), encodes the processed frames as JPEG images, and serves them over HTTP. This test verifies that both face tracking and streaming functionalities are working correctly.

Dependencies:
    - OpenCV (cv2) and NumPy for image processing.
    - Flask for serving the video stream.
    - Picamera2 for interfacing with the PiCamera.
    - cvzone for face detection and annotation.
    - Custom modules from .utils and ..Camera.pi_camera_streamer for calibration and video streaming.
"""


from flask import Flask, Response
import time
import logging
import numpy as np
import cv2
from .utils import detect_faces
from ..Camera.pi_camera_streamer import CalibratedCamera, PiVideo
from time import sleep

import cvzone
from cvzone.FaceDetectionModule import FaceDetector

face_detector = FaceDetector(minDetectionCon=0.5, modelSelection=0)



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
                    
                    nparr = np.frombuffer(frame_data, dtype=np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
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

                            # Display x and y position below the bounding box
                            position_text = f"x: {x}, y: {y}"
                            cv2.putText(
                                frame, 
                                position_text, 
                                (x, y + h + 30),  # Position below the bounding box
                                cv2.FONT_HERSHEY_PLAIN, 
                                2, 
                                (0, 255, 0), 
                                2
                            )
                    ret, jpeg = cv2.imencode('.jpg', frame)
                    if ret:
                        frame_data = jpeg.tobytes()
                    
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

if __name__ == "__main__":
    
    # Current settings for the pi camera 
    CAMERA_MATRIX_PATH = "src/Camera/calibration/pi_cal/cameraMatrix.pkl"
    CAMERA_DISTORTION_PATH = "src/Camera/calibration/pi_cal/dist.pkl"

    calibrated_camera = CalibratedCamera(
        cam_mat_path=CAMERA_MATRIX_PATH,
        cam_dist_path=CAMERA_DISTORTION_PATH,
        frame_width=1080,
        frame_height=720
    )
    stream = PiVideo(
        calibrated_camera=calibrated_camera,
    framerate=30,
    brightness=0.0,
    contrast=1.0
    ).start()
    app = create_app(stream)
    app.run(host='0.0.0.0', port=5000)