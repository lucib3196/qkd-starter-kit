"""
camera_test_stream.py

This script is a simple test application that captures images from a Raspberry Pi Camera using Picamera2 and streams them over HTTP via Flask.
It continuously captures frames, encodes them as JPEG, and serves them at the /video_feed endpoint. Note that this test does not integrate the full pi_camera_streamer functionality; it is solely intended to verify basic camera operation and streaming.
"""

import cv2
from picamera2 import Picamera2
from flask import Flask, Response

app = Flask(__name__)
picam2 = Picamera2()

# Start the camera
picam2.start()

def generate():
    """Continuously capture frames, encode them to JPEG, and yield as multipart response."""
    while True:
        frame = picam2.capture_array()
        ret, jpeg = cv2.imencode('.jpg', frame)
        print(picam2.camera_configuration())
        if ret:
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/video_feed')
def video_feed():
    """HTTP endpoint that streams the JPEG frames as a multipart response."""
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
