import cv2
from gpiozero import AngularServo
import numpy as np
import time
from . import WebcamVideoStreamThreaded, FPS, putIterationsPerSec, VideoShow

from gpiozero.pins.pigpio import PiGPIOFactory
from gpiozero import Device
Device.pin_factory = PiGPIOFactory()



HAAR_CASCADE_FACE = r'src/FaceDetection/haarcascade_frontalface_default.xml'
HAAR_CASCADE_EYES = r'src/FaceDetection/haarcascade_eye.xml'

# Intialize the face classifier
faceClassifier = cv2.CascadeClassifier(HAAR_CASCADE_FACE)
eyeClassifier = cv2.CascadeClassifier(HAAR_CASCADE_EYES)

print(eyeClassifier)
def detect_faces(img_frame):
    # Convert to gray scale for faster processing
    gray = cv2.cvtColor(img_frame, cv2.COLOR_BGR2GRAY)

    # Ensure classifiers are loaded
    if faceClassifier.empty() or eyeClassifier.empty():
        raise ValueError("Classifier files are not loaded correctly!")

    # Detect faces
    faces = faceClassifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3)

    # Loop through detected faces
    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(img_frame, (x, y), (x + w, y + h), (255, 0, 255), 2)

        # Region of interest for eyes
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img_frame[y:y + h, x:x + w]

        # Detect eyes within the face ROI
        eyes = eyeClassifier.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            # Draw rectangle around each detected eye
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 127, 255), 2)

    # Add text to display the number of faces detected
    text = f"Number of Faces Detected = {len(faces)}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_frame, text, (10, 30), font, 1, (255, 0, 0), 2)

    return img_frame  # Return the processed frame

def main(src=0):
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
            frame = detect_faces(frame)
            frame_with_fps = putIterationsPerSec(frame, fps_counter.fps())
            # video_display.frame = frame_with_fps

        # Update FPS counter
        fps_counter.update()
if __name__ == "__main__":
    """
    Main function to start the threaded video stream with ArUco marker detection.
    """
    main(0)