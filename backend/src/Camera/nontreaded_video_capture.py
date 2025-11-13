from datetime import datetime
import cv2
from .fps import FPS, putIterationsPerSec

"""
Overview:
This script provides a non-threaded implementation to test camera functionality.
It captures video frames using OpenCV, overlays the iterations per second (FPS) on the video frames, and displays the video in real-time.

Key Features:
1. Non-Threaded Video Capture:
   - Captures video frames directly in the main thread for simplicity.

2. FPS Overlay:
   - Displays the current iterations per second (FPS) on the video frames.

3. Usability:
   - Press 'q' to exit the video stream.

Modules Used:
- datetime: Used indirectly through the FPS class for measuring elapsed time.
- OpenCV (cv2): Used for capturing video frames and displaying them.
- Custom fps module: Provides FPS tracking and overlay functionality.

Usage:
This script is suitable for simple camera testing or as a baseline comparison with threaded implementations.
"""

def noThreading(source=0):
    """
    Function to test the camera functionality without threading. Captures video from the camera,
    overlays the iterations per second (FPS) on the video frames, and displays the video.

    Parameters:
    - source (int or str): The video source (0 for default camera, or file path for video).

    Usage:
    - Press 'q' to exit the video stream.
    """
    # Initialize video capture
    cap = cv2.VideoCapture(source)

    # Start the FPS counter
    fps_counter = FPS().start()

    while True:
        # Capture a frame
        grabbed, frame = cap.read()

        # Exit if frame not grabbed or 'q' key is pressed
        if not grabbed or cv2.waitKey(1) == ord("q"):
            break

        # Overlay the iterations per second (FPS) on the frame
        frame = putIterationsPerSec(frame, fps_counter.fps())

        # Display the frame
        cv2.imshow("Video", frame)

        # Update the FPS counter
        fps_counter.update()

if __name__ == '__main__':
    """
    Main function to execute the non-threaded camera test.

    - Starts video capture without threading.
    - Displays the live video with FPS overlay.
    - Provides a basic test to verify camera functionality and FPS tracking.
    """
    noThreading()