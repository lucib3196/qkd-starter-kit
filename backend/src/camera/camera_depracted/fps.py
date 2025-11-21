import datetime
import cv2

"""
Overview:
This script provides utilities to measure and display the performance of video processing tasks. 
It includes a class to calculate frames per second (FPS) and a function to overlay the FPS 
or iterations per second on a video frame.

Key Features:
1. FPS Measurement:
   - Tracks the number of frames processed and calculates the elapsed time.
   - Provides methods to compute the current FPS.

2. FPS Overlay:
   - Adds the iterations per second as text on the video frames.

Modules Used:
- datetime: For measuring the elapsed time during FPS calculations.
- OpenCV (cv2): For overlaying text on video frames.

Usage:
This module can be imported and used as part of a video processing pipeline to track and display performance metrics.
"""

class FPS:
    """
    A class for measuring frames per second (FPS) in video processing tasks.

    Attributes:
    - _start (datetime): The time when the FPS counter starts.
    - _end (datetime): The time when the FPS counter stops (currently unused).
    - _numFrames (int): The number of frames processed since the counter started.
    """

    def __init__(self):
        """
        Initializes the FPS counter with default values.
        """
        self._start = None
        self._end = None
        self._numFrames = 0

    def start(self):
        """
        Starts the FPS counter by recording the current time.

        Returns:
        - self: The instance of the class to allow method chaining.
        """
        self._start = datetime.datetime.now()
        return self

    def update(self):
        """
        Increments the frame count by 1. Should be called after processing each frame.
        """
        self._numFrames += 1

    def elapsed(self):
        """
        Calculates the total elapsed time since the counter started.

        Returns:
        - float: Elapsed time in seconds.
        """
        return (datetime.datetime.now() - self._start).total_seconds()

    def fps(self):
        """
        Calculates the current frames per second (FPS).

        Returns:
        - float: Current FPS based on the number of frames processed and elapsed time.
        """
        return self._numFrames / self.elapsed()


def putIterationsPerSec(frame, iterations_per_sec):
    """
    Adds the iterations per second text to the lower-left corner of a video frame.

    Parameters:
    - frame (numpy.ndarray): The video frame to modify.
    - iterations_per_sec (float): The current iterations per second to display.

    Returns:
    - frame (numpy.ndarray): The modified video frame with text overlay.
    """
    cv2.putText(
        frame,
        "{:.0f} iterations/sec".format(iterations_per_sec),
        (10, 450),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
    )
    return frame
