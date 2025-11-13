"""
Overview:
This script tests camera functionality and threading performance.
It creates a threaded video stream using OpenCV to capture video frames from a webcam
and overlays the frames per second (FPS) on the video.

Key Features:
1. **Threaded Video Capture:**
   - Continuously captures frames in a separate thread to enhance performance.
   - Reduces latency caused by frame grabbing in the main thread.

2. **FPS Overlay:**
   - Calculates and displays the current FPS on each video frame.

3. **Usability:**
   - Press 'q' to exit the video stream.

Modules Used:
- threading: For running the video capture in a separate thread.
- OpenCV (cv2): For capturing and displaying video frames.
- Custom `fps` module: Tracks and overlays FPS on the video frames.
"""

from threading import Thread
import cv2
import os
import pickle
from .fps import FPS, putIterationsPerSec

# Paths for camera calibration data
CAMERA_MATRIX_PATH = "src/Camera/cameraMatrix.pkl"
CAMERA_DISTORTION_PATH = "src/Camera/dist.pkl"

def define_camera_settings(camera_matrix_path, camera_distortion_path):
    """
    Load the camera calibration data from the specified file paths.

    Parameters:
    - camera_matrix_path (str): Path to the camera matrix file.
    - camera_distortion_path (str): Path to the distortion coefficients file.

    Returns:
    - tuple: Camera matrix and distortion coefficients.
    """
    with open(camera_matrix_path, "rb") as file:
        camera_matrix = pickle.load(file)

    with open(camera_distortion_path, "rb") as file:
        camera_distortion_coefficients = pickle.load(file)

    return camera_matrix, camera_distortion_coefficients

def load_camera_calibration():
    """
    Load camera calibration data using the predefined paths.

    Returns:
    - tuple: Camera matrix and distortion coefficients.
    """
    camera_matrix_path = os.path.join(os.getcwd(), CAMERA_MATRIX_PATH)
    camera_distortion_path = os.path.join(os.getcwd(), CAMERA_DISTORTION_PATH)

    print(f"Camera matrix path: {camera_matrix_path}")  # Debugging
    print(f"Camera distortion path: {camera_distortion_path}")  # Debugging

    return define_camera_settings(camera_matrix_path, camera_distortion_path)



class WebcamVideoStreamThreaded():
    """
    A class for threaded video capture from a webcam.

    Attributes:
    - stream (cv2.VideoCapture): The video capture object.
    - grabbed (bool): Indicates if the frame was successfully grabbed.
    - frame (numpy.ndarray): The current frame from the video stream.
    - stopped (bool): Flag to stop the video stream thread.
    """

    def __init__(self, src=0):
        """
        Initialize the video stream and read the first frame.

        Parameters:
        - src (int or str): Video source (0 for default camera, or file path).
        """
        self.stream = cv2.VideoCapture(src)
        if not self.stream.isOpened():
            print("Error: Unable to access the camera.")
            exit()
        self.frame_width = self.stream.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.frame_height = self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT)

        self.camera_matrix, self.camera_dist = load_camera_calibration()
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        """
        Start the thread to read frames from the video stream.

        Returns:
        - self: The instance of the class to allow method chaining.
        """
        self.video_thread = Thread(target=self.get, args=()).start()
        return self
    

    def get(self):
        """
        Continuously grab frames from the video stream until stopped.
        """
        while not self.stopped:
            if not self.grabbed:
                self.grabbed.release()
                self.stop()
            else:
                self.grabbed, self.frame = self.stream.read()
                self.calibrate_camera()
                self.undistort_frame()

    def calibrate_camera(self):
        """
        Calculate the optimal new camera matrix for undistortion.
        """
        height, width = self.frame.shape[:2]
        self.new_camera_mtx, self.roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.camera_dist, (width, height), 1, (width, height)
        )
        self.width = width
        self.height = height
        return self


    def undistort_frame(self):
        """
        Apply undistortion to the current frame and crop the result.
        """
        undistorted_frame = cv2.undistort(
            self.frame, self.camera_matrix, self.camera_dist, None, self.new_camera_mtx
        )
        x, y, w, h = self.roi
        self.undistorted_frame = undistorted_frame[y : y + h, x : x + w]

    def stop(self):
        """
        Stop the video stream thread by setting the stopped flag to True.
        """
        self.stopped = True
        
    
        

def thread_video_get(src=0):
    """
    Function to test the camera functionality using threading. Captures video from the camera,
    displays it in a window, and overlays the iterations per second (FPS) on the video frame.

    Parameters:
    - src (int or str): The video source (0 for default camera, or file path).

    Usage:
    - Press 'q' to exit the video stream.
    """
    video_getter = WebcamVideoStreamThreaded(src).start()
    fps_counter = FPS().start()

    while True:
        if (cv2.waitKey(1) == ord("q")) or video_getter.stopped:
            video_getter.stop()
            break

        frame = video_getter.frame
        frame = putIterationsPerSec(frame, fps_counter.fps())
        cv2.imshow("Video", frame)
        fps_counter.update()
        
        
class VideoShow:
    """
    Handles displaying frames in a separate thread.

    Attributes:
    - frame: Current frame to be displayed.
    - stopped: Flag to stop the thread.
    """
    def __init__(self, frame=None):
        self.frame = frame
        self.stopped = False

    def start(self,):
        """
        Starts the thread to display video frames.

        Returns:
        - self: Instance of the VideoShow class for method chaining.
        """
        self.video_thread=Thread(target=self.show, args=()).start()
        return self

    def show(self):
        """
        Continuously displays the video frame until stopped.
        """
        while not self.stopped:
            if self.frame is not None:
                cv2.imshow('Video', self.frame)
            if cv2.waitKey(1) == ord("q"):
                self.stopped = True

    def stop(self):
        """
        Stops the video display thread by setting the stopped flag to True.
        """
        self.stopped = True




if __name__ == "__main__":
    """
    Main function to execute the threaded camera test.

    - Starts a threaded video stream.
    - Displays the live video with FPS overlay.
    - Provides a basic test to verify camera functionality and threading.
    """
    thread_video_get()