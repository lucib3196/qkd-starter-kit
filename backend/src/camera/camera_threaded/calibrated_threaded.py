from threading import Thread
import cv2
from src.camera.camera_utils import load_camera_calibration
from pydantic import BaseModel
from pathlib import Path


class CalibrationSettings(BaseModel):
    camera_matrix_path: str | Path
    camera_distortion_path: str | Path


class CalibratedThreadedStream:
    """
    A class for threaded video capture from a webcam or usb camera.

    Attributes:
    - stream (cv2.VideoCapture): The video capture object.
    - grabbed (bool): Indicates if the frame was successfully grabbed.
    - frame (numpy.ndarray): The current frame from the video stream.
    - stopped (bool): Flag to stop the video stream thread.
    """

    def __init__(self, src=0, calibration_settings: CalibrationSettings | None = None):
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

        if not calibration_settings:
            raise ValueError("Calibration settings must be set")

        self.camera_matrix, self.camera_dist = load_camera_calibration(
            calibration_settings.camera_matrix_path,
            calibration_settings.camera_distortion_path,
        )

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
                self.grabbed.release()  # type: ignore
                self.stop()
            else:
                self.grabbed, self.frame = self.stream.read()
                self.calibrate_camera()
                self.undistort_frame()

    def stop(self):
        """
        Stop the video stream thread by setting the stopped flag to True.
        """
        self.stopped = True

    def calibrate_camera(self):
        """
        Calculate the optimal new camera matrix for undistortion.
        """
        height, width = self.frame.shape[:2]
        self.new_camera_mtx, self.roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix,
            self.camera_dist,
            (width, height),
            1,
            (width, height),
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


def thread_video_get(src=0):
    """
    Function to test the camera functionality using threading. Captures video from the camera,
    displays it in a window, and overlays the iterations per second (FPS) on the video frame.

    Parameters:
    - src (int or str): The video source (0 for default camera, or file path).

    Usage:
    - Press 'q' to exit the video stream.
    """
    video_getter = CalibratedThreadedStream(src).start()

    while True:
        if (cv2.waitKey(1) == ord("q")) or video_getter.stopped:
            video_getter.stop()
            break

        frame = video_getter.frame

        cv2.imshow("Video", frame)


if __name__ == "__main__":
    """
    Main function to execute the threaded camera test.

    - Starts a threaded video stream.
    - Displays the live video with FPS overlay.
    - Provides a basic test to verify camera functionality and threading.
    """
    thread_video_get()
