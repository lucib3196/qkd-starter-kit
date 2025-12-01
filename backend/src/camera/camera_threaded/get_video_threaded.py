from threading import Thread
import cv2
from src.fps.fps import putIterationsPerSec, FPS
from cv2.typing import MatLike
from .models import CalibrationSettings
from src.camera.camera_utils import load_camera_calibration
from threading import Lock


class VideoGet:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()

    def stop(self):
        self.stopped = True
        return self


class VideoGetCalibrated:
    def __init__(self, src=0, calibration_settings: CalibrationSettings | None = None):
        self.stream = cv2.VideoCapture(src)

        self.stopped = False

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
        self.calibrate_camera()
        self.lock = Lock()
        
    def start(self):
        self.thread = Thread(target=self.get, args=(),daemon=True).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
                break
            else:
                with self.lock:
                    (self.grabbed, self.frame) = self.stream.read()
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


class VideoShow:
    def __init__(self, frame: MatLike | None = None):
        self.frame = frame
        self.stopped = False

    def start(self):
        Thread(target=self.show, args=()).start()
        return self

    def show(self):
        while not self.stopped:
            if self.frame is not None:
                cv2.imshow("Video", self.frame)
                if cv2.waitKey(1) == ord("q"):
                    self.stopped = True

    def stop(self):
        self.stopped = True
        cv2.destroyAllWindows()


def threaded_video_get(source=0):
    """
    Dedicated thread for grabbing video frames with VideoGet object.
    Main thread shows video frames.
    """

    video_getter = VideoGet(source).start()
    video_show = VideoShow(video_getter.frame).start()

    counter = FPS().start()
    while True:
        if video_show.stopped or video_getter.stopped:
            video_getter.stop()
            video_show.stop()
            break

        frame = video_getter.frame
        putIterationsPerSec(frame, counter.fps())
        video_show.frame = frame
        counter.update()


if __name__ == "__main__":
    threaded_video_get()
