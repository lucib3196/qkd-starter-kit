import cv2
from src.camera.camera_threaded.models import CalibrationSettings
from src.camera.camera_threaded.get_video_threaded import VideoGetCalibrated, VideoShow
from src.fps.fps import putIterationsPerSec, FPS
from pathlib import Path
import time

import cv2
import numpy as np


from src.aruco_marker.utils import *
from src.camera.camera_threaded import CalibrationSettings
from src.camera.camera_threaded.get_video_threaded import VideoGetCalibrated
from src.controls.pid import PIDController
from src.fps.fps import putIterationsPerSec, FPS

from src.aruco_marker.utils import (
    detect_markers,
    track_and_render_marker,
)
from pathlib import Path
import numpy as np

# Servo angle limits
PAN_MIN = -135
PAN_MAX = 135
TILT_MIN = -135
TILT_MAX = 135


# Create the pid and tilt controllers
pan_controller = PIDController(Kp=1, Ki=0.0, Kd=1)
tilt_controller = PIDController(Kp=1, Ki=0, Kd=2)


def main(source=0):
    try:
        start_time = time.time()
        matrix = Path(
            r"src\camera\camera_calibration\calibration\pical20284096\cameraMatrix.pkl"
        ).resolve()
        distortion = Path(
            r"src\camera\camera_calibration\calibration\pical20284096\distortion.pkl"
        ).resolve()

        settings = CalibrationSettings(
            camera_matrix_path=matrix, camera_distortion_path=distortion
        )
        video_stream = VideoGetCalibrated(source, calibration_settings=settings).start()
        video_show = VideoShow(video_stream.frame).start()
        fps = FPS().start()
        print("Camera started")
        current_pan = 0
        current_tilt = 0

        # Data columns: Time, Pan_Error (deg), Pan_Angle, Tilt_Error (deg), Tilt_Angle
        data_array = np.empty((0, 5))

        while True:
            elapsed_time = time.time() - start_time
            # Stop threads if either thread signals to stop
            if video_stream.stopped:
                break

            frame = video_stream.frame
            if frame is not None:
                pass
                # Optionally turn the image into a gray scale
                # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # ArUco marker detection
                ## The aruco marker must match a dict type
                aruco_dict_type = cv2.aruco.DICT_6X6_250
                aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
                parameters = cv2.aruco.DetectorParameters()

                markers = detect_markers(frame, aruco_dict, parameters)
                if markers:
                    for m in markers:
                        # Returns the transformation matrix
                        T = track_and_render_marker(
                            frame,
                            m[0],
                            m[1],
                            video_stream.camera_matrix,
                            video_stream.camera_dist,
                        )
                        # Gets the orientation matrix
                        R = T[:-1, -1]
                        x, y, z = R

                        pan_error_rad = np.arctan2(x, z)
                        pan_error_deg = np.degrees(pan_error_rad)

                        tilt_error_rad = np.arctan2(y, z)
                        tilt_error_deg = np.degrees(tilt_error_rad)

                        pan_correction = pan_controller.compute(
                            pan_error_deg, elapsed_time
                        )
                        tilt_correction = tilt_controller.compute(
                            tilt_error_deg, elapsed_time
                        )

                        new_pan = current_pan - pan_correction
                        new_tilt = current_tilt - tilt_correction

                        # Clip angles to valid range
                        current_pan = np.clip(new_pan, PAN_MIN, PAN_MAX)
                        current_tilt = np.clip(new_tilt, TILT_MIN, TILT_MAX)

                        data_entry = np.array(
                            [
                                [
                                    elapsed_time,
                                    np.abs(pan_error_deg),
                                    new_pan,
                                    np.abs(tilt_error_deg),
                                    new_tilt,
                                ]
                            ]
                        )
                        data_array = np.vstack((data_array, data_entry))

                video_show.frame = frame
                putIterationsPerSec(frame, fps.fps())
                fps.update()
                time.sleep(0.01)

            # Exit the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break

    except Exception as e:
        # Log or print the exception for debugging
        print(f"An error occurred: {e}")
        cv2.destroyAllWindows()

    finally:
        # Clean up resources
        if "video_stream" in locals() and video_stream is not None:  # type: ignore
            video_stream.stop()  # type: ignore
        cv2.destroyAllWindows()
        print("Resources have been released. Exiting gracefully.")


if __name__ == "__main__":
    main()
