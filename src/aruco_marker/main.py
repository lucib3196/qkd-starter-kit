import cv2
from src.camera.camera_threaded import CalibratedThreadedStream, CalibrationSettings
from .utils import (
    detect_markers,
    get_marker_corners,
    get_marker_center,
    draw_corners_circ,
    track_and_render_marker,
)
from pathlib import Path


def main(source=0):
    try:
        matrix = Path(
            r"src\camera\camera_calibration\calibration\pical20284096\cameraMatrix.pkl"
        ).resolve()
        distortion = Path(
            r"src\camera\camera_calibration\calibration\pical20284096\distortion.pkl"
        ).resolve()

        settings = CalibrationSettings(
            camera_matrix_path=matrix, camera_distortion_path=distortion
        )
        video_stream = CalibratedThreadedStream(
            source, calibration_settings=settings
        ).start()
        print("Camera started")

        # ArUco marker detection
        ## The aruco marker must match a dict type
        aruco_dict_type = cv2.aruco.DICT_6X6_250
        aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
        parameters = cv2.aruco.DetectorParameters()

        while True:
            # Stop threads if either thread signals to stop
            if video_stream.stopped:
                break
            with video_stream.lock:
                frame = (
                    video_stream.frame.copy()
                    if video_stream.frame is not None
                    else None
                )
            if frame is None:
                continue

            # Optionally turn the image into a gray scale
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            markers = detect_markers(frame, aruco_dict, parameters)
            if markers:
                for m in markers:
                    track_and_render_marker(
                        frame,
                        m[0],
                        m[1],
                        video_stream.camera_matrix,
                        video_stream.camera_dist,
                    )
            video_stream.frame = frame
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
