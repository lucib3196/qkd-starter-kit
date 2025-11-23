import cv2
from src.camera.camera_threaded import CalibratedThreadedStream, CalibrationSettings
from .utils import (
    detect_markers,
    get_marker_corners,
    get_marker_center,
    estimate_pose_and_transformation_matrix,
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

        while True:
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
                        print(f"Found marker {m[0]} with id of {m[1]}")
                        coord = get_marker_corners(m[0])
                        print("These are the marker corners", coord)
                        center = get_marker_center(m[0])
                        print("This is the center", center)
                        cv2.putText(
                            video_stream.frame,
                            "Hello",
                            (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                        )
                        data = estimate_pose_and_transformation_matrix(
                            m[0], video_stream.camera_matrix, video_stream.camera_dist
                        )
                        print(data)

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
