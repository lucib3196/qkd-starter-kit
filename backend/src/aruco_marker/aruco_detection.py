import cv2
from src.camera.camera_threaded import CalibratedThreadedStream, CalibrationSettings
from .utils import find_marker, get_marker_coord, get_marker_center
from pathlib import Path

# Constants
aruco_dict_type = cv2.aruco.DICT_6X6_250


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
            print(frame, "This is the value of the frame")
            if frame is not None:
                pass

                # # Logic to display stuff
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # ArUco marker detection
                aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
                parameters = cv2.aruco.DetectorParameters()

                markers = find_marker(gray, aruco_dict, parameters)
                print("Marker detection")
                marker_arr = get_marker_coord(markers)
                if marker_arr:
                    for marker in marker_arr:
                        center = get_marker_center(marker)
                        print(center)

            # Exit the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except Exception as e:
        # Log or print the exception for debugging
        print(f"An error occurred: {e}")

    finally:
        # Clean up resources
        if "video_stream" in locals() and video_stream is not None:  # type: ignore
            video_stream.stop()  # type: ignore
        cv2.destroyAllWindows()
        print("Resources have been released. Exiting gracefully.")


if __name__ == "__main__":
    main()
