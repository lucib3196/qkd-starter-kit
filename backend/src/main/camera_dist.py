import cv2
import time
import numpy as np
# Import marker detection and camera feed utilities
from ..ArUcoMarker.utils import find_marker, track_and_render_marker, draw_center_frame
from . import WebcamVideoStreamThreaded, VideoShow

ARUCO_DICT_TYPE = cv2.aruco.DICT_ARUCO_ORIGINAL
MARKER_LENGTH = 0.046  # meters

def main(src=0):
    print("Starting Camera Calibration Mode (No Control)")
    start_time = time.time()
    # Data columns: Time, Pan_Error (deg), Tilt_Error (deg), tvec_x, tvec_y, tvec_z
    data_array = np.empty((0, 6))
    
    video_stream = WebcamVideoStreamThreaded(src).start()
    video_display = VideoShow(video_stream.frame).start()
    
    try:
        while True:
            elapsed_time = time.time() - start_time
            if video_stream.stopped:
                break
            frame = video_stream.frame
            if frame is None:
                continue
            
            frame_height, frame_width = frame.shape[:2]
            center = (frame_width // 2, frame_height // 2)
            draw_center_frame(frame, center)
            
            aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
            parameters = cv2.aruco.DetectorParameters()
            marker_array = find_marker(frame, aruco_dict, parameters)
            
            if marker_array:
                for marker, marker_id in marker_array:
                    transformation_matrix = track_and_render_marker(
                        frame,
                        marker,
                        marker_id,
                        camera_matrix=video_stream.camera_matrix,
                        distortion_coefficient=video_stream.camera_dist,
                        marker_length=MARKER_LENGTH
                    )
                    tvec = transformation_matrix[:-1, -1]
                    x, y, z = tvec
                    
                    # Compute error angles for calibration
                    pan_error_rad = np.arctan2(x, z)
                    pan_error_deg = np.degrees(pan_error_rad)
                    tilt_error_rad = np.arctan2(y, z)
                    tilt_error_deg = np.degrees(tilt_error_rad)
                    
                    # Log calibration data
                    data_entry = np.array([[elapsed_time, np.abs(pan_error_deg), np.abs(tilt_error_deg),
                                            x, y, z]])
                    data_array = np.vstack((data_array, data_entry))
            
            video_display.frame = frame

    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        video_stream.stop()
        video_display.stop()
        cv2.destroyAllWindows()
        np.savetxt("data_calib.csv", data_array, delimiter=",",
                   header="Time,Pan_Error,Tilt_Error,tvec_x,tvec_y,tvec_z", comments="", fmt="%.5f")
        print("Data saved to data_calib.csv")

if __name__ == "__main__":
    main()
