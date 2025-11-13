import cv2
import time
import numpy as np
from gpiozero import Device, AngularServo
from gpiozero.pins.pigpio import PiGPIOFactory
from ..ArUcoMarker.utils import find_marker, track_and_render_marker, draw_center_frame
from . import WebcamVideoStreamThreaded, VideoShow
from ..PID import PIDController

Device.pin_factory = PiGPIOFactory()

ARUCO_DICT_TYPE = cv2.aruco.DICT_ARUCO_ORIGINAL
MARKER_LENGTH = 0.046  # meters

# Tilt servo limits and pin
TILT_MIN = 0
TILT_MAX = 180
TILT_SERVO_PIN = 27

# Initialize tilt servo (500µs to 2500µs pulse width)
tilt_servo = AngularServo(TILT_SERVO_PIN, min_angle=TILT_MIN, max_angle=TILT_MAX,
                          min_pulse_width=0.5/1000, max_pulse_width=2.5/1000)

# Initialize tilt PID controller with gain 1 (Kp=1, Ki=0, Kd=0)
tilt_controller = PIDController(Kp=1.0, Ki=0, Kd=0)

def main(src=0):
    print("Starting Tilt-Only Tracking")
    start_time = time.time()
    # Data columns: Time, Tilt_Error (deg), Tilt_Angle
    data_array = np.empty((0, 3))
    
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
                    
                    # Calculate tilt error (in degrees)
                    tilt_error_rad = np.arctan2(y, z)
                    tilt_error_deg = np.degrees(tilt_error_rad)
                    
                    current_tilt = tilt_servo.angle if tilt_servo.angle is not None else 90
                    
                    # Compute correction using the tilt PID controller
                    tilt_correction = tilt_controller.update_PD(tilt_error_deg)
                    new_tilt = current_tilt - tilt_correction
                    new_tilt = np.clip(new_tilt, TILT_MIN, TILT_MAX)
                    
                    tilt_servo.angle = new_tilt
                    
                    # Log time, absolute error, and new tilt angle
                    data_entry = np.array([[elapsed_time, np.abs(tilt_error_deg), new_tilt]])
                    data_array = np.vstack((data_array, data_entry))
            
            video_display.frame = frame

    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        video_stream.stop()
        video_display.stop()
        cv2.destroyAllWindows()
        np.savetxt("data_tilt.csv", data_array, delimiter=",",
                   header="Time,Tilt_Error,Tilt_Angle", comments="", fmt="%.5f")
        print("Data saved to data_tilt.csv")

if __name__ == "__main__":
    main()
