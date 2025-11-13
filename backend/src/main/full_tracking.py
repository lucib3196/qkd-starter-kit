import cv2
import time
import numpy as np
from gpiozero import Device, AngularServo
from gpiozero.pins.pigpio import PiGPIOFactory
# Import marker detection and camera feed utilities
from ..ArUcoMarker.utils import find_marker, track_and_render_marker, draw_center_frame
from . import WebcamVideoStreamThreaded, VideoShow
from ..PID import PIDController

# Configure gpiozero to use the PiGPIOFactory
Device.pin_factory = PiGPIOFactory()

# Constants
ARUCO_DICT_TYPE = cv2.aruco.DICT_ARUCO_ORIGINAL
MARKER_LENGTH = 0.046  # meters

# Servo angle limits
PAN_MIN = -135
PAN_MAX = 135
TILT_MIN = -135
TILT_MAX = 135

# Servo pins
PAN_SERVO_PIN = 17
TILT_SERVO_PIN = 27

# Initialize servos with a pulse width range of 500µs to 2500µs
pan_servo = AngularServo(PAN_SERVO_PIN, min_angle=PAN_MIN, max_angle=PAN_MAX)
tilt_servo = AngularServo(TILT_SERVO_PIN, min_angle=TILT_MIN, max_angle=TILT_MAX)

# Initialize PID controllers (tuning parameters can be adjusted)
pan_controller  = PIDController(Kp=1 ,Ki =0.0 ,Kd= 1)
tilt_controller = PIDController(Kp =1 ,Ki =0 ,Kd =2)

def main(src=0):
    print("Starting Full Pan–Tilt Tracking")
    start_time = time.time()
    # Data columns: Time, Pan_Error (deg), Pan_Angle, Tilt_Error (deg), Tilt_Angle
    data_array = np.empty((0, 5))
    
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
            
            # Marker detection
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
                    # Extract translation vector (tvec)
                    tvec = transformation_matrix[:-1, -1]
                    x, y, z = tvec
                    
                    # Compute error angles (in degrees)
                    pan_error_rad = np.arctan2(x, z)
                    pan_error_deg = np.degrees(pan_error_rad)
                    
                    tilt_error_rad = np.arctan2(y, z)
                    tilt_error_deg = np.degrees(tilt_error_rad)
                    
                    # Get current servo angles (set defaults if not yet set)
                    current_pan = pan_servo.angle if pan_servo.angle is not None else 0
                    current_tilt = tilt_servo.angle if tilt_servo.angle is not None else 90  # assume center for tilt
                    
                    # Compute corrections using PID controllers
                    pan_correction = pan_controller.compute(pan_error_deg, elapsed_time)
                    tilt_correction = tilt_controller.compute(tilt_error_deg,elapsed_time)
                    
                    # Update servo angles
                    new_pan = current_pan - pan_correction
                    new_tilt = current_tilt - tilt_correction
                    
                    # Clip angles to valid range
                    new_pan = np.clip(new_pan, PAN_MIN, PAN_MAX)
                    new_tilt = np.clip(new_tilt, TILT_MIN, TILT_MAX)
                    
                    pan_servo.angle = new_pan
                    tilt_servo.angle = new_tilt
                    
                    # Log data: time, pan error, pan angle, tilt error, tilt angle
                    data_entry = np.array([[elapsed_time, np.abs(pan_error_deg), new_pan,
                                            np.abs(tilt_error_deg), new_tilt]])
                    data_array = np.vstack((data_array, data_entry))
            
            video_display.frame = frame

    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        video_stream.stop()
        video_display.stop()
        cv2.destroyAllWindows()
        np.savetxt("data_main.csv", data_array, delimiter=",",
                   header="Time,Pan_Error,Pan_Angle,Tilt_Error,Tilt_Angle", comments="", fmt="%.5f")
        print("Data saved to data_main.csv")

if __name__ == "__main__":
    main()
