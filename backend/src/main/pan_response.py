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

# Pan servo limits and pin
PAN_MIN = -90
PAN_MAX = 90
PAN_SERVO_PIN = 17

# Initialize pan servo (500µs to 2500µs pulse width)
pan_servo = AngularServo(PAN_SERVO_PIN, min_angle=PAN_MIN, max_angle=PAN_MAX,
                         min_pulse_width=0.5/1000, max_pulse_width=2.5/1000)

# Initialize pan PID controller with gain 1 (Kp=1, Ki=0, Kd=0)
pan_controller = PIDController(Kp=0.5, Ki=0, Kd=0.5)

def main(src=0):
    print("Starting Pan-Only Tracking")
    start_time = time.time()
    # Data columns: Time, Pan_Error (deg), Pan_Angle
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
                    
                    # Calculate pan error (in degrees)
                    pan_error_rad = np.arctan2(x, z)
                    pan_error_deg = np.degrees(pan_error_rad)
                    
                    current_pan = pan_servo.angle if pan_servo.angle is not None else 0
                    
                    # Compute correction using the pan PID controller
                    pan_correction = pan_controller.update_PD(pan_error_deg)
                    new_pan = current_pan - pan_correction
                    new_pan = np.clip(new_pan, PAN_MIN, PAN_MAX)
                    
                    pan_servo.angle = new_pan
                    
                    # Log time, absolute error, and new pan angle
                    data_entry = np.array([[elapsed_time, np.abs(pan_error_deg), new_pan]])
                    data_array = np.vstack((data_array, data_entry))
            
            video_display.frame = frame

    except Exception as e:
        print(f"Error occurred: {e}")
    finally:
        video_stream.stop()
        video_display.stop()
        cv2.destroyAllWindows()
        np.savetxt("data_pan.csv", data_array, delimiter=",",
                   header="Time,Pan_Error,Pan_Angle", comments="", fmt="%.5f")
        print("Data saved to data_pan.csv")

if __name__ == "__main__":
    main()
