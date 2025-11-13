import cv2
## Import Utilitis Marker Detection and Camera Feed
from ..ArUcoMarker.utils import (
    find_marker,
    track_and_render_marker, draw_center_frame
)
from . import WebcamVideoStreamThreaded,VideoShow
import time 
import numpy as np
from gpiozero import Device, AngularServo

# Configure gpiozero to use the PiGPIOFactory
from gpiozero.pins.pigpio import PiGPIOFactory
Device.pin_factory = PiGPIOFactory()
from ..PID import PIDController
#Constants 
ARUCO_DICT_TYPE = cv2.aruco.DICT_ARUCO_ORIGINAL
MARKER_LENGTH = 0.046 # Meter


## Define Servo Angle 
# Constant
BASE_MIN = 0
BASE_MAX = 180

BASE_MIN_TOP = -90
BASE_MAX_TOP = 90


# Initialize servo controls 
# Pan Servo controls horizontal movement 
pan_servo_pin = 17
pan_servo = AngularServo(pan_servo_pin, min_angle=BASE_MIN_TOP, max_angle=BASE_MAX_TOP.min_pulse_width=0.5/1000, max_pulse_width=2.5/1000)

# Tilt Servo controls vertical movement 
tilt_servo_pin = 27
tilt_servo = AngularServo(tilt_servo_pin, min_angle=BASE_MIN, max_angle=BASE_MAX,min_pulse_width=0.5/1000, max_pulse_width=2.5/1000))

## Define the PID Controllers for both the pan and tilt this needs to be adjusted as needed
pan_controller  = PIDController(Kp=0.5 ,Ki =0 ,Kd =0.5)
tilt_controller = PIDController(Kp =0.5 ,Ki =0 ,Kd =0.5)

def main(src=0):
    print("Running Program")
    # Define an array to store data
    start_time = time.time()
    num_samples = 3
    data_array = np.zeros((0,num_samples))
    try:
        # Intialize camera and threads
        video_stream = WebcamVideoStreamThreaded(src).start()
        video_display = VideoShow(video_stream.frame).start()
        while True:
            current_time = time.time()-start_time
            
            if video_stream.stopped:
                break
            
            frame = video_stream.frame
            
            if frame is not None:
                frame_height, frame_width = video_stream.frame_height, video_stream.frame_width
                center = (int(frame_width // 2), int(frame_height // 2))
                
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
                            marker_length = MARKER_LENGTH
                        )   
                        print(transformation_matrix)
                        tvec = transformation_matrix[:-1,-1] # Last column all the rows
                        
                        x, y, z = tvec

                        # Print current pan servo angle
                        print(tvec)
                        print(f"Current Pan Servo Angle: {pan_servo.angle:.2f}")

                        # Compute pan angle error
                        pan_angle = np.arctan2(x, z)
                        error_angle_deg = np.degrees(pan_angle)
                        print("hi")
                        # Compute tilt angle error 
                        tilt_angle = np.arctan2(y,z)
                        error_angle_tilt= np.degrees(tilt_angle)

                        # Print calculated error angle
                        print(f"Calculated Error Angle: {error_angle_deg:.2f}")

                        # Compute turn correction
                        turn_x = pan_controller.update_PD(error_angle_deg)
                        new_pan_angle = pan_servo.angle - turn_x

                        # Compute title angle angle correction 
                        turn_y = tilt_controller.update_PD(error_angle_tilt)
                        new_tilt_angle = tilt_servo.angle -turn_y

                        # Print updated pan angle before clipping
                        print(f"New Pan Angle (Before Clipping): {new_pan_angle:.2f}")
                        print(f"New Tilt Angle (Before Clipping): {new_tilt_angle:.2f}")
                        # Ensure new pan angle stays within limits
                        new_pan_angle = np.clip(new_pan_angle, BASE_MIN, BASE_MAX)

                        # Print final adjusted pan angle
                        print(f"Final Pan Angle (After Clipping): {new_pan_angle:.2f}")
                        print(f"Final Tilt Angle (After Clipping): {new_tilt_angle:.2f}")
                        # Apply to pan servo
                        pan_servo.angle = int(new_pan_angle)
                        tilt_servo.angle = int(new_tilt_angle)

                        new_data = np.array([current_time, np.abs(error_angle_deg), pan_servo.angle])
                        data_array = np.vstack((data_array,new_data))
                        
                        
                video_display.frame = frame
        video_stream.video_thread.join()
        video_display.video_thread.join()
    except Exception as e:
        print(f"An Error Occurred {e}")
    finally:
        # Clean up 
        if video_stream:
            video_stream.stop()
        
        cv2.destroyAllWindows()
        np.savetxt("data_main.csv", data_array, delimiter=",", header="Time,Error,Pan_Angle", comments="", fmt="%.5f")
        
if __name__ == "__main__":
    main()