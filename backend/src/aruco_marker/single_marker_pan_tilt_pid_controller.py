# Standard library imports
import math
import threading
import time
import traceback
import logging
import os
# Third-party library imports
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from gpiozero.pins.pigpio import PiGPIOFactory
from gpiozero import Device, AngularServo
from gpiozero.tools import sin_values
from signal import pause
from datetime import date,datetime
# Configure gpiozero to use the PiGPIOFactory
Device.pin_factory = PiGPIOFactory()
# Local imports
from . import (
    WebcamVideoStreamThreaded,
    FPS,
    putIterationsPerSec,
    VideoShow
)
from .utils import (
    get_corner_and_center,
    find_marker,
    draw_square_frame,
    draw_id,
    estimate_marker_pose_single,
    correct_white_balance, 
    draw_center_frame,
    calc_dist
)
from ..PID import PIDController

# Logging Configuration
# This sets the logging level, determining the lowest level of messages that will be logged.
# 
# Levels (from lowest to highest severity):
# - DEBUG: Detailed information, typically for diagnosing problems.
# - INFO: General information about program execution.
# - WARNING: Indicates something unexpected or potential issues.
# - ERROR: A serious issue that prevents a part of the program from functioning.
# - CRITICAL: A very severe error indicating the program may not continue running.
#
# If set to DEBUG, all messages (DEBUG, INFO, WARNING, ERROR, CRITICAL) will be shown.
# If set to INFO, only INFO and more severe messages (INFO, WARNING, ERROR, CRITICAL) will be shown.
# If set to WARNING, only WARNING and more severe messages (WARNING, ERROR, CRITICAL) will be shown, and so on.
current_date = datetime.now()
base_dir = os.path.dirname(__file__)
# Define the logs directory path
log_path = os.path.join(base_dir, 'logs')
# Create the logs directory if it doesn't exist
if not os.path.exists(log_path):
    os.mkdir(log_path)
# Define the log file path
log_file = os.path.join(log_path, f'single_marker_pid_log_{current_date}.log')
# Configure logging
logging.basicConfig(
    filename=log_file,            # Full path to the log file
    level=logging.INFO,           # Minimum log level to save (e.g., INFO)
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
)
logging.info("Logging system initialized.")

## Define processes 
### If these are set to true then we will be showing plots and or video feed
### We may want to toggle based on performance issues, ie the more things that are happening the more processing power we need
SHOW_PLOTS = True
SHOW_VIDEO_FEED = True

# Define Global Variables and Constants
MASTER_DATA = np.array([])
current_date = date.today()
ARUCO_DICT_TYPE = cv2.aruco.DICT_6X6_250
FRAME_SKIP = 5
frame_count = 0



# Servo Angle Limits
BASE_MIN = -135  # Servo Angle Min
BASE_MAX = 135   # Servo Angle Max

# Initialize servo controls 
# Pan Servo controls horizontal movement 
pan_servo_pin = 17
pan_servo = AngularServo(pan_servo_pin, min_angle=BASE_MIN, max_angle=BASE_MAX)

# Tilt Servo controls vertical movement 
tilt_servo_pin = 27
tilt_servo = AngularServo(tilt_servo_pin, min_angle=BASE_MIN, max_angle=BASE_MAX)

## Define the PID Controllers for both the pan and tilt this needs to be adjusted as needed
pan_controller  = PIDController(Kp=30 ,Ki =0 ,Kd =0)
tilt_controller = PIDController(Kp =0.5 ,Ki =0 ,Kd =0 )

msg_n = 25
message_str = f"""
Pan and Tilt Mechanism: 
Current Date:  {current_date}\n
{'-'*msg_n + 'Running Parameters'+ '-'*msg_n}\n
Pan Servo: Pin {pan_servo_pin} PID Values:  {pan_controller.get_specs()} Angle Range {BASE_MIN}-{BASE_MAX} Current Angle {tilt_servo.angle} \n 
Tilt Servo: Pin {tilt_servo_pin} PID Values:  {tilt_controller.get_specs()} Angle Range {BASE_MIN}-{BASE_MAX} Current Angle {pan_servo.angle} \n 
Threading Events (Boolean) \n Show Plots: {SHOW_PLOTS} \n Show Feed {SHOW_VIDEO_FEED}
{'-'*msg_n}\n
"""
logging.info(message_str)

# def scout_mode(stop_event):
    # print('Looking...')
    # try:
    #     # Assign sine wave generator to the servo source
    #     pan_servo.source = sin_values()
    #     pan_servo.source_delay = 0.001  # Adjust delay between source updates
    #     tilt_servo.source = sin_values()
    #     tilt_servo.source_delay = 0.001
    #     while not stop_event.is_set():
    #         print('Scanning...')
            
            
    #     print('Found Marker. Locking In.')

    # except Exception as e:
    #     print(f"Error in scout_mode: {e}")

    # finally:
    #     # Stop the servo and clean up
    #     pan_servo.detach()
    #     print("Servo stopped.")
    # pass
    
    
    
def plot_data(position_data, time_data, angle_data, error_data, current_index):
    """
    Plots the data using Matplotlib in a (usually) separate thread.
    Now 'position_data', 'time_data', etc. are lists instead of NumPy arrays.
    We ignore 'current_index' because with lists, the length is our "index".
    """

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 12))

    # 1) Distance vs. Time
    axes[0].set_title('Distance Target to Center')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Position Error')
    line_dist, = axes[0].plot([], [], 'r-', lw=2)

    # 2) Pan and Tilt Angles vs. Time
    axes[1].set_title('Pan and Tilt Servo Angles')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Angle (deg)')
    pan_angle_line, = axes[1].plot([], [], 'b-', lw=2, label='Pan Angle')
    tilt_angle_line, = axes[1].plot([], [], 'g-', lw=2, label='Tilt Angle')
    axes[1].legend()

    # 3) Pan and Tilt Error vs. Time
    axes[2].set_title('Pan and Tilt Error (%)')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Error (%)')
    pan_angle_line_err, = axes[2].plot([], [], 'b-', lw=2, label='Pan Angle Error')
    tilt_angle_line_err, = axes[2].plot([], [], 'g-', lw=2, label='Tilt Angle Error')
    axes[2].legend()

    def update_plot(frame):
        """
        Updates the lines with the latest data each frame (interval=100ms).
        Convert lists -> arrays for plotting.
        We ignore 'current_index' because the list length is used instead.
        """

        # Convert lists to arrays
        if len(time_data) == 0:
            # If no data yet, nothing to plot
            return (line_dist, pan_angle_line, tilt_angle_line,
                    pan_angle_line_err, tilt_angle_line_err)

        x_data = np.array(time_data)
        pos_data = np.array(position_data)
        ang_data = np.array(angle_data)   # shape: (N, 2)
        err_data = np.array(error_data)   # shape: (N, 2)

        print(ang_data)

        # 1) Distance
        line_dist.set_data(x_data, pos_data)
        axes[0].relim()
        axes[0].autoscale_view()

        # 2) Servo Angles
        pan_angle_line.set_data(x_data, ang_data[:, 0])
        tilt_angle_line.set_data(x_data, ang_data[:, 1])
        axes[1].relim()
        axes[1].autoscale_view()

        # 3) Pan/Tilt Errors (in %)
        
        pan_angle_line_err.set_data(x_data, np.abs(err_data[:, 0]) * 100)
        tilt_angle_line_err.set_data(x_data, np.abs(err_data[:, 1]) * 100)
        axes[2].relim()
        axes[2].autoscale_view()

        return (line_dist, pan_angle_line, tilt_angle_line,
                pan_angle_line_err, tilt_angle_line_err)

    ani = FuncAnimation(fig, update_plot, interval=100)
    plt.show()

def main(src=0):
    # --- 1) Use lists instead of NumPy arrays ---
    POSITION_DATA = []
    TIME_DATA = []
    ANGLE_DATA = []
    ERROR = []

    # We'll keep a maximum of 50 data points.
    MAX_POINTS = 50

    start_time = time.time()
    frame_count = 0
    index = 0  # We'll keep this if you want, but won't really use it for lists.

    try:
        # Initialize camera and display threads (stubs or actual)
        video_stream = WebcamVideoStreamThreaded(src).start()
        video_display = VideoShow(video_stream.frame).start()
        fps_counter = FPS().start()

        # 2) If we show plots, start them in a separate thread
        if SHOW_PLOTS:
            plot_thread = threading.Thread(
                target=plot_data,
                args=(POSITION_DATA, TIME_DATA, ANGLE_DATA, ERROR, index),
                daemon=True
            )
            plot_thread.start()

        # 3) Main loop: Acquire data
        while True:
            frame_count += 1
            current_time = time.time() - start_time

            if video_stream.stopped or video_display.stopped:
                break

            frame = video_stream.frame

            # Skip frames
            if frame_count % FRAME_SKIP != 0:
                continue

            if frame is not None:
                # Marker detection, etc.
                frame_height, frame_width = video_stream.frame_height, video_stream.frame_width
                center = (int(frame_width // 2), int(frame_height // 2))

                draw_center_frame(frame, center)
                aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
                parameters = cv2.aruco.DetectorParameters()
                marker_array = find_marker(frame, aruco_dict, parameters)

                if marker_array:
                    for marker, marker_id in marker_array:
                        coord = get_corner_and_center(marker)
                        draw_square_frame(frame, coordinates=coord)
                        center_aruco = coord[-1]
                        total_dist = calc_dist(center, center_aruco)

                        dist_from_camera = estimate_marker_pose_single(
                            frame,
                            marker,
                            marker_id=marker_id,
                            camera_matrix=video_stream.camera_matrix,
                            distortion_coeff=video_stream.camera_dist
                        )

                        x, y = center_aruco
                        err_x = float(x - center[0]) /center[0]
                        err_y = float(y - center[1]/ center[1])


                        turn_x = pan_controller.compute(err_x, current_time)
                        turn_y = tilt_controller.compute(err_y, current_time)

                        cam_pan = -turn_x
                        cam_tilt = -turn_y

                        print(f'This is the current angle {pan_servo.angle}')
                        print(f'This is how many degrees to change {cam_pan}')

                        cam_tilt = max(0, min(BASE_MAX, cam_tilt))
                        print(f'This is the new pan angle {cam_pan}')

                        pan_servo.angle = int(cam_pan)
                        tilt_servo.angle = int(cam_tilt)

                        # Logging
                        logging.info(f"Time: {current_time:.2f}s | Error X: {err_x:.2f}, Error Y: {err_y:.2f}")
                        logging.info(f"Pan Angle: {pan_servo.angle}, Tilt Angle: {tilt_servo.angle}")

                        # 4) Append data for plotting (as lists)
                        TIME_DATA.append(current_time)
                        POSITION_DATA.append(total_dist)
                        ANGLE_DATA.append([pan_servo.angle, tilt_servo.angle])  # or (pan_servo.angle, tilt_servo.angle)
                        ERROR.append([err_x, err_y])

                        # 5) Keep only the last MAX_POINTS
                        if len(TIME_DATA) > MAX_POINTS:
                            TIME_DATA.pop(0)
                            POSITION_DATA.pop(0)
                            ANGLE_DATA.pop(0)
                            ERROR.pop(0)

                # Display FPS
                frame_with_fps = putIterationsPerSec(frame, fps_counter.fps())
                video_display.frame = frame_with_fps

            # Update FPS counter
            fps_counter.update()

        # Stop camera threads
        video_stream.video_thread.join()
        video_display.video_thread.join()
        plot_thread.join()

    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()

    finally:
        # Clean up
        if video_stream:
            video_stream.stop()
        if video_display:
            video_display.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()