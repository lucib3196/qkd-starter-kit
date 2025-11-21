# Standard library imports
import math
import threading
import time
import traceback

# Third-party library imports
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from gpiozero.pins.pigpio import PiGPIOFactory
from gpiozero import Device, AngularServo

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
    estimate_marker_pose_multiple
)

# Define Global Variables and Constants
ARUCO_DICT_TYPE = cv2.aruco.DICT_6X6_250
POSITION_DATA = []
TIME_DATA = []
ANGLE_DATA = []
ERROR = 

# Servo Angle Limits
BASE_MIN = -135  # Servo Angle Min
BASE_MAX = 135   # Servo Angle Max

# Initialize Servos
# Pan Servo controls horizontal movement (e.g., GPIO pin 17)
pan_servo = AngularServo(17, min_angle=BASE_MIN, max_angle=BASE_MAX)

# Tilt Servo controls vertical movement (e.g., GPIO pin 18)
upper_servo = AngularServo(27, min_angle=BASE_MIN, max_angle=BASE_MAX)

# Function to calculate distance from center to target
def calc_dist(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return math.hypot(x1 - x2, y1 - y2)

def calc_dist_error(p1, p2, actual):
    return abs(actual - calc_dist(p1, p2)) / actual

# def white_balance_frame(frame):
#     """
#     Perform white balance on an input video frame using the Gray World Assumption.
#
#     Parameters:
#         frame (numpy.ndarray): Input video frame (BGR format).
#
#     Returns:
#         numpy.ndarray: White-balanced video frame.
#     """
#     # Convert frame to float32 for calculations
#     frame = frame.astype(np.float32)
    
#     # Compute the average value for each channel
#     avg_b = np.mean(frame[:, :, 0])  # Blue channel
#     avg_g = np.mean(frame[:, :, 1])  # Green channel
#     avg_r = np.mean(frame[:, :, 2])  # Red channel
    
#     # Compute the overall average
#     avg_gray = (avg_b + avg_g + avg_r) / 3
    
#     # Scaling factors for each channel
#     scale_b = avg_gray / avg_b
#     scale_g = avg_gray / avg_g
#     scale_r = avg_gray / avg_r
    
#     # Apply scaling to each channel
#     frame[:, :, 0] *= scale_b  # Scale blue channel
#     frame[:, :, 1] *= scale_g  # Scale green channel
#     frame[:, :, 2] *= scale_r  # Scale red channel
    
#     # Clip values to valid range [0, 255] and convert back to uint8
#     frame = np.clip(frame, 0, 255).astype(np.uint8)
    
#     return frame

# Function to calculate delta theta
# def calc_desired_theta(x_pos, z_pos):
#     return None

# Draw center point
def draw_center_frame(frame, center):
    cv2.circle(frame, center, 5, (255, 0, 0), 1)

def plot_data():
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 12))

    # First Subplot: Measures distance between center and marker position
    axes[0].set_title('Distance Target to Center')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Position Error')
    line_dist, = axes[0].plot([], [], 'r-', lw=2)  # Initialize an empty red line

    # Second Subplot: Measures changes in angle
    axes[1].set_title('Pan and Tilt Servo Angles')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Angle (deg)')
    pan_angle_line, = axes[1].plot([], [], 'b-', lw=2, label='Pan Angle')   # Blue line for Pan
    tilt_angle_line, = axes[1].plot([], [], 'g-', lw=2, label='Tilt Angle') # Green line for Tilt
    axes[1].legend()


    axes[2].set_title('Pan and Tilt Error ')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Angle (deg)')
    pan_angle_line, = axes[1].plot([], [], 'b-', lw=2, label='Pan Angle Error')   # Blue line for Pan
    tilt_angle_line, = axes[1].plot([], [], 'g-', lw=2, label='Tilt Angle Error') # Green line for Tilt
    axes[2].legend()



    def update_plot(frame):
        # Update the data for the first subplot
        line_dist.set_data(TIME_DATA, POSITION_DATA)
        axes[0].relim()
        axes[0].autoscale_view()

        # Update the data for the second subplot
        pan_angle_line.set_data(TIME_DATA, [angle[0] for angle in ANGLE_DATA])
        tilt_angle_line.set_data(TIME_DATA, [angle[1] for angle in ANGLE_DATA])
        axes[1].relim()
        axes[1].autoscale_view()

        # Update the date for the third plot
        pan_angle_line.set_data(TIME_DATA, [angle[0] for angle in ANGLE_DATA])
        tilt_angle_line.set_data(TIME_DATA, [angle[1] for angle in ANGLE_DATA])
        axes[1].relim()
        axes[1].autoscale_view()

        return line_dist, pan_angle_line, tilt_angle_line

    # Create FuncAnimation for non-blocking updates
    ani = FuncAnimation(fig, update_plot, interval=100)
    plt.show()

def main(src=0):
    start_time = time.time()
    video_stream = None
    video_display = None
    plot_thread = None
    fps_counter = None

    try:
        # Initialize camera and FPS counter
        video_stream = WebcamVideoStreamThreaded(src).start()
        video_display = VideoShow(video_stream.frame).start()
        fps_counter = FPS().start()

        # Start a plot thread
        plot_thread = threading.Thread(target=plot_data, daemon=True)
        plot_thread.start()

        while True:
            current_time = time.time() - start_time

            if video_stream.stopped or video_display.stopped:
                break

            frame = video_stream.frame

            if frame is not None:
                frame_height, frame_width = video_stream.frame_height, video_stream.frame_width
                center = (int(frame_width // 2), int(frame_height // 2))

                # Draw center of the frame
                draw_center_frame(frame, center)

                # Aruco marker detection
                aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT_TYPE)
                parameters = cv2.aruco.DetectorParameters()
                marker_array = find_marker(frame, aruco_dict, parameters)

                if marker_array:
                    for marker, marker_id in marker_array:
                        coord = get_corner_and_center(marker)
                        draw_square_frame(frame, coordinates=coord)
                        center_aruco = coord[-1]  # Last value is the center
                        dist = calc_dist(center, center_aruco)
                        dist_from_camera = estimate_marker_pose_single(
                            frame,
                            marker,
                            marker_id=marker_id,
                            camera_matrix=video_stream.camera_matrix,
                            distortion_coeff=video_stream.camera_dist
                        )
                        cv2.putText(
                            frame,
                            f'Current error: {dist:.2f}',
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 255, 255),
                            3
                        )

                        # Extract x and y positions of the marker center
                        x, y = center_aruco

                        # Calculate offsets from the center of the frame
                        turn_x = float(x - (frame_width / 2))
                        turn_y = float(y - (frame_height / 2))

                        # Convert to percentage offset
                        turn_x /= float(frame_width / 2)
                        turn_y /= float(frame_height / 2)
                        print(f'Turn x {turn_x}')
                        # Scale offset to degrees (Proportional factor)
                        proportional_factor = 40
                        turn_x *= proportional_factor  # Horizontal Field of View scaling
                        turn_y *= (proportional_factor+60)  # Vertical Field of View scaling
                        cam_pan = -turn_x
                        cam_tilt = turn_y

                        print(f"Pan Adjustment: {cam_pan}, Tilt Adjustment: {cam_tilt}")

                        # Clamp Pan/Tilt to 0 to 180 degrees
                        cam_pan = max(0, min(270, cam_pan))
                        cam_tilt = max(0, min(270, cam_tilt))

                        # Update the servos
                        # Adjust by subtracting 90 to center the servo movement
                        pan_servo.angle = int(cam_pan )
                        upper_servo.angle = int(cam_tilt)

                        print(f"Pan Servo Angle: {pan_servo.angle}, Tilt Servo Angle: {upper_servo.angle}")

                        # Append data for plotting
                        TIME_DATA.append(current_time)
                        POSITION_DATA.append(dist)
                        ANGLE_DATA.append((pan_servo.angle, upper_servo.angle))

                        # Maintain a maximum of 100 data points
                        if len(TIME_DATA) > 100:
                            TIME_DATA.pop(0)
                            POSITION_DATA.pop(0)
                            ANGLE_DATA.pop(0)

                # Display FPS on the frame
                frame_with_fps = putIterationsPerSec(frame, fps_counter.fps())
                video_display.frame = frame_with_fps

            # Update FPS counter
            fps_counter.update()
        video_stream.video_thread.join()
        video_display.video_thread.join()
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
    finally:
        # Ensure threads are stopped and resources are cleaned up
        if video_stream:
            video_stream.stop()
        if video_display:
            video_display.stop()
        cv2.destroyAllWindows()

        # # Join threads if they were started
        # video_stream.video_thread.join()
        # video_display.video_thread.join()

if __name__ == '__main__':
    main()
