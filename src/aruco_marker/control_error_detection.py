from . import WebcamVideoStreamThreaded, FPS, putIterationsPerSec, VideoShow
import traceback
import numpy as np
import cv2 
from .utils import get_corner_and_center, find_marker, draw_square_frame
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import time

# Constants 
aruco_dict_type = cv2.aruco.DICT_6X6_250

# Global Variables 
position_data = []
time_data = []
running = True

def update_plot(i):
    '''Function to update the plot'''
    plt.cla()
    plt.plot(time_data, position_data, label='Position vs. Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Position')
    plt.legend(loc='upper left')

def calculate_positions_error(p1,p2):
    x1,y1 = p1
    x2,y2 = p2
    dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
    return dist
def draw_center_frame(frame, center):
    cv2.circle(frame, center, 5, (255, 0, 0), 1)

def main(src=0):
    start_time = time.time()
    try:
        # Initialize camera and FPS counter
        video_stream = WebcamVideoStreamThreaded(src).start()
        video_display = VideoShow(video_stream.frame).start()
        fps_counter = FPS().start()

        time_data = []  # Initialize time data list
        position_data = []  # Initialize position data list

        # Start a separate thread for real-time plotting
        def plot_thread():
            fig, ax = plt.subplots()
            ax.set_title("Real-Time Error Plot")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Position Error")
            line, = ax.plot([], [], 'r-', lw=2)  # Initialize an empty line

            def update_plot(frame):
                # Update plot data
                line.set_data(time_data, position_data)
                ax.relim()
                ax.autoscale_view()
                return line,

            # FuncAnimation for non-blocking updates
            ani = FuncAnimation(fig, update_plot, interval=100)
            plt.show()

        # Start the plot in a separate thread
        plot_thread = threading.Thread(target=plot_thread, daemon=True)
        plot_thread.start()

        while True:
            # Calculate elapsed time
            current_time = time.time() - start_time

            # Stop the loop if video stream/display is stopped
            if video_stream.stopped or video_display.stopped:
                video_stream.stop()
                video_display.stop()
                break

            frame = video_stream.frame

            if frame is not None:
                height, width = video_stream.frame_height, video_stream.frame_width
                center = (int(width // 2), int(height // 2))

                # Draw center of the frame
                draw_center_frame(frame, center)

                # Aruco marker detection
                aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
                parameters = cv2.aruco.DetectorParameters()
                marker_array = find_marker(frame, aruco_dict, parameters)

                if marker_array:
                    for marker, id in marker_array:
                        coord = get_corner_and_center(marker)
                        draw_square_frame(frame, coordinates=coord)

                        center_aruco = coord[-1]  # Last value is the center
                        error = calculate_positions_error(center, center_aruco)
                        cv2.putText(
                            frame,
                            f'Current error: {error:.2f}',
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 255, 255),
                            3
                        )
                        # Update data lists
                        time_data.append(current_time)
                        position_data.append(error)

                        # Maintain only the last 100 entries
                        if len(time_data) > 100:
                            time_data.pop(0)
                            position_data.pop(0)

                # Display FPS on the frame
                frame_with_fps = putIterationsPerSec(frame, fps_counter.fps())
                video_display.frame = frame_with_fps

            # Update FPS counter
            fps_counter.update()

        # Ensure threads are stopped and joined
        video_stream.video_thread.join()
        video_display.video_thread.join()
    except Exception as e:
        print(f"An error occurred: {e}")
        traceback.print_exc()
    finally:
        # Ensure resources are cleaned up
        video_stream.stop()
        video_display.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()