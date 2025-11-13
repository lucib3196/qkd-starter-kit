import cv2
import cv2.aruco as aruco
import time 
from . import WebcamVideoStreamThreaded, FPS, putIterationsPerSec,VideoShow

# Constants 
aruco_dict_type = cv2.aruco.DICT_6X6_250

def get_marker(frame, aruco_dict, parameters):
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    corners, ids, rejected = detector.detectMarkers(frame)
    if ids is not None: 
        ids_sorted = []
        for id in ids:
            ids_sorted.append(id[0])
    else:
        ids_sorted = ids
    return corners, ids_sorted

def get_marker_coord(markers, ids, point = 0):
    """Get the coordinate points of a given marker

    Args:
        markers (_type_): the marker coordinates
        ids (_type_): A list of ids
        point (int, optional): _description_. Defaults to 0 which corresponds to first corner, 1 corresponds to 2nd corrner etc
        which goes top left, top right, bottom right and bottom left
    """
    arr = []
    for marker in markers:
        arr.append([int(marker[0][point][0]),int(marker[0][point][1])])
    return arr, ids
        
def get_marker_center(marker):
    # Get the corner to calculate the center
    top_left,ids = get_marker_coord(marker,1,point=0)
    top_right,ids = get_marker_coord(marker, 1, point=1)
    bottom_right,ids = get_marker_coord(marker,1, point = 2)
    bottom_left = get_marker_coord(marker, 1, point=4)
    if top_left:
        center_X=(top_left[0][0]+top_right[0][0]+top_left[0][0]+top_right[0][0])*0.25
        center_Y=(top_left[0][1]+top_right[0][1]+top_left[0][1]+top_right[0][1])*0.25
        marker_center=[[int(center_X),int(center_Y)]]
    else:
        marker_center=[[0,0]]
    return marker_center

def draw_corners_circ(frame, corners):
    """Draws a circle on the corner of a marker specifically the top left one

    Args:
        frame (_type_): _description_
        corners (_type_): _description_
    """
    for corner in corners:
        cv2.circle(frame,(corner[0],corner[1]),10,(0,255,0),thickness=-1)       
    
        
    
    
def main(source=0):
    try:
        # Initialize the video stream and display threads
        video_stream = WebcamVideoStreamThreaded(source).start()
        video_display = VideoShow(video_stream.frame).start()
        fps_counter = FPS().start()

        while True:
            # Stop threads if either thread signals to stop
            if video_stream.stopped or video_display.stopped:
                break

            frame = video_stream.frame
            if frame is not None:
                # Logic to display stuff
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # ArUco marker detection
                aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
                parameters = cv2.aruco.DetectorParameters()

                markers, ids_sorted = get_marker(gray, aruco_dict, parameters)
                marker_arr,ids = get_marker_coord(markers, ids_sorted)
                if (marker_arr and ids):
                    for marker in marker_arr:
                        center = get_marker_center(marker)
                        print(center)

                # Add FPS to the frame
                frame_with_fps = putIterationsPerSec(frame, fps_counter.fps())
                video_display.frame = frame_with_fps

                # Update FPS counter
                fps_counter.update()

            # Exit the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        # Log or print the exception for debugging
        print(f"An error occurred: {e}")

    finally:
        # Clean up resources
        if 'video_stream' in locals() and video_stream is not None:
            video_stream.stop()
        if 'video_display' in locals() and video_display is not None:
            video_display.stop()
        cv2.destroyAllWindows()
        print("Resources have been released. Exiting gracefully.")


    
    
if __name__ == '__main__':
    main()