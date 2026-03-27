from pathlib import Path
from threading import Thread
from contextlib import asynccontextmanager
import time
import cv2
import uvicorn
from src.camera.camera_threaded import CalibrationSettings
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
import numpy as np
from src.fps.fps import putIterationsPerSec, FPS
from src.camera.camera_threaded.get_video_threaded import VideoGetCalibrated
from src.controls.pid import PIDController
from .utils import detect_markers, track_and_render_marker


# Create the pid and tilt controllers
pan_controller = PIDController(Kp=1, Ki=0.0, Kd=1)
tilt_controller = PIDController(Kp=1, Ki=0, Kd=2)

# Define the aruco marker data
aruco_dict_type = cv2.aruco.DICT_6X6_250
aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
parameters = cv2.aruco.DetectorParameters()

# Define the camera settings
matrix = Path(
    r"src\camera\camera_calibration\calibration\pical20284096\cameraMatrix.pkl"
).resolve()
distortion = Path(
    r"src\camera\camera_calibration\calibration\pical20284096\distortion.pkl"
).resolve()
# Load up calibration settings
settings = CalibrationSettings(
    camera_matrix_path=matrix, camera_distortion_path=distortion
)


# Servo angle limits
PAN_MIN = -135
PAN_MAX = 135
TILT_MIN = -135
TILT_MAX = 135


@asynccontextmanager
async def lifespan(app: FastAPI):
    global video_getter, counter, processing_thread, running, aruco_dict, parameters, start_time, data_array
    # Data columns: Time, Pan_Error (deg), Pan_Angle, Tilt_Error (deg), Tilt_Angle
    data_array = np.empty((0, 5))
    running = True

    # Start camera thread
    video_getter = VideoGetCalibrated(0, settings).start()
    counter = FPS().start()
    start_time = time.time()
    print("Camera started")

    # Processing thread
    processing_thread = Thread(target=processing_loop, daemon=True)
    processing_thread.start()

    yield  # App is running!

    # Shutdown
    running = False
    if video_getter:
        video_getter.stop()

    processing_thread.join(timeout=2)
    np.savetxt(
        "data_main.csv",
        data_array,
        delimiter=",",
        header="Time,Pan_Error,Pan_Angle,Tilt_Error,Tilt_Angle",
        comments="",
        fmt="%.5f",
    )


app = FastAPI(lifespan=lifespan)


@app.get("/stream")
async def stream_video():
    def generator():
        while not video_getter.stopped:
            frame = video_getter.frame
            if frame is None:
                time.sleep(0.01)
                continue

            ret, buffer = cv2.imencode(".jpg", frame)
            if not ret:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
            )

            time.sleep(0.05)

    return StreamingResponse(
        generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


def processing_loop():
    global running, video_getter, counter, start_time, data_array

    current_pan = 0
    current_tilt = 0

    while running:
        elapsed_time = time.time() - start_time
        if video_getter.stopped or not running:
            running = False
            break

        frame = video_getter.frame
        if frame is not None:
            markers = detect_markers(frame, aruco_dict, parameters)
            if markers:
                for m in markers:
                    # Returns the transformation matrix
                    T = track_and_render_marker(
                        frame,
                        m[0],
                        m[1],
                        video_getter.camera_matrix,
                        video_getter.camera_dist,
                    )
                    # Gets the orientation matrix
                    R = T[:-1, -1]
                    x, y, z = R

                    pan_error_rad = np.arctan2(x, z)
                    pan_error_deg = np.degrees(pan_error_rad)

                    tilt_error_rad = np.arctan2(y, z)
                    tilt_error_deg = np.degrees(tilt_error_rad)

                    pan_correction = pan_controller.compute(pan_error_deg, elapsed_time)
                    tilt_correction = tilt_controller.compute(
                        tilt_error_deg, elapsed_time
                    )

                    new_pan = current_pan - pan_correction
                    new_tilt = current_tilt - tilt_correction

                    # Clip angles to valid range
                    current_pan = np.clip(new_pan, PAN_MIN, PAN_MAX)
                    current_tilt = np.clip(new_tilt, TILT_MIN, TILT_MAX)

                    data_entry = np.array(
                        [
                            [
                                elapsed_time,
                                np.abs(pan_error_deg),
                                new_pan,
                                np.abs(tilt_error_deg),
                                new_tilt,
                            ]
                        ]
                    )
                    data_array = np.vstack((data_array, data_entry))

            putIterationsPerSec(frame, counter.fps())
            counter.update()

            time.sleep(0.01)


@app.get("/", response_class=HTMLResponse)
def index():
    html_path = (
        Path("src/camera/camera_threaded/video_stream.html").resolve().read_text()
    )
    return HTMLResponse(content=html_path)


if __name__ == "__main__":
    uvicorn.run(
        "src.aruco_marker.base_stream:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
