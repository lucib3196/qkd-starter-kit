from pathlib import Path
from threading import Thread
from contextlib import asynccontextmanager
import time
import cv2
import uvicorn

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse

from src.fps.fps import putIterationsPerSec, FPS
from src.camera.camera_threaded.get_video_threaded import VideoGet

from .utils import (
    detect_markers,
    get_marker_corners,
    get_marker_center,
    estimate_pose_and_transformation_matrix,
)



@asynccontextmanager
async def lifespan(app: FastAPI):
    global video_getter, counter, processing_thread, running

    running = True

    # Start camera thread
    video_getter = VideoGet(0).start()
    counter = FPS().start()

    # Processing thread
    processing_thread = Thread(target=processing_loop, daemon=True)
    processing_thread.start()

    yield  # App is running!

    # Shutdown
    running = False
    if video_getter:
        video_getter.stop()


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

            time.sleep(0.01)

    return StreamingResponse(
        generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


def processing_loop():
    global running, video_getter, counter

    while running:
        if video_getter.stopped:
            running = False
            break

        frame = video_getter.frame
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
        "src.camera.camera_threaded.threaded_stream:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
