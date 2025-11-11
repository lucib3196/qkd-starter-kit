from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import cv2
import asyncio
import uvicorn
from fastapi.responses import HTMLResponse

app = FastAPI()


async def frame_generator():
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break
        # Convert the np array to a jpg
        _, buffer = cv2.imencode(".jpg", frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )
        await asyncio.sleep(1/10)
    cap.release()


@app.get("/stream")
async def stream_video():
    return StreamingResponse(
        frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/", response_class=HTMLResponse)
def index():
    """Main page displaying the live Pi Camera stream."""
    html_content = """
    <!DOCTYPE html>
    <html>
        <head>
            <title>USB Camera Stream</title>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                body {
                    margin: 0;
                    padding: 0;
                    background-color: #000;
                    color: #fff;
                    font-family: Arial, sans-serif;
                    text-align: center;
                }
                h1 {
                    margin-top: 20px;
                    font-size: 1.8em;
                    color: #00ff90;
                }
                .container {
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    min-height: 80vh;
                }
                img {
                    max-width: 100%;
                    height: auto;
                    border: 3px solid #00ff90;
                    border-radius: 8px;
                    box-shadow: 0 0 20px #00ff90;
                }
            </style>
        </head>
        <body>
            <h1>Raspberry Pi Live Camera Stream (USB)</h1>
            <div class="container">
                <img src="/stream" alt="Camera Stream" />
            </div>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    # Make sure pigpiod is running:
    # sudo systemctl start pigpiod
    uvicorn.run(
        "src.camera.camera_usb_streaming:app",  # module:app name
        host="0.0.0.0",
        port=8000,
        reload=True,  # hot reload during development
    )
