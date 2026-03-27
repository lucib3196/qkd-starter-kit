
from time import sleep
from pathlib import Path

try:
    from picamera2 import Picamera2 # type: ignore
except ImportError:
    from src.mocking.mock_picamera import Picamera2
    print("Mocking Picamera")


def picamera_capture(path: str | Path):
    """
    Capture images interactively using the Raspberry Pi Camera (Picamera2).

    Press:
      - 'c' to capture and save an image
      - 'q' to quit the capture loop
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    picam2 = Picamera2()

    # Configure camera need to figure out what exactly this does
    config = picam2.create_video_configuration(
        main={"size": (1280, 720), "format": "XBGR8888"} # type: ignore
    )
    picam2.configure(config)
    picam2.start()

    print("📷 Camera started. Adjusting exposure...")
    sleep(2)  # Allow camera to stabilize

    num = 0
    try:
        while True:
            key = input("Press 'c' to capture an image or 'q' to quit: ").strip().lower()

            if key == "c":
                filename = f"image_{num}.jpg"
                save_path = path / filename
                picam2.capture_file(str(save_path)) # type: ignore
                print(f"✅ Image saved: {save_path}")
                num += 1

            elif key == "q":
                print(" Exiting capture loop...")
                break

            else:
                print(" Invalid input. Press 'c' to capture or 'q' to quit.")

    finally:
        picam2.stop()
        print("Camera stopped successfully.")


if __name__ == "__main__":
    path = Path(__file__).parent / "captures"
    picamera_capture(path)
