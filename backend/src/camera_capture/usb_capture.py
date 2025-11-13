import cv2
from pathlib import Path

def usb_capture(path: str | Path):
    """
    Capture images from a USB camera feed.
    Press 's' to save an image, or 'Esc' to exit.
    """
    path = Path(path)
    if not path.exists():
        raise ValueError(f"Path {path} does not exist")

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Error: Could not open camera.")
        return

    print("Camera started. Press 's' to save an image, 'Esc' to quit.")
    num = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            print("Failed to grab frame. Exiting...")
            break

        cv2.imshow("USB Camera Feed", frame)
        k = cv2.waitKey(5) & 0xFF

        if k == 27:  # ESC key
            print("\nExiting...")
            break
        elif k == ord("s"):
            image_path = path / f"img_{num}.png"
            cv2.imwrite(str(image_path), frame)
            print(f"✅ Image saved: {image_path}")
            num += 1

    cam.release()
    cv2.destroyAllWindows()
    print("Camera feed closed.")

if __name__ == "__main__":
    path = Path(__file__).parent
    usb_capture(path)



if __name__ == "__main__":
    path = Path(__file__).parent
    usb_capture(path)
