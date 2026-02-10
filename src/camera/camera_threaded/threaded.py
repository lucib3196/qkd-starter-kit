from threading import Thread
import cv2
import numpy as np

class ThreadedStream:
    """
    Threaded video capture that works on:
    - Raspberry Pi (real camera)
    - Windows (mock mode)
    """

    def __init__(self, src=0, mock=False):
        self.stopped = False
        self.mock = mock

        if mock:
            # Create a fake frame for Windows development
            self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
            self.grabbed = True
        else:
            self.stream = cv2.VideoCapture(src)
            if not self.stream.isOpened():
                print("Error: Unable to access camera.")
                self.mock = True
                self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
                self.grabbed = True
            else:
                self.grabbed, self.frame = self.stream.read()

    def start(self):
        """Start both the reader and display threads."""
        Thread(target=self.update, args=(), daemon=True).start()
        Thread(target=self.show, args=(), daemon=True).start()
        return self

    def update(self):
        """Continuously grab frames."""
        while not self.stopped:
            if self.mock:
                # Just generate a black image or test pattern
                self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
            else:
                self.grabbed, self.frame = self.stream.read()
                if not self.grabbed:
                    print("Camera stream ended.")
                    self.stop()
                    break

    def show(self):
        """Display frames continuously."""
        while not self.stopped:
            if self.frame is not None:
                cv2.imshow("Video", self.frame)

            # Quit with q
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop()

    def stop(self):
        """Safely stops threads."""
        self.stopped = True
        if not self.mock:
            self.stream.release()
        cv2.destroyAllWindows()


        
        
if __name__ == "__main__":
    """
    Main function to execute the threaded camera test.

    - Starts a threaded video stream.
    - Displays the live video with FPS overlay.
    - Provides a basic test to verify camera functionality and threading.
    """
    stream  = ThreadedStream()
    stream.start()
    stream.show()
