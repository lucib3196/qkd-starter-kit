from threading import Thread
import cv2
from src.fps.fps import putIterationsPerSec, FPS
from cv2.typing import MatLike


class VideoGet:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()

    def stop(self):
        self.stopped = True
        return self


class VideoShow:
    def __init__(self, frame: MatLike | None = None):
        self.frame = frame
        self.stopped = False

    def start(self):
        Thread(target=self.show, args=()).start()
        return self

    def show(self):
        while not self.stopped:
            if self.frame is not None:
                cv2.imshow("Video", self.frame)
                if cv2.waitKey(1) == ord("q"):
                    self.stopped = True

    def stop(self):
        self.stopped = True
        cv2.destroyAllWindows()


def threaded_video_get(source=0):
    """
    Dedicated thread for grabbing video frames with VideoGet object.
    Main thread shows video frames.
    """

    video_getter = VideoGet(source).start()
    video_show = VideoShow(video_getter.frame).start()

    counter = FPS().start()
    while True:
        if video_show.stopped or video_getter.stopped:
            video_getter.stop()
            video_show.stop()
            break

        frame = video_getter.frame
        putIterationsPerSec(frame, counter.fps())
        video_show.frame = frame
        counter.update()


if __name__ == "__main__":
    threaded_video_get()


