class Picamera2:
    def __init__(self):
        print("Mock Picamera2 loaded (Windows development mode)")

    def start(self):
        print("Mock camera start")

    def capture_array(self):
        print("Mock capture")
        return None

    def create_video_configuration(**kwargs):
        return None

    def configure(self,*args,**kwargs,):
        return None
    def capture_file(self,**args):
        return ""