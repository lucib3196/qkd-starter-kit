from gpiozero import Servo
from time import sleep
from src.config import servo_settings

bs_settings = servo_settings.base_servo

servo = Servo(
    bs_settings.pin,
    min_pulse_width=bs_settings.min_pulse_width,
    max_pulse_width=bs_settings.max_pulse_width,
)

while True:
    servo.min()
    print("Min")
    sleep(2)
    servo.mid()
    print("Mid")
    sleep(2)
    print("Max")
    servo.max()
    sleep(2)
