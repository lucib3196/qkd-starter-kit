from gpiozero import Servo
from gpiozero.tools import sin_values
from signal import pause
from time import sleep
from gpiozero.pins.pigpio import PiGPIOFactory
from gpiozero import Device
Device.pin_factory = PiGPIOFactory()


servo = Servo(27, min_pulse_width=0.5/1000, max_pulse_width=2.5/1000)

while True:
    # servo.min()
    # print("Min")
    # sleep(2)
    servo.mid()
    print("Mid")
    # sleep(2)
    # print("Max")
    # servo.max()
    # sleep(2)