from src.config.settings import servo_settings

from time import sleep
from gpiozero import AngularServo, Device
from gpiozero.pins.pigpio import PiGPIOFactory

Device.pin_factory = PiGPIOFactory()


# Load servo settings
base_servo_settings = servo_settings.base_servo
tilt_servo_settings = servo_settings.tilt_servo

base_servo = AngularServo(
    pin=base_servo_settings.pin,
    min_angle=base_servo_settings.min_angle,
    max_angle=base_servo_settings.max_angle,
    min_pulse_width=base_servo_settings.min_pulse_width,
    max_pulse_width=base_servo_settings.max_pulse_width,
)

tilt_servo = AngularServo(
    pin=tilt_servo_settings.pin,
    min_angle=tilt_servo_settings.min_angle,
    max_angle=tilt_servo_settings.max_angle,
    min_pulse_width=tilt_servo_settings.min_pulse_width,
    max_pulse_width=tilt_servo_settings.max_pulse_width,
)
