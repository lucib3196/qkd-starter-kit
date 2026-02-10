from . import base_servo, tilt_servo
from time import sleep


# Not specific to a servo just meant for a test
angle_sweep = [0, 90, 180]

try:
    # Initialize the servo at a starting position
    base_servo.angle = 0
    tilt_servo.angle = 0
    while True:
        for angle in angle_sweep:
            base_servo.angle = angle
            print(f"Base servo set to {angle}°")
            sleep(2)
        for angle in angle_sweep:
            tilt_servo.angle = angle
            print(f"Tilt servo set to {angle}°")
            sleep(2)
except KeyboardInterrupt:
    print("\nTest interrupted. Exiting...")
