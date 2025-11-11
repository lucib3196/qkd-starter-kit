from gpiozero import AngularServo
from time import sleep
# from gpiozero import Device
# from gpiozero.pins.pigpio import PiGPIOFactory

# Device.pin_factory = PiGPIOFactory()
"""
This script demonstrates basic control of two servos using the GPIO Zero library.

Servos:
- Base Servo: Controls the base and oscillates between -90° and 90°.
- Upper Servo: Controls the upper section and oscillates between -45° and 45°.

How it works:
1. The script initializes two AngularServo objects, each tied to specific GPIO pins.
2. In an infinite loop, it moves the base servo and upper servo to their minimum and maximum angles with a delay.
3. Use this script to verify servo connections and test basic movement.

Note:
- Ensure the Raspberry Pi GPIO pins are properly connected to the servo's signal pin.
- Provide external power to the servos if needed, as GPIO pins cannot supply sufficient current for servos under load.
- If running this script over SSH, use `sudo` to ensure GPIO permissions.

Requirements:
- GPIO Zero library: Install it with `pip install gpiozero`
"""
from gpiozero.pins.pigpio import PiGPIOFactory
from gpiozero import Device
Device.pin_factory = PiGPIOFactory()
# Constant
BASE_MIN = 0
BASE_MAX = 180

BASE_MIN_TOP = -90
BASE_MAX_TOP = 90
# # Initialize the base servo on GPIO pin 17 with a range from -90 to 90 degrees
base_servo = AngularServo(17,min_angle=BASE_MIN,max_angle = BASE_MAX, min_pulse_width=0.5/1000, max_pulse_width=2.5/1000)

# Initialize the upper servo on GPIO pin 27 with a range from -45 to 45 degrees
upper_servo = AngularServo(27,min_angle=-90,max_angle = 90, min_pulse_width=0.5/1000, max_pulse_width=2.5/1000)

# Infinite loop to test the servos
try:
    while True:
        # # Move the base servo to its minimum angle (-90 degrees)
        base_servo.angle = BASE_MIN
        print(f"Base servo set to {BASE_MIN} degrees")
        sleep(2)  # Wait for 2 seconds

        # Move the base servo to its maximum angle (90 degrees)
        base_servo.angle = BASE_MAX
        print(f"Base servo set to {BASE_MAX} degrees")
        sleep(2)  # Wait for 2 seconds

        # Move the upper servo to its minimum angle (-45 degrees)
        upper_servo.angle = BASE_MIN_TOP+45
        print(f"Upper servo set to {upper_servo.angle} degrees")
        sleep(2)  # Wait for 2 seconds

        # Move the upper servo to its maximum angle (45 degrees)
        upper_servo.angle = BASE_MAX_TOP-30
        print(f"Upper servo set to {upper_servo.angle} degrees")
        sleep(2)  # Wait for 2 seconds

except KeyboardInterrupt:
    # Handle exit gracefully on Ctrl+C
    print("\nTest interrupted. Exiting...")

