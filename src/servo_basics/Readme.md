## 🧩 Pin Layout and Servo Configuration Guide

For reference on **GPIO pin layouts**, see the official GPIO Zero documentation:
🔗 [GPIO Zero Pin Layouts](https://gpiozero.readthedocs.io/en/stable/recipes.html)

For **basic servo control examples**, refer to:
🔗 [GPIO Zero Servo Recipes](https://gpiozero.readthedocs.io/en/stable/recipes.html#servo)

---

### ⚙️ Servo Configuration

All servo configuration settings are defined in:
`src/config/servo_config.yaml`

This file specifies parameters such as:

* **Pin number**
* **Minimum and maximum angles**
* **Neutral position**
* **Minimum and maximum pulse widths**

---

### ⚠️ Important Notes

To ensure accurate and smooth servo motion:

* Always verify the **minimum** and **maximum pulse width** values against your servo’s **datasheet**.
* Typical values for a standard **SG90 (9g) micro servo** are:

  * **Minimum pulse width:** 500 µs (0.0005 s)
  * **Maximum pulse width:** 2400 µs (0.0024 s)
* Using incorrect pulse width values can cause reduced range of motion or jitter.

---

### Example

In your YAML configuration (`servo_config.yaml`):

```yaml
servos:
  pan:
    pin: 17
    min_angle: -90
    max_angle: 90
    min_pulse_width: 0.0005
    max_pulse_width: 0.0024
  tilt:
    pin: 18
    min_angle: -45
    max_angle: 45
    min_pulse_width: 0.0005
    max_pulse_width: 0.0024
```

## Code examples
- dual_servo_sweep.py — Demonstrates coordinated sweeping motion for both pan and tilt servos by specifying target angles.

- servo_sweep.py — Shows a basic base servo sweep between its minimum, midpoint, and maximum positions (no angular input required).

- api_control.py — Launches a FastAPI server that allows remote servo control via HTTP requests.
