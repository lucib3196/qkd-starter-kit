
---

# Getting Started

## 1. Enter the Backend Directory

```bash
cd backend
```

## 2. Create and Activate a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Project Structure

## Camera Module

Modules for capturing, processing, and streaming video using **PiCamera2** or **USB webcams**.

* **camera_calibration**
  Tools & scripts for camera calibration, including calibration image capture.

* **camera_capture**
  General-purpose capture module supporting both PiCamera and USB camera input.

* **camera_deprecated**
  Old camera code kept for reference; includes the legacy threaded PiCamera implementation.

* **camera_feed**
  Examples showing how to locally stream camera output or expose it via a FastAPI endpoint.

* **camera_threaded**
  Main camera implementation using threads for improved performance; supports calibration settings.

* **camera_utils**
  Utility functions shared across camera modules.

---

##  ARUCO Marker Detection

* **base**
  Basic ArUco marker detection using a *local* USB camera feed.

* **base_stream**
  Experimental version of `base` that works with streamed video input.

* **main**
  Concise wrapper implementation of the ArUco detection pipeline.

> Other ArUco-related scripts are outdated and should be cleaned up or removed.

---

## Controls

* **PID**
  A reusable PID controller class used for servo alignment, tracking, and general control loops.

---

##  Mocking

* **mock_picamera**
  A fake PiCamera class used to avoid import errors when developing on non-Raspberry Pi systems.

---

## 🛠️ PiControl (Legacy)

* Older collection of scripts for tracking and servo control.
  Will be reorganized and merged into the updated main tracking workflow.

---

## Main Tracking System

Current versions of the tracking logic, including servo integration.

* **stream_tracking**
  Full streaming-based tracker with servo initialization and closed-loop control.

* **tracking**
  Local webcam-based tracking implementation for PC-based testing.

---

## Servo Basics

Example scripts demonstrating how to initialize and drive servos.



---

## Face Tracking

Real-time face detection and streaming through Flask.

* **picamera_facedetection**
  Detects faces from PiCamera, overlays annotations, and streams via HTTP.

---

## Servo Control Tests

Scripts for quickly verifying servo motion.

* **servo_test**
  Moves servos to their min/max angles.

* **servo_test2**
  Continuously sweeps servos between angle limits.

---

## Wiring Guide
![image](https://github.com/user-attachments/assets/2eee6460-e1cd-4281-bdd4-0497255fffef)


* **Pan servo** → GPIO17 (pin 11)
* **Tilt servo** → GPIO13 (pin 13)
* **Ground** → Pin 9
* **Camera** → PiCamera port
* **Power supply** → DC supply to servos

---

## Resources

* [gpiozero documentation](https://gpiozero.readthedocs.io/en/stable/)
* [OpenCV](https://opencv.org/)
```

