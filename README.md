# Smart Wildlife Detection & Trap System  
**YOLO + Raspberry Pi + Ultrasonic Sensor + Stepper Motor**

This project is a **Raspberry Pi–based automated wildlife detection and trap control system** that combines **computer vision (YOLO)**, an **ultrasonic distance sensor**, a **stepper motor–controlled trap**, and a **live MJPEG video stream served via Flask**.

The system is designed to operate **only during daylight hours**, detect specific animals (e.g., iguanas or cats), reduce false positives using **debounced detections**, and safely activate a mechanical trap when all conditions are met.

---
## Features

-  **Live MJPEG camera stream** with YOLO bounding boxes
- **YOLO object detection** using a custom-trained model
- **Ultrasonic sensor trigger** for proximity-based activation
-  **Detection debounce logic** to minimize false positives
-  **Stepper motor trap control** with auto-reset
-  **Daylight-only operation** using Astral sun calculations
- **Flask REST API** for manual trap control
-  **Automatic image capture and storage**
-  Optimized for **real-time performance on Raspberry Pi**

---

##  Hardware Requirements

- Raspberry Pi (Pi 4 recommended)
- Raspberry Pi Camera Module (Picamera2 / libcamera)
- Ultrasonic Distance Sensor (HC-SR04)
- 28BYJ-48 Stepper Motor
- ULN2003 Stepper Motor Driver
- External 5V power supply for stepper motor
- Jumper wires and mounting hardware

---

##  Software Requirements

- Raspberry Pi OS (64-bit recommended)
- Python 3.9 or newer
