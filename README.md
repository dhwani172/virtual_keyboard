# Gesture Virtual Keyboard (OpenCV + MediaPipe)

A fast, gesture-driven on-screen keyboard built with OpenCV + MediaPipe.  
Supports **pinch** and **dwell** input, low-latency camera capture, One-Euro pointer smoothing, and a clean purple UI.

https://github.com/dhwani172/virtual_keyboard

---

## Features
- **Pinch** input with hysteresis + short hold (avoids double-fires)
- **Dwell** input with visible progress ring
- **One-Euro filter** for smooth + responsive cursor
- **Low-latency capture** (MSMF + MJPG, tiny buffer)
- **Downscaled processing** (fast) + **magnet** snapping to key centers
- Toggleable camera overlay (tracking still works while hidden)

---

## Quick Start
```bash
# 1) Create and activate a venv (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2) Install deps
pip install -r requirements.txt

# 3) Run
python virtual_keyboard.py
