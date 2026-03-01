# 🚗 Driver Drowsiness Detection System

A real-time driver drowsiness detection system using **computer vision** and **machine learning**. The system monitors a driver's face through a camera, detects signs of drowsiness (eye closure, yawning, head nodding), and triggers an alarm to alert the driver.

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10+-00A67E)
![Flask](https://img.shields.io/badge/Flask-3.0+-000?logo=flask)

---

## 🎯 Features

- **Eye Aspect Ratio (EAR)** — Detects prolonged eye closure indicating drowsiness
- **Mouth Aspect Ratio (MAR)** — Detects yawning as a sign of fatigue
- **Head Pose Estimation** — Detects head nodding/drooping
- **Drowsiness Scoring** — Weighted 0–100 score combining all signals
- **Real-time Alarm** — Audible alert when drowsiness is detected
- **Blink & Yawn Counting** — Tracks frequency of blinks and yawns
- **Two Modes**:
  - **Standalone Mode** — OpenCV window with HUD overlay
  - **Web Dashboard** — Premium dark-themed browser UI with live video

---

## 📐 Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌──────────────┐
│   Webcam /  │     │  MediaPipe Face   │     │  Drowsiness  │
│  Camera     │────>│  Mesh (468 pts)   │────>│  Algorithms  │
└─────────────┘     └──────────────────┘     └──────┬───────┘
                                                     │
                             ┌───────────────────────┼──────────────┐
                             │                       │              │
                        ┌────▼────┐           ┌──────▼──────┐  ┌───▼───┐
                        │  EAR    │           │    MAR      │  │ Head  │
                        │ (Eyes)  │           │  (Mouth)    │  │ Pose  │
                        └────┬────┘           └──────┬──────┘  └───┬───┘
                             │                       │             │
                             └───────────┬───────────┘─────────────┘
                                         │
                                  ┌──────▼──────┐
                                  │  Drowsiness │
                                  │   Score     │──────> Alarm
                                  │  (0-100)    │
                                  └─────────────┘
```

---

## 🔧 Tech Stack

| Technology | Purpose |
|-----------|---------|
| **Python 3.8+** | Core programming language |
| **OpenCV** | Video capture, image processing, display |
| **MediaPipe** | Face mesh landmark detection (478 points) |
| **NumPy** | Numerical computations |
| **Flask** | Web dashboard server |

---

## 🚀 Setup & Installation

### 1. Clone / navigate to the project

```bash
cd driver-drowsiness-detection
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### Standalone Mode (OpenCV Window)

```bash
python main.py
```

**Controls:**
| Key | Action |
|-----|--------|
| `Q` | Quit |
| `R` | Reset all counters |

### Web Dashboard Mode

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

---

## ⚙️ Configuration

All thresholds are configurable in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `EAR_THRESHOLD` | 0.25 | Eye closed when EAR < this |
| `EAR_CONSEC_FRAMES` | 20 | Frames to confirm drowsiness |
| `MAR_THRESHOLD` | 0.70 | Yawn detected when MAR > this |
| `ALARM_FREQUENCY` | 2500 Hz | Alarm beep frequency |
| `CAMERA_INDEX` | 0 | Camera device index |

---

## 🧠 How It Works

### Eye Aspect Ratio (EAR)

The EAR formula measures eye openness using 6 landmarks per eye:

```
EAR = (||p2 - p6|| + ||p3 - p5||) / (2 × ||p1 - p4||)
```

- **Open eye**: EAR ≈ 0.3–0.4
- **Closed eye**: EAR < 0.25
- **Drowsiness**: EAR < 0.25 for 20+ consecutive frames

### Mouth Aspect Ratio (MAR)

```
MAR = vertical_distance / horizontal_distance
```

- **Closed mouth**: MAR ≈ 0.2–0.4
- **Yawn**: MAR > 0.7

### Drowsiness Score

Weighted combination with smooth averaging:

```
Score = (EAR_weight × eye_score) + (MAR_weight × mouth_score) + (Head_weight × head_score)
```

| Score Range | Status |
|-----------|--------|
| 0–20 | ✅ ALERT |
| 20–40 | ⚡ MILD |
| 40–65 | ⚠️ MODERATE |
| 65–100 | 🚨 SEVERE |

---

## 🍓 Raspberry Pi Deployment

This project is compatible with Raspberry Pi:

1. Install on Raspberry Pi OS:
   ```bash
   sudo apt update && sudo apt install python3-pip python3-venv
   pip3 install opencv-python-headless mediapipe flask numpy
   ```

2. Connect a USB webcam or use the Pi Camera Module.

3. For the Pi Camera, change `CAMERA_INDEX = 0` in `config.py`.

4. Run in web dashboard mode for best performance:
   ```bash
   python3 app.py
   ```

5. Access the dashboard from any device on the same network:
   - Change `FLASK_HOST = "0.0.0.0"` in `config.py`
   - Open `http://<pi-ip>:5000`

---

## 📂 Project Structure

```
driver-drowsiness-detection/
├── main.py              # Standalone OpenCV mode
├── app.py               # Flask web dashboard mode
├── detector.py          # Core detection engine
├── config.py            # Configuration & thresholds
├── requirements.txt     # Python dependencies
├── README.md            # This file
├── static/
│   ├── css/style.css    # Dashboard styles
│   └── js/app.js        # Dashboard interactivity
└── templates/
    └── dashboard.html   # Dashboard HTML template
```

---

## 📄 License

This project is for educational and research purposes.
