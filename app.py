"""
Driver Drowsiness Detection System — Flask Web Dashboard.
Provides a browser-based monitoring interface with live video stream
and real-time metrics.

Usage:
    python app.py
    Then open http://127.0.0.1:5000 in your browser.
"""

import threading
import cv2
from flask import Flask, render_template, Response, jsonify
import config
from detector import DrowsinessDetector

app = Flask(__name__)

# ── Global state ──
detector = DrowsinessDetector()
camera = None
camera_lock = threading.Lock()
output_frame = None
frame_lock = threading.Lock()


def initialize_camera():
    """Initialize the camera capture."""
    global camera
    camera = cv2.VideoCapture(config.CAMERA_INDEX)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    camera.set(cv2.CAP_PROP_FPS, config.FPS)
    return camera.isOpened()


def detection_loop():
    """Background thread: capture frames and run detection."""
    global output_frame
    while True:
        with camera_lock:
            if camera is None or not camera.isOpened():
                continue
            ret, frame = camera.read()

        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        annotated = detector.process_frame(frame)

        with frame_lock:
            output_frame = annotated.copy()


def generate_frames():
    """Generator that yields MJPEG frames for the video stream."""
    global output_frame
    while True:
        with frame_lock:
            if output_frame is None:
                continue
            ret, buffer = cv2.imencode(".jpg", output_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])

        if not ret:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


# ─── Routes ──────────────────────────────────────────────────────

@app.route("/")
def index():
    """Serve the dashboard page."""
    return render_template("dashboard.html")


@app.route("/video_feed")
def video_feed():
    """MJPEG video stream endpoint."""
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/stats")
def api_stats():
    """Return current detection statistics as JSON."""
    return jsonify(detector.get_stats())


@app.route("/api/reset", methods=["POST"])
def api_reset():
    """Reset all detection counters."""
    detector.reset()
    return jsonify({"status": "ok"})


# ─── Main ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  DRIVER DROWSINESS DETECTION SYSTEM")
    print("  Web Dashboard Mode")
    print("=" * 60)
    print()

    if not initialize_camera():
        print("[ERROR] Could not open camera!")
        print("  The dashboard will still load, but without video.")
    else:
        print(f"[INFO] Camera opened (index {config.CAMERA_INDEX})")

    # Start detection in background thread
    t = threading.Thread(target=detection_loop, daemon=True)
    t.start()
    print("[INFO] Detection thread started")

    url = f"http://{config.FLASK_HOST}:{config.FLASK_PORT}"
    print(f"\n[INFO] Dashboard available at: {url}")
    print("[INFO] Press Ctrl+C to stop\n")

    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG,
        threaded=True,
    )

