"""
Driver Drowsiness Detection System — Standalone Mode.
Runs the detection using an OpenCV window with real-time video from the webcam.

Usage:
    python main.py

Controls:
    Q - Quit
    R - Reset counters
"""

import sys
import cv2
import config
from detector import DrowsinessDetector


def play_alarm():
    """Play an alarm sound (Windows beep or console bell)."""
    try:
        import winsound
        winsound.Beep(config.ALARM_FREQUENCY, config.ALARM_DURATION)
    except ImportError:
        # Fallback for non-Windows systems
        print("\a", end="", flush=True)


def main():
    print("=" * 60)
    print("  DRIVER DROWSINESS DETECTION SYSTEM")
    print("  Standalone Mode (OpenCV Window)")
    print("=" * 60)
    print()
    print(f"  Camera Index : {config.CAMERA_INDEX}")
    print(f"  EAR Threshold: {config.EAR_THRESHOLD}")
    print(f"  MAR Threshold: {config.MAR_THRESHOLD}")
    print()
    print("  Controls:")
    print("    Q - Quit")
    print("    R - Reset counters")
    print("=" * 60)

    # Initialize camera
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        print("\n[ERROR] Could not open camera. Please check your webcam connection.")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, config.FPS)

    # Initialize detector
    detector = DrowsinessDetector()
    print("\n[INFO] Detection started. Press Q to quit.\n")

    window_name = "Driver Drowsiness Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, config.FRAME_WIDTH, config.FRAME_HEIGHT)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARN] Failed to capture frame. Retrying...")
                continue

            # Mirror the frame for natural interaction
            frame = cv2.flip(frame, 1)

            # Process frame through the detection pipeline
            annotated_frame = detector.process_frame(frame)

            # Check for alarm
            if detector.should_alarm():
                play_alarm()

            # Display the frame
            cv2.imshow(window_name, annotated_frame)

            # Keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == ord("Q"):
                print("\n[INFO] Quit requested. Shutting down...")
                break
            elif key == ord("r") or key == ord("R"):
                detector.reset()
                print("[INFO] Counters reset.")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted. Shutting down...")

    finally:
        cap.release()
        detector.release()
        cv2.destroyAllWindows()
        print("[INFO] Resources released. Goodbye!")


if __name__ == "__main__":
    main()

