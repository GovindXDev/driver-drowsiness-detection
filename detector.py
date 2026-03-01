"""
Core Drowsiness Detection Engine.
Uses MediaPipe FaceLandmarker (Tasks API) for facial landmark detection and
computes Eye Aspect Ratio (EAR), Mouth Aspect Ratio (MAR), and head pose
to determine drowsiness level.
"""

import os
import time
import math
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)
import config

# Path to the face landmark model
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")


class DrowsinessDetector:
    """Real-time drowsiness detection using facial landmarks."""

    def __init__(self):
        # ── MediaPipe FaceLandmarker setup (Tasks API) ──
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.face_landmarker = FaceLandmarker.create_from_options(options)

        # ── State tracking ──
        self.ear_counter = 0          # Consecutive frames with low EAR
        self.mar_counter = 0          # Consecutive frames with high MAR
        self.blink_counter = 0        # Total blinks detected
        self.yawn_counter = 0         # Total yawns detected
        self.drowsy_alert = False     # Currently in drowsy state
        self.yawn_alert = False       # Currently yawning
        self.drowsiness_score = 0.0   # 0-100 scale
        self.status = "ALERT"         # Current status string

        # ── Metrics ──
        self.current_ear = 0.0
        self.current_mar = 0.0
        self.head_pitch = 0.0
        self.face_detected = False

        # ── Timing ──
        self.last_alarm_time = 0
        self.start_time = time.time()
        self.frame_count = 0
        self.fps = 0.0
        self._fps_start = time.time()
        self._fps_frame_count = 0

        # ── Score history for smoothing ──
        self._score_history = []
        self._max_history = 30

        # ── Event log ──
        self.events = []
        self._max_events = 50

    # ─────────────────────────────────────────────────────────────
    #  Geometry helpers
    # ─────────────────────────────────────────────────────────────

    @staticmethod
    def _distance(p1, p2):
        """Euclidean distance between two 2D points."""
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    # ─────────────────────────────────────────────────────────────
    #  Eye Aspect Ratio (EAR)
    # ─────────────────────────────────────────────────────────────

    def _compute_ear(self, landmarks, indices, w, h):
        """
        Compute Eye Aspect Ratio for a given set of eye landmark indices.

        EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
        """
        pts = []
        for idx in indices:
            lm = landmarks[idx]
            pts.append((lm.x * w, lm.y * h))

        # Vertical distances
        v1 = self._distance(pts[1], pts[5])  # p2 - p6
        v2 = self._distance(pts[2], pts[4])  # p3 - p5
        # Horizontal distance
        h1 = self._distance(pts[0], pts[3])  # p1 - p4

        if h1 == 0:
            return 0.0
        return (v1 + v2) / (2.0 * h1)

    def _get_ear(self, landmarks, w, h):
        """Compute average EAR for both eyes."""
        left_ear = self._compute_ear(landmarks, config.LEFT_EYE_INDICES, w, h)
        right_ear = self._compute_ear(landmarks, config.RIGHT_EYE_INDICES, w, h)
        return (left_ear + right_ear) / 2.0

    # ─────────────────────────────────────────────────────────────
    #  Mouth Aspect Ratio (MAR)
    # ─────────────────────────────────────────────────────────────

    def _get_mar(self, landmarks, w, h):
        """
        Compute Mouth Aspect Ratio.
        MAR = vertical_distance / horizontal_distance
        """
        left_corner = landmarks[78]
        right_corner = landmarks[308]
        h_dist = self._distance(
            (left_corner.x * w, left_corner.y * h),
            (right_corner.x * w, right_corner.y * h),
        )

        top_points = [13, 312, 311, 310]
        bottom_points = [14, 317, 402, 318]

        v_dist_total = 0.0
        count = 0
        for t_idx, b_idx in zip(top_points, bottom_points):
            t = landmarks[t_idx]
            b = landmarks[b_idx]
            v_dist_total += self._distance(
                (t.x * w, t.y * h), (b.x * w, b.y * h)
            )
            count += 1

        if h_dist == 0:
            return 0.0
        return (v_dist_total / count) / h_dist

    # ─────────────────────────────────────────────────────────────
    #  Head Pose Estimation (simplified via nose & chin)
    # ─────────────────────────────────────────────────────────────

    def _get_head_pitch(self, landmarks, w, h):
        """Estimate head pitch using nose tip, forehead, and chin."""
        nose_tip = landmarks[1]
        forehead = landmarks[10]
        chin = landmarks[152]

        nose = (nose_tip.x * w, nose_tip.y * h)
        fore = (forehead.x * w, forehead.y * h)
        ch = (chin.x * w, chin.y * h)

        face_height = self._distance(fore, ch)
        if face_height == 0:
            return 0.0

        nose_to_mid_y = nose[1] - (fore[1] + ch[1]) / 2
        pitch = (nose_to_mid_y / face_height) * 90.0
        return pitch

    # ─────────────────────────────────────────────────────────────
    #  Drowsiness Score Computation
    # ─────────────────────────────────────────────────────────────

    def _compute_drowsiness_score(self):
        """Compute a weighted drowsiness score (0–100)."""
        eye_score = min(100, (self.ear_counter / config.EAR_CONSEC_FRAMES) * 100)
        mouth_score = min(100, (self.mar_counter / config.MAR_CONSEC_FRAMES) * 100)
        head_score = min(100, max(0, (abs(self.head_pitch) - 5) / config.HEAD_NOD_THRESHOLD * 100))

        raw_score = (
            eye_score * config.SCORE_EAR_WEIGHT
            + mouth_score * config.SCORE_MAR_WEIGHT
            + head_score * config.SCORE_HEAD_WEIGHT
        )

        self._score_history.append(raw_score)
        if len(self._score_history) > self._max_history:
            self._score_history.pop(0)

        return sum(self._score_history) / len(self._score_history)

    def _get_status(self, score):
        """Map drowsiness score to a status label."""
        if score < config.SCORE_ALERT:
            return "ALERT"
        elif score < config.SCORE_MILD:
            return "MILD DROWSINESS"
        elif score < config.SCORE_MODERATE:
            return "MODERATE DROWSINESS"
        else:
            return "SEVERE DROWSINESS"

    def _get_status_color(self):
        """Get BGR color for current status."""
        if self.status == "ALERT":
            return config.COLOR_GREEN
        elif self.status == "MILD DROWSINESS":
            return config.COLOR_YELLOW
        elif self.status == "MODERATE DROWSINESS":
            return config.COLOR_ORANGE
        else:
            return config.COLOR_RED

    # ─────────────────────────────────────────────────────────────
    #  Alarm logic
    # ─────────────────────────────────────────────────────────────

    def should_alarm(self):
        """Check if alarm should be triggered."""
        if not config.ALARM_ENABLED:
            return False
        if self.drowsy_alert and (time.time() - self.last_alarm_time) > config.ALARM_COOLDOWN:
            self.last_alarm_time = time.time()
            return True
        return False

    def _add_event(self, event_type, message):
        """Add an event to the log."""
        timestamp = time.strftime("%H:%M:%S")
        self.events.insert(0, {
            "time": timestamp,
            "type": event_type,
            "message": message,
        })
        if len(self.events) > self._max_events:
            self.events.pop()

    # ─────────────────────────────────────────────────────────────
    #  Main processing pipeline
    # ─────────────────────────────────────────────────────────────

    def process_frame(self, frame):
        """
        Process a single video frame and return annotated frame + metrics.

        Args:
            frame: BGR image from OpenCV.

        Returns:
            annotated_frame: Frame with overlays drawn.
        """
        h, w, _ = frame.shape
        self.frame_count += 1

        # FPS calculation
        self._fps_frame_count += 1
        elapsed = time.time() - self._fps_start
        if elapsed >= 1.0:
            self.fps = self._fps_frame_count / elapsed
            self._fps_frame_count = 0
            self._fps_start = time.time()

        # Convert BGR to RGB and create MediaPipe Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect face landmarks
        result = self.face_landmarker.detect(mp_image)

        self.face_detected = False

        if result.face_landmarks:
            for face_landmarks_list in result.face_landmarks:
                self.face_detected = True
                landmarks = face_landmarks_list  # List of NormalizedLandmark

                # ── Compute metrics ──
                self.current_ear = self._get_ear(landmarks, w, h)
                self.current_mar = self._get_mar(landmarks, w, h)
                self.head_pitch = self._get_head_pitch(landmarks, w, h)

                # ── Eye closure detection ──
                if self.current_ear < config.EAR_THRESHOLD:
                    self.ear_counter += 1
                    if self.ear_counter >= config.EAR_CONSEC_FRAMES:
                        if not self.drowsy_alert:
                            self._add_event("DROWSY", "Eye closure detected!")
                        self.drowsy_alert = True
                else:
                    if config.EAR_BLINK_FRAMES <= self.ear_counter < config.EAR_CONSEC_FRAMES:
                        self.blink_counter += 1
                    if self.drowsy_alert:
                        self._add_event("RECOVERED", "Driver became alert")
                    self.drowsy_alert = False
                    self.ear_counter = 0

                # ── Yawn detection ──
                if self.current_mar > config.MAR_THRESHOLD:
                    self.mar_counter += 1
                    if self.mar_counter >= config.MAR_CONSEC_FRAMES:
                        if not self.yawn_alert:
                            self.yawn_counter += 1
                            self._add_event("YAWN", f"Yawn detected (total: {self.yawn_counter})")
                        self.yawn_alert = True
                else:
                    self.yawn_alert = False
                    self.mar_counter = 0

                # ── Drowsiness score ──
                self.drowsiness_score = self._compute_drowsiness_score()
                self.status = self._get_status(self.drowsiness_score)

                # ── Draw face mesh ──
                self._draw_landmarks(frame, landmarks, w, h)

        else:
            self.face_detected = False
            self.ear_counter = min(self.ear_counter + 1, config.EAR_CONSEC_FRAMES)

        # ── Draw HUD overlay ──
        self._draw_hud(frame, w, h)

        return frame

    # ─────────────────────────────────────────────────────────────
    #  Drawing helpers
    # ─────────────────────────────────────────────────────────────

    def _draw_landmarks(self, frame, landmarks, w, h):
        """Draw eye and mouth contours on the frame."""
        # Draw eye contours
        for indices in [config.LEFT_EYE_INDICES, config.RIGHT_EYE_INDICES]:
            pts = []
            for idx in indices:
                lm = landmarks[idx]
                pts.append((int(lm.x * w), int(lm.y * h)))
            pts = np.array(pts, dtype=np.int32)
            color = config.COLOR_RED if self.drowsy_alert else config.COLOR_GREEN
            cv2.polylines(frame, [pts], True, color, 2)

        # Draw mouth contour
        mouth_indices = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
                         308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
        mouth_pts = []
        for idx in mouth_indices:
            lm = landmarks[idx]
            mouth_pts.append((int(lm.x * w), int(lm.y * h)))
        mouth_pts = np.array(mouth_pts, dtype=np.int32)
        mouth_color = config.COLOR_ORANGE if self.yawn_alert else config.COLOR_CYAN
        cv2.polylines(frame, [mouth_pts], True, mouth_color, 2)

    def _draw_hud(self, frame, w, h):
        """Draw the heads-up display with metrics and status."""
        status_color = self._get_status_color()

        # ── Top status bar ──
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 50), config.COLOR_DARK_BG, -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        cv2.putText(frame, f"Status: {self.status}", (10, 35),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(frame, f"FPS: {self.fps:.0f}", (w - 120, 35),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.COLOR_WHITE, 1)

        # ── Left metrics panel ──
        panel_y = 60
        metrics = [
            (f"EAR: {self.current_ear:.3f}", config.COLOR_RED if self.current_ear < config.EAR_THRESHOLD else config.COLOR_GREEN),
            (f"MAR: {self.current_mar:.3f}", config.COLOR_ORANGE if self.current_mar > config.MAR_THRESHOLD else config.COLOR_GREEN),
            (f"Score: {self.drowsiness_score:.0f}/100", status_color),
            (f"Blinks: {self.blink_counter}", config.COLOR_CYAN),
            (f"Yawns: {self.yawn_counter}", config.COLOR_CYAN),
        ]

        overlay2 = frame.copy()
        cv2.rectangle(overlay2, (0, 55), (200, 55 + len(metrics) * 30 + 10), config.COLOR_DARK_BG, -1)
        cv2.addWeighted(overlay2, 0.6, frame, 0.4, 0, frame)

        for text, color in metrics:
            cv2.putText(frame, text, (10, panel_y + 20),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            panel_y += 28

        # ── Drowsiness score bar ──
        bar_x, bar_y, bar_w, bar_h = w - 40, 60, 20, 200
        overlay3 = frame.copy()
        cv2.rectangle(overlay3, (bar_x - 5, bar_y - 5), (bar_x + bar_w + 5, bar_y + bar_h + 25), config.COLOR_DARK_BG, -1)
        cv2.addWeighted(overlay3, 0.6, frame, 0.4, 0, frame)

        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (60, 60, 60), -1)

        fill_h = int((self.drowsiness_score / 100.0) * bar_h)
        if fill_h > 0:
            cv2.rectangle(
                frame,
                (bar_x, bar_y + bar_h - fill_h),
                (bar_x + bar_w, bar_y + bar_h),
                status_color, -1
            )
        cv2.putText(frame, f"{self.drowsiness_score:.0f}%", (bar_x - 8, bar_y + bar_h + 20),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.4, config.COLOR_WHITE, 1)

        # ── Alert banner ──
        if self.drowsy_alert:
            overlay4 = frame.copy()
            cv2.rectangle(overlay4, (0, h - 60), (w, h), config.COLOR_RED, -1)
            cv2.addWeighted(overlay4, 0.7, frame, 0.3, 0, frame)
            text = "!!! DROWSINESS DETECTED - PLEASE TAKE A BREAK !!!"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            text_x = (w - text_size[0]) // 2
            cv2.putText(frame, text, (text_x, h - 25),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, config.COLOR_WHITE, 2)

        # ── No face warning ──
        if not self.face_detected:
            cv2.putText(frame, "NO FACE DETECTED", (w // 2 - 120, h // 2),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, config.COLOR_YELLOW, 2)

    # ─────────────────────────────────────────────────────────────
    #  Stats for web dashboard
    # ─────────────────────────────────────────────────────────────

    def get_stats(self):
        """Return current detection stats as dictionary."""
        return {
            "ear": round(self.current_ear, 3),
            "mar": round(self.current_mar, 3),
            "head_pitch": round(self.head_pitch, 1),
            "drowsiness_score": round(self.drowsiness_score, 1),
            "status": self.status,
            "blinks": self.blink_counter,
            "yawns": self.yawn_counter,
            "drowsy_alert": self.drowsy_alert,
            "yawn_alert": self.yawn_alert,
            "face_detected": self.face_detected,
            "fps": round(self.fps, 1),
            "uptime": int(time.time() - self.start_time),
            "events": self.events[:10],
        }

    def reset(self):
        """Reset all counters and state."""
        self.ear_counter = 0
        self.mar_counter = 0
        self.blink_counter = 0
        self.yawn_counter = 0
        self.drowsy_alert = False
        self.yawn_alert = False
        self.drowsiness_score = 0.0
        self.status = "ALERT"
        self._score_history.clear()
        self.events.clear()
        self._add_event("SYSTEM", "Detection reset")

    def release(self):
        """Release MediaPipe resources."""
        self.face_landmarker.close()
