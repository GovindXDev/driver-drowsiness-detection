"""
Configuration settings for Driver Drowsiness Detection System.
All tunable thresholds and parameters are centralized here.
"""

# ─── Camera Settings ────────────────────────────────────────────
CAMERA_INDEX = 0              # Default camera (0 = built-in webcam)
FRAME_WIDTH = 640             # Capture width
FRAME_HEIGHT = 480            # Capture height
FPS = 30                      # Target frames per second

# ─── Eye Aspect Ratio (EAR) Settings ────────────────────────────
EAR_THRESHOLD = 0.25          # Below this value, eyes are considered closed
EAR_CONSEC_FRAMES = 20        # Consecutive frames below threshold to trigger drowsiness
EAR_BLINK_FRAMES = 3          # Consecutive frames for a normal blink

# MediaPipe Face Mesh landmark indices for eyes
# Right eye landmarks (from the subject's perspective)
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
# Left eye landmarks (from the subject's perspective)
LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

# ─── Mouth Aspect Ratio (MAR) Settings ──────────────────────────
MAR_THRESHOLD = 0.7           # Above this value, mouth is considered open (yawn)
MAR_CONSEC_FRAMES = 10        # Consecutive frames for yawn detection

# MediaPipe Face Mesh landmark indices for mouth
UPPER_LIP_INDICES = [13, 312, 311, 310, 415, 308]
LOWER_LIP_INDICES = [14, 317, 402, 318, 324, 78]
MOUTH_CORNER_INDICES = [78, 308]  # Left and right corners
MOUTH_VERTICAL_INDICES = [13, 14]  # Top and bottom center

# ─── Head Pose Settings ─────────────────────────────────────────
HEAD_NOD_THRESHOLD = 15.0     # Degrees of pitch to consider head nodding
HEAD_NOD_CONSEC_FRAMES = 15   # Consecutive frames of nodding

# ─── Drowsiness Score Settings ───────────────────────────────────
SCORE_EAR_WEIGHT = 0.50       # Weight for eye closure in drowsiness score
SCORE_MAR_WEIGHT = 0.25       # Weight for yawning
SCORE_HEAD_WEIGHT = 0.25      # Weight for head nodding

# Score thresholds for drowsiness levels
SCORE_ALERT = 20              # Below: ALERT (green)
SCORE_MILD = 40               # Below: MILD DROWSINESS (yellow)
SCORE_MODERATE = 65           # Below: MODERATE DROWSINESS (orange)
# Above MODERATE: SEVERE DROWSINESS (red)

# ─── Alarm Settings ─────────────────────────────────────────────
ALARM_ENABLED = True
ALARM_FREQUENCY = 2500        # Hz for beep alarm
ALARM_DURATION = 500          # ms for beep alarm
ALARM_COOLDOWN = 3.0          # Seconds between alarm triggers

# ─── Flask Web App Settings ─────────────────────────────────────
FLASK_HOST = "127.0.0.1"
FLASK_PORT = 5000
FLASK_DEBUG = False

# ─── Display Colors (BGR format for OpenCV) ──────────────────────
COLOR_GREEN = (0, 200, 0)
COLOR_YELLOW = (0, 220, 255)
COLOR_ORANGE = (0, 140, 255)
COLOR_RED = (0, 0, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_CYAN = (255, 255, 0)
COLOR_DARK_BG = (30, 30, 30)

