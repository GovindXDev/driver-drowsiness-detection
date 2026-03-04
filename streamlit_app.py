"""
Driver Drowsiness Detection System — Streamlit + WebRTC Cloud App.
Uses streamlit-webrtc to stream webcam from the user's browser,
processes each frame through the drowsiness detector, and displays
real-time annotated video with metrics.

Deploy on Streamlit Community Cloud:
    1. Push this repo to GitHub
    2. Go to https://share.streamlit.io
    3. Select your repo and set streamlit_app.py as the main file

Local usage:
    streamlit run streamlit_app.py
"""

import av
import cv2
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase

from detector import DrowsinessDetector
import config

# ─── Page Configuration ──────────────────────────────────────────
st.set_page_config(
    page_title="Driver Drowsiness Detection",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS for a premium dark look ──────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }
    .main-header h1 {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00d2ff 0%, #3a7bd5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .main-header p {
        color: #888;
        font-size: 0.95rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #2a2a4a;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.7rem;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0, 210, 255, 0.15);
    }
    .metric-label {
        font-size: 0.75rem;
        font-weight: 500;
        color: #8892b0;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.3rem;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #e6f1ff;
    }

    .status-alert { color: #00e676; }
    .status-mild { color: #ffea00; }
    .status-moderate { color: #ff9100; }
    .status-severe { color: #ff1744; animation: pulse 1s ease-in-out infinite; }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    .alert-banner {
        padding: 0.8rem 1.2rem;
        border-radius: 10px;
        text-align: center;
        font-weight: 600;
        font-size: 0.95rem;
        margin-bottom: 0.7rem;
        animation: slideIn 0.3s ease-out;
    }
    @keyframes slideIn {
        from { transform: translateY(-10px); opacity: 0; }
        to { transform: translateY(0); opacity: 1; }
    }
    .alert-green {
        background: linear-gradient(135deg, rgba(0,230,118,0.15), rgba(0,230,118,0.05));
        border: 1px solid rgba(0,230,118,0.3);
        color: #00e676;
    }
    .alert-yellow {
        background: linear-gradient(135deg, rgba(255,234,0,0.15), rgba(255,234,0,0.05));
        border: 1px solid rgba(255,234,0,0.3);
        color: #ffea00;
    }
    .alert-orange {
        background: linear-gradient(135deg, rgba(255,145,0,0.15), rgba(255,145,0,0.05));
        border: 1px solid rgba(255,145,0,0.3);
        color: #ff9100;
    }
    .alert-red {
        background: linear-gradient(135deg, rgba(255,23,68,0.15), rgba(255,23,68,0.05));
        border: 1px solid rgba(255,23,68,0.3);
        color: #ff1744;
    }

    .event-item {
        font-size: 0.8rem;
        padding: 0.35rem 0;
        border-bottom: 1px solid #1a1a2e;
        color: #8892b0;
    }

    .sidebar .stButton > button {
        width: 100%;
        border-radius: 8px;
        font-weight: 600;
    }

    .score-bar-container {
        background: #1a1a2e;
        border-radius: 8px;
        height: 12px;
        overflow: hidden;
        margin-top: 0.3rem;
    }
    .score-bar-fill {
        height: 100%;
        border-radius: 8px;
        transition: width 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)


# ─── WebRTC Video Processor ─────────────────────────────────────
class DrowsinessVideoProcessor(VideoProcessorBase):
    """Process each WebRTC video frame through the drowsiness detector."""

    def __init__(self):
        self.detector = DrowsinessDetector()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Receive a video frame, process it, and return annotated frame."""
        img = frame.to_ndarray(format="bgr24")

        # Mirror for natural interaction
        img = cv2.flip(img, 1)

        # Run detection pipeline
        annotated = self.detector.process_frame(img)

        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


# ─── Helper functions ────────────────────────────────────────────

def get_status_class(status: str) -> str:
    """Map status string to CSS class."""
    mapping = {
        "ALERT": "status-alert",
        "MILD DROWSINESS": "status-mild",
        "MODERATE DROWSINESS": "status-moderate",
        "SEVERE DROWSINESS": "status-severe",
    }
    return mapping.get(status, "status-alert")


def get_alert_class(status: str) -> str:
    """Map status string to alert banner CSS class."""
    mapping = {
        "ALERT": "alert-green",
        "MILD DROWSINESS": "alert-yellow",
        "MODERATE DROWSINESS": "alert-orange",
        "SEVERE DROWSINESS": "alert-red",
    }
    return mapping.get(status, "alert-green")


def get_alert_icon(status: str) -> str:
    """Map status string to an emoji icon."""
    mapping = {
        "ALERT": "✅",
        "MILD DROWSINESS": "⚠️",
        "MODERATE DROWSINESS": "🟠",
        "SEVERE DROWSINESS": "🚨",
    }
    return mapping.get(status, "✅")


def get_score_bar_color(score: float) -> str:
    """Return gradient color for the score bar."""
    if score < config.SCORE_ALERT:
        return "#00e676"
    elif score < config.SCORE_MILD:
        return "#ffea00"
    elif score < config.SCORE_MODERATE:
        return "#ff9100"
    else:
        return "#ff1744"


# ─── App Layout ──────────────────────────────────────────────────

# Header
st.markdown("""
<div class="main-header">
    <h1>🚗 Driver Drowsiness Detection</h1>
    <p>Real-time drowsiness monitoring using AI-powered facial analysis</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">EAR Threshold</div>
        <div class="metric-value">{config.EAR_THRESHOLD}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">MAR Threshold</div>
        <div class="metric-value">{config.MAR_THRESHOLD}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📖 How It Works")
    st.markdown("""
    1. **Allow camera** access when prompted
    2. Position your face in front of the camera
    3. The system monitors:
       - 👁️ **Eye closure** (EAR metric)
       - 🥱 **Yawning** (MAR metric)
       - 📐 **Head nodding** (pitch angle)
    4. Drowsiness is scored 0–100
    """)

    st.markdown("---")
    st.markdown("### 🔧 Detection Parameters")
    st.markdown(f"""
    | Parameter | Value |
    |---|---|
    | EAR Threshold | `{config.EAR_THRESHOLD}` |
    | MAR Threshold | `{config.MAR_THRESHOLD}` |
    | Eye Frames | `{config.EAR_CONSEC_FRAMES}` |
    | Yawn Frames | `{config.MAR_CONSEC_FRAMES}` |
    | Head Nod° | `{config.HEAD_NOD_THRESHOLD}°` |
    """)

    st.markdown("---")
    st.markdown(
        "<p style='text-align:center; color:#555; font-size:0.8rem;'>"
        "Built with Streamlit + MediaPipe<br>"
        "© 2026 Driver Safety System</p>",
        unsafe_allow_html=True,
    )


# ── Main Content ─────────────────────────────────────────────────

# STUN/TURN server config for WebRTC connectivity on cloud
RTC_CONFIGURATION = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
    ]
}

col_video, col_info = st.columns([3, 1])

with col_video:
    ctx = webrtc_streamer(
        key="drowsiness-detection",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=DrowsinessVideoProcessor,
        media_stream_constraints={
            "video": {
                "width": {"ideal": config.FRAME_WIDTH},
                "height": {"ideal": config.FRAME_HEIGHT},
                "frameRate": {"ideal": config.FPS},
            },
            "audio": False,
        },
        async_processing=True,
    )

with col_info:
    if ctx.video_processor:
        processor = ctx.video_processor
        detector = processor.detector
        stats = detector.get_stats()

        # Status banner
        status = stats.get("status", "ALERT")
        icon = get_alert_icon(status)
        alert_cls = get_alert_class(status)
        status_cls = get_status_class(status)
        st.markdown(
            f'<div class="alert-banner {alert_cls}">'
            f'{icon} {status}</div>',
            unsafe_allow_html=True,
        )

        # Drowsiness Score
        score = stats.get("drowsiness_score", 0)
        bar_color = get_score_bar_color(score)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Drowsiness Score</div>
            <div class="metric-value" style="color:{bar_color}">{score:.0f}<span style="font-size:0.9rem;color:#555">/100</span></div>
            <div class="score-bar-container">
                <div class="score-bar-fill" style="width:{min(score, 100):.0f}%;background:{bar_color}"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Metrics
        ear = stats.get("ear", 0)
        mar = stats.get("mar", 0)
        ear_color = "#ff1744" if ear < config.EAR_THRESHOLD else "#00e676"
        mar_color = "#ff9100" if mar > config.MAR_THRESHOLD else "#00e676"

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">👁️ Eye Aspect Ratio</div>
            <div class="metric-value" style="color:{ear_color}">{ear:.3f}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">👄 Mouth Aspect Ratio</div>
            <div class="metric-value" style="color:{mar_color}">{mar:.3f}</div>
        </div>
        """, unsafe_allow_html=True)

        blinks = stats.get("blinks", 0)
        yawns = stats.get("yawns", 0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">👁️ Blinks / 🥱 Yawns</div>
            <div class="metric-value">{blinks} <span style="font-size:0.9rem;color:#555">/</span> {yawns}</div>
        </div>
        """, unsafe_allow_html=True)

        # Reset button
        if st.button("🔄 Reset Counters", use_container_width=True):
            detector.reset()
            st.rerun()
    else:
        st.markdown("""
        <div class="metric-card" style="text-align:center; padding:2rem 1rem;">
            <div style="font-size:2.5rem; margin-bottom:0.5rem;">📷</div>
            <div class="metric-label" style="font-size:0.9rem;">
                Click <b>START</b> to begin<br>drowsiness detection
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.info("👆 Press the **START** button on the video panel to enable your webcam.")


# ── Footer ───────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#555; font-size:0.8rem;'>"
    "⚠️ This system is for <b>educational/demonstration purposes</b> only. "
    "Do not rely on it for actual driving safety.</p>",
    unsafe_allow_html=True,
)
