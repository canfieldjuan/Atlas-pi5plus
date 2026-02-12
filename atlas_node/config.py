"""Atlas Edge Node configuration -- loaded from environment variables."""

import os
from pathlib import Path

BASE_DIR = Path("/opt/atlas-node")
MODEL_DIR = BASE_DIR / "models"

# --- Node identity ---
NODE_ID = os.getenv("NODE_ID", "edge-opi5plus-01")

# --- WebSocket ---
ATLAS_BRAIN_HOST = os.getenv("ATLAS_BRAIN_HOST", "atlas-brain.tailc7bd29.ts.net")
ATLAS_BRAIN_PORT = os.getenv("ATLAS_BRAIN_PORT", "8000")
ATLAS_WS_URL = os.getenv(
    "ATLAS_WS_URL",
    "ws://{}:{}/api/v1/ws/edge/{}".format(ATLAS_BRAIN_HOST, ATLAS_BRAIN_PORT, NODE_ID),
)
WS_RECONNECT_BASE = float(os.getenv("WS_RECONNECT_BASE", "1.0"))
WS_RECONNECT_MAX = float(os.getenv("WS_RECONNECT_MAX", "60.0"))
WS_PING_INTERVAL = float(os.getenv("WS_PING_INTERVAL", "60"))
WS_PING_TIMEOUT = float(os.getenv("WS_PING_TIMEOUT", "120"))

# --- Vision (YOLO World) ---
YOLO_WORLD_MODEL_PATH = os.getenv(
    "YOLO_WORLD_MODEL_PATH",
    str(MODEL_DIR / "yolo-world" / "yolo_world_v2s_i8.rknn"),
)
CLIP_TEXT_MODEL_PATH = os.getenv(
    "CLIP_TEXT_MODEL_PATH",
    str(MODEL_DIR / "yolo-world" / "clip_text_fp32.rknn"),
)
CAMERA_DEVICE = int(os.getenv("CAMERA_DEVICE", "0"))
CAMERA_WIDTH = int(os.getenv("CAMERA_WIDTH", "640"))
CAMERA_HEIGHT = int(os.getenv("CAMERA_HEIGHT", "480"))
RTSP_STREAM_URL = os.getenv("RTSP_STREAM_URL", "rtsp://localhost:8554/cam1")
VISION_FPS = float(os.getenv("VISION_FPS", "5"))
YOLO_CONF_THRESHOLD = float(os.getenv("YOLO_CONF_THRESHOLD", "0.25"))
YOLO_NMS_THRESHOLD = float(os.getenv("YOLO_NMS_THRESHOLD", "0.45"))
YOLO_INPUT_SIZE = int(os.getenv("YOLO_INPUT_SIZE", "640"))

# --- Speech (STT) ---
STT_MODEL_DIR = os.getenv(
    "STT_MODEL_DIR",
    str(MODEL_DIR / "sensevoice"),
)
AUDIO_DEVICE_INDEX = os.getenv("AUDIO_DEVICE_INDEX", None)  # None = auto-detect
if AUDIO_DEVICE_INDEX is not None:
    AUDIO_DEVICE_INDEX = int(AUDIO_DEVICE_INDEX)
AUDIO_INPUT_DEVICE_NAME = os.getenv("AUDIO_INPUT_DEVICE_NAME", "USB 2.0 Camera")
AUDIO_SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
AUDIO_CHANNELS = int(os.getenv("AUDIO_CHANNELS", "1"))
AUDIO_CHUNK_SECONDS = float(os.getenv("AUDIO_CHUNK_SECONDS", "5.0"))

# --- Face Detection / Recognition ---
FACE_ENABLED = os.getenv("FACE_ENABLED", "true").lower() in ("true", "1", "yes")
FACE_DET_MODEL_PATH = os.getenv(
    "FACE_DET_MODEL_PATH",
    str(MODEL_DIR / "face" / "RetinaFace_mobile320.rknn"),
)
FACE_REC_MODEL_PATH = os.getenv(
    "FACE_REC_MODEL_PATH",
    str(MODEL_DIR / "face" / "w600k_mbf_i8.rknn"),
)
FACE_DB_DIR = os.getenv("FACE_DB_DIR", str(BASE_DIR / "face_db"))
FACE_MATCH_THRESHOLD = float(os.getenv("FACE_MATCH_THRESHOLD", "0.40"))
FACE_DET_CONF_THRESHOLD = float(os.getenv("FACE_DET_CONF_THRESHOLD", "0.5"))
FACE_DET_NMS_THRESHOLD = float(os.getenv("FACE_DET_NMS_THRESHOLD", "0.4"))
FACE_COOLDOWN_SECONDS = float(os.getenv("FACE_COOLDOWN_SECONDS", "5.0"))
FACE_AUTO_ENROLL = os.getenv("FACE_AUTO_ENROLL", "true").lower() in ("true", "1", "yes")
FACE_UNKNOWN_DIR = os.getenv("FACE_UNKNOWN_DIR", str(BASE_DIR / "face_unknown"))

# --- Speaker Identification ---
SPEAKER_ENABLED = os.getenv("SPEAKER_ENABLED", "true").lower() in ("true", "1", "yes")
SPEAKER_MODEL_PATH = os.getenv(
    "SPEAKER_MODEL_PATH",
    str(MODEL_DIR / "speaker" / "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"),
)
SPEAKER_DB_DIR = os.getenv("SPEAKER_DB_DIR", str(BASE_DIR / "speaker_db"))
SPEAKER_MATCH_THRESHOLD = float(os.getenv("SPEAKER_MATCH_THRESHOLD", "0.6"))
SPEAKER_COOLDOWN_SECONDS = float(os.getenv("SPEAKER_COOLDOWN_SECONDS", "10.0"))

# --- Gait Recognition ---
GAIT_ENABLED = os.getenv("GAIT_ENABLED", "true").lower() in ("true", "1", "yes")
GAIT_POSE_MODEL_PATH = os.getenv(
    "GAIT_POSE_MODEL_PATH",
    str(MODEL_DIR / "pose" / "yolov8n-pose.rknn"),
)
GAIT_DB_DIR = os.getenv("GAIT_DB_DIR", str(BASE_DIR / "gait_db"))
GAIT_MATCH_THRESHOLD = float(os.getenv("GAIT_MATCH_THRESHOLD", "0.75"))
GAIT_MIN_FRAMES = int(os.getenv("GAIT_MIN_FRAMES", "40"))
GAIT_MAX_FRAMES = int(os.getenv("GAIT_MAX_FRAMES", "60"))
GAIT_COOLDOWN_SECONDS = float(os.getenv("GAIT_COOLDOWN_SECONDS", "15.0"))
GAIT_EMBEDDING_DIM = int(os.getenv("GAIT_EMBEDDING_DIM", "256"))
GAIT_VISIBILITY_THRESHOLD = float(os.getenv("GAIT_VISIBILITY_THRESHOLD", "0.3"))

# --- Person Tracking ---
TRACK_CONFIRM_HITS = int(os.getenv("TRACK_CONFIRM_HITS", "3"))
TRACK_LOST_AGE = int(os.getenv("TRACK_LOST_AGE", "15"))
TRACK_DELETE_AGE = int(os.getenv("TRACK_DELETE_AGE", "30"))
TRACK_IOU_THRESHOLD = float(os.getenv("TRACK_IOU_THRESHOLD", "0.3"))

# --- Motion Detection ---
MOTION_ENABLED = os.getenv("MOTION_ENABLED", "true").lower() in ("true", "1", "yes")
MOTION_SENSITIVITY = float(os.getenv("MOTION_SENSITIVITY", "0.005"))
MOTION_MIN_AREA = int(os.getenv("MOTION_MIN_AREA", "500"))
MOTION_COOLDOWN_SECONDS = float(os.getenv("MOTION_COOLDOWN_SECONDS", "10.0"))
MOTION_LEARNING_RATE = float(os.getenv("MOTION_LEARNING_RATE", "0.005"))
MOTION_WARMUP_FRAMES = int(os.getenv("MOTION_WARMUP_FRAMES", "30"))

# --- Event Store ---
EVENT_DB_PATH = os.getenv("EVENT_DB_PATH", str(BASE_DIR / "data" / "events.db"))
EVENT_RETENTION_DAYS = int(os.getenv("EVENT_RETENTION_DAYS", "30"))

# --- Dashboard ---
DASHBOARD_ENABLED = os.getenv("DASHBOARD_ENABLED", "true").lower() in ("true", "1", "yes")
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "8080"))
DASHBOARD_HOST = os.getenv("DASHBOARD_HOST", "0.0.0.0")

# --- Identity Sync ---
IDENTITY_SYNC_ENABLED = os.getenv("IDENTITY_SYNC_ENABLED", "true").lower() in ("true", "1", "yes")
IDENTITY_SYNC_INTERVAL = float(os.getenv("IDENTITY_SYNC_INTERVAL", "300"))  # 5 min periodic re-sync
IDENTITY_WATCH_INTERVAL = float(os.getenv("IDENTITY_WATCH_INTERVAL", "10"))  # poll for local registrations

# --- Wake Word (OpenWakeWord) ---
WAKEWORD_ENABLED = os.getenv("WAKEWORD_ENABLED", "false").lower() in ("true", "1", "yes")
WAKEWORD_MODEL_PATH = os.getenv(
    "WAKEWORD_MODEL_PATH",
    str(MODEL_DIR / "wakeword" / "hey_jarvis_v0.1.onnx"),
)
WAKEWORD_THRESHOLD = float(os.getenv("WAKEWORD_THRESHOLD", "0.5"))
WAKEWORD_LISTEN_SECONDS = float(os.getenv("WAKEWORD_LISTEN_SECONDS", "10.0"))
WAKEWORD_PREBUFFER_FRAMES = int(os.getenv("WAKEWORD_PREBUFFER_FRAMES", "10"))

# --- TTS ---
TTS_ENGINE = os.getenv("TTS_ENGINE", "piper")  # "piper" or "kokoro"
STREAMING_TTS_ENABLED = os.getenv("STREAMING_TTS_ENABLED", "true").lower() in ("true", "1", "yes")

# --- Local Skills ---
SKILLS_TIMEZONE = os.getenv("SKILLS_TIMEZONE", "America/Chicago")
SKILLS_MAX_TIMERS = int(os.getenv("SKILLS_MAX_TIMERS", "10"))

# --- Camera Skill ---
CAMERA_DEFAULT_MONITOR = int(os.getenv("CAMERA_DEFAULT_MONITOR", "1"))
CAMERA_MONITOR_MAP = os.getenv("CAMERA_MONITOR_MAP", "1=HDMI-1,2=HDMI-2")
CAMERA_MPV_IPC_DIR = os.getenv("CAMERA_MPV_IPC_DIR", "/tmp/atlas-mpv")

# --- Local LLM (Phi-3 via llama-server) ---
LOCAL_LLM_ENABLED = os.getenv("LOCAL_LLM_ENABLED", "true").lower() in ("true", "1", "yes")
LOCAL_LLM_PORT = int(os.getenv("LOCAL_LLM_PORT", "8081"))
LOCAL_LLM_MAX_TOKENS = int(os.getenv("LOCAL_LLM_MAX_TOKENS", "200"))
LOCAL_LLM_TIMEOUT = float(os.getenv("LOCAL_LLM_TIMEOUT", "120"))
LOCAL_LLM_TEMPERATURE = float(os.getenv("LOCAL_LLM_TEMPERATURE", "0.7"))

# --- LLM Routing ---
# "brain"  = Brain primary, local LLM fallback when Brain is offline
# "local"  = Local LLM primary for speech queries (Brain still gets vision/security)
LLM_ROUTE = os.getenv("LLM_ROUTE", "brain")
