# GUI & Camera Feed Audit (2026-02-14)

Comprehensive audit of all GUI implementations, camera feed displays, and web
dashboards across Edge (Pi) and Brain.

## Inventory

### Edge Node (Orange Pi 5 Plus)

| Component | File | Type | Status |
|-----------|------|------|--------|
| Camera Skill | `atlas_node/skills/camera_skill.py` | Voice -> MPV fullscreen on Pi HDMI | **Active** |
| Security Dashboard | `web/dashboard.html` + `atlas_node/dashboard.py` | aiohttp :8080, WebRTC + events + stats | **Active** |
| Monitor Display | `web/monitor.html` | Full-screen WebRTC, Brain WS control | **Dead code** |
| Startup Banner | `atlas_node/startup_display.py` | ANSI text on /dev/tty1 | **Active** |

### Brain Server

| Component | File | Type | Status |
|-----------|------|------|--------|
| Camera Display Tool | `atlas_brain/tools/display.py` | Voice -> FFplay on Brain monitors | **Broken** (atlas_vision not running) |
| YOLO Webcam Stream | `atlas_brain/api/video.py` | MJPEG :8000/api/v1/video/webcam | **Active** |
| atlas_vision Service | `atlas_video-processing/atlas_vision/` | Standalone FastAPI :5002 | **Not deployed** |
| React Dashboard | `atlas-ui/` | Vite + React + TypeScript | **Dormant** (built, not served) |

### Shared Infrastructure

| Component | Config | Ports | Status |
|-----------|--------|-------|--------|
| MediaMTX | `/opt/mediamtx/` (Docker) | RTSP :8554, WebRTC :8889, HLS :8888, API :9997 | **Active** |
| cam1-capture | systemd service | FFmpeg -> RTSP | **Active** |

---

## Redundancy Findings

### 1. Edge MPV Skill vs Brain FFplay Tool -- NOT REDUNDANT

- **Edge skill**: Runs MPV on Pi HDMI from RTSP (MediaMTX). Voice-triggered locally.
- **Brain tool**: Runs FFplay on Brain PC monitors from atlas_vision MJPEG. LLM-triggered.
- Different machines, different displays, different video sources. Keep both.
- **Issue**: Brain tool is broken because atlas_vision (:5002) is not running.

### 2. dashboard.html vs monitor.html -- monitor.html is DEAD CODE

- **dashboard.html**: Full admin UI with WebRTC video, live detections panel, events
  timeline, system stats. Served at `:8080/`.
- **monitor.html**: Bare full-screen WebRTC viewer with Brain WS control surface
  (`show_stream`, `show_url` messages). Designed as a wall-mount display.
- **Problems with monitor.html**:
  - Not served by any route in `dashboard.py` (no handler maps to it)
  - Connects to `ws://atlas-brain:8000/api/v1/ws/monitor` which does NOT exist
  - The Brain-side WebSocket endpoint was never implemented
- **Verdict**: monitor.html is dead code. Remove or implement the Brain endpoint.

### 3. Brain video.py vs atlas_vision -- OVERLAPPING, atlas_vision NOT DEPLOYED

- **Brain `api/video.py`**: MJPEG webcam stream with YOLO-World detection. Loads
  YOLOv8/YOLO-World models in the Brain process. Only working video endpoint.
- **atlas_vision** (`atlas_video-processing/`): Standalone FastAPI service with full
  camera management, face/gait recognition streams, YOLO detection. Has its own
  requirements.txt and entry point.
- **Current state**:
  - atlas_vision is not running (no process, no systemd service, no Docker entry)
  - Brain video.py has deprecated endpoints (410 responses) pointing to atlas_vision
  - Brain display.py (FFplay tool) references atlas_vision URLs on :5002 that nothing serves
  - YOLO model loaded in Brain process duplicates what atlas_vision would provide
- **Verdict**: Planned separation of concerns (Brain = orchestration, atlas_vision =
  computation) was never completed. Only video.py works today.

### 4. React atlas-ui -- DORMANT

- Built `dist/` directory exists but dev server is not running.
- Source files last modified 33+ days ago (Jan 12).
- No deployment config (no systemd, no nginx, no Docker).
- **Verdict**: Abandoned in favor of voice/CLI interaction.

---

## Action Items

| Priority | Item | Scope |
|----------|------|-------|
| Low | Remove or serve `web/monitor.html` | Edge |
| Medium | Deploy atlas_vision as systemd service OR remove dead references | Brain |
| Medium | Fix `tools/display.py` FFplay tool (points to non-running :5002) | Brain |
| Low | Decide: keep or remove React `atlas-ui` | Brain |
| Low | Consolidate YOLO loading (video.py vs atlas_vision) | Brain |

---

## Architecture (Current Working State)

```
Edge Node (Orange Pi 5+)
  atlas-node.service
    +-- VisionPipeline: RKNN YOLO World + face/gait on NPU
    +-- SpeechPipeline: SenseVoice STT + Silero VAD + speaker ID
    +-- CameraSkill: voice -> MPV fullscreen on HDMI (RTSP source)
    +-- DashboardServer: :8080 (dashboard.html + REST + WS)
    +-- WS Client -> Brain :8000
  cam1-capture.service
    +-- FFmpeg /dev/video0 -> MediaMTX RTSP :8554/cam1
  MediaMTX (Docker)
    +-- RTSP :8554, WebRTC :8889, HLS :8888

Brain Server (PC)
  uvicorn atlas_brain.main:app :8000
    +-- api/video.py: MJPEG webcam + YOLO-World detection
    +-- api/edge/websocket.py: edge node WS handler
    +-- tools/display.py: FFplay on Brain monitors (BROKEN)
    +-- LangGraph agent + Qwen3-30B-A3B via Ollama
  NOT RUNNING:
    +-- atlas_vision (:5002) -- planned but undeployed
    +-- atlas-ui (React) -- built but not served
```
