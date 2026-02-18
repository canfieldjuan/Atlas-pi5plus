# Security Scope Remediation Tracker

Date opened: 2026-02-17
Node truth repo: Atlas-pi5plus (`/opt/atlas-node`)

## Scope Goal

Primary mission for this node is security sensing, eventing, escalation, and operator visibility.

## Issue Backlog

| ID | Severity | Status | Issue | Evidence | Target Files |
|----|----------|--------|-------|----------|--------------|
| SEC-001 | High | In Progress | Dashboard and stream surfaces are exposed without auth and broad bind/CORS | `atlas_node/config.py:134`, `atlas_node/dashboard.py:170`, `/opt/mediamtx/mediamtx.yml:20` | `atlas_node/config.py`, `atlas_node/dashboard.py`, `/opt/mediamtx/mediamtx.yml`, `web/dashboard.html` |
| SEC-002 | High | Closed | Person-triggered stream starts but is not explicitly released, causing long-running recording/streaming | `atlas_node/vision.py:575`, `atlas_node/stream_manager.py:57`, `atlas_node/vision.py:549` | `atlas_node/vision.py`, `atlas_node/stream_manager.py` |
| SEC-003 | High | Closed | Brain sends `escalation_alert` but edge node has no handler | Brain: `atlas_brain/escalation/evaluator.py:227`; Edge handlers: `atlas_node/main.py:194-203` | `atlas_node/main.py` |
| SEC-004 | Medium | Open | General assistant scope (time/timer/math/chat) exceeds strict security-node mission | `atlas_node/skills/__init__.py:26-30`, `atlas_node/local_llm.py:19` | `atlas_node/skills/__init__.py`, `atlas_node/config.py`, `atlas_node/speech.py` |
| SEC-005 | Medium | Closed | Unknown face auto-enrollment defaults on, increasing identity noise/privacy risk | `atlas_node/config.py:73`, `atlas_node/face.py:347` | `atlas_node/config.py`, `atlas_node/face.py` |
| SEC-006 | Medium | Open | Recording footprint growth needs policy guardrails and controls | `/opt/mediamtx/mediamtx.yml:41-45`, recordings footprint observed | `/opt/mediamtx/mediamtx.yml`, ops policy docs |

## Execution Rules

- One issue per commit.
- Validate each touched file before moving to next issue.
- Preserve public contracts and existing function signatures.
- Use configurable settings for new behavior.

## Current Work Item

- Continue: `SEC-001` (MediaMTX exposure controls + runtime token enablement)

## Remediation Log

- 2026-02-17: SEC-003 closed. Edge now handles Brain escalation_alert messages in atlas_node/main.py.
- 2026-02-17: SEC-002 closed. Stream idle logic now uses last person detection request time to auto-release person_detect stream when detections stop.
- 2026-02-17: SEC-001 partial. Added optional dashboard token auth and configurable CORS origin; dashboard frontend now forwards token query param to API/WS endpoints.
- 2026-02-17: SEC-001 runtime hardening enabled on this node. DASHBOARD_API_TOKEN was set in .env.local and validated with 401 (no token) / 200 (token) on /api/status.
- 2026-02-17: SEC-001 partial. MediaMTX API/RTSP/HLS listeners moved to loopback (`127.0.0.1`) and runtime-validated after container restart; WebRTC remains exposed on :8889 for operator dashboard viewing.
- 2026-02-17: SEC-005 closed. FACE_AUTO_ENROLL default changed from true to false; unknown-face auto-enrollment now requires explicit opt-in via environment config.
