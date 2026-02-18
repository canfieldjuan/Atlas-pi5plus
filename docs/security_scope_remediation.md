# Security Scope Remediation Tracker

Date opened: 2026-02-17
Node truth repo: Atlas-pi5plus (`/opt/atlas-node`)

## Scope Goal

Primary mission for this node is security sensing, eventing, escalation, and operator visibility.

## Issue Backlog

| ID | Severity | Status | Issue | Evidence | Target Files |
|----|----------|--------|-------|----------|--------------|
| SEC-001 | High | Open | Dashboard and stream surfaces are exposed without auth and broad bind/CORS | `atlas_node/config.py:134`, `atlas_node/dashboard.py:170`, `/opt/mediamtx/mediamtx.yml:20` | `atlas_node/config.py`, `atlas_node/dashboard.py`, `/opt/mediamtx/mediamtx.yml` |
| SEC-002 | High | Open | Person-triggered stream starts but is not explicitly released, causing long-running recording/streaming | `atlas_node/vision.py:575`, `atlas_node/stream_manager.py:57`, `atlas_node/vision.py:549` | `atlas_node/vision.py`, `atlas_node/stream_manager.py` |
| SEC-003 | High | Closed | Brain sends `escalation_alert` but edge node has no handler | Brain: `atlas_brain/escalation/evaluator.py:227`; Edge handlers: `atlas_node/main.py:194-203` | `atlas_node/main.py` |
| SEC-004 | Medium | Open | General assistant scope (time/timer/math/chat) exceeds strict security-node mission | `atlas_node/skills/__init__.py:26-30`, `atlas_node/local_llm.py:19` | `atlas_node/skills/__init__.py`, `atlas_node/config.py`, `atlas_node/speech.py` |
| SEC-005 | Medium | Open | Unknown face auto-enrollment defaults on, increasing identity noise/privacy risk | `atlas_node/config.py:73`, `atlas_node/face.py:347` | `atlas_node/config.py`, `atlas_node/face.py` |
| SEC-006 | Medium | Open | Recording footprint growth needs policy guardrails and controls | `/opt/mediamtx/mediamtx.yml:41-45`, recordings footprint observed | `/opt/mediamtx/mediamtx.yml`, ops policy docs |

## Execution Rules

- One issue per commit.
- Validate each touched file before moving to next issue.
- Preserve public contracts and existing function signatures.
- Use configurable settings for new behavior.

## Current Work Item

- Next: `SEC-002` (fix person-triggered stream release behavior)

## Remediation Log

- 2026-02-17: SEC-003 closed. Edge now handles Brain escalation_alert messages in atlas_node/main.py.
