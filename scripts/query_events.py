#!/usr/bin/env python3
"""Query events from the Brain alerts API.

Usage:
    python query_events.py --last 1h              # last hour
    python query_events.py --last 24h --person juan
    python query_events.py --last 7d --type motion_detected
    python query_events.py --last 1h --limit 50
"""

import argparse
import json
import os
import sys
import urllib.request
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atlas_node import config


def parse_duration(s: str) -> float:
    """Parse duration string like '1h', '24h', '7d', '30m' to hours."""
    s = s.strip().lower()
    if s.endswith("d"):
        return float(s[:-1]) * 24
    elif s.endswith("h"):
        return float(s[:-1])
    elif s.endswith("m"):
        return float(s[:-1]) / 60
    else:
        return float(s)


def main():
    parser = argparse.ArgumentParser(description="Query Atlas event history from Brain")
    parser.add_argument("--last", default="1h",
                        help="Time window (e.g. '1h', '24h', '7d', '30m')")
    parser.add_argument("--person", default=None,
                        help="Filter by person name")
    parser.add_argument("--type", default=None, dest="event_type",
                        help="Filter by event type (motion_detected, person_entered, person_left, unknown_face)")
    parser.add_argument("--limit", type=int, default=100,
                        help="Max results (default: 100)")
    args = parser.parse_args()

    hours = parse_duration(args.last)
    since_minutes = int(hours * 60)

    params = {
        "event_type": "security",
        "node_id": config.NODE_ID,
        "include_acknowledged": "true",
        "since_minutes": str(since_minutes),
        "limit": str(args.limit),
    }

    query_string = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"{config.BRAIN_API_BASE}/alerts?{query_string}"

    try:
        req = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.URLError as e:
        print(f"Error: Cannot reach Brain API at {config.BRAIN_API_BASE}")
        print(f"  {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    alerts = data.get("alerts", [])

    # Client-side filters not supported by API
    if args.event_type:
        alerts = [a for a in alerts
                  if (a.get("event_data") or {}).get("event") == args.event_type]
    if args.person:
        person_lower = args.person.lower()
        alerts = [a for a in alerts
                  if person_lower in (a.get("event_data") or {}).get("name", "").lower()]

    if not alerts:
        print(f"No events found in the last {args.last}")
        return

    print(f"Found {len(alerts)} events in the last {args.last}:\n")

    for a in alerts:
        ed = a.get("event_data") or {}
        ts = a.get("triggered_at", "")[:19]
        event = ed.get("event", a.get("rule_name", "").replace("edge_security_", ""))
        confidence = ed.get("confidence", ed.get("combined_confidence", 0))
        name = ed.get("name", "")
        track_id = ed.get("track_id", "-")

        print(f"  [{ts}] {event:<20s}  "
              f"conf={confidence:.3f}  "
              f"name={name or '-':<15s}  "
              f"track={track_id}")


if __name__ == "__main__":
    main()
