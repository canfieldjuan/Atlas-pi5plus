#!/usr/bin/env python3
"""Query the local event store.

Usage:
    python query_events.py --last 1h              # last hour
    python query_events.py --last 24h --person juan
    python query_events.py --last 7d --type motion_detected
    python query_events.py --last 1h --limit 50
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atlas_node.event_store import EventStore


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
    parser = argparse.ArgumentParser(description="Query Atlas edge node event history")
    parser.add_argument("--last", default="1h",
                        help="Time window (e.g. '1h', '24h', '7d', '30m')")
    parser.add_argument("--person", default=None,
                        help="Filter by person name")
    parser.add_argument("--type", default=None, dest="event_type",
                        help="Filter security events by type (motion_detected, person_entered, person_left, unknown_face)")
    parser.add_argument("--limit", type=int, default=100,
                        help="Max results (default: 100)")
    parser.add_argument("--db", default=None,
                        help="Path to events.db (default: from config)")
    args = parser.parse_args()

    hours = parse_duration(args.last)

    store = EventStore(db_path=args.db)
    store.init()

    results = store.query_recent(
        hours=hours,
        person=args.person,
        event_type=args.event_type,
        limit=args.limit,
    )

    if not results:
        print(f"No events found in the last {args.last}")
        store.close()
        return

    print(f"Found {len(results)} events in the last {args.last}:\n")

    for r in results:
        ts = r["timestamp"][:19]  # trim microseconds
        if r["table"] == "recognition":
            print(f"  [{ts}] RECOGNITION  {r['person_name']:<15s}  "
                  f"type={r['recognition_type']:<10s}  "
                  f"conf={r['confidence']:.3f}  "
                  f"track={r.get('track_id', '-')}")
        else:
            print(f"  [{ts}] SECURITY     {r['event_type']:<20s}  "
                  f"conf={r.get('confidence', 0):.3f}  "
                  f"track={r.get('track_id', '-')}")

    store.close()


if __name__ == "__main__":
    main()
