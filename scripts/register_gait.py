#!/usr/bin/env python3
"""Register a gait identity by capturing walking from the RTSP stream.

Usage:
    python register_gait.py juan --duration 10    # capture 10s of walking

Runs YOLO World (person detection) + YOLOv8-pose per frame, extracts gait features,
and saves to gait_db/<name>.npy.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Register a gait pattern for identification")
    parser.add_argument("name", help="Identity name (e.g. 'juan')")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Capture duration in seconds (default: 10)")
    parser.add_argument("--rtsp-url", default="rtsp://localhost:8554/cam1",
                        help="RTSP stream URL")
    parser.add_argument("--gait-db", default="/opt/atlas-node/gait_db",
                        help="Gait database directory")
    parser.add_argument("--pose-model",
                        default="/opt/atlas-node/models/pose/yolov8n-pose.rknn",
                        help="YOLOv8n-pose RKNN model path")
    parser.add_argument("--fps", type=float, default=5.0,
                        help="Capture FPS (default: 5)")
    args = parser.parse_args()

    # Open RTSP stream
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    cap = cv2.VideoCapture(args.rtsp_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"ERROR: Cannot open RTSP stream: {args.rtsp_url}")
        sys.exit(1)
    print(f"Opened RTSP stream: {args.rtsp_url}")

    # Init pose estimator
    from atlas_node.gait import (
        GaitRecognizer, GaitAnalyzer, GaitDatabase,
        _letterbox, _decode_pose_outputs, _nms, _POSE_INPUT_SIZE,
        compute_gait_embedding,
    )
    from rknnlite.api import RKNNLite

    print(f"Loading YOLOv8n-pose: {args.pose_model}")
    rknn_pose = RKNNLite()
    ret = rknn_pose.load_rknn(args.pose_model)
    if ret != 0:
        print(f"ERROR: Failed to load pose model (code {ret})")
        sys.exit(1)
    ret = rknn_pose.init_runtime(core_mask=RKNNLite.NPU_CORE_1)
    if ret != 0:
        print(f"ERROR: Failed to init pose runtime (code {ret})")
        sys.exit(1)
    print("Pose model ready")

    # Capture frames and extract keypoints
    interval = 1.0 / args.fps
    total_frames = int(args.duration * args.fps)
    keypoints_seq = []

    print(f"\nWalk in front of the camera for {args.duration}s ({total_frames} frames at {args.fps} FPS)...")
    print("Starting capture in 3 seconds...")
    time.sleep(3)
    print("GO!")

    for frame_idx in range(total_frames):
        t0 = time.monotonic()

        ret, frame = cap.read()
        if not ret:
            print(f"WARNING: Frame read failed at {frame_idx}")
            continue

        h, w = frame.shape[:2]

        # Run pose estimation
        img, scale, pad = _letterbox(frame)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_input = np.expand_dims(img_rgb, axis=0)

        outputs = rknn_pose.inference(inputs=[img_input])
        boxes, scores, kpts = _decode_pose_outputs(outputs)

        # Filter and NMS
        mask = scores >= 0.35
        boxes = boxes[mask]
        scores = scores[mask]
        kpts = kpts[mask]

        if len(boxes) > 0:
            keep = _nms(boxes, scores, 0.45)
            boxes = boxes[keep]
            kpts = kpts[keep]

            # Undo letterbox
            dw, dh = pad
            boxes[:, [0, 2]] = np.clip((boxes[:, [0, 2]] - dw) / scale, 0, w)
            boxes[:, [1, 3]] = np.clip((boxes[:, [1, 3]] - dh) / scale, 0, h)
            kpts[:, :, 0] = (kpts[:, :, 0] - dw) / scale
            kpts[:, :, 1] = (kpts[:, :, 1] - dh) / scale

            # Use the largest person detection
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            best = int(np.argmax(areas))
            keypoints_seq.append(kpts[best])

            print(f"\r  Frame {frame_idx + 1}/{total_frames}: "
                  f"{len(boxes)} person(s), {len(keypoints_seq)} valid keyframes",
                  end="", flush=True)
        else:
            print(f"\r  Frame {frame_idx + 1}/{total_frames}: "
                  f"no person detected, {len(keypoints_seq)} valid keyframes",
                  end="", flush=True)

        elapsed = time.monotonic() - t0
        if elapsed < interval:
            time.sleep(interval - elapsed)

    print()
    cap.release()
    rknn_pose.release()

    print(f"\nCollected {len(keypoints_seq)} keypoint frames")

    if len(keypoints_seq) < 20:
        print("ERROR: Not enough frames with valid poses (need at least 20)")
        sys.exit(1)

    # Extract 256-dim gait embedding
    features = compute_gait_embedding(keypoints_seq)
    if features is None:
        print("ERROR: Failed to extract gait embedding (not enough visible keypoints)")
        sys.exit(1)

    print(f"Gait embedding: dim={len(features)}, norm={np.linalg.norm(features):.3f}")
    print(f"  First 12 values: {features[:12]}")
    print(f"  Non-zero dims: {np.count_nonzero(features)}/256")

    # Save
    from pathlib import Path
    db_dir = Path(args.gait_db)
    db_dir.mkdir(parents=True, exist_ok=True)
    out_path = db_dir / f"{args.name}.npy"
    np.save(out_path, features)
    print(f"\nRegistered gait '{args.name}' -> {out_path}")
    print("Done!")


if __name__ == "__main__":
    main()
