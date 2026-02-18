#!/usr/bin/env python3
"""Register a face identity by capturing from the RTSP stream or a static image.

Usage:
    python register_face.py alice                    # capture from RTSP stream
    python register_face.py alice --image photo.jpg  # use a specific image file

Saves a 512-dim embedding as face_db/<name>.npy.
"""

import argparse
import os
import sys

# Add project root to path so we can import atlas_node
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np


def grab_frame_rtsp(url: str):
    """Capture a single frame from the RTSP stream."""
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print(f"ERROR: Cannot open RTSP stream: {url}")
        sys.exit(1)
    # Read a few frames to skip initial keyframe delay
    for _ in range(5):
        ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        print("ERROR: Failed to capture frame from stream")
        sys.exit(1)
    return frame


def grab_frame_device(device: int = 0, width: int = 1280, height: int = 720):
    """Capture a single frame directly from /dev/videoN."""
    cap = cv2.VideoCapture(device)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        print(f"ERROR: Cannot open /dev/video{device}")
        sys.exit(1)
    # Read a few frames to let auto-exposure settle
    for _ in range(10):
        ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        print("ERROR: Failed to capture frame from camera")
        sys.exit(1)
    return frame


def main():
    parser = argparse.ArgumentParser(description="Register a face for recognition")
    parser.add_argument("name", help="Identity name (e.g. 'alice')")
    parser.add_argument("--image", help="Path to image file (default: capture from RTSP)")
    parser.add_argument("--rtsp-url", default="",
                        help="RTSP stream URL (default: use /dev/video0 directly)")
    parser.add_argument("--device", type=int, default=0,
                        help="Camera device index (default: 0 = /dev/video0)")
    parser.add_argument("--face-db", default="/opt/atlas-node/face_db",
                        help="Face database directory")
    parser.add_argument("--det-model", default="/opt/atlas-node/models/face/RetinaFace_mobile320.rknn",
                        help="RetinaFace RKNN model path")
    parser.add_argument("--rec-model", default="/opt/atlas-node/models/face/w600k_mbf_i8.rknn",
                        help="MobileFaceNet RKNN model path")
    args = parser.parse_args()

    # Get frame
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"ERROR: Cannot read image: {args.image}")
            sys.exit(1)
        print(f"Loaded image: {args.image} ({frame.shape[1]}x{frame.shape[0]})")
    elif args.rtsp_url:
        print(f"Capturing from RTSP stream: {args.rtsp_url}")
        frame = grab_frame_rtsp(args.rtsp_url)
        print(f"Captured frame: {frame.shape[1]}x{frame.shape[0]}")
    else:
        print(f"Capturing from /dev/video{args.device}...")
        frame = grab_frame_device(args.device)
        print(f"Captured frame: {frame.shape[1]}x{frame.shape[0]}")

    # Import and initialize face recognizer (uses NPU)
    from atlas_node.face import FaceRecognizer, FaceDatabase, _align_face

    print("Loading face models on NPU...")
    rec = FaceRecognizer()
    rec.init()

    # Detect faces
    faces = rec._detect_faces(frame)
    if not faces:
        print("ERROR: No faces detected in the image")
        rec.release()
        sys.exit(1)

    print(f"Detected {len(faces)} face(s)")

    if len(faces) > 1:
        # Pick the largest face (by area)
        areas = [(f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]) for f in faces]
        best = int(np.argmax(areas))
        print(f"Using largest face (index {best}, area={areas[best]:.0f}px)")
        face = faces[best]
    else:
        face = faces[0]

    # Get embedding
    embedding = rec._get_embedding(frame, face["landmarks"])
    print(f"Embedding: shape={embedding.shape}, norm={np.linalg.norm(embedding):.3f}")

    # Save
    db = FaceDatabase(args.face_db)
    db.register(args.name, embedding)
    print(f"Registered '{args.name}' -> {args.face_db}/{args.name}.npy")

    rec.release()
    print("Done!")


if __name__ == "__main__":
    main()
