#!/usr/bin/env python3
"""Register a speaker identity by recording from the microphone or from an audio file.

Usage:
    python register_speaker.py juan --duration 5      # record 5s from mic
    python register_speaker.py juan --audio voice.wav  # use existing audio file

Saves a speaker embedding as speaker_db/<name>.npy.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def record_audio(duration: float, sample_rate: int = 16000, device_index=None) -> np.ndarray:
    """Record audio from the microphone."""
    import pyaudio

    pa = pyaudio.PyAudio()
    print(f"Recording {duration}s of audio... Speak now!")
    stream = pa.open(
        format=pyaudio.paFloat32,
        channels=1,
        rate=sample_rate,
        input=True,
        input_device_index=device_index,
        frames_per_buffer=int(sample_rate * 0.1),
    )

    frames = []
    total_samples = int(sample_rate * duration)
    read_size = int(sample_rate * 0.1)
    collected = 0
    while collected < total_samples:
        chunk = min(read_size, total_samples - collected)
        raw = stream.read(chunk, exception_on_overflow=False)
        frames.append(np.frombuffer(raw, dtype=np.float32))
        collected += chunk

    stream.stop_stream()
    stream.close()
    pa.terminate()
    print("Recording complete.")

    return np.concatenate(frames)


def load_audio_file(path: str, sample_rate: int = 16000) -> np.ndarray:
    """Load audio from a WAV file and resample to target rate if needed."""
    import wave

    with wave.open(path, "rb") as wf:
        assert wf.getnchannels() == 1, f"Expected mono audio, got {wf.getnchannels()} channels"
        file_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    sample_width = wf.getsampwidth()
    if sample_width == 2:
        audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(raw, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    # Simple resample if needed
    if file_rate != sample_rate:
        import scipy.signal
        audio = scipy.signal.resample(
            audio, int(len(audio) * sample_rate / file_rate)
        ).astype(np.float32)
        print(f"Resampled from {file_rate}Hz to {sample_rate}Hz")

    return audio


def main():
    parser = argparse.ArgumentParser(description="Register a speaker for voice identification")
    parser.add_argument("name", help="Speaker identity name (e.g. 'juan')")
    parser.add_argument("--audio", help="Path to WAV audio file (default: record from mic)")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Recording duration in seconds (default: 5)")
    parser.add_argument("--sample-rate", type=int, default=16000,
                        help="Audio sample rate (default: 16000)")
    parser.add_argument("--device", type=int, default=None,
                        help="Audio input device index (default: system default)")
    parser.add_argument("--speaker-db", default="/opt/atlas-node/speaker_db",
                        help="Speaker database directory")
    parser.add_argument("--model", default="/opt/atlas-node/models/speaker/"
                        "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx",
                        help="Speaker embedding model path")
    args = parser.parse_args()

    # Get audio
    if args.audio:
        print(f"Loading audio from: {args.audio}")
        audio = load_audio_file(args.audio, args.sample_rate)
    else:
        audio = record_audio(args.duration, args.sample_rate, args.device)

    duration = len(audio) / args.sample_rate
    print(f"Audio: {duration:.2f}s, {len(audio)} samples")

    if duration < 1.0:
        print("ERROR: Audio too short (minimum 1 second required)")
        sys.exit(1)

    # Extract embedding
    import sherpa_onnx

    print(f"Loading speaker model: {args.model}")
    ext_config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
        model=args.model,
        num_threads=2,
    )
    extractor = sherpa_onnx.SpeakerEmbeddingExtractor(ext_config)

    stream = extractor.create_stream()
    stream.accept_waveform(sample_rate=args.sample_rate, waveform=audio)
    stream.input_finished()

    if not extractor.is_ready(stream):
        print("ERROR: Extractor not ready -- audio may be too short or invalid")
        sys.exit(1)

    embedding = np.array(extractor.compute(stream), dtype=np.float32)
    print(f"Embedding: dim={len(embedding)}, norm={np.linalg.norm(embedding):.3f}")

    # Save
    from pathlib import Path
    db_dir = Path(args.speaker_db)
    db_dir.mkdir(parents=True, exist_ok=True)
    out_path = db_dir / f"{args.name}.npy"
    np.save(out_path, embedding)
    print(f"Registered speaker '{args.name}' -> {out_path}")
    print("Done!")


if __name__ == "__main__":
    main()
