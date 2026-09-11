#!/usr/bin/env python3
"""Generate or checksum the fixed local fixtures used by all 48 trials."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
import wave
import zlib
from fractions import Fraction
from pathlib import Path
from typing import Any

FIXTURE_ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = FIXTURE_ROOT / "manifest.json"
GENERATED_FILES = ("document.html", "image.png", "audio.wav", "video.mp4")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    body = kind + payload
    return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)


def _write_document(path: Path) -> None:
    path.write_text("<html><body><p>alpha retrieval paragraph.</p><p>beta second paragraph.</p></body></html>\n")


def _write_png(path: Path) -> None:
    width, height = 32, 20
    rows = b"".join(b"\x00" + bytes((255, 0, 0)) * width for _ in range(height))
    payload = b"\x89PNG\r\n\x1a\n"
    payload += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    payload += _png_chunk(b"IDAT", zlib.compress(rows, level=9))
    payload += _png_chunk(b"IEND", b"")
    path.write_bytes(payload)


def _write_wav(path: Path) -> None:
    sample_rate = 8000
    samples = [int(12000 * math.sin(2 * math.pi * 440 * index / sample_rate)) for index in range(sample_rate)]
    frames = b"".join(struct.pack("<h", sample) for sample in samples)
    with wave.open(str(path), "wb") as audio_file:
        audio_file.setnchannels(1)
        audio_file.setsampwidth(2)
        audio_file.setframerate(sample_rate)
        audio_file.writeframes(frames)


def _write_video(path: Path) -> None:
    try:
        import av
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "Generating video.mp4 requires the PyAV and NumPy dependencies installed by Pixeltable"
        ) from exc

    container = av.open(str(path), "w")
    container.metadata.clear()
    stream = container.add_stream("mpeg4", rate=2)
    stream.width = 32
    stream.height = 24
    stream.pix_fmt = "yuv420p"
    stream.time_base = Fraction(1, 2)
    for index in range(4):
        pixels = np.full((24, 32, 3), (index * 60, 0, 255 - index * 60), dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
        frame.pts = index
        frame.time_base = Fraction(1, 2)
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def _manifest() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "fixtures": {
            "document.html": {
                "bytes": (FIXTURE_ROOT / "document.html").stat().st_size,
                "sha256": _sha256(FIXTURE_ROOT / "document.html"),
                "expectation": "two paragraphs; beta appears only in the second",
            },
            "image.png": {
                "bytes": (FIXTURE_ROOT / "image.png").stat().st_size,
                "sha256": _sha256(FIXTURE_ROOT / "image.png"),
                "expectation": "32x20 RGB red image",
            },
            "audio.wav": {
                "bytes": (FIXTURE_ROOT / "audio.wav").stat().st_size,
                "sha256": _sha256(FIXTURE_ROOT / "audio.wav"),
                "expectation": "one second, mono, 8000 Hz, 440 Hz tone",
            },
            "video.mp4": {
                "bytes": (FIXTURE_ROOT / "video.mp4").stat().st_size,
                "sha256": _sha256(FIXTURE_ROOT / "video.mp4"),
                "expectation": "four 32x24 frames at 2 fps",
            },
        },
    }


def write_fixtures() -> None:
    FIXTURE_ROOT.mkdir(parents=True, exist_ok=True)
    _write_document(FIXTURE_ROOT / "document.html")
    _write_png(FIXTURE_ROOT / "image.png")
    _write_wav(FIXTURE_ROOT / "audio.wav")
    _write_video(FIXTURE_ROOT / "video.mp4")
    MANIFEST_PATH.write_text(json.dumps(_manifest(), indent=2, sort_keys=True) + "\n")


def check_fixtures() -> list[str]:
    errors: list[str] = []
    try:
        manifest = json.loads(MANIFEST_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        return [f"cannot read fixture manifest: {exc}"]
    entries = manifest.get("fixtures", {})
    for name in GENERATED_FILES:
        path = FIXTURE_ROOT / name
        if not path.is_file():
            errors.append(f"missing fixture: {name}")
            continue
        entry = entries.get(name, {})
        if path.stat().st_size != entry.get("bytes"):
            errors.append(f"byte-size mismatch: {name}")
        if _sha256(path) != entry.get("sha256"):
            errors.append(f"checksum mismatch: {name}")
    unknown = set(entries) - set(GENERATED_FILES)
    if unknown:
        errors.append(f"unknown fixture manifest entries: {', '.join(sorted(unknown))}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="Regenerate the committed fixture bytes and manifest.")
    mode.add_argument("--check", action="store_true", help="Verify committed fixture checksums without regenerating.")
    args = parser.parse_args()
    if args.write:
        try:
            write_fixtures()
        except (OSError, RuntimeError) as exc:
            print(f"fixture generation failed: {exc}", file=sys.stderr)
            return 1
    errors = check_fixtures()
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print(json.dumps(_manifest(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
