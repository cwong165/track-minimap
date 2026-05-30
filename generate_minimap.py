#!/usr/bin/env python3
"""
GPS Mini-Map Overlay Generator
Thunderhill Raceway – May 29 2026

Parses GPS log files and renders a real-time animated minimap video:
a red dot moving along the track trace.

Overlay in your video editor:
  - Black background: use Screen blend mode → dark areas vanish
  - Magenta background (--bg magenta): chroma-key it out

Usage:
    pip install -r requirements.txt
    python generate_minimap.py [options]

Options:
    --gps-folder PATH   Folder with GPS .txt files  [TH-May-29-2026/GPS]
    --output PATH       Output video file            [TH-May-29-2026/minimap_overlay.mp4]
    --size N            Minimap size in pixels        [500]
    --fps N             Output frames per second      [30]
    --bg black|magenta  Background color              [black]
    --trail N           Fading trail length, seconds  [3.0]
"""

import os
import re
import glob
import argparse
from datetime import datetime, timedelta

import numpy as np
import cv2


# ── GPS parsing ────────────────────────────────────────────────────────────────

_GPS_RE = re.compile(
    r"(\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})\s+"
    r"N:([\d.]+)\s+W:([\d.]+)\s+"
    r"(\d+)\s+km/h"
)


def parse_gps_folder(folder: str) -> list[tuple]:
    """Return [(datetime, lat, lon, speed_kmh), ...] sorted by time, deduplicated."""
    points = []
    for path in sorted(glob.glob(os.path.join(folder, "*.txt"))):
        with open(path) as fh:
            for line in fh:
                m = _GPS_RE.search(line)
                if m:
                    ts = datetime.strptime(m.group(1), "%Y/%m/%d %H:%M:%S")
                    points.append((ts, float(m.group(2)), float(m.group(3)), int(m.group(4))))

    points.sort(key=lambda p: p[0])
    seen, deduped = set(), []
    for p in points:
        if p[0] not in seen:
            seen.add(p[0])
            deduped.append(p)
    return deduped


# ── Coordinate → pixel mapping ─────────────────────────────────────────────────

class CoordMapper:
    """Maps GPS (lat, lon) to pixel (x, y) preserving real-world aspect ratio."""

    def __init__(self, points: list[tuple], width: int, height: int, padding: int):
        lats = [p[1] for p in points]
        lons = [p[2] for p in points]
        lat_min, lat_max = min(lats), max(lats)
        lon_min, lon_max = min(lons), max(lons)

        # Correct longitude for latitude compression
        cos_lat = np.cos(np.radians((lat_min + lat_max) / 2))
        lat_range = lat_max - lat_min
        lon_range = (lon_max - lon_min) * cos_lat

        usable_w = width - 2 * padding
        usable_h = height - 2 * padding
        scale = min(usable_w / lon_range, usable_h / lat_range)

        # Center track within canvas
        self._x0 = padding + (usable_w - lon_range * scale) / 2
        self._y0 = padding + (usable_h - lat_range * scale) / 2
        self._scale = scale
        self._cos_lat = cos_lat
        self._lat_max = lat_max
        self._lon_max = lon_max

    def __call__(self, lat: float, lon: float) -> tuple[int, int]:
        x = int(self._x0 + (self._lon_max - lon) * self._cos_lat * self._scale)
        y = int(self._y0 + (self._lat_max - lat) * self._scale)
        return x, y


# ── Interpolation ──────────────────────────────────────────────────────────────

class GpsTimeline:
    """Fast binary-search interpolation over GPS points."""

    def __init__(self, points: list[tuple]):
        self.t0 = points[0][0]
        self.t_end = points[-1][0]
        self.duration = (self.t_end - self.t0).total_seconds()

        self._times = np.array([(p[0] - self.t0).total_seconds() for p in points])
        self._lats  = np.array([p[1] for p in points])
        self._lons  = np.array([p[2] for p in points])
        self._spds  = np.array([p[3] for p in points], dtype=float)

    def at(self, t_sec: float) -> tuple[float, float, float]:
        """Return (lat, lon, speed_kmh) at t_sec seconds from session start."""
        t_sec = np.clip(t_sec, 0, self.duration)
        idx = int(np.searchsorted(self._times, t_sec, side="right")) - 1
        idx = max(0, min(idx, len(self._times) - 2))

        t0, t1 = self._times[idx], self._times[idx + 1]
        alpha = (t_sec - t0) / (t1 - t0) if t1 > t0 else 0.0

        lat = self._lats[idx] + alpha * (self._lats[idx + 1] - self._lats[idx])
        lon = self._lons[idx] + alpha * (self._lons[idx + 1] - self._lons[idx])
        spd = self._spds[idx] + alpha * (self._spds[idx + 1] - self._spds[idx])
        return lat, lon, spd


# ── Drawing helpers ────────────────────────────────────────────────────────────

def build_track_base(points: list[tuple], mapper: CoordMapper,
                     size: int, bg: tuple[int, int, int]) -> np.ndarray:
    """Draw the full GPS trace as a dim reference line."""
    canvas = np.full((size, size, 3), bg, dtype=np.uint8)
    px_pts = [mapper(p[1], p[2]) for p in points]
    for i in range(len(px_pts) - 1):
        cv2.line(canvas, px_pts[i], px_pts[i + 1], (55, 55, 55), 3, cv2.LINE_AA)
    return canvas


def draw_frame(base: np.ndarray, trail: list[tuple[int, int]],
               pos: tuple[int, int], speed: float) -> np.ndarray:
    frame = base.copy()

    # Fading trail
    n = len(trail)
    for j in range(n - 1):
        alpha = (j + 1) / n
        intensity = int(80 + 100 * alpha)   # dim red → bright red
        thickness = max(1, int(5 * alpha))
        color = (0, 0, intensity)            # BGR red
        cv2.line(frame, trail[j], trail[j + 1], color, thickness, cv2.LINE_AA)

    # Current position dot
    cv2.circle(frame, pos, 9, (0, 0, 255), -1, cv2.LINE_AA)   # red fill
    cv2.circle(frame, pos, 9, (255, 255, 255), 1, cv2.LINE_AA) # white ring

    # Speed readout
    label = f"{int(speed)} km/h"
    cv2.putText(frame, label, (10, base.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    return frame


# ── Main render loop ───────────────────────────────────────────────────────────

def render(points: list[tuple], args) -> None:
    size     = args.size
    fps      = args.fps
    bg_color = (0, 0, 0) if args.bg == "black" else (255, 0, 255)

    mapper   = CoordMapper(points, size, size, padding=30)
    timeline = GpsTimeline(points)
    base     = build_track_base(points, mapper, size, bg_color)

    total_frames  = int(timeline.duration * fps)
    trail_max     = int(args.trail * fps)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (size, size))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open output file: {args.output}")

    print(f"  {len(points):,} GPS points  |  {timeline.duration/60:.1f} min  |  {total_frames:,} frames @ {fps}fps")
    print(f"  Output → {args.output}")

    trail: list[tuple[int, int]] = []

    for fi in range(total_frames):
        t_sec = fi / fps
        lat, lon, spd = timeline.at(t_sec)
        pos = mapper(lat, lon)

        trail.append(pos)
        if len(trail) > trail_max:
            trail.pop(0)

        writer.write(draw_frame(base, trail, pos, spd))

        if fi % (fps * 30) == 0:
            pct = 100 * fi / total_frames
            print(f"  [{pct:5.1f}%]  t={t_sec/60:.1f}min  {int(spd)} km/h")

    writer.release()
    print("  Done.")

    # Save static trace PNG alongside the video
    trace_path = os.path.splitext(args.output)[0] + "_trace.png"
    trace_img  = build_track_base(points, mapper, size, bg_color)
    start_px   = mapper(points[0][1], points[0][2])
    cv2.circle(trace_img, start_px, 7, (0, 255, 0), -1)
    cv2.putText(trace_img, "START", (start_px[0] + 11, start_px[1] + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 0), 1, cv2.LINE_AA)
    cv2.imwrite(trace_path, trace_img)
    print(f"  Trace PNG → {trace_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate GPS mini-map overlay video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--gps-folder", default=r"TH-May-29-2026\GPS",
                        help="Folder containing GPS .txt log files")
    parser.add_argument("--output",     default=r"TH-May-29-2026\minimap_overlay.mp4",
                        help="Output video path (.mp4)")
    parser.add_argument("--size",       type=int,   default=500,
                        help="Minimap canvas size in pixels (square)")
    parser.add_argument("--fps",        type=int,   default=30,
                        help="Output video frame rate")
    parser.add_argument("--bg",         choices=["black", "magenta"], default="black",
                        help="Background: black (Screen blend) or magenta (chroma-key)")
    parser.add_argument("--trail",      type=float, default=3.0,
                        help="Fading trail length in seconds")
    args = parser.parse_args()

    print(f"Parsing GPS files from: {args.gps_folder}")
    points = parse_gps_folder(args.gps_folder)
    if not points:
        raise SystemExit("No GPS points found — check --gps-folder path")
    print(f"Loaded {len(points):,} GPS points")

    render(points, args)


if __name__ == "__main__":
    main()
