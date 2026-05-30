#!/usr/bin/env python3
"""
GPS Mini-Map Overlay Generator
Thunderhill Raceway – May 29 2026

Two modes
─────────
  GPS-trace mode (default):
    Auto-renders the track from GPS data.  No calibration needed.
    python generate_minimap.py

  Map-image mode (recommended):
    Uses your actual track diagram (TH.png) as the background.
    Requires a calibration.json produced by calibrate.py first.
    python calibrate.py          # one-time click step
    python generate_minimap.py --map TH.png

Overlay the output video in your editor:
  Black background → Screen blend mode (dark areas vanish)
  Magenta (--bg magenta) → chroma-key

Options:
    --gps-folder PATH   Folder with GPS .txt files  [TH-May-29-2026/GPS]
    --output PATH       Output video file            [TH-May-29-2026/minimap_overlay.mp4]
    --map PATH          Track map PNG (TH.png)       [off]
    --calib PATH        Calibration JSON file        [calibration.json]
    --size N            Canvas size in pixels (sq)  [500]
    --fps N             Output frame rate            [30]
    --bg black|magenta  Background style             [black]
    --trail N           Trail length, seconds        [3.0]
"""

import os
import re
import glob
import json
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


def parse_gps_folder(folder: str) -> list:
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


# ── GPS timeline (binary-search interpolation) ─────────────────────────────────

class GpsTimeline:
    def __init__(self, points: list):
        self.t0       = points[0][0]
        self.t_end    = points[-1][0]
        self.duration = (self.t_end - self.t0).total_seconds()
        self._times   = np.array([(p[0] - self.t0).total_seconds() for p in points])
        self._lats    = np.array([p[1] for p in points])
        self._lons    = np.array([p[2] for p in points])
        self._spds    = np.array([p[3] for p in points], dtype=float)

    def at(self, t_sec: float):
        t_sec = float(np.clip(t_sec, 0, self.duration))
        idx = int(np.searchsorted(self._times, t_sec, side="right")) - 1
        idx = max(0, min(idx, len(self._times) - 2))
        t0, t1 = self._times[idx], self._times[idx + 1]
        a = (t_sec - t0) / (t1 - t0) if t1 > t0 else 0.0
        return (
            self._lats[idx] + a * (self._lats[idx + 1] - self._lats[idx]),
            self._lons[idx] + a * (self._lons[idx + 1] - self._lons[idx]),
            self._spds[idx] + a * (self._spds[idx + 1] - self._spds[idx]),
        )


# ── Coordinate mappers ─────────────────────────────────────────────────────────

class AutoMapper:
    """Fits GPS bounding box to canvas with correct aspect ratio. No calibration needed."""

    def __init__(self, points: list, width: int, height: int, padding: int):
        lats = [p[1] for p in points]
        lons = [p[2] for p in points]
        lat_min, lat_max = min(lats), max(lats)
        lon_min, lon_max = min(lons), max(lons)
        cos_lat = np.cos(np.radians((lat_min + lat_max) / 2))

        lon_range = (lon_max - lon_min) * cos_lat
        lat_range = lat_max - lat_min
        usable_w  = width  - 2 * padding
        usable_h  = height - 2 * padding
        scale     = min(usable_w / lon_range, usable_h / lat_range)

        self._x0      = padding + (usable_w - lon_range * scale) / 2
        self._y0      = padding + (usable_h - lat_range * scale) / 2
        self._scale   = scale
        self._cos_lat = cos_lat
        self._lat_max = lat_max
        self._lon_max = lon_max

    def __call__(self, lat: float, lon: float):
        x = int(self._x0 + (self._lon_max - lon) * self._cos_lat * self._scale)
        y = int(self._y0 + (self._lat_max - lat) * self._scale)
        return x, y


class CalibrationMapper:
    """
    2-point affine calibration.  Handles any map rotation/scale automatically
    using a complex-number similarity transform.
    """

    def __init__(self, calib: list):
        if len(calib) < 2:
            raise ValueError("Need at least 2 calibration points")
        p0, p1 = calib[0], calib[1]

        cos_lat = np.cos(np.radians((p0["lat"] + p1["lat"]) / 2))

        # GPS displacement from p0→p1 in (east, north) space
        # W-longitude: decreasing value = going east
        gps_e = -(p1["lon"] - p0["lon"]) * cos_lat   # positive = east
        gps_n =   p1["lat"] - p0["lat"]               # positive = north

        # Pixel displacement p0→p1 (screen: right = +x, DOWN = +y)
        pix_dx = p1["px"] - p0["px"]
        pix_dy = p1["py"] - p0["py"]

        # Complex representation: GPS in math coords, pixel with y flipped to math coords
        z_gps = complex(gps_e, gps_n)
        z_pix = complex(pix_dx, -pix_dy)   # flip y: screen-down → math-up

        # Similarity transform: z_pix = ratio * z_gps
        self._ratio   = z_pix / z_gps
        self._cos_lat = cos_lat
        self._lat0    = p0["lat"]
        self._lon0    = p0["lon"]
        self._px0     = p0["px"]
        self._py0     = p0["py"]

    def __call__(self, lat: float, lon: float):
        gps_e = -(lon - self._lon0) * self._cos_lat
        gps_n =   lat - self._lat0
        z = self._ratio * complex(gps_e, gps_n)
        x = int(self._px0 + z.real)
        y = int(self._py0 - z.imag)    # flip back: math-up → screen-down
        return x, y


# ── Background canvas builders ─────────────────────────────────────────────────

def make_trace_background(points: list, mapper, size: int, bg: tuple) -> np.ndarray:
    """Draw GPS trace as dim lines on a solid background."""
    canvas = np.full((size, size, 3), bg, dtype=np.uint8)
    prev = None
    for p in points:
        cur = mapper(p[1], p[2])
        if prev is not None:
            cv2.line(canvas, prev, cur, (55, 55, 55), 3, cv2.LINE_AA)
        prev = cur
    return canvas


def make_map_background(map_path: str, mapper, points: list,
                        size: int, bg: tuple) -> np.ndarray:
    """
    Load TH.png, invert it (white→black, track lines→white),
    then resize to `size × size` using the mapper's coordinate range to crop.
    """
    img = cv2.imread(map_path)
    if img is None:
        raise FileNotFoundError(f"Cannot open map: {map_path}")

    # Invert: white background → black, black track lines → white/gray
    inverted = 255 - img

    # Tone down the track lines so the red dot stands out
    # Scale from 0–255 to 0–140 (dim white) so dot is clearly brighter
    inverted = (inverted.astype(np.float32) * 0.55).astype(np.uint8)

    # Find the pixel extent of the GPS data in map-image coordinates
    all_px = [mapper(p[1], p[2]) for p in points]
    xs = [p[0] for p in all_px]
    ys = [p[1] for p in all_px]

    pad   = 40
    x_min = max(0,              min(xs) - pad)
    x_max = min(inverted.shape[1], max(xs) + pad)
    y_min = max(0,              min(ys) - pad)
    y_max = min(inverted.shape[0], max(ys) + pad)

    crop = inverted[y_min:y_max, x_min:x_max]
    if crop.size == 0:
        raise RuntimeError("Calibration mapped all GPS points outside the image — check calibration.json")

    # Resize crop to the output canvas size
    resized = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)

    # Build a thin wrapper mapper that accounts for the crop + resize
    scale_x = size / (x_max - x_min)
    scale_y = size / (y_max - y_min)
    x_min_f, y_min_f = float(x_min), float(y_min)

    return resized, scale_x, scale_y, x_min_f, y_min_f


# ── Frame drawing ──────────────────────────────────────────────────────────────

def draw_frame(base: np.ndarray, trail: list, pos, speed: float) -> np.ndarray:
    frame = base.copy()
    n = len(trail)

    for j in range(n - 1):
        a = (j + 1) / n
        cv2.line(frame, trail[j], trail[j + 1],
                 (0, 0, int(80 + 130 * a)),
                 max(1, int(5 * a)), cv2.LINE_AA)

    cv2.circle(frame, pos, 9, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, pos, 9, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(frame, f"{int(speed)} km/h", (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
    return frame


# ── Main render loop ───────────────────────────────────────────────────────────

def render(points: list, args) -> None:
    size      = args.size
    fps       = args.fps
    bg_color  = (0, 0, 0) if args.bg == "black" else (255, 0, 255)
    timeline  = GpsTimeline(points)

    use_map = bool(args.map)

    # ── Build mapper ──────────────────────────────────────────────────────────
    if use_map:
        calib_path = args.calib
        if not os.path.isfile(calib_path):
            raise SystemExit(
                f"\ncalibration.json not found at '{calib_path}'.\n"
                "Run:  python calibrate.py\nto click the GPS points on TH.png first."
            )
        with open(calib_path) as f:
            calib = json.load(f)
        mapper = CalibrationMapper(calib)
    else:
        mapper = AutoMapper(points, size, size, padding=30)

    # ── Build background ──────────────────────────────────────────────────────
    if use_map:
        base, sx, sy, x_off, y_off = make_map_background(
            args.map, mapper, points, size, bg_color
        )

        # Wrap the mapper to account for crop + resize
        raw_mapper = mapper
        def mapper(lat, lon):
            rx, ry = raw_mapper(lat, lon)
            return (int((rx - x_off) * sx), int((ry - y_off) * sy))
    else:
        base = make_trace_background(points, mapper, size, bg_color)

    # ── Write video ───────────────────────────────────────────────────────────
    total_frames = int(timeline.duration * fps)
    trail_max    = int(args.trail * fps)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, fps, (size, size))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open output: {args.output}")

    print(f"  {len(points):,} GPS points  |  {timeline.duration / 60:.1f} min  |  "
          f"{total_frames:,} frames @ {fps} fps")
    print(f"  {'Map: ' + args.map if use_map else 'GPS-trace background'}")
    print(f"  Output → {args.output}")

    trail: list = []
    for fi in range(total_frames):
        lat, lon, spd = timeline.at(fi / fps)
        pos = mapper(lat, lon)

        trail.append(pos)
        if len(trail) > trail_max:
            trail.pop(0)

        writer.write(draw_frame(base, trail, pos, spd))

        if fi % (fps * 30) == 0:
            print(f"  [{100 * fi / total_frames:5.1f}%]  "
                  f"t={fi / fps / 60:.1f} min  {int(spd)} km/h")

    writer.release()
    print("  Done.")

    # ── Static trace PNG ──────────────────────────────────────────────────────
    trace_path = os.path.splitext(args.output)[0] + "_trace.png"
    trace = base.copy()
    if not use_map:
        start = mapper(points[0][1], points[0][2])
        cv2.circle(trace, start, 7, (0, 255, 0), -1)
        cv2.putText(trace, "START", (start[0] + 10, start[1] + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 0), 1, cv2.LINE_AA)
    cv2.imwrite(trace_path, trace)
    print(f"  Trace PNG → {trace_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate GPS mini-map overlay video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run calibrate.py first when using --map.",
    )
    parser.add_argument("--gps-folder", default=r"TH-May-29-2026\GPS")
    parser.add_argument("--output",     default=r"TH-May-29-2026\minimap_overlay.mp4")
    parser.add_argument("--map",        default="",        help="Track map PNG (TH.png)")
    parser.add_argument("--calib",      default="calibration.json")
    parser.add_argument("--size",       type=int,   default=500)
    parser.add_argument("--fps",        type=int,   default=30)
    parser.add_argument("--bg",         choices=["black", "magenta"], default="black")
    parser.add_argument("--trail",      type=float, default=3.0)
    args = parser.parse_args()

    print(f"Parsing GPS: {args.gps_folder}")
    points = parse_gps_folder(args.gps_folder)
    if not points:
        raise SystemExit("No GPS points found — check --gps-folder")
    print(f"Loaded {len(points):,} points")

    render(points, args)


if __name__ == "__main__":
    main()
