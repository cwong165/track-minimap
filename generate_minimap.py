#!/usr/bin/env python3
"""
GPS Mini-Map Overlay Generator
Thunderhill Raceway - May 29 2026

How the dot stays on the track
-------------------------------
1. AffineMapper: least-squares fit over all calibration corners gives a
   GPS -> canvas-pixel transform that handles any map rotation/scale.

2. TrackSnapper: converts the entire GPS session log to canvas pixels
   (via AffineMapper), lightly smooths out noise, then for every video
   frame snaps the current GPS position to the nearest segment on that
   full pixel polyline.

   The polyline comes from the actual driven path, so every real curve
   the car took is captured -- not just the straight lines between your
   calibration corners.  The dot physically cannot leave the track line
   because it is always projected onto it.

Workflow:
  1. py calibrate.py          # one-time: click corners, paste GPS block
  2. py generate_minimap.py --map TH.png

Overlay in editor: Screen blend mode (black bg disappears).

Options:
  --gps-folder PATH   GPS log folder            [TH-May-29-2026/GPS]
  --output     PATH   Output video              [TH-May-29-2026/minimap_overlay.mp4]
  --map        PATH   Track diagram PNG         [required]
  --calib      PATH   Calibration JSON          [calibration.json]
  --size       N      Canvas size px (square)   [500]
  --fps        N      Frame rate                [30]
  --trail      N      Trail length, seconds     [2.0]
  --smooth     N      GPS noise smoothing wnd   [5]
  --bg         black|magenta                    [black]
"""

import os
import re
import glob
import json
import argparse
from datetime import datetime

import numpy as np
import cv2


# ---------------------------------------------------------------------------
# GPS parsing
# ---------------------------------------------------------------------------

_GPS_RE = re.compile(
    r"(\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})\s+"
    r"N:([\d.]+)\s+W:([\d.]+)\s+"
    r"(\d+)\s+km/h"
)


def parse_gps_folder(folder: str) -> list:
    """Return [(datetime, lat, lon), ...] sorted and deduped.
    Longitude stored as negative (standard signed convention)."""
    rows = []
    for path in sorted(glob.glob(os.path.join(folder, "*.txt"))):
        with open(path) as fh:
            for line in fh:
                m = _GPS_RE.search(line)
                if m:
                    ts = datetime.strptime(m.group(1), "%Y/%m/%d %H:%M:%S")
                    rows.append((ts, float(m.group(2)), -float(m.group(3))))
    rows.sort(key=lambda r: r[0])
    seen, out = set(), []
    for r in rows:
        if r[0] not in seen:
            seen.add(r[0])
            out.append(r)
    return out


# ---------------------------------------------------------------------------
# GPS timeline  (binary-search interpolation)
# ---------------------------------------------------------------------------

class GpsTimeline:
    def __init__(self, points: list):
        self.t0       = points[0][0]
        self.duration = (points[-1][0] - self.t0).total_seconds()
        self._times   = np.array([(p[0] - self.t0).total_seconds() for p in points])
        self._lats    = np.array([p[1] for p in points])
        self._lons    = np.array([p[2] for p in points])

    def at(self, t_sec: float):
        t_sec = float(np.clip(t_sec, 0, self.duration))
        idx   = int(np.searchsorted(self._times, t_sec, side="right")) - 1
        idx   = max(0, min(idx, len(self._times) - 2))
        t0, t1 = self._times[idx], self._times[idx + 1]
        a = (t_sec - t0) / (t1 - t0) if t1 > t0 else 0.0
        return (
            float(self._lats[idx] + a * (self._lats[idx + 1] - self._lats[idx])),
            float(self._lons[idx] + a * (self._lons[idx + 1] - self._lons[idx])),
        )


# ---------------------------------------------------------------------------
# AffineMapper  (calibration corners -> least-squares GPS->pixel transform)
# ---------------------------------------------------------------------------

class AffineMapper:
    """
    Fits a 2-D affine transform (GPS metric -> canvas pixel) from N
    calibration corners using least squares.  Works correctly for any
    map rotation or scale, and improves with more corners.

    Longitude uses standard signed convention (-122.33 for western US).
    """

    def __init__(self, corners: list):
        mean_lat  = np.mean([c["lat"] for c in corners])
        self._cos = np.cos(np.radians(mean_lat))
        self._R   = 111_111.0          # metres per degree

        # GPS -> metric (east positive, north positive)
        E  = np.array([c["lon"] * self._cos * self._R for c in corners])
        N  = np.array([c["lat"]               * self._R for c in corners])
        PX = np.array([c["px"] for c in corners], dtype=float)
        PY = np.array([c["py"] for c in corners], dtype=float)

        M = np.column_stack([E, N, np.ones(len(corners))])
        self._cx, *_ = np.linalg.lstsq(M, PX, rcond=None)
        self._cy, *_ = np.linalg.lstsq(M, PY, rcond=None)

    def __call__(self, lat: float, lon: float):
        """Returns (px, py) as floats in canvas space."""
        v = np.array([lon * self._cos * self._R,
                      lat               * self._R,
                      1.0])
        return float(self._cx @ v), float(self._cy @ v)


# ---------------------------------------------------------------------------
# TrackSnapper  (snaps any pixel position to the driven GPS trace)
# ---------------------------------------------------------------------------

class TrackSnapper:
    """
    Converts the full GPS session log to a dense pixel polyline via
    AffineMapper, then snaps any query position to the nearest segment
    on that polyline.

    Because the polyline comes from the actual driven path every real
    curve is captured.  Snapping to a segment (not just the nearest
    point) ensures smooth, continuous motion along the track.
    """

    def __init__(self, gps_points: list, mapper: AffineMapper, smooth: int = 5):
        # Convert every GPS reading to canvas pixels
        px = np.array([mapper(p[1], p[2])[0] for p in gps_points])
        py = np.array([mapper(p[1], p[2])[1] for p in gps_points])

        # Light smoothing to suppress GPS noise (preserves curvature)
        if smooth > 1:
            k = np.ones(smooth) / smooth
            px = np.convolve(px, k, mode="same")
            py = np.convolve(py, k, mode="same")

        self._px = px
        self._py = py
        self._n  = len(px)
        print(f"  Track polyline: {self._n} points from GPS session data")

    def __call__(self, raw_px: float, raw_py: float) -> tuple:
        """Snap (raw_px, raw_py) to nearest segment on the GPS trace polyline."""
        dx    = self._px - raw_px
        dy    = self._py - raw_py
        dist2 = dx * dx + dy * dy

        # Examine segments adjacent to the 8 nearest polyline points
        k = min(8, self._n - 1)
        candidates = np.argpartition(dist2, k)[:k]

        best_d2 = np.inf
        best_x  = self._px[0]
        best_y  = self._py[0]

        for idx in candidates:
            for j in (int(idx) - 1, int(idx)):
                if j < 0 or j >= self._n - 1:
                    continue
                ax, ay   = self._px[j],     self._py[j]
                bx, by   = self._px[j + 1], self._py[j + 1]
                dab_x    = bx - ax
                dab_y    = by - ay
                len2     = dab_x * dab_x + dab_y * dab_y
                if len2 < 0.1:
                    continue
                t = ((raw_px - ax) * dab_x + (raw_py - ay) * dab_y) / len2
                t = max(0.0, min(1.0, t))
                nx = ax + t * dab_x
                ny = ay + t * dab_y
                d2 = (raw_px - nx) ** 2 + (raw_py - ny) ** 2
                if d2 < best_d2:
                    best_d2 = d2
                    best_x, best_y = nx, ny

        return int(round(best_x)), int(round(best_y))


# ---------------------------------------------------------------------------
# Canvas builder
# ---------------------------------------------------------------------------

def build_canvas(map_path: str, calib_path: str,
                 canvas_size: int, bg_color: tuple):
    """
    Returns (canvas_bgr, AffineMapper).

    TH.png is letterboxed into a square dark canvas with colours
    inverted (white->black bg, track lines->white/gray).
    """
    img = cv2.imread(map_path)
    if img is None:
        raise FileNotFoundError(f"Cannot open map: {map_path}")

    with open(calib_path) as f:
        raw = json.load(f)
    if len(raw) < 3:
        raise ValueError("calibration.json needs at least 3 corners")

    h, w   = img.shape[:2]
    scale  = min(canvas_size / w, canvas_size / h)
    sw, sh = int(w * scale), int(h * scale)
    off_x  = (canvas_size - sw) // 2
    off_y  = (canvas_size - sh) // 2

    inverted = 255 - cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    inverted = (inverted.astype(np.float32) * 0.60).astype(np.uint8)

    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    if bg_color != (0, 0, 0):
        canvas[:] = bg_color
    canvas[off_y:off_y + sh, off_x:off_x + sw] = inverted

    # Scale calibration pixel coords from original image space -> canvas space
    corners = [
        {
            "lat": c["lat"],
            "lon": c["lon"],
            "px":  c["px_orig"] * scale + off_x,
            "py":  c["py_orig"] * scale + off_y,
        }
        for c in raw
    ]

    return canvas, AffineMapper(corners)


# ---------------------------------------------------------------------------
# Frame drawing
# ---------------------------------------------------------------------------

def draw_frame(base: np.ndarray, trail: list, pos: tuple) -> np.ndarray:
    frame = base.copy()
    n = len(trail)
    for j in range(n - 1):
        a = (j + 1) / n
        cv2.line(frame, trail[j], trail[j + 1],
                 (0, 0, int(60 + 140 * a)),
                 max(1, int(4 * a)), cv2.LINE_AA)
    cv2.circle(frame, pos, 8, (0,   0, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, pos, 8, (255, 255, 255),  1, cv2.LINE_AA)
    return frame


# ---------------------------------------------------------------------------
# Render loop
# ---------------------------------------------------------------------------

def render(gps_points: list, args) -> None:
    bg_color = (0, 0, 0) if args.bg == "black" else (255, 0, 255)

    print("Building canvas and calibration transform...")
    canvas, mapper = build_canvas(args.map, args.calib, args.size, bg_color)

    print("Building track polyline from GPS session data...")
    snapper = TrackSnapper(gps_points, mapper, smooth=args.smooth)

    timeline     = GpsTimeline(gps_points)
    total_frames = int(timeline.duration * args.fps)
    trail_max    = int(args.trail * args.fps)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, args.fps,
                             (args.size, args.size))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open output: {args.output}")

    print(f"  {len(gps_points):,} GPS pts | "
          f"{timeline.duration / 60:.1f} min | "
          f"{total_frames:,} frames @ {args.fps} fps")
    print(f"  Output: {args.output}")

    trail: list = []
    for fi in range(total_frames):
        lat, lon = timeline.at(fi / args.fps)
        raw_px, raw_py = mapper(lat, lon)
        pos = snapper(raw_px, raw_py)

        trail.append(pos)
        if len(trail) > trail_max:
            trail.pop(0)

        writer.write(draw_frame(canvas, trail, pos))

        if fi % (args.fps * 30) == 0:
            print(f"  [{100 * fi / total_frames:5.1f}%]  "
                  f"t={fi / args.fps / 60:.1f} min")

    writer.release()
    print("  Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="GPS mini-map with track-snapped dot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--gps-folder", default=r"TH-May-29-2026\GPS")
    p.add_argument("--output",     default=r"TH-May-29-2026\minimap_overlay.mp4")
    p.add_argument("--map",        required=True, help="Track diagram PNG (TH.png)")
    p.add_argument("--calib",      default="calibration.json")
    p.add_argument("--size",       type=int,   default=500)
    p.add_argument("--fps",        type=int,   default=30)
    p.add_argument("--trail",      type=float, default=2.0)
    p.add_argument("--smooth",     type=int,   default=5,
                   help="GPS noise smoothing window (readings). "
                        "Higher = smoother but slightly less responsive.")
    p.add_argument("--bg",         choices=["black", "magenta"], default="black")
    args = p.parse_args()

    print(f"Parsing GPS: {args.gps_folder}")
    points = parse_gps_folder(args.gps_folder)
    if not points:
        raise SystemExit("No GPS points found -- check --gps-folder")
    print(f"Loaded {len(points):,} GPS points")

    if not os.path.isfile(args.calib):
        raise SystemExit(
            f"\nNo calibration at '{args.calib}'.\nRun:  py calibrate.py  first."
        )

    render(points, args)


if __name__ == "__main__":
    main()
