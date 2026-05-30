#!/usr/bin/env python3
"""
GPS Mini-Map Overlay Generator
Thunderhill Raceway – May 29 2026

The red dot is always snapped to the nearest point on the track polyline
defined by your calibration corners — GPS noise cannot push it off the track.

Workflow:
  1. py calibrate.py          # one-time: click corners, enter GPS coords
  2. py generate_minimap.py --map TH.png

Overlay in editor: Screen blend mode (black bg disappears).

Options:
  --gps-folder PATH   GPS log folder            [TH-May-29-2026/GPS]
  --output PATH       Output video              [TH-May-29-2026/minimap_overlay.mp4]
  --map PATH          Track diagram PNG         [required for map mode]
  --calib PATH        Calibration JSON          [calibration.json]
  --size N            Canvas width px           [500]
  --fps N             Frame rate                [30]
  --trail N           Trail seconds             [2.0]
  --bg black|magenta  Background                [black]
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
    rows = []
    for path in sorted(glob.glob(os.path.join(folder, "*.txt"))):
        with open(path) as fh:
            for line in fh:
                m = _GPS_RE.search(line)
                if m:
                    ts  = datetime.strptime(m.group(1), "%Y/%m/%d %H:%M:%S")
                    rows.append((ts, float(m.group(2)), -float(m.group(3))))
    rows.sort(key=lambda r: r[0])
    seen, out = set(), []
    for r in rows:
        if r[0] not in seen:
            seen.add(r[0])
            out.append(r)
    return out


# ── GPS timeline ──────────────────────────────────────────────────────────────

class GpsTimeline:
    def __init__(self, points: list):
        self.t0       = points[0][0]
        self.duration = (points[-1][0] - self.t0).total_seconds()
        self._times   = np.array([(p[0] - self.t0).total_seconds() for p in points])
        self._lats    = np.array([p[1] for p in points])
        self._lons    = np.array([p[2] for p in points])

    def at(self, t_sec: float):
        """Linearly interpolated (lat, lon) at t_sec from session start."""
        t_sec = float(np.clip(t_sec, 0, self.duration))
        idx = int(np.searchsorted(self._times, t_sec, side="right")) - 1
        idx = max(0, min(idx, len(self._times) - 2))
        t0, t1 = self._times[idx], self._times[idx + 1]
        a = (t_sec - t0) / (t1 - t0) if t1 > t0 else 0.0
        return (
            float(self._lats[idx] + a * (self._lats[idx + 1] - self._lats[idx])),
            float(self._lons[idx] + a * (self._lons[idx + 1] - self._lons[idx])),
        )


# ── Track projector ───────────────────────────────────────────────────────────

class TrackProjector:
    """
    Snaps (lat, lon) to the nearest point on the closed track polyline.

    GPS coordinates are converted to metric east/north offsets so that
    1 unit = 1 metre in all directions. This gives correct segment distances
    even for diagonal segments.
    """

    def __init__(self, corners: list):
        """
        corners: list of dicts {lat, lon, px, py}
        px/py are already in canvas (output) pixel space.
        """
        n = len(corners)
        self.n = n
        mean_lat = np.mean([c["lat"] for c in corners])
        self._cos = np.cos(np.radians(mean_lat))
        R = 111_111.0   # metres per degree

        # GPS metric positions — lon is standard negative (W hemisphere)
        # More positive lon = more east; ge increases eastward.
        self._ge = np.array([c["lon"] * self._cos * R for c in corners])
        self._gn = np.array([c["lat"]               * R for c in corners])

        # Canvas pixel positions
        self._px = np.array([c["px"] for c in corners], dtype=float)
        self._py = np.array([c["py"] for c in corners], dtype=float)

    def __call__(self, lat: float, lon: float) -> tuple[int, int]:
        R = 111_111.0
        pe = lon * self._cos * R
        pn = lat               * R

        best_d2   = np.inf
        best_x = self._px[0]
        best_y = self._py[0]

        for i in range(self.n):
            j = (i + 1) % self.n          # closed loop

            dae = self._ge[j] - self._ge[i]
            dan = self._gn[j] - self._gn[i]
            len2 = dae * dae + dan * dan

            t = 0.0 if len2 < 1e-6 else (
                ((pe - self._ge[i]) * dae + (pn - self._gn[i]) * dan) / len2
            )
            t = max(0.0, min(1.0, t))

            # Nearest GPS point on segment
            ne = self._ge[i] + t * dae
            nn = self._gn[i] + t * dan
            d2 = (pe - ne) ** 2 + (pn - nn) ** 2

            if d2 < best_d2:
                best_d2 = d2
                best_x  = self._px[i] + t * (self._px[j] - self._px[i])
                best_y  = self._py[i] + t * (self._py[j] - self._py[i])

        return int(round(best_x)), int(round(best_y))


# ── Canvas builder ────────────────────────────────────────────────────────────

def build_canvas_and_projector(map_path: str, calib_path: str,
                                canvas_size: int, bg_color: tuple):
    """
    Loads TH.png + calibration.json.
    Returns (canvas_bgr, TrackProjector).

    The map is letterboxed into a canvas_size × canvas_size square.
    Colors are inverted: black background, white/gray track lines.
    """
    img = cv2.imread(map_path)
    if img is None:
        raise FileNotFoundError(f"Cannot open map: {map_path}")

    with open(calib_path) as f:
        raw = json.load(f)
    if len(raw) < 3:
        raise ValueError("calibration.json has fewer than 3 corners — re-run calibrate.py")

    h, w = img.shape[:2]
    scale    = min(canvas_size / w, canvas_size / h)
    sw, sh   = int(w * scale), int(h * scale)
    off_x    = (canvas_size - sw) // 2
    off_y    = (canvas_size - sh) // 2

    # Invert: white→black, black lines→white; then dim lines so dot stands out
    inverted = (255 - cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA))
    inverted = (inverted.astype(np.float32) * 0.60).astype(np.uint8)

    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    if bg_color != (0, 0, 0):
        canvas[:] = bg_color
    canvas[off_y:off_y + sh, off_x:off_x + sw] = inverted

    # Scale corner pixel coords from original image → canvas
    corners = [
        {
            "lat": c["lat"],
            "lon": c["lon"],
            "px":  c["px_orig"] * scale + off_x,
            "py":  c["py_orig"] * scale + off_y,
        }
        for c in raw
    ]

    return canvas, TrackProjector(corners)


# ── Frame rendering ───────────────────────────────────────────────────────────

def draw_frame(base: np.ndarray, trail: list, pos: tuple) -> np.ndarray:
    frame = base.copy()
    n = len(trail)

    # Fading trail (dark red → bright red)
    for j in range(n - 1):
        a = (j + 1) / n
        cv2.line(frame, trail[j], trail[j + 1],
                 (0, 0, int(60 + 140 * a)),
                 max(1, int(4 * a)), cv2.LINE_AA)

    # Dot: red fill + white ring
    cv2.circle(frame, pos, 8, (0,   0, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, pos, 8, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


# ── Render loop ───────────────────────────────────────────────────────────────

def render(gps_points: list, args) -> None:
    bg_color = (0, 0, 0) if args.bg == "black" else (255, 0, 255)
    canvas, projector = build_canvas_and_projector(
        args.map, args.calib, args.size, bg_color
    )

    timeline     = GpsTimeline(gps_points)
    total_frames = int(timeline.duration * args.fps)
    trail_max    = int(args.trail * args.fps)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, args.fps,
                             (args.size, args.size))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot write: {args.output}")

    print(f"  {len(gps_points):,} GPS points | {timeline.duration / 60:.1f} min "
          f"| {total_frames:,} frames @ {args.fps} fps")
    print(f"  Map: {args.map}")
    print(f"  Output: {args.output}")

    trail: list[tuple] = []

    for fi in range(total_frames):
        lat, lon = timeline.at(fi / args.fps)
        pos = projector(lat, lon)

        trail.append(pos)
        if len(trail) > trail_max:
            trail.pop(0)

        writer.write(draw_frame(canvas, trail, pos))

        if fi % (args.fps * 30) == 0:
            print(f"  [{100 * fi / total_frames:5.1f}%]  t={fi / args.fps / 60:.1f} min")

    writer.release()
    print("  Done.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="GPS mini-map overlay with track-snapped dot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--gps-folder", default=r"TH-May-29-2026\GPS")
    p.add_argument("--output",     default=r"TH-May-29-2026\minimap_overlay.mp4")
    p.add_argument("--map",        required=True, help="Track diagram PNG (TH.png)")
    p.add_argument("--calib",      default="calibration.json")
    p.add_argument("--size",       type=int,   default=500,
                   help="Canvas size (square, px)")
    p.add_argument("--fps",        type=int,   default=30)
    p.add_argument("--trail",      type=float, default=2.0,
                   help="Trail length in seconds")
    p.add_argument("--bg",         choices=["black", "magenta"], default="black")
    args = p.parse_args()

    print(f"Parsing GPS: {args.gps_folder}")
    points = parse_gps_folder(args.gps_folder)
    if not points:
        raise SystemExit("No GPS points found — check --gps-folder")
    print(f"Loaded {len(points):,} GPS points")

    if not os.path.isfile(args.calib):
        raise SystemExit(
            f"\nNo calibration found at '{args.calib}'.\n"
            "Run:  py calibrate.py   first."
        )

    render(points, args)


if __name__ == "__main__":
    main()
