#!/usr/bin/env python3
"""
GPS Mini-Map Overlay Generator
Thunderhill Raceway - May 29 2026

GPS cleaning philosophy
-----------------------
Remove outliers hard -- no data is better than bad data.
  Pass 1  Drop any reading that implies >280 km/h from the previous one.
  Pass 2  Drop any reading whose arc position goes backward by >20 m.

Gap filling
-----------
Missing chunks are bridged by interpolating the arc-length position
(how far around the track the car is) between the last good reading
and the next good reading.  The dot moves smoothly along the track
through any gap rather than teleporting or jumping.

Dot
---
Always snapped to the nearest segment of the GPS-driven track polyline.
Cannot leave the track, cannot go backward, cannot teleport.

Workflow:
  1. py calibrate.py
  2. py generate_minimap.py --map TH.png

Options:
  --gps-folder PATH   GPS log folder            [TH-May-29-2026/GPS]
  --output     PATH   Output video              [TH-May-29-2026/minimap_overlay.mp4]
  --map        PATH   Track diagram PNG         [required]
  --calib      PATH   Calibration JSON          [calibration.json]
  --size       N      Canvas size px (square)   [500]
  --fps        N      Frame rate                [30]
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
    r"N:([\d.]+)\s+W:([\d.]+)"
)


def _dedup(rows):
    seen, out = set(), []
    for r in rows:
        if r[0] not in seen:
            seen.add(r[0])
            out.append(r)
    return out


def parse_gps_folder(folder: str) -> list:
    rows = []
    for path in sorted(glob.glob(os.path.join(folder, "*.txt"))):
        with open(path) as fh:
            for line in fh:
                m = _GPS_RE.search(line)
                if m:
                    ts = datetime.strptime(m.group(1), "%Y/%m/%d %H:%M:%S")
                    rows.append((ts, float(m.group(2)), -float(m.group(3))))
    rows.sort(key=lambda r: r[0])
    return _dedup(rows)


# ---------------------------------------------------------------------------
# GPS cleaning
# ---------------------------------------------------------------------------

def _build_corner_arc(corners, cos_lat, R=111_111.0):
    """Corner polyline in GPS metric space. Returns (E, N, arc, total_arc)."""
    n = len(corners)
    E = np.array([c["lon"] * cos_lat * R for c in corners])
    N = np.array([c["lat"]               * R for c in corners])
    arc = np.zeros(n)
    for i in range(1, n):
        arc[i] = arc[i - 1] + np.hypot(E[i] - E[i - 1], N[i] - N[i - 1])
    total_arc = arc[-1] + np.hypot(E[0] - E[-1], N[0] - N[-1])
    return E, N, arc, total_arc


def _project_arc(lat, lon, E, N, arc, total_arc, cos_lat, R=111_111.0):
    """Arc-length position (metres) of GPS point projected onto corner polyline."""
    pe = lon * cos_lat * R
    pn = lat             * R
    n  = len(E)
    best_d2 = np.inf
    best_arc = 0.0
    for i in range(n):
        j     = (i + 1) % n
        arc_i = arc[i]
        arc_j = arc[j] if j > 0 else total_arc
        dae = E[j] - E[i]; dan = N[j] - N[i]
        len2 = dae * dae + dan * dan
        t = 0.0 if len2 < 1e-6 else max(0.0, min(1.0,
            ((pe - E[i]) * dae + (pn - N[i]) * dan) / len2))
        ne = E[i] + t * dae; nn = N[i] + t * dan
        d2 = (pe - ne) ** 2 + (pn - nn) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best_arc = arc_i + t * (arc_j - arc_i)
    return best_arc


def clean_gps(raw: list, corners: list, clean_path: str,
              max_speed_kmh: float = 280.0,
              backward_tol_m: float = 20.0) -> list:
    """
    Aggressively remove outliers.  No data is better than bad data.

    Pass 1 - Teleport: drop if implied speed from previous reading > max_speed_kmh.
    Pass 2 - Forward-only: project onto corner arc; drop if going backward
             by more than backward_tol_m.  Lap wrap-around allowed.

    Gaps left by removed readings are bridged in TrackTimeline by arc
    interpolation -- the dot fakes smooth motion between the surrounding
    good positions.
    """
    mean_lat = np.mean([c["lat"] for c in corners])
    cos_lat  = np.cos(np.radians(mean_lat))
    R        = 111_111.0
    max_ms   = max_speed_kmh / 3.6
    E, N, arc_pts, total_arc = _build_corner_arc(corners, cos_lat, R)

    # Pass 1: teleport removal
    p1 = [raw[0]]
    n_tp = 0
    for curr in raw[1:]:
        prev = p1[-1]
        dt   = (curr[0] - prev[0]).total_seconds()
        if dt <= 0:
            n_tp += 1; continue
        de = (curr[2] - prev[2]) * cos_lat * R
        dn = (curr[1] - prev[1]) * R
        if np.hypot(de, dn) / dt <= max_ms:
            p1.append(curr)
        else:
            n_tp += 1

    # Pass 2: forward-only
    p2       = [p1[0]]
    prev_arc = _project_arc(p1[0][1], p1[0][2], E, N, arc_pts, total_arc, cos_lat, R)
    n_bk     = 0
    for p in p1[1:]:
        a          = _project_arc(p[1], p[2], E, N, arc_pts, total_arc, cos_lat, R)
        lap_done   = (prev_arc > total_arc * 0.75) and (a < total_arc * 0.25)
        if lap_done or a >= prev_arc - backward_tol_m:
            p2.append(p)
            prev_arc = a
        else:
            n_bk += 1

    # Save clean file
    parent = os.path.dirname(clean_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(clean_path, "w") as f:
        for p in p2:
            ts = p[0].strftime("%Y/%m/%d %H:%M:%S")
            f.write(f"{ts} N:{p[1]:.6f} W:{-p[2]:.6f}\n")

    print(f"  Teleports removed : {n_tp}")
    print(f"  Backward removed  : {n_bk}")
    print(f"  Kept              : {len(p2):,} / {len(raw):,} readings")
    print(f"  Clean GPS saved   : {clean_path}")
    return p2


# ---------------------------------------------------------------------------
# AffineMapper  (GPS -> canvas pixel via least-squares over calibration corners)
# ---------------------------------------------------------------------------

class AffineMapper:
    def __init__(self, corners: list):
        mean_lat  = np.mean([c["lat"] for c in corners])
        self._cos = np.cos(np.radians(mean_lat))
        self._R   = 111_111.0
        E  = np.array([c["lon"] * self._cos * self._R for c in corners])
        N  = np.array([c["lat"]               * self._R for c in corners])
        PX = np.array([c["px"] for c in corners], dtype=float)
        PY = np.array([c["py"] for c in corners], dtype=float)
        M  = np.column_stack([E, N, np.ones(len(corners))])
        self._cx, *_ = np.linalg.lstsq(M, PX, rcond=None)
        self._cy, *_ = np.linalg.lstsq(M, PY, rcond=None)

    def __call__(self, lat: float, lon: float):
        v = np.array([lon * self._cos * self._R, lat * self._R, 1.0])
        return float(self._cx @ v), float(self._cy @ v)


# ---------------------------------------------------------------------------
# TrackSnapper  (pixel polyline from GPS session + arc-aware snapping)
# ---------------------------------------------------------------------------

class TrackSnapper:
    """
    Builds a pixel polyline from the cleaned GPS session (via AffineMapper),
    smooths it, then snaps positions to it.

    Also maintains a cumulative arc-length array so callers can:
      snap_with_arc(px, py)  -> (snapped_px, snapped_py, arc_metres)
      at_arc(arc)            -> (px, py)  -- position at given arc length
    """

    def __init__(self, gps_points: list, mapper: AffineMapper, smooth: int = 5):
        px = np.array([mapper(p[1], p[2])[0] for p in gps_points])
        py = np.array([mapper(p[1], p[2])[1] for p in gps_points])
        if smooth > 1:
            k  = np.ones(smooth) / smooth
            px = np.convolve(px, k, mode="same")
            py = np.convolve(py, k, mode="same")
        self._px = px
        self._py = py
        self._n  = len(px)

        # Cumulative arc lengths along the pixel polyline
        diffs        = np.hypot(np.diff(px), np.diff(py))
        self._arc    = np.zeros(self._n)
        self._arc[1:] = np.cumsum(diffs)
        self.total_arc = float(self._arc[-1])

        print(f"  Track polyline : {self._n} points, "
              f"total arc = {self.total_arc:.0f} px")

    # -- internal nearest-segment search ------------------------------------

    def _find_segment(self, raw_px: float, raw_py: float):
        """Returns (best_x, best_y, arc) of the snapped point."""
        dx    = self._px - raw_px
        dy    = self._py - raw_py
        dist2 = dx * dx + dy * dy
        k     = min(8, self._n - 1)
        cands = np.argpartition(dist2, k)[:k]

        best_d2 = np.inf
        best_x  = self._px[0]
        best_y  = self._py[0]
        best_arc = 0.0

        for idx in cands:
            for j in (int(idx) - 1, int(idx)):
                if j < 0 or j >= self._n - 1:
                    continue
                ax, ay  = self._px[j],     self._py[j]
                bx, by  = self._px[j + 1], self._py[j + 1]
                dab_x   = bx - ax
                dab_y   = by - ay
                len2    = dab_x * dab_x + dab_y * dab_y
                if len2 < 0.1:
                    continue
                t = ((raw_px - ax) * dab_x + (raw_py - ay) * dab_y) / len2
                t = max(0.0, min(1.0, t))
                nx  = ax + t * dab_x
                ny  = ay + t * dab_y
                d2  = (raw_px - nx) ** 2 + (raw_py - ny) ** 2
                if d2 < best_d2:
                    best_d2  = d2
                    best_x, best_y = nx, ny
                    best_arc = self._arc[j] + t * (self._arc[j + 1] - self._arc[j])

        return best_x, best_y, best_arc

    # -- public API ---------------------------------------------------------

    def __call__(self, raw_px: float, raw_py: float) -> tuple:
        x, y, _ = self._find_segment(raw_px, raw_py)
        return int(round(x)), int(round(y))

    def snap_with_arc(self, raw_px: float, raw_py: float) -> tuple:
        """Returns (px, py, arc) -- snapped pixel + arc length along polyline."""
        x, y, arc = self._find_segment(raw_px, raw_py)
        return int(round(x)), int(round(y)), arc

    def at_arc(self, arc_target: float) -> tuple:
        """
        Pixel position at arc_target along the track.
        Handles cumulative (multi-lap) arc values by wrapping with modulo.
        """
        arc_mod = arc_target % self.total_arc
        idx = int(np.searchsorted(self._arc, arc_mod, side="right")) - 1
        idx = max(0, min(idx, self._n - 2))
        span = self._arc[idx + 1] - self._arc[idx]
        t    = (arc_mod - self._arc[idx]) / span if span > 1e-6 else 0.0
        t    = max(0.0, min(1.0, t))
        x    = self._px[idx] + t * (self._px[idx + 1] - self._px[idx])
        y    = self._py[idx] + t * (self._py[idx + 1] - self._py[idx])
        return int(round(x)), int(round(y))


# ---------------------------------------------------------------------------
# TrackTimeline  (time -> pixel, with gap-bridging via arc interpolation)
# ---------------------------------------------------------------------------

class TrackTimeline:
    """
    Maps session time -> pixel position.

    Each clean GPS reading is projected onto the track to get an arc-length
    position.  Between readings (including across large gaps from removed
    outliers) the arc position is linearly interpolated and resolved to a
    pixel via TrackSnapper.at_arc().

    The dot always moves forward along the track, even through data gaps.
    """

    def __init__(self, clean_points: list, mapper: AffineMapper,
                 snapper: TrackSnapper):
        self.t0       = clean_points[0][0]
        self.duration = (clean_points[-1][0] - self.t0).total_seconds()

        # Per-reading: time and snapped arc position
        times = []
        arcs  = []
        for p in clean_points:
            rpx, rpy = mapper(p[1], p[2])
            _, _, arc = snapper.snap_with_arc(rpx, rpy)
            times.append((p[0] - self.t0).total_seconds())
            arcs.append(arc)

        # Make arcs monotonically increasing across lap boundaries
        arcs = np.array(arcs, dtype=float)
        total = snapper.total_arc
        for i in range(1, len(arcs)):
            # Lap completion: arc dropped from near end back to near start
            if arcs[i] < arcs[i - 1] - total * 0.5:
                arcs[i:] += total

        self._times   = np.array(times)
        self._arcs    = arcs
        self._snapper = snapper

    def at(self, t_sec: float) -> tuple:
        """Returns pixel (x, y) for the given session time (seconds from start)."""
        t_sec = float(np.clip(t_sec, 0, self.duration))
        idx   = int(np.searchsorted(self._times, t_sec, side="right")) - 1
        idx   = max(0, min(idx, len(self._times) - 2))

        t0, t1     = self._times[idx], self._times[idx + 1]
        arc0, arc1 = self._arcs[idx],  self._arcs[idx + 1]

        a    = (t_sec - t0) / (t1 - t0) if t1 > t0 else 0.0
        arc  = arc0 + a * (arc1 - arc0)
        return self._snapper.at_arc(arc)


# ---------------------------------------------------------------------------
# Canvas builder
# ---------------------------------------------------------------------------

def build_canvas(map_path: str, calib_path: str,
                 canvas_size: int, bg_color: tuple):
    img = cv2.imread(map_path)
    if img is None:
        raise FileNotFoundError(f"Cannot open map: {map_path}")
    with open(calib_path) as f:
        raw_corners = json.load(f)
    if len(raw_corners) < 3:
        raise ValueError("Need >= 3 corners in calibration.json")

    h, w   = img.shape[:2]
    scale  = min(canvas_size / w, canvas_size / h)
    sw, sh = int(w * scale), int(h * scale)
    off_x  = (canvas_size - sw) // 2
    off_y  = (canvas_size - sh) // 2

    inv = 255 - cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)
    inv = (inv.astype(np.float32) * 0.60).astype(np.uint8)

    canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
    if bg_color != (0, 0, 0):
        canvas[:] = bg_color
    canvas[off_y:off_y + sh, off_x:off_x + sw] = inv

    corners = [
        {"lat": c["lat"], "lon": c["lon"],
         "px": c["px_orig"] * scale + off_x,
         "py": c["py_orig"] * scale + off_y}
        for c in raw_corners
    ]
    return canvas, AffineMapper(corners), raw_corners


# ---------------------------------------------------------------------------
# Frame drawing
# ---------------------------------------------------------------------------

def draw_frame(base: np.ndarray, pos: tuple) -> np.ndarray:
    frame = base.copy()
    cv2.circle(frame, pos, 8, (0,   0, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, pos, 8, (255, 255, 255),  1, cv2.LINE_AA)
    return frame


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def render(raw_points: list, args) -> None:
    bg_color = (0, 0, 0) if args.bg == "black" else (255, 0, 255)

    print("Building canvas...")
    canvas, mapper, raw_corners = build_canvas(
        args.map, args.calib, args.size, bg_color)

    # Clean GPS -- always regenerated
    gps_parent = os.path.dirname(os.path.normpath(args.gps_folder))
    clean_path = os.path.join(gps_parent, "GPS_clean.txt")
    print("Cleaning GPS data...")
    clean = clean_gps(raw_points, raw_corners, clean_path)

    print("Building track polyline...")
    snapper  = TrackSnapper(clean, mapper, smooth=args.smooth)

    print("Building track timeline (arc interpolation)...")
    timeline = TrackTimeline(clean, mapper, snapper)

    total_frames = int(timeline.duration * args.fps)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, args.fps,
                             (args.size, args.size))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open output: {args.output}")

    print(f"  Duration : {timeline.duration / 60:.1f} min")
    print(f"  Frames   : {total_frames:,} @ {args.fps} fps")
    print(f"  Output   : {args.output}")

    for fi in range(total_frames):
        pos = timeline.at(fi / args.fps)
        writer.write(draw_frame(canvas, pos))
        if fi % (args.fps * 30) == 0:
            print(f"  [{100 * fi / total_frames:5.1f}%]  "
                  f"t = {fi / args.fps / 60:.1f} min")

    writer.release()
    print("  Done.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="GPS mini-map -- clean, forward-only, gap-bridging",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--gps-folder", default=r"TH-May-29-2026\GPS")
    p.add_argument("--output",     default=r"TH-May-29-2026\minimap_overlay.mp4")
    p.add_argument("--map",        required=True)
    p.add_argument("--calib",      default="calibration.json")
    p.add_argument("--size",       type=int, default=500)
    p.add_argument("--fps",        type=int, default=30)
    p.add_argument("--smooth",     type=int, default=5)
    p.add_argument("--bg",         choices=["black", "magenta"], default="black")
    args = p.parse_args()

    if not os.path.isfile(args.calib):
        raise SystemExit(f"No calibration at '{args.calib}'. Run: py calibrate.py")

    print(f"Parsing GPS: {args.gps_folder}")
    raw = parse_gps_folder(args.gps_folder)
    if not raw:
        raise SystemExit("No GPS points found -- check --gps-folder")
    print(f"Raw GPS points: {len(raw):,}")

    render(raw, args)


if __name__ == "__main__":
    main()
