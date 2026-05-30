#!/usr/bin/env python3
"""
GPS Mini-Map Overlay Generator
Thunderhill Raceway - May 29 2026

Pipeline
--------
1. Parse raw GPS from all log files.
2. Clean: remove teleports (>280 km/h) and backward motion (>20 m reverse).
   Gaps left by removed readings are handled automatically -- the interpolator
   bridges across them, faking smooth forward motion between the surrounding
   good readings.
3. Extract one reference lap (first ~1 track-length of clean readings).
   Building the track polyline from a single loop avoids the multi-lap
   snap ambiguity that caused the dot to jump.
4. Render: GpsTimeline interpolates lat/lon across the full session (including
   any gaps), AffineMapper converts to canvas pixels, TrackSnapper projects
   onto the single-loop polyline.  The dot cannot leave the track or reverse.

Saves GPS_clean.txt beside the GPS folder every run.

Options:
  --gps-folder PATH   [TH-May-29-2026/GPS]
  --output     PATH   [TH-May-29-2026/minimap_overlay.mp4]
  --map        PATH   track diagram PNG (required)
  --calib      PATH   [calibration.json]
  --size       N      canvas size px square  [500]
  --fps        N      [30]
  --smooth     N      GPS noise smoothing    [5]
  --bg         black|magenta                 [black]
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
    r"(\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})\s+N:([\d.]+)\s+W:([\d.]+)"
)


def _dedup(rows):
    seen, out = set(), []
    for r in rows:
        if r[0] not in seen:
            seen.add(r[0])
            out.append(r)
    return out


def parse_gps_folder(folder: str) -> list:
    """Read every *.txt in folder. Lon stored as negative (standard sign)."""
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

def _corner_arc(corners, cos_lat, R=111_111.0):
    """Closed polyline through calibration corners. Returns (E, N, arc, total)."""
    n = len(corners)
    E = np.array([c["lon"] * cos_lat * R for c in corners])
    N = np.array([c["lat"]               * R for c in corners])
    arc = np.zeros(n)
    for i in range(1, n):
        arc[i] = arc[i - 1] + np.hypot(E[i] - E[i - 1], N[i] - N[i - 1])
    total = arc[-1] + np.hypot(E[0] - E[-1], N[0] - N[-1])
    return E, N, arc, total


def _project_arc(lat, lon, E, N, arc, total, cos_lat, R=111_111.0):
    """Arc-length position (m) of GPS point projected onto corner polyline."""
    pe, pn = lon * cos_lat * R, lat * R
    n = len(E)
    best_d2, best_arc = np.inf, 0.0
    for i in range(n):
        j     = (i + 1) % n
        arc_j = arc[j] if j > 0 else total
        dae, dan = E[j] - E[i], N[j] - N[i]
        len2 = dae * dae + dan * dan
        t = 0.0 if len2 < 1e-6 else max(0.0, min(1.0,
            ((pe - E[i]) * dae + (pn - N[i]) * dan) / len2))
        ne, nn = E[i] + t * dae, N[i] + t * dan
        d2 = (pe - ne) ** 2 + (pn - nn) ** 2
        if d2 < best_d2:
            best_d2  = d2
            best_arc = arc[i] + t * (arc_j - arc[i])
    return best_arc


def clean_gps(raw: list, corners: list, clean_path: str,
              max_speed_kmh: float = 280.0,
              backward_tol_m: float = 20.0) -> list:
    """
    Remove outliers hard.  No data > bad data.
    Gaps are bridged later by GpsTimeline interpolation.

    Pass 1: drop teleports  (implied speed > max_speed_kmh)
    Pass 2: drop backward   (arc goes > backward_tol_m in reverse)
    """
    mean_lat = np.mean([c["lat"] for c in corners])
    cos_lat  = np.cos(np.radians(mean_lat))
    R        = 111_111.0
    max_ms   = max_speed_kmh / 3.6
    E, N, arc_pts, total = _corner_arc(corners, cos_lat, R)

    # Pass 1 — teleport
    p1, n_tp = [raw[0]], 0
    for curr in raw[1:]:
        prev = p1[-1]
        dt = (curr[0] - prev[0]).total_seconds()
        if dt <= 0:
            n_tp += 1; continue
        de = (curr[2] - prev[2]) * cos_lat * R
        dn = (curr[1] - prev[1]) * R
        if np.hypot(de, dn) / dt <= max_ms:
            p1.append(curr)
        else:
            n_tp += 1

    # Pass 2 — forward-only
    p2, n_bk = [p1[0]], 0
    prev_a = _project_arc(p1[0][1], p1[0][2], E, N, arc_pts, total, cos_lat, R)
    for p in p1[1:]:
        a        = _project_arc(p[1], p[2], E, N, arc_pts, total, cos_lat, R)
        lap_done = (prev_a > total * 0.75) and (a < total * 0.25)
        if lap_done or a >= prev_a - backward_tol_m:
            p2.append(p); prev_a = a
        else:
            n_bk += 1

    # Save
    if (parent := os.path.dirname(clean_path)):
        os.makedirs(parent, exist_ok=True)
    with open(clean_path, "w") as f:
        for p in p2:
            f.write(f"{p[0].strftime('%Y/%m/%d %H:%M:%S')} "
                    f"N:{p[1]:.6f} W:{-p[2]:.6f}\n")

    print(f"  Teleports removed : {n_tp}")
    print(f"  Backward removed  : {n_bk}")
    print(f"  Kept : {len(p2):,} / {len(raw):,}  |  GPS_clean.txt saved")
    return p2


# ---------------------------------------------------------------------------
# Reference lap  (single-loop track polyline, avoids multi-lap snap issue)
# ---------------------------------------------------------------------------

def extract_reference_lap(clean: list, approx_length_m: float,
                           cos_lat: float, R: float = 111_111.0) -> list:
    """
    Walk clean GPS readings from the start until we have covered
    approximately one lap's worth of cumulative distance.

    Why: building TrackSnapper from all laps creates an N-loop polyline.
    Snapping to it is ambiguous -- the same physical location appears N
    times and the nearest-segment search can pick any of them, causing
    the dot to jump wildly.  A single-loop polyline has each track location
    exactly once, so snapping is always unambiguous.
    """
    dist, lap = 0.0, [clean[0]]
    for i in range(1, len(clean)):
        prev, curr = clean[i - 1], clean[i]
        de = (curr[2] - prev[2]) * cos_lat * R
        dn = (curr[1] - prev[1]) * R
        dist += np.hypot(de, dn)
        lap.append(curr)
        if dist >= approx_length_m:
            break
    print(f"  Reference lap : {len(lap)} pts, "
          f"{dist:.0f} m GPS distance  (track ~{approx_length_m:.0f} m)")
    return lap


# ---------------------------------------------------------------------------
# GPS timeline  (interpolates lat/lon across whole session; bridges gaps)
# ---------------------------------------------------------------------------

class GpsTimeline:
    """
    Binary-search interpolation over all clean GPS readings.

    When outliers were removed, consecutive timestamps have a larger gap.
    GpsTimeline.at(t) interpolates lat/lon linearly between the surrounding
    good readings, effectively faking smooth motion through any hole.
    """

    def __init__(self, points: list):
        self.t0       = points[0][0]
        self.duration = (points[-1][0] - self.t0).total_seconds()
        self._times   = np.array([(p[0] - self.t0).total_seconds() for p in points])
        self._lats    = np.array([p[1] for p in points])
        self._lons    = np.array([p[2] for p in points])

    def at(self, t_sec: float):
        t_sec = float(np.clip(t_sec, 0.0, self.duration))
        idx   = int(np.searchsorted(self._times, t_sec, side="right")) - 1
        idx   = max(0, min(idx, len(self._times) - 2))
        t0, t1 = self._times[idx], self._times[idx + 1]
        a = (t_sec - t0) / (t1 - t0) if t1 > t0 else 0.0
        return (
            float(self._lats[idx] + a * (self._lats[idx + 1] - self._lats[idx])),
            float(self._lons[idx] + a * (self._lons[idx + 1] - self._lons[idx])),
        )


# ---------------------------------------------------------------------------
# AffineMapper  (GPS -> canvas pixel via least-squares over all calib corners)
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
# TrackSnapper  (single-loop pixel polyline; nearest-segment projection)
# ---------------------------------------------------------------------------

class TrackSnapper:
    """
    Built from ONE reference lap so the polyline is a single closed loop.
    Every physical track location appears exactly once -- snapping is
    always unambiguous regardless of which lap the car is currently on.
    """

    def __init__(self, ref_lap: list, mapper: AffineMapper, smooth: int = 5):
        px = np.array([mapper(p[1], p[2])[0] for p in ref_lap])
        py = np.array([mapper(p[1], p[2])[1] for p in ref_lap])
        if smooth > 1:
            k  = np.ones(smooth) / smooth
            px = np.convolve(px, k, mode="same")
            py = np.convolve(py, k, mode="same")
        self._px = px
        self._py = py
        self._n  = len(px)
        print(f"  Track polyline : {self._n} points (single lap)")

    def __call__(self, raw_px: float, raw_py: float) -> tuple:
        dx    = self._px - raw_px
        dy    = self._py - raw_py
        dist2 = dx * dx + dy * dy
        k     = min(8, self._n - 1)
        cands = np.argpartition(dist2, k)[:k]

        best_d2 = np.inf
        best_x  = self._px[0]
        best_y  = self._py[0]

        for idx in cands:
            for j in (int(idx) - 1, int(idx)):
                if j < 0 or j >= self._n - 1:
                    continue
                ax, ay  = self._px[j],     self._py[j]
                bx, by  = self._px[j + 1], self._py[j + 1]
                dab_x, dab_y = bx - ax, by - ay
                len2    = dab_x * dab_x + dab_y * dab_y
                if len2 < 0.1:
                    continue
                t  = ((raw_px - ax) * dab_x + (raw_py - ay) * dab_y) / len2
                t  = max(0.0, min(1.0, t))
                nx = ax + t * dab_x
                ny = ay + t * dab_y
                d2 = (raw_px - nx) ** 2 + (raw_py - ny) ** 2
                if d2 < best_d2:
                    best_d2 = d2
                    best_x, best_y = nx, ny

        return int(round(best_x)), int(round(best_y))


# ---------------------------------------------------------------------------
# Image-pixel snap  (BFS Voronoi -- dot lands exactly on the drawn line)
# ---------------------------------------------------------------------------

def _build_image_snap(img_bgr: np.ndarray, sw: int, sh: int,
                      off_x: int, off_y: int, canvas_size: int) -> tuple:
    """
    Build a nearest-track-pixel lookup table for every canvas pixel.

    Algorithm: BFS outward from each track-line pixel.  Each non-track
    pixel inherits the coordinates of the nearest track pixel it was
    reached from.  O(canvas_size^2) -- runs once at startup.

    In TH.png the track lines are BLACK on a WHITE background.
    We threshold the original (non-inverted) image at gray < 100.

    Returns (nearest_x, nearest_y) arrays of shape (canvas_size, canvas_size).
    Pixels with no track pixel in the image have value -1 (use GPS pos as fallback).
    """
    from collections import deque

    gray = cv2.cvtColor(
        cv2.resize(img_bgr, (sw, sh), interpolation=cv2.INTER_AREA),
        cv2.COLOR_BGR2GRAY,
    )
    track_mask = gray < 100   # True where track line drawn

    H = W = canvas_size
    nx = np.full((H, W), -1, dtype=np.int16)
    ny = np.full((H, W), -1, dtype=np.int16)
    vis = np.zeros((H, W), dtype=bool)

    q = deque()
    ys, xs = np.where(track_mask)
    for iy, ix in zip(ys.tolist(), xs.tolist()):
        cx, cy = ix + off_x, iy + off_y
        if 0 <= cx < W and 0 <= cy < H and not vis[cy, cx]:
            nx[cy, cx] = cx
            ny[cy, cx] = cy
            vis[cy, cx] = True
            q.append((cy, cx))

    DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
    while q:
        y, x = q.popleft()
        for dy, dx in DIRS:
            yy, xx = y + dy, x + dx
            if 0 <= yy < H and 0 <= xx < W and not vis[yy, xx]:
                vis[yy, xx] = True
                nx[yy, xx] = nx[y, x]
                ny[yy, xx] = ny[y, x]
                q.append((yy, xx))

    n_track = int(vis.sum())   # all pixels got assigned (vis covers whole canvas after BFS)
    print(f"  Image snap map : {n_track:,} pixels assigned  "
          f"({int((track_mask > 0).sum()):,} track-line pixels in source)")
    return nx, ny


# ---------------------------------------------------------------------------
# Canvas
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

    print("  Building image snap map (BFS)...")
    snap_x, snap_y = _build_image_snap(img, sw, sh, off_x, off_y, canvas_size)

    corners = [
        {"lat": c["lat"], "lon": c["lon"],
         "px": c["px_orig"] * scale + off_x,
         "py": c["py_orig"] * scale + off_y}
        for c in raw_corners
    ]
    return canvas, AffineMapper(corners), raw_corners, snap_x, snap_y


# ---------------------------------------------------------------------------
# Frame
# ---------------------------------------------------------------------------

def draw_frame(base: np.ndarray, pos: tuple) -> np.ndarray:
    frame = base.copy()
    cv2.circle(frame, pos, 8, (0,   0, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, pos, 8, (255, 255, 255),  1, cv2.LINE_AA)
    return frame


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def render(raw: list, args) -> None:
    bg_color = (0, 0, 0) if args.bg == "black" else (255, 0, 255)

    print("Building canvas...")
    canvas, mapper, raw_corners, snap_x, snap_y = build_canvas(
        args.map, args.calib, args.size, bg_color)

    gps_parent = os.path.dirname(os.path.normpath(args.gps_folder))
    clean_path = os.path.join(gps_parent, "GPS_clean.txt")
    print("Cleaning GPS...")
    clean = clean_gps(raw, raw_corners, clean_path)

    # Approximate track length from calibration corners
    mean_lat = np.mean([c["lat"] for c in raw_corners])
    cos_lat  = np.cos(np.radians(mean_lat))
    R        = 111_111.0
    _, _, _, track_len_m = _corner_arc(raw_corners, cos_lat, R)

    print("Extracting reference lap for track shape...")
    ref_lap = extract_reference_lap(clean, track_len_m, cos_lat, R)

    print("Building track snapper...")
    snapper  = TrackSnapper(ref_lap, mapper, smooth=args.smooth)

    # Full session timeline (GpsTimeline bridges gaps via lat/lon interpolation)
    timeline     = GpsTimeline(clean)
    total_frames = int(timeline.duration * args.fps)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, args.fps,
                             (args.size, args.size))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot write: {args.output}")

    print(f"  {timeline.duration / 60:.1f} min  |  "
          f"{total_frames:,} frames @ {args.fps} fps")
    print(f"  Output: {args.output}")

    S = args.size - 1   # clamp bound
    for fi in range(total_frames):
        lat, lon       = timeline.at(fi / args.fps)
        raw_px, raw_py = mapper(lat, lon)

        # Step 1: GPS polyline snap -- correct direction / lap ordering
        gx, gy = snapper(raw_px, raw_py)

        # Step 2: image pixel snap -- land exactly on the drawn track line
        cx = int(max(0, min(S, gx)))
        cy = int(max(0, min(S, gy)))
        ix, iy = int(snap_x[cy, cx]), int(snap_y[cy, cx])
        pos = (ix, iy) if ix >= 0 else (gx, gy)

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
        description="GPS mini-map: clean GPS, single-lap snap, gap-bridging",
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
