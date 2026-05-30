#!/usr/bin/env python3
"""
GPS Mini-Map Overlay Generator
Thunderhill Raceway - May 29 2026

GPS cleaning pipeline (runs every time before rendering):
  Pass 1 - Teleport removal:
    Any reading that requires the car to travel faster than 280 km/h
    from the previous reading is dropped.
  Pass 2 - Forward-only:
    Each reading is projected onto the calibration corner polyline to get
    a 1-D arc-length position on the track.  Readings that go backward by
    more than ~50 m (GPS noise floor) are dropped.  Lap wrap-around is
    detected automatically so the finish-line crossing is never treated as
    backward motion.
  Output: GPS_clean.txt saved next to the GPS folder for inspection.

The dot is then snapped to the nearest segment of a pixel polyline built
from the cleaned GPS trace -- it can never leave the track line.

Workflow:
  1. py calibrate.py          # one-time
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

# Intentionally does NOT require km/h so it also reads the clean file.
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
    """Read all *.txt files in folder. Lon stored as negative (standard)."""
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


def parse_gps_file(path: str) -> list:
    """Read a single GPS file (raw or clean)."""
    rows = []
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
    """
    Build the calibration corner polyline in GPS metric space (metres).
    Returns (E, N, arc, total_arc) where:
      E[i], N[i]  = east/north metres of corner i
      arc[i]      = cumulative arc length from corner 0 to corner i
      total_arc   = full lap length (including closing segment)
    """
    n = len(corners)
    E = np.array([c["lon"] * cos_lat * R for c in corners])
    N = np.array([c["lat"]               * R for c in corners])

    arc = np.zeros(n)
    for i in range(1, n):
        arc[i] = arc[i - 1] + np.hypot(E[i] - E[i - 1], N[i] - N[i - 1])

    # closing segment (last corner back to first)
    total_arc = arc[-1] + np.hypot(E[0] - E[-1], N[0] - N[-1])
    return E, N, arc, total_arc


def _project_arc(lat, lon, E, N, arc, total_arc, cos_lat, R=111_111.0):
    """
    Project GPS point onto the corner polyline.
    Returns the arc-length position (metres) of the nearest point.
    """
    pe = lon * cos_lat * R
    pn = lat             * R
    n  = len(E)

    best_d2  = np.inf
    best_arc = 0.0

    for i in range(n):
        j     = (i + 1) % n
        arc_i = arc[i]
        arc_j = arc[j] if j > 0 else total_arc

        dae  = E[j] - E[i]
        dan  = N[j] - N[i]
        len2 = dae * dae + dan * dan
        t    = 0.0 if len2 < 1e-6 else (
            ((pe - E[i]) * dae + (pn - N[i]) * dan) / len2
        )
        t = max(0.0, min(1.0, t))

        ne = E[i] + t * dae
        nn = N[i] + t * dan
        d2 = (pe - ne) ** 2 + (pn - nn) ** 2

        if d2 < best_d2:
            best_d2  = d2
            best_arc = arc_i + t * (arc_j - arc_i)

    return best_arc


def clean_gps(raw: list, corners: list, clean_path: str,
              max_speed_kmh: float = 280.0,
              backward_tol_m: float = 50.0) -> list:
    """
    Two-pass GPS filter.

    Pass 1 - Teleport removal
      Drop any reading where the implied speed from the previous kept
      reading exceeds max_speed_kmh.

    Pass 2 - Forward-only
      Project each reading onto the corner polyline arc.  If the new arc
      position is more than backward_tol_m behind the previous, skip it.
      Lap wrap-around (arc drops from near total back to near zero) is
      detected and allowed.

    Saves the cleaned readings to clean_path.
    Returns the cleaned list.
    """
    mean_lat = np.mean([c["lat"] for c in corners])
    cos_lat  = np.cos(np.radians(mean_lat))
    R        = 111_111.0
    max_ms   = max_speed_kmh / 3.6

    E, N, arc_pts, total_arc = _build_corner_arc(corners, cos_lat, R)

    # ── Pass 1: teleport removal ─────────────────────────────────────────────
    p1 = [raw[0]]
    n_teleport = 0
    for curr in raw[1:]:
        prev = p1[-1]
        dt   = (curr[0] - prev[0]).total_seconds()
        if dt <= 0:
            n_teleport += 1
            continue
        de = (curr[2] - prev[2]) * cos_lat * R
        dn = (curr[1] - prev[1]) * R
        if np.hypot(de, dn) / dt <= max_ms:
            p1.append(curr)
        else:
            n_teleport += 1

    # ── Pass 2: forward-only ─────────────────────────────────────────────────
    p2 = [p1[0]]
    prev_arc = _project_arc(p1[0][1], p1[0][2],
                            E, N, arc_pts, total_arc, cos_lat, R)
    n_backward = 0

    for p in p1[1:]:
        a = _project_arc(p[1], p[2], E, N, arc_pts, total_arc, cos_lat, R)

        # Lap completion: arc wraps from near end back to near start
        lap_complete = (prev_arc > total_arc * 0.75) and (a < total_arc * 0.25)

        if lap_complete or a >= prev_arc - backward_tol_m:
            p2.append(p)
            prev_arc = a
        else:
            n_backward += 1

    # ── Save clean file ──────────────────────────────────────────────────────
    parent = os.path.dirname(clean_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    with open(clean_path, "w") as f:
        for p in p2:
            ts      = p[0].strftime("%Y/%m/%d %H:%M:%S")
            lon_pos = -p[2]          # back to positive W for storage
            f.write(f"{ts} N:{p[1]:.6f} W:{lon_pos:.6f}\n")

    print(f"  Pass 1 removed {n_teleport:,} teleports")
    print(f"  Pass 2 removed {n_backward:,} backward readings")
    print(f"  {len(raw):,} -> {len(p2):,} readings kept "
          f"({len(raw) - len(p2):,} dropped)")
    print(f"  Clean GPS saved: {clean_path}")
    return p2


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
# AffineMapper  (GPS -> canvas pixel, least-squares over calibration corners)
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

        M = np.column_stack([E, N, np.ones(len(corners))])
        self._cx, *_ = np.linalg.lstsq(M, PX, rcond=None)
        self._cy, *_ = np.linalg.lstsq(M, PY, rcond=None)

    def __call__(self, lat: float, lon: float):
        v = np.array([lon * self._cos * self._R,
                      lat             * self._R,
                      1.0])
        return float(self._cx @ v), float(self._cy @ v)


# ---------------------------------------------------------------------------
# TrackSnapper  (snaps pixel position to driven GPS trace polyline)
# ---------------------------------------------------------------------------

class TrackSnapper:
    """
    Converts the full (cleaned) GPS session to a dense pixel polyline,
    smooths it, then snaps any query position to the nearest segment.
    The dot physically cannot leave the track line.
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
        print(f"  Track polyline: {self._n} points")

    def __call__(self, raw_px: float, raw_py: float) -> tuple:
        dx    = self._px - raw_px
        dy    = self._py - raw_py
        dist2 = dx * dx + dy * dy

        k          = min(8, self._n - 1)
        candidates = np.argpartition(dist2, k)[:k]

        best_d2 = np.inf
        best_x  = self._px[0]
        best_y  = self._py[0]

        for idx in candidates:
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
                    best_d2 = d2
                    best_x, best_y = nx, ny

        return int(round(best_x)), int(round(best_y))


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
        raise ValueError("Need at least 3 corners in calibration.json")

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

    corners = [
        {
            "lat": c["lat"],
            "lon": c["lon"],
            "px":  c["px_orig"] * scale + off_x,
            "py":  c["py_orig"] * scale + off_y,
        }
        for c in raw_corners
    ]

    return canvas, AffineMapper(corners), raw_corners


# ---------------------------------------------------------------------------
# Frame drawing  (dot only, no trail)
# ---------------------------------------------------------------------------

def draw_frame(base: np.ndarray, pos: tuple) -> np.ndarray:
    frame = base.copy()
    cv2.circle(frame, pos, 8, (0,   0, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, pos, 8, (255, 255, 255),  1, cv2.LINE_AA)
    return frame


# ---------------------------------------------------------------------------
# Render loop
# ---------------------------------------------------------------------------

def render(clean_points: list, args) -> None:
    bg_color = (0, 0, 0) if args.bg == "black" else (255, 0, 255)

    print("Building canvas and calibration transform...")
    canvas, mapper, raw_corners = build_canvas(
        args.map, args.calib, args.size, bg_color
    )

    # Determine clean GPS path
    gps_parent = os.path.dirname(os.path.normpath(args.gps_folder))
    clean_path = os.path.join(gps_parent, "GPS_clean.txt")

    # Clean the GPS data (always regenerated so the file stays fresh)
    print("Cleaning GPS data...")
    clean_points = clean_gps(
        clean_points, raw_corners, clean_path,
        max_speed_kmh=280.0,
        backward_tol_m=50.0,
    )

    print("Building track polyline from cleaned GPS data...")
    snapper = TrackSnapper(clean_points, mapper, smooth=args.smooth)

    timeline     = GpsTimeline(clean_points)
    total_frames = int(timeline.duration * args.fps)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.output, fourcc, args.fps,
                             (args.size, args.size))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open output: {args.output}")

    print(f"  {len(clean_points):,} pts | "
          f"{timeline.duration / 60:.1f} min | "
          f"{total_frames:,} frames @ {args.fps} fps")
    print(f"  Output: {args.output}")

    for fi in range(total_frames):
        lat, lon    = timeline.at(fi / args.fps)
        raw_px, raw_py = mapper(lat, lon)
        pos         = snapper(raw_px, raw_py)
        writer.write(draw_frame(canvas, pos))

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
        description="GPS mini-map with cleaned, track-snapped dot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--gps-folder", default=r"TH-May-29-2026\GPS")
    p.add_argument("--output",     default=r"TH-May-29-2026\minimap_overlay.mp4")
    p.add_argument("--map",        required=True)
    p.add_argument("--calib",      default="calibration.json")
    p.add_argument("--size",       type=int,   default=500)
    p.add_argument("--fps",        type=int,   default=30)
    p.add_argument("--smooth",     type=int,   default=5,
                   help="Pixel polyline smoothing window (GPS readings)")
    p.add_argument("--bg",         choices=["black", "magenta"], default="black")
    args = p.parse_args()

    if not os.path.isfile(args.calib):
        raise SystemExit(f"No calibration at '{args.calib}'. Run: py calibrate.py")

    print(f"Parsing GPS: {args.gps_folder}")
    raw_points = parse_gps_folder(args.gps_folder)
    if not raw_points:
        raise SystemExit("No GPS points found -- check --gps-folder")
    print(f"Loaded {len(raw_points):,} raw GPS points")

    render(raw_points, args)


if __name__ == "__main__":
    main()
