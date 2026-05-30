#!/usr/bin/env python3
"""
Corner calibration tool for GPS mini-map.

PHASE 1  (window)
  Click every corner/waypoint on TH.png in lap order (CCW).
  Each click drops a numbered dot.
  Z = undo last click
  D = done — move to phase 2

PHASE 2  (terminal)
  For each corner in order, look at your dash cam footage and type the
  GPS shown on-screen when you passed that corner.
  Accept either:
    39.54075 122.33146
    N:39.54075 W:122.33146

Output: calibration.json  (used by generate_minimap.py --map TH.png)
"""

import json
import re
import sys

import cv2
import numpy as np

MAP_PATH   = "TH.png"
CALIB_FILE = "calibration.json"

# ── state ────────────────────────────────────────────────────────────────────
corners: list[dict] = []          # {px_orig, py_orig} filled in phase 1
img_orig: np.ndarray = None        # original full-res image
img_disp: np.ndarray = None        # scaled display image
disp_scale: float = 1.0
phase: int = 1

WIN = "PHASE 1 — click corners CCW | Z=undo | D=done"


# ── drawing ───────────────────────────────────────────────────────────────────

def redraw(highlight: int = -1) -> None:
    canvas = img_disp.copy()
    n = len(corners)
    for i, c in enumerate(corners):
        px = int(c["px_orig"] * disp_scale)
        py = int(c["py_orig"] * disp_scale)
        col   = (0, 255, 120) if i == highlight else (0, 80, 255)
        ring  = (255, 255, 255)
        cv2.circle(canvas, (px, py), 8, col, -1, cv2.LINE_AA)
        cv2.circle(canvas, (px, py), 8, ring, 1,  cv2.LINE_AA)
        cv2.putText(canvas, str(i + 1), (px + 10, py + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2, cv2.LINE_AA)
        if i > 0:
            prev = corners[i - 1]
            cv2.line(canvas,
                     (int(prev["px_orig"] * disp_scale), int(prev["py_orig"] * disp_scale)),
                     (px, py), (0, 200, 80), 1, cv2.LINE_AA)
    # Close loop preview
    if n >= 3:
        first = corners[0]
        last  = corners[-1]
        cv2.line(canvas,
                 (int(last["px_orig"]  * disp_scale), int(last["py_orig"]  * disp_scale)),
                 (int(first["px_orig"] * disp_scale), int(first["py_orig"] * disp_scale)),
                 (0, 120, 60), 1, cv2.LINE_AA)
    cv2.imshow(WIN, canvas)


def on_mouse(event, x, y, _flags, _param) -> None:
    if phase != 1 or event != cv2.EVENT_LBUTTONDOWN:
        return
    corners.append({"px_orig": x / disp_scale, "py_orig": y / disp_scale})
    print(f"  corner {len(corners):2d}  px=({x / disp_scale:.0f}, {y / disp_scale:.0f})")
    redraw()


# ── GPS input ─────────────────────────────────────────────────────────────────

def parse_gps(raw: str):
    """Return (lat, lon) floats or None."""
    nums = re.findall(r"\d+\.\d+", raw)
    if len(nums) >= 2:
        return float(nums[0]), float(nums[1])
    return None


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    global img_orig, img_disp, disp_scale, phase

    img = cv2.imread(MAP_PATH)
    if img is None:
        sys.exit(f"Cannot open {MAP_PATH!r} — run from D:\\trackproj\\")

    img_orig = img
    h, w = img.shape[:2]
    max_side = 1100
    disp_scale = min(max_side / w, max_side / h)
    img_disp = cv2.resize(img, (int(w * disp_scale), int(h * disp_scale)),
                          interpolation=cv2.INTER_AREA)

    print(__doc__)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN, on_mouse)
    redraw()

    # ── Phase 1: clicking ────────────────────────────────────────────────────
    while phase == 1:
        key = cv2.waitKey(50) & 0xFF
        if key == ord("z") and corners:
            corners.pop()
            print(f"  undo → {len(corners)} corners")
            redraw()
        elif key == ord("d"):
            if len(corners) < 3:
                print("  need at least 3 corners — keep clicking")
            else:
                phase = 2
        elif key == 27:
            print("Aborted.")
            cv2.destroyAllWindows()
            sys.exit(0)

    print(f"\n{len(corners)} corners locked.\n")

    # ── Phase 2: GPS input ────────────────────────────────────────────────────
    print("Phase 2 — type the GPS coordinate shown on your dash cam")
    print("when you passed each corner.  Format: 39.54075 122.33146\n")

    for i, c in enumerate(corners):
        redraw(highlight=i)
        cv2.waitKey(1)

        while True:
            try:
                raw = input(f"  Corner {i + 1:2d} lat lon › ").strip()
            except EOFError:
                break
            result = parse_gps(raw)
            if result:
                c["lat"], c["lon"] = result
                print(f"            → N:{c['lat']}  W:{c['lon']}")
                break
            print("    couldn't parse — try: 39.54075 122.33146")

    cv2.destroyAllWindows()

    # Save
    out = [
        {"px_orig": c["px_orig"], "py_orig": c["py_orig"],
         "lat": c["lat"], "lon": c["lon"]}
        for c in corners
    ]
    with open(CALIB_FILE, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nSaved {len(out)} corners → {CALIB_FILE}")
    print("Run:  py generate_minimap.py --map TH.png")


if __name__ == "__main__":
    main()
