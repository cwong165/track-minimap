#!/usr/bin/env python3
"""
Click-to-calibrate: mark GPS points on TH.png to build calibration.json.

Usage:
    pip install opencv-python
    python calibrate.py

Instructions:
  1. The track map opens in a window.
  2. Click each labelled GPS point IN ORDER (order listed in the window title).
  3. Press Z to undo the last click.
  4. When all points are clicked the file is saved automatically.
  5. Press ESC at any time to quit without saving.

Output: calibration.json (used by generate_minimap.py --map TH.png)
"""

import json
import sys
import cv2
import numpy as np

MAP_PATH   = "TH.png"
CALIB_FILE = "calibration.json"

# GPS points to calibrate — taken from dash cam GPS overlay
# Identify where you were on TH.png at each timestamp and click there.
POINTS = [
    {
        "label": "1 — 07:35:00  18 km/h  (just rolling out, first corner approach)",
        "lat": 39.540752,
        "lon": 122.331459,
    },
    {
        "label": "2 — 07:35:19  92 km/h  (back straight, going fast)",
        "lat": 39.536877,
        "lon": 122.331230,
    },
]

# ── globals ─────────────────────────────────────────────────────────────────

clicks: list[dict] = []
img_orig: np.ndarray = None
WIN = "Calibrate — click points in order  |  Z=undo  ESC=quit"


def redraw() -> None:
    display = img_orig.copy()
    n = len(POINTS)

    for i, c in enumerate(clicks):
        col = (0, 180, 0)
        cv2.circle(display, (c["px"], c["py"]), 9, col, -1, cv2.LINE_AA)
        cv2.circle(display, (c["px"], c["py"]), 9, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(display, str(i + 1), (c["px"] + 11, c["py"] + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)

    if len(clicks) < n:
        label = POINTS[len(clicks)]["label"]
        bar = np.full((46, display.shape[1], 3), (30, 30, 30), dtype=np.uint8)
        cv2.putText(bar, f"Click point {label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 1, cv2.LINE_AA)
        display = np.vstack([bar, display])
    else:
        bar = np.full((46, display.shape[1], 3), (0, 80, 0), dtype=np.uint8)
        cv2.putText(bar, "All points set — saving calibration.json…", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 120), 1, cv2.LINE_AA)
        display = np.vstack([bar, display])

    cv2.imshow(WIN, display)


def on_mouse(event, x, y, flags, _param) -> None:
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    idx = len(clicks)
    if idx >= len(POINTS):
        return
    p = POINTS[idx]
    clicks.append({"label": p["label"], "lat": p["lat"], "lon": p["lon"],
                   "px": x, "py": y - 46})   # offset for the instruction bar
    print(f"  ✓ Point {idx + 1}: pixel ({x}, {y - 46})  GPS N:{p['lat']} W:{p['lon']}")
    redraw()

    if len(clicks) == len(POINTS):
        save_and_quit()


def save_and_quit() -> None:
    with open(CALIB_FILE, "w") as f:
        json.dump(clicks, f, indent=2)
    print(f"\nSaved → {CALIB_FILE}")
    for c in clicks:
        print(f"  {c['label']}")
        print(f"    GPS  N:{c['lat']}  W:{c['lon']}")
        print(f"    Pixel ({c['px']}, {c['py']})")
    cv2.waitKey(1200)
    cv2.destroyAllWindows()
    sys.exit(0)


def main() -> None:
    global img_orig

    img = cv2.imread(MAP_PATH)
    if img is None:
        sys.exit(f"Cannot open {MAP_PATH} — run from D:\\trackproj\\")

    # Scale down if the image is enormous
    h, w = img.shape[:2]
    max_px = 1100
    if max(h, w) > max_px:
        s = max_px / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

    img_orig = img.copy()

    print(__doc__)
    print("Points to click:")
    for i, p in enumerate(POINTS):
        print(f"  {i + 1}. {p['label']}")
    print()

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN, on_mouse)
    redraw()

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == 27:           # ESC
            print("Aborted — nothing saved.")
            break
        if key == ord("z") and clicks:
            removed = clicks.pop()
            print(f"  Undo: removed point at ({removed['px']}, {removed['py']})")
            redraw()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
