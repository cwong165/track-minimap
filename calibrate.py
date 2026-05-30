#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Corner calibration — click, paste GPS block, save.

STEP 1  Click each corner on the map in lap order (CCW).
        Each dot is labelled T1, T2 …
        Z = undo last click

STEP 2  Ctrl+V  to paste the whole GPS coordinate block into the
        panel at the bottom.  Format accepted (one per line):
            39.536907, -122.331192
            39.536239, -122.326820
            ...
        The right half of the panel shows the matched pairs live.

STEP 3  Press D to save calibration.json.

Run from D:\\trackproj\\
"""

import json
import os
import re
import subprocess
import sys

import cv2
import numpy as np

MAP_PATH   = "TH.png"
CALIB_FILE = "calibration.json"

MAX_W      = 1000   # max display width
MAX_MAP_H  = 650    # max map height (leaves room for panel below)
PANEL_H    = 185    # height of the GPS text panel

WIN = "Calibrate  |  click corners (CCW)  |  Ctrl+V paste GPS  |  Z undo  |  D save"

# ── globals ──────────────────────────────────────────────────────────────────
corners: list      = []   # [{px_orig, py_orig}, ...]
text_buffer: str   = ""
coords: list       = []   # [(lat, lon), ...] parsed live
map_img: np.ndarray = None
disp_scale: float  = 1.0
map_w: int = 0
map_h: int = 0
frame_n: int = 0


# ── helpers ───────────────────────────────────────────────────────────────────

def get_clipboard() -> str:
    try:
        r = subprocess.run(
            ["powershell", "-command", "Get-Clipboard"],
            capture_output=True, text=True, timeout=4, encoding="utf-8",
        )
        return r.stdout
    except Exception:
        return ""


def parse_coords(text: str) -> list:
    pat = re.compile(r"(-?\d{2,3}\.\d+)\s*,\s*(-?\d{2,3}\.\d+)")
    out = []
    for m in pat.finditer(text):
        lat = float(m.group(1))
        lon = float(m.group(2))
        if lon > 0:
            lon = -lon   # W:122.33 style (no minus) → standard -122.33
        out.append((lat, lon))
    return out


def label_bg(canvas, text, x, y, font, scale, col, thick):
    """Draw text with a dark backing rectangle so it's readable on any background."""
    (tw, th), bl = cv2.getTextSize(text, font, scale, thick)
    cv2.rectangle(canvas, (x - 2, y - th - 2), (x + tw + 2, y + bl + 1),
                  (0, 0, 0), -1)
    cv2.putText(canvas, text, (x, y), font, scale, col, thick, cv2.LINE_AA)


# ── frame builder ─────────────────────────────────────────────────────────────

def build_frame() -> np.ndarray:
    global frame_n
    frame_n += 1

    canvas = np.zeros((map_h + PANEL_H, map_w, 3), dtype=np.uint8)
    canvas[:map_h] = map_img

    # ── corners on map
    n_corners = len(corners)
    for i, c in enumerate(corners):
        px = int(c["px_orig"] * disp_scale)
        py = int(c["py_orig"] * disp_scale)
        matched = i < len(coords)
        col = (0, 220, 80) if matched else (30, 100, 255)

        # dot
        cv2.circle(canvas, (px, py), 8, col, -1, cv2.LINE_AA)
        cv2.circle(canvas, (px, py), 8, (255, 255, 255), 1, cv2.LINE_AA)

        # label with background
        label_bg(canvas, f"T{i+1}", px + 10, py + 5,
                 cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 2)

        # line to previous
        if i > 0:
            p = corners[i - 1]
            cv2.line(canvas,
                     (int(p["px_orig"] * disp_scale), int(p["py_orig"] * disp_scale)),
                     (px, py), (0, 160, 50), 1, cv2.LINE_AA)

    # close-loop preview
    if n_corners >= 3:
        f, l = corners[0], corners[-1]
        cv2.line(canvas,
                 (int(l["px_orig"] * disp_scale), int(l["py_orig"] * disp_scale)),
                 (int(f["px_orig"] * disp_scale), int(f["py_orig"] * disp_scale)),
                 (0, 90, 30), 1, cv2.LINE_AA)

    # ── panel background
    py0 = map_h
    cv2.rectangle(canvas, (0, py0), (map_w, py0 + PANEL_H), (26, 26, 26), -1)
    cv2.line(canvas, (0, py0), (map_w, py0), (80, 80, 80), 1)

    # header / status bar
    n_matched = len(coords)
    ok = n_matched >= n_corners > 0
    s_col = (60, 240, 120) if ok else (40, 180, 255)
    s_text = (f"Ctrl+V = paste  |  C = clear paste  |  Z = undo corner  |  D = save   "
              f"[{n_matched} / {n_corners} matched]")
    cv2.putText(canvas, s_text, (8, py0 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, s_col, 1, cv2.LINE_AA)
    cv2.line(canvas, (0, py0 + 28), (map_w, py0 + 28), (55, 55, 55), 1)

    mid = map_w // 2

    # left half: raw pasted text
    cv2.putText(canvas, "pasted input", (8, py0 + 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, (90, 90, 90), 1, cv2.LINE_AA)
    raw_lines = text_buffer.replace("\r", "").splitlines()
    for i, line in enumerate(raw_lines[:8]):
        cv2.putText(canvas, line[:48], (8, py0 + 58 + i * 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (155, 155, 155), 1, cv2.LINE_AA)

    # blinking cursor
    cursor_row = min(len(raw_lines), 8)
    if (frame_n // 10) % 2 == 0:
        cy = py0 + 58 + cursor_row * 16
        cv2.line(canvas, (8, cy - 11), (8, cy + 1), (180, 180, 180), 1)

    # right half: parsed & matched
    cv2.line(canvas, (mid, py0 + 28), (mid, py0 + PANEL_H), (50, 50, 50), 1)
    cv2.putText(canvas, "matched corners", (mid + 8, py0 + 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.34, (90, 90, 90), 1, cv2.LINE_AA)
    for i, (lat, lon) in enumerate(coords[:8]):
        col = (70, 240, 70) if i < n_corners else (100, 100, 100)
        text = f"T{i+1}: {lat:.5f},  {lon:.6f}"
        cv2.putText(canvas, text, (mid + 8, py0 + 58 + i * 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1, cv2.LINE_AA)

    return canvas


# ── mouse ─────────────────────────────────────────────────────────────────────

def on_mouse(event, x, y, _flags, _param) -> None:
    if event != cv2.EVENT_LBUTTONDOWN or y >= map_h:
        return
    corners.append({"px_orig": round(x / disp_scale, 2),
                    "py_orig": round(y / disp_scale, 2)})
    print(f"  T{len(corners):2d}  @ ({x / disp_scale:.0f}, {y / disp_scale:.0f})")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    global map_img, disp_scale, map_w, map_h, text_buffer, coords

    # Skip if calibration already exists
    if os.path.isfile(CALIB_FILE):
        print(f"\n{CALIB_FILE} already exists.")
        print("Delete it and re-run calibrate.py if you want to redo it.")
        print("Otherwise run:  py generate_minimap.py --map TH.png")
        sys.exit(0)

    img = cv2.imread(MAP_PATH)
    if img is None:
        sys.exit(f"Cannot open {MAP_PATH!r} — run from D:\\trackproj\\")

    h, w = img.shape[:2]
    disp_scale = min(MAX_W / w, MAX_MAP_H / h)
    map_w = int(w * disp_scale)
    map_h = int(h * disp_scale)
    map_img = cv2.resize(img, (map_w, map_h), interpolation=cv2.INTER_AREA)

    print(__doc__)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, map_w, map_h + PANEL_H)
    cv2.setMouseCallback(WIN, on_mouse)

    while True:
        cv2.imshow(WIN, build_frame())
        key = cv2.waitKey(50) & 0xFF

        if key == 22:                        # Ctrl+V
            text_buffer = get_clipboard()
            coords = parse_coords(text_buffer)
            print(f"  Pasted -> {len(coords)} GPS coordinates parsed")

        elif key == ord("c"):                # clear pasted text
            text_buffer = ""
            coords = []
            print("  Cleared pasted coordinates")

        elif key == ord("z") and corners:    # undo
            corners.pop()
            print(f"  Undo → {len(corners)} corners remaining")

        elif key == ord("d"):                # done / save
            if not corners:
                print("  Click at least one corner first.")
            elif len(coords) < len(corners):
                print(f"  {len(coords)} GPS coords for {len(corners)} corners "
                      f"— paste more (or remove corners with Z).")
            else:
                break

        elif key == 27:                      # ESC
            print("Aborted — nothing saved.")
            cv2.destroyAllWindows()
            sys.exit(0)

    cv2.destroyAllWindows()

    # Build and save calibration
    out = [
        {
            "turn":    i + 1,
            "label":   f"T{i + 1}",
            "px_orig": c["px_orig"],
            "py_orig": c["py_orig"],
            "lat":     lat,
            "lon":     lon,          # standard negative convention  e.g. -122.33
        }
        for i, (c, (lat, lon)) in enumerate(zip(corners, coords))
    ]

    with open(CALIB_FILE, "w") as f:
        json.dump(out, f, indent=2)

    print(f"\nSaved {len(out)} corners → {CALIB_FILE}")
    for e in out:
        print(f"  {e['label']:3s}  GPS ({e['lat']:.5f}, {e['lon']:.6f})"
              f"  → pixel ({e['px_orig']:.0f}, {e['py_orig']:.0f})")
    print("\nNext:  py generate_minimap.py --map TH.png")


if __name__ == "__main__":
    main()
