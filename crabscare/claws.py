"""Yellow-claw detection.

A claw is a small patch much more yellow than the sand around it. Yellowness
is Lab b* (the yellow-blue axis) minus the local background median, scaled up
in shadow by (sunlit sand brightness / local sand brightness), because shade
dulls colour roughly in proportion to how much darker it is.

Blobs are grouped at the LOW threshold and keep their peak yellowness. The scan
records every candidate with peak >= candidate_floor; whether a candidate is a
claw (peak >= high) is decided afterwards, so `high` can be calibrated without
rescanning. Spots that are yellow in nearly every frame (pebbles, shells) are
marked static and ignored.
"""

import time
from dataclasses import asdict, dataclass

import cv2
import numpy as np
import pandas as pd

from . import scene


@dataclass(frozen=True)
class DetectorParams:
    bg_kernel: int = 41            # px; local background window (claws are ~10-20 px)
    max_shadow_gain: float = 3.0   # cap on the shadow correction
    low: float = 15                # corrected yellowness to belong to a blob
    candidate_floor: float = 18    # blobs with a lower peak are not recorded at all
    # Both cutoffs set with calibrate.py on trial 53, midway between the noise in
    # crab-free frames and the weakest hand-confirmed claw:
    high: float = 28               # peak yellowness: noise max 27.2, weakest claw 30
    max_green: float = -3.5        # median a* relative to sand: claws are lime-yellow (-4 to
                                   # -10); rust, pebbles, shadow-edge noise are not (-3 to +4)
    min_area: int = 15             # px at the LOW threshold
    max_area: int = 800            # px; rejects large yellow objects
    sample_fps: float = 5.0        # frames analysed per second of video
    static_frac: float = 0.9       # yellow in at least this fraction of frames -> static
    static_dilate_px: int = 12     # a pebble's detected centre can sit several px off its core


def _quarter_median(ch, k):
    small = cv2.resize(ch, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    return cv2.medianBlur(small, int(round(k / 4)) | 1)


def yellowness(lab, p=DetectorParams()):
    L, b = lab[:, :, 0], lab[:, :, 2]
    h, w = b.shape
    b_bg = cv2.resize(_quarter_median(b, p.bg_kernel), (w, h), interpolation=cv2.INTER_LINEAR)
    L_bg = _quarter_median(L, p.bg_kernel).astype(np.float32)
    L_sun = np.percentile(L_bg, 75)  # brightness of sunlit sand in this frame
    gain = np.clip(L_sun / np.maximum(L_bg, 1.0), 1.0, p.max_shadow_gain)
    gain = cv2.resize(gain, (w, h), interpolation=cv2.INTER_LINEAR)
    return (b.astype(np.float32) - b_bg.astype(np.float32)) * gain


def greenness(lab, p=DetectorParams()):
    """Lab a* relative to the local background; negative means greener than the sand."""
    a = lab[:, :, 1]
    a_bg = cv2.resize(_quarter_median(a, p.bg_kernel), (a.shape[1], a.shape[0]),
                      interpolation=cv2.INTER_LINEAR)
    return cv2.subtract(a, a_bg, dtype=cv2.CV_16S)


def find_blobs(rel, a_rel, p=DetectorParams()):
    """Candidate blobs as (x, y, area, peak yellowness, median relative a*) tuples."""
    low_mask = (rel >= p.low).astype(np.uint8)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(low_mask, connectivity=8)
    blobs = []
    for i in np.unique(labels[rel >= p.candidate_floor]):
        area = stats[i, cv2.CC_STAT_AREA]
        if i == 0 or not p.min_area <= area <= p.max_area:
            continue
        x, y, w, h = stats[i, :4]
        box = labels[y:y + h, x:x + w] == i
        peak = rel[y:y + h, x:x + w][box].max()
        green = np.median(a_rel[y:y + h, x:x + w][box])
        blobs.append((float(cents[i, 0]), float(cents[i, 1]), int(area), float(peak), float(green)))
    return blobs


def is_claw(peak, green, p=DetectorParams()):
    return (peak >= p.high) & (green <= p.max_green)


def detect(frame_bgr, p=DetectorParams()):
    """Claws in a single frame (no static masking); for quick checks."""
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    return [bl for bl in find_blobs(yellowness(lab, p), greenness(lab, p), p)
            if is_claw(bl[3], bl[4], p)]


def scan_video(video_path, static_white, p=DetectorParams(), sp=scene.SceneParams(),
               progress_every_s=300):
    """Scan frames at p.sample_fps.

    Returns (candidates, frames, static_core): one row per candidate blob, one
    row per analysed frame (with the view-blocked flag), and the undilated mask
    of pixels that are yellow in nearly every frame (see flag_static).
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    step = max(1, round(fps / p.sample_fps))
    acc, n_open = None, 0
    blob_rows, frame_rows = [], []
    idx, t0, next_report = 0, time.time(), progress_every_s
    while True:
        if idx % step:
            if not cap.grab():  # decode but skip colour conversion
                break
            idx += 1
            continue
        ok, frame = cap.read()
        if not ok:
            break
        t = idx / fps
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        white = scene.new_white_fraction_lab(lab, static_white, sp)
        blocked = white > sp.blocked_frac
        rel = yellowness(lab, p)
        if not blocked:
            low = (rel >= p.low).astype(np.uint16)
            acc = low if acc is None else acc + low
            n_open += 1
        frame_rows.append((idx, t, white, blocked))
        blob_rows.extend((idx, t, *bl) for bl in find_blobs(rel, greenness(lab, p), p))
        if t >= next_report:
            print(f"  {t / 60:5.1f} min of video done ({time.time() - t0:.0f} s elapsed)", flush=True)
            next_report += progress_every_s
        idx += 1
    cap.release()

    static_core = (acc >= p.static_frac * n_open).astype(np.uint8)
    cands = pd.DataFrame(blob_rows, columns=["frame", "t", "x", "y", "area", "peak", "green"])
    frames = pd.DataFrame(frame_rows, columns=["frame", "t", "new_white", "blocked"])
    return cands, frames, static_core


def flag_static(cands, static_core, p=DetectorParams()):
    """Mark candidates centred within static_dilate_px of an always-yellow spot."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * p.static_dilate_px + 1,) * 2)
    static = cv2.dilate((static_core > 0).astype(np.uint8), k)
    xi = cands["x"].round().astype(int).clip(0, static.shape[1] - 1)
    yi = cands["y"].round().astype(int).clip(0, static.shape[0] - 1)
    return cands.assign(static=static[yi, xi].astype(bool))


def claw_counts(cands, frames, p=DetectorParams(), sp=scene.SceneParams()):
    """Claws per analysed frame; NaN where the view is blocked. Needs flag_static first.

    Frames within sp.blocked_guard_s of a blocked frame are also treated as
    blocked: an object entering or leaving is partly in view before it crosses
    the blocked threshold.
    """
    claws = cands[is_claw(cands["peak"], cands["green"], p) & ~cands["static"]]
    n = claws.groupby("frame").size()
    out = frames.copy()
    step = out["t"].diff().median()
    win = 2 * int(round(sp.blocked_guard_s / step)) + 1
    out["blocked"] = out["blocked"].astype(int).rolling(win, center=True, min_periods=1).max() > 0
    out["n_claws"] = out["frame"].map(n).fillna(0).astype(float)
    out.loc[out["blocked"], "n_claws"] = np.nan
    return out


def draw(frame_bgr, blobs, label=None):
    out = frame_bgr.copy()
    for x, y, area, *_ in blobs:
        r = int(max(10, np.sqrt(area) * 1.5))
        cv2.circle(out, (int(x), int(y)), r, (255, 0, 255), 2)
    if label:
        cv2.putText(out, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
    return out


def params_dict(p):
    return asdict(p)
