"""Choose the claw thresholds from labelled data.

For each yellowness threshold, report how many noise candidates survive in the
crab-free stretches (false detections) and how many hand-confirmed claws are
still detected, with and without the greenness check.

Usage: python calibrate.py "C:\\fiddlercrab\\crab triar53.MP4" calibration\\crab_triar53.json
"""

import json
import sys

import cv2
import numpy as np
import pandas as pd

from crabscare.claws import DetectorParams, flag_static
from crabscare.video import output_dir

video, labels_path = sys.argv[1], sys.argv[2]
p = DetectorParams()
out = output_dir(video)
labels = json.loads(open(labels_path).read())
static_core = cv2.imread(str(out / "static_yellow_core.png"), cv2.IMREAD_GRAYSCALE)
cands = flag_static(pd.read_csv(out / "candidates.csv"), static_core, p)
frames = pd.read_csv(out / "frames.csv")

usable = cands[~cands["static"] & ~cands["frame"].isin(frames.loc[frames["blocked"], "frame"])]
in_neg = np.zeros(len(usable), dtype=bool)
neg_frames = 0
for lo, hi in labels["negative_windows_s"]:
    in_neg |= usable["t"].between(lo, hi).to_numpy()
    neg_frames += int(frames["t"].between(lo, hi).sum())
neg = usable[in_neg]

pos = []
for t, x, y in labels["positive_claws"]:
    near = usable[(usable["t"] - t).abs() < 0.11]
    d = np.hypot(near["x"] - x, near["y"] - y)
    if len(d) and d.min() < 15:
        r = near.loc[d.idxmin()]
        pos.append((r["peak"], r["green"]))
    else:
        pos.append((0.0, 0.0))
pos = pd.DataFrame(pos, columns=["peak", "green"])

print(f"Crab-free frames: {neg_frames}; noise candidates there: {len(neg)}")
print("Noisiest candidates in crab-free frames:")
print(neg.sort_values("peak", ascending=False).head(8).round(1).to_string(index=False))
print(f"\nHand-confirmed claws: {len(pos)}; greenness range {pos['green'].min():.0f} to {pos['green'].max():.0f}")
print(f"\nGreenness check: median a* relative to sand <= {p.max_green}")
print("threshold | false detections: yellow only, +green | claws detected: yellow only, +green")
neg_green = neg["green"] <= p.max_green
pos_green = pos["green"] <= p.max_green
for thr in range(18, 41, 2):
    print(f"{thr:9d} | {int((neg['peak'] >= thr).sum()):17d} {int(((neg['peak'] >= thr) & neg_green).sum()):7d}"
          f" | {int((pos['peak'] >= thr).sum()):17d}/{len(pos)} {int(((pos['peak'] >= thr) & pos_green).sum()):4d}/{len(pos)}")
