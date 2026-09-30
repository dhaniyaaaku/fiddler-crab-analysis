"""Stage 2: build the scene model and scan a whole video for claw candidates.

Usage: python run_scan.py "C:\\fiddlercrab\\crab triar53.MP4"
"""

import json
import sys
import time

import cv2

from crabscare import scene
from crabscare.claws import DetectorParams, params_dict, scan_video
from crabscare.video import output_dir

video = sys.argv[1]
p, sp = DetectorParams(), scene.SceneParams()
out = output_dir(video)
t0 = time.time()

print("Building clean background ...", flush=True)
bg = scene.median_background(video, sp.n_background_frames)
static_white = scene.static_white_mask(bg, sp)
cv2.imwrite(str(out / "background.png"), bg)
cv2.imwrite(str(out / "static_white.png"), static_white * 255)

print(f"Scanning at {p.sample_fps} frames/s ...", flush=True)
cands, frames, static_core = scan_video(video, static_white, p, sp)
cands.to_csv(out / "candidates.csv", index=False)
frames.to_csv(out / "frames.csv", index=False)
cv2.imwrite(str(out / "static_yellow_core.png"), static_core * 255)
(out / "params.json").write_text(json.dumps({"detector": params_dict(p), "scene": vars(sp)}, indent=2))

print(f"{len(frames)} frames ({frames['blocked'].sum()} blocked), "
      f"{len(cands)} candidates; {time.time() - t0:.0f} s total")
print(f"Saved to {out.resolve()}")
