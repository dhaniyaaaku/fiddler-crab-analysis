"""Stage 4: find the scares in a scanned video and save a review clip of each.

Usage: python run_scares.py "C:\\fiddlercrab\\crab triar53.MP4"
"""

import subprocess
import sys

import cv2
import pandas as pd

from crabscare.audio import transients
from crabscare.claws import DetectorParams, claw_counts, flag_static
from crabscare.scare import ScareParams, find_scares
from crabscare.video import ffmpeg_exe, output_dir


def mmss(s):
    return f"{int(s // 60)}:{s % 60:05.2f}"


video = sys.argv[1]
out = output_dir(video)
p, sp = DetectorParams(), ScareParams()

static_core = cv2.imread(str(out / "static_yellow_core.png"), cv2.IMREAD_GRAYSCALE)
cands = flag_static(pd.read_csv(out / "candidates.csv"), static_core, p)
counts = claw_counts(cands, pd.read_csv(out / "frames.csv"), p)
counts.to_csv(out / "claw_counts.csv", index=False)
tr = transients(out / "audio.wav")
scares, scored = find_scares(tr, counts, sp)
scored.to_csv(out / "sound_candidates.csv", index=False)
scares.to_csv(out / "scares.csv", index=False)

print(f"{len(tr)} sharp sounds; {len(scares)} confirmed scare(s)")
for i, s in scares.iterrows():
    print(f"  scare {i + 1} at {mmss(s['t'])}: {s['before']:.0f} males up -> {s['after']:.0f} "
          f"within {sp.after_s[1]:.0f} s ({s['drop_frac']:.0%}); sounds in group: {s['sounds_in_group']}")
    clip = out / f"scare{i + 1}_{mmss(s['t']).replace(':', 'm')}s.mp4"
    subprocess.run([ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
                    "-ss", f"{max(0, s['t'] - 3):.2f}", "-i", str(video), "-t", "10",
                    "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac", str(clip)], check=True)

rejected = scored[~scored["scare"] & (scored["before"] >= sp.min_before)]
print("\nClosest rejected sounds (highest drop fraction):")
for _, r in rejected.sort_values("drop_frac", ascending=False).head(5).iterrows():
    print(f"  {mmss(r['t'])}: {r['before']:.0f} -> {r['after']:.0f} ({r['drop_frac']:.0%}), "
          f"jump {r['jump_db']:.0f} dB")
