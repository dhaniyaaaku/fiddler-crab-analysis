"""Stage 3: draw detections on the saved sample frames for checking by eye.

Usage: python run_check_frames.py "C:\\fiddlercrab\\crab triar53.MP4"
"""

import sys

import cv2

from crabscare.claws import detect, draw
from crabscare.video import output_dir

out = output_dir(sys.argv[1])
check_dir = out / "check"
check_dir.mkdir(exist_ok=True)
for f in sorted((out / "frames").glob("*.png")):
    frame = cv2.imread(str(f))
    blobs = detect(frame)
    cv2.imwrite(str(check_dir / f.name), draw(frame, blobs, f"{f.stem}  claws: {len(blobs)}"))
print(f"Saved to {check_dir.resolve()}")
