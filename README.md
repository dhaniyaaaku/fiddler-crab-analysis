# Fiddler crab scare-response analysis

Automated analysis of overhead trial videos of male fiddler crabs. The goal is to measure how long each male stays in his burrow after a scare (a textbook bang), with and without females present, without watching every video by hand.

## How it works

1. **Scene model.** A clean background is built from the per-pixel median of frames spread across the video. Frames where a large new white object covers the view (the trial card, the notebook used for the scare) are marked as blocked and skipped.
2. **Claw detection.** Males are found by their yellow major claw. A claw must be much more yellow than the surrounding sand (Lab b\*, corrected for shadow) and slightly greener than it (Lab a\*), which separates claws from orange pebbles, rust on the quadrat joints and shadow edges. Spots that are yellow in nearly every frame are ignored.
3. **Scare detection.** Every sharp sound in the audio is a candidate. It counts as a scare only if at least 70% of the visible males disappear within 4 seconds of it. The rule never uses how long the males stay down, since that is the quantity being measured.
4. **Re-emergence timing** per burrow (in progress).

Detection thresholds are set with `calibrate.py` from labelled data (`calibration/`): stretches of video where every crab is underground, and claws confirmed by eye.

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

FFmpeg is bundled through `imageio-ffmpeg`; nothing else needs installing.

## Usage

Run the stages in order for each video:

```
python -m crabscare.video "path\to\video.MP4"    # video info, audio track, sample frames
python run_scan.py "path\to\video.MP4"           # claw scan, ~13 min for a 29 min video
python run_scares.py "path\to\video.MP4"         # scare times and a review clip of each
```

Checking and calibration:

```
python run_check_frames.py "path\to\video.MP4"   # detections drawn on the sample frames
python calibrate.py "path\to\video.MP4" calibration\crab_triar53.json
```

Results are written to `output/<video name>/`.

## Status

- Tested on one trial (53): one scare found at 9:14.87 among 322 sharp sounds; no false claw detections in crab-free stretches; 27 of 28 hand-confirmed claws detected.
- All thresholds were calibrated on that same trial, so accuracy still has to be measured on videos not used for tuning.
- Very small claws can be missed at 720p.
