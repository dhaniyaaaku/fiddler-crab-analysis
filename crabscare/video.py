"""Video input: metadata, audio extraction and frame grabbing."""

import json
import re
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg

AUDIO_SR = 16000  # Hz; plenty for detecting impulsive sounds such as the bang


def ffmpeg_exe():
    return imageio_ffmpeg.get_ffmpeg_exe()


def slug(video_path):
    """Folder-safe name for a video, e.g. 'crab triar53.MP4' -> 'crab_triar53'."""
    return re.sub(r"[^A-Za-z0-9]+", "_", Path(video_path).stem).strip("_")


def output_dir(video_path, root="output"):
    d = Path(root) / slug(video_path)
    d.mkdir(parents=True, exist_ok=True)
    return d


def probe(video_path):
    """Return basic video properties and whether an audio stream exists."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Cannot open {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    info = {
        "path": str(video_path),
        "fps": fps,
        "n_frames": n_frames,
        "duration_s": n_frames / fps if fps else None,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()

    # ffmpeg prints stream info to stderr and exits non-zero without an output file
    res = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(video_path)],
                         capture_output=True, text=True)
    info["has_audio"] = "Audio:" in res.stderr
    return info


def extract_audio(video_path, wav_path, sr=AUDIO_SR):
    """Write the audio track as mono 16-bit WAV."""
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
           "-i", str(video_path), "-vn", "-ac", "1", "-ar", str(sr),
           "-acodec", "pcm_s16le", str(wav_path)]
    subprocess.run(cmd, check=True)
    return wav_path


def grab_frames(video_path, times_s, out_dir):
    """Save the frame nearest each time (seconds) as PNG; returns the file paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    paths = []
    for t in times_s:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
        ok, frame = cap.read()
        if not ok:
            continue
        p = out_dir / f"t{int(t // 60):02d}m{t % 60:05.2f}s.png"
        cv2.imwrite(str(p), frame)
        paths.append(p)
    cap.release()
    return paths


def run_stage1(video_path, sample_times_s):
    out = output_dir(video_path)
    info = probe(video_path)
    (out / "info.json").write_text(json.dumps(info, indent=2))
    if info["has_audio"]:
        extract_audio(video_path, out / "audio.wav")
    grab_frames(video_path, sample_times_s, out / "frames")
    return info, out


if __name__ == "__main__":
    import sys
    video = sys.argv[1]
    # A spread of times across the trial; the scare is roughly 9 min in this video
    times = [5, 93, 300, 530, 545, 560, 575, 600, 660, 900, 1200, 1500, 1740]
    info, out = run_stage1(video, times)
    print(json.dumps(info, indent=2))
    print("Output:", out.resolve())
