"""Scene model: permanent white objects, and frames where the view is blocked.

The clean background is the per-pixel median of frames spread across the
whole video, so anything transient (crabs, hands, the notebook, the card)
drops out. Permanent white things in it (the quadrat, shells) are masked;
white appearing anywhere else means an object is between camera and ground.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class SceneParams:
    n_background_frames: int = 60
    white_L: int = 235          # Lab lightness (0-255); sunlit sand is ~150-175, paper and PVC ~255
    white_chroma: int = 20      # max distance of a*, b* from neutral (128)
    static_dilate_px: int = 7   # grow the permanent-white mask to cover small camera jitter
    blocked_frac: float = 0.03  # frame counts as blocked above this fraction of new white pixels
                                # (the pipe alone is ~1%, the card and notebook far more)
    blocked_guard_s: float = 0.4  # also distrust frames this close to a blocked frame


def median_background(video_path, n=SceneParams.n_background_frames):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for idx in np.linspace(0, total - 1, n + 2)[1:-1].astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, f = cap.read()
        if ok:
            frames.append(f)
    cap.release()
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


def white_mask_lab(lab, p=SceneParams()):
    L = lab[:, :, 0]
    a = cv2.absdiff(lab[:, :, 1], 128)
    b = cv2.absdiff(lab[:, :, 2], 128)
    return ((L >= p.white_L) & (a <= p.white_chroma) & (b <= p.white_chroma)).astype(np.uint8)


def white_mask(frame_bgr, p=SceneParams()):
    return white_mask_lab(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB), p)


def static_white_mask(background_bgr, p=SceneParams()):
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * p.static_dilate_px + 1,) * 2)
    return cv2.dilate(white_mask(background_bgr, p), k)


def new_white_fraction_lab(lab, static_white, p=SceneParams()):
    """Fraction of the frame that is white but not permanently white."""
    w = white_mask_lab(lab, p)
    w[static_white > 0] = 0
    return float(w.mean())


def new_white_fraction(frame_bgr, static_white, p=SceneParams()):
    return new_white_fraction_lab(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB), static_white, p)
