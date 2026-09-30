"""Audio transients: candidate times for the textbook bang.

Every sharp sound is only a candidate; the claw count decides which ones are
real scares (see scare.py). The score is how far the loudness jumps above the
level of the preceding second, so steady noise such as wind or talking scores
low and sudden impacts score high.
"""

import numpy as np
import pandas as pd
from scipy import signal
from scipy.io import wavfile

HOP_S = 0.01          # loudness resolution
HIGHPASS_HZ = 500     # impacts are broadband; this drops wind rumble and much of the voice
BASELINE_S = 1.0      # "before" window for the jump score
MIN_GAP_S = 0.5       # peaks closer than this are one event


def loudness_db(wav_path):
    sr, x = wavfile.read(wav_path)
    x = x.astype(np.float32) / 32768.0
    sos = signal.butter(4, HIGHPASS_HZ, btype="highpass", fs=sr, output="sos")
    x = signal.sosfilt(sos, x)
    hop = int(sr * HOP_S)
    n = len(x) // hop
    energy = (x[: n * hop].reshape(n, hop) ** 2).mean(axis=1)
    return 10 * np.log10(energy + 1e-12)


def transients(wav_path, min_jump_db=12.0):
    """Sharp sounds as a table: time (s), jump above the previous second (dB), level (dB)."""
    db = loudness_db(wav_path)
    base_n = int(BASELINE_S / HOP_S)
    # median of the preceding second, lagged so the onset itself isn't included
    baseline = pd.Series(db).shift(5).rolling(base_n, min_periods=base_n // 2).median().to_numpy()
    jump = np.nan_to_num(db - baseline, nan=0.0)
    peaks, props = signal.find_peaks(jump, height=min_jump_db, distance=int(MIN_GAP_S / HOP_S))
    return pd.DataFrame({"t": peaks * HOP_S, "jump_db": props["peak_heights"], "level_db": db[peaks]})
