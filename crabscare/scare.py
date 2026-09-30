"""Scare finder: a sharp sound counts as a scare only if the visible males vanish.

For each audio transient, compare the claw count just before it with the
lowest count in the few seconds after it. A real scare sends most visible
males underground within seconds; noise leaves the count unchanged.

The rule deliberately looks only at how fast and how completely the males
disappear, never at how long they stay down: hiding time is the quantity being
measured, so detecting scares by it would bias the results. Nothing assumes
how many scares a video has. Frames where the view is blocked (card, notebook)
have no count and are skipped.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScareParams:
    before_s: tuple = (-6.0, -0.5)   # window for the "before" count, relative to the sound
    after_s: tuple = (0.5, 4.0)      # males must be gone within this window
    smooth_s: float = 1.0            # rolling median to suppress one-frame flicker
    min_before: float = 2.0          # need at least this many males up to judge a drop
    min_drop_frac: float = 0.7       # fraction of visible males that must disappear
    merge_s: float = 10.0            # confirmed sounds closer than this are one scare


def smoothed_counts(counts, p=ScareParams()):
    step = counts["t"].diff().median()
    win = max(1, int(round(p.smooth_s / step)))
    return counts.assign(n=counts["n_claws"].rolling(win, center=True, min_periods=1).median())


def score_candidates(transients, counts, p=ScareParams()):
    """Add before/after claw counts and the drop fraction to each audio transient."""
    c = smoothed_counts(counts, p)
    t, n = c["t"].to_numpy(), c["n"].to_numpy()

    def window(t0, lo, hi, fn):
        sel = (t >= t0 + lo) & (t <= t0 + hi) & ~np.isnan(n)
        return float(fn(n[sel])) if sel.any() else np.nan

    rows = []
    for tt in transients["t"]:
        before = window(tt, *p.before_s, np.median)
        after = window(tt, *p.after_s, np.min)
        frac = (before - after) / before if before > 0 else 0.0
        rows.append((before, after, frac))
    out = transients.copy()
    out[["before", "after", "drop_frac"]] = rows
    out["scare"] = (out["before"] >= p.min_before) & (out["drop_frac"] >= p.min_drop_frac)
    return out


def find_scares(transients, counts, p=ScareParams()):
    """Confirmed scares, one row per event.

    Confirmed sounds within merge_s of each other form one event. A quieter
    sound just before the bang can also look confirmed (the males vanish inside
    its "after" window), so the event is timed at its loudest sound; all the
    sounds in the group are kept for review.
    """
    scored = score_candidates(transients, counts, p)
    hits = scored[scored["scare"]].sort_values("t")
    group = (hits["t"].diff() > p.merge_s).cumsum()
    events = []
    for _, g in hits.groupby(group):
        best = g.loc[g["jump_db"].idxmax()]
        events.append({"t": best["t"], "jump_db": best["jump_db"],
                       "before": best["before"], "after": best["after"],
                       "drop_frac": best["drop_frac"],
                       "sounds_in_group": ", ".join(f"{x:.2f}" for x in g["t"])})
    return pd.DataFrame(events), scored
