"""Daily per-user activity matrices and the temporal features (Q3).

Every stream (the 6 target activities and the relational streams of relational.py) is stored as a users x days matrix of event
counts plus its cumulative sums, so that the count of any window ending at a cutoff is an O(1) lookup per user.

Collection gaps: the push stream has days / hours in which nothing was collected.  A count over a window that contains such
gaps is not "0 pushes" but "unknown", so every window feature of a stream is divided by the share of the window that was really
collected (coverage-adjusted count) and becomes NaN when less than MIN_COVER of the window was collected.  For a stream without
gaps the coverage is 1 and all formulas reduce to the plain counts.
"""
import numpy as np
import pandas as pd
from .config import LABELS, DAY0, NB
from .data import hourly_alive

WINDOWS = [1, 2, 3, 7, 14, 28, 56, 90]
MIN_COVER = 0.25
COLLAB = [k for k in LABELS if k != "push"]          # the five collaboration activities (no collection gaps found)


class Daily:
    def __init__(self, T, ev, adjust=True):
        users = T["Users"]
        self.cohort_ids = users.loc[users.is_cohort, "user_id"].values
        self.cidx = np.full(len(users), -1, np.int32)
        self.cidx[self.cohort_ids] = np.arange(len(self.cohort_ids))
        self.n = len(self.cohort_ids)
        self.adjust = adjust
        self.C, self.S, self.last, self.E, self.SE = {}, {}, {}, {}, {}
        self.alive = hourly_alive(ev["push"])                                   # hourly push-collection indicator
        self.push_cover = self.alive.reshape(NB, 24).mean(1)                    # share of each daily bin that was collected
        for k in LABELS:
            self.add_stream(k, ev[k].u.values, ev[k].t.values, self.push_cover if k == "push" else None)
        self._register("total", sum(self.C[k] for k in COLLAB), None)           # all collaboration activity of a user

    # ------------------------------------------------------------------------------------------------------------------
    def _register(self, name, C, cover):
        self.C[name] = C
        self.S[name] = np.concatenate([np.zeros((self.n, 1), np.int32), np.cumsum(C, axis=1, dtype=np.int32)], axis=1)
        self.last[name] = np.maximum.accumulate(np.where(C > 0, np.arange(NB, dtype=np.int16)[None, :], np.int16(-1)), axis=1)
        e = np.ones(NB) if (cover is None or not self.adjust) else np.asarray(cover, np.float64)
        self.E[name] = e
        self.SE[name] = np.concatenate([[0.0], np.cumsum(e)])

    def add_stream(self, name, user_rows, times, cover=None):
        """Add an event stream: Users row ids + event times -> daily count matrix of the cohort users."""
        row = self.cidx[np.asarray(user_rows).astype(np.int64)]
        b = ((pd.to_datetime(times) - DAY0) // pd.Timedelta(days=1)).values.astype(np.int64)
        ok = (row >= 0) & (b >= 0) & (b < NB)
        C = np.bincount(row[ok] * NB + b[ok], minlength=self.n * NB).reshape(self.n, NB).astype(np.int32)
        self._register(name, C, cover)

    def bin_of(self, cutoff):
        return int((cutoff - DAY0) // pd.Timedelta(days=1))

    # ---- adjusted building blocks --------------------------------------------------------------------------------------
    def window(self, k, rows, b, w):
        """Coverage-adjusted event count of the w days before bin b.

        If less than MIN_COVER of a window of >= 7 days was collected (e.g. the push stream is down for a week), the count is
        estimated from the first longer window (28 / 56 / 90 days) that was collected well enough: events per collected day x w.
        Shorter windows give NaN.  (This case never occurs in the training samples, it only makes the prediction at a cutoff
        that follows a long outage well defined.)"""
        lo = max(b - w, 0)
        raw = (self.S[k][rows, b] - self.S[k][rows, lo]).astype(np.float32)
        e, n = self.SE[k][b] - self.SE[k][lo], b - lo
        if e >= n - 1e-6:
            return raw
        if e >= MIN_COVER * n:
            return raw * np.float32(n / e)
        if n >= 7:
            for w2 in (28, 56, 90):
                lo2 = max(b - w2, 0)
                e2, n2 = self.SE[k][b] - self.SE[k][lo2], b - lo2
                if w2 > w and e2 >= MIN_COVER * n2:
                    return ((self.S[k][rows, b] - self.S[k][rows, lo2]) / np.float32(e2) * np.float32(n)).astype(np.float32)
        return np.full(len(rows), np.nan, np.float32)

    def days_since_last(self, k, rows, b):
        """Collected days since the last event (999 = never)."""
        last = self.last[k][rows, b - 1].astype(np.int64)
        since = self.SE[k][b] - self.SE[k][np.maximum(last + 1, 0)]
        return np.where(last < 0, 999.0, since).astype(np.float32)

    def active_days(self, k, rows, b, w=28):
        a = (self.C[k][rows, b - w:b] > 0).sum(1).astype(np.float32)
        e = self.SE[k][b] - self.SE[k][b - w]
        if e >= w - 1e-6:
            return a
        if e < MIN_COVER * w:
            return np.full(len(rows), np.nan, np.float32)
        return np.minimum(a * (w / e), w).astype(np.float32)

    def ewma(self, k, rows, b, half_life, span=60):
        """Exponentially decayed activity (weights 0.5 ** (age / half_life)) of the last `span` days."""
        wts = 0.5 ** (np.arange(1, span + 1) / half_life)[::-1]
        num = self.C[k][rows, b - span:b] @ wts
        den, tot = float(self.E[k][b - span:b] @ wts), float(wts.sum())
        if den >= tot - 1e-9:
            return num.astype(np.float32)
        if den <= 1e-9:
            return np.full(len(rows), np.nan, np.float32)
        return (num * (tot / den)).astype(np.float32)                              # extrapolate from the collected part of the span

    # ---- the temporal feature block ------------------------------------------------------------------------------------
    def temporal_features(self, rows, b):
        """User-only features from data strictly before the cutoff (bin b starts at the cutoff)."""
        f = {}
        for k in LABELS + ["total"]:
            for w in WINDOWS:
                f[f"{k}_n{w}d"] = self.window(k, rows, b, w)                       # multi-window counts
            f[f"{k}_active_days28"] = self.active_days(k, rows, b, 28)
            f[f"{k}_days_since_last"] = self.days_since_last(k, rows, b)           # recency
            for h in (3, 14):
                f[f"{k}_ewma_h{h}"] = self.ewma(k, rows, b, h)                     # exponentially decayed activity
            for i in range(8):
                f[f"{k}_week_lag{i}"] = self.window(k, rows, b - 7 * i, 7)         # weekly counts of the last 8 weeks
            f[f"{k}_trend"] = (f[f"{k}_n7d"] + 0.5) / (f[f"{k}_n28d"] / 4 + 0.5)  # last week vs the 4-week average
        return pd.DataFrame(f)


def global_context(X_temporal, cutoffs, daily=None):
    """Cohort-level context at each cutoff (identical for all users of that cutoff).

    * mean activity of the users of the cutoff in the last 7 / 28 days (glob_*): lets the model see level shifts of the whole
      platform / of the data collection (the weekly positive rates move by a factor of 5-10 over the project period);
    * share of the last 3 / 7 / 28 days in which the push stream was being collected (glob_push_cover_*), if `daily` is given:
      the counts above are adjusted for gaps, but a recent gap is itself informative, because poor collection tends to persist."""
    cols = [f"{k}_n7d" for k in LABELS] + [f"{k}_n28d" for k in LABELS]
    cuts = np.asarray(cutoffs)
    g = X_temporal[cols].groupby(cuts).mean()
    g.columns = ["glob_" + c for c in cols]
    if daily is not None:
        for w in (3, 7, 28):
            g[f"glob_push_cover_{w}d"] = [float(daily.push_cover[daily.bin_of(pd.Timestamp(c)) - w:daily.bin_of(pd.Timestamp(c))].mean()) for c in g.index]
    return g.loc[cuts].reset_index(drop=True).astype(np.float32)
