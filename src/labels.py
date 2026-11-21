"""Test population and labels (both validated against the official files, see scripts/validate_task_definition.py)."""
import numpy as np
import pandas as pd
from .config import *


def eligible_users(ev, cohort_ids, list_time):
    """Cohort users with >= 5 activity records in the 90 days before `list_time` (reproduces test_input.csv exactly)."""
    lo = list_time - pd.Timedelta(days=ELIGIBLE_LOOKBACK_DAYS)
    parts = [e.loc[(e.t > lo) & (e.t <= list_time), "u"] for e in ev.values()]
    cnt = pd.concat(parts).value_counts()
    return np.intersect1d(cnt[cnt >= ELIGIBLE_MIN_RECORDS].index.values, cohort_ids)


def window_labels(ev, user_ids, window_start):
    """0/1 matrix (len(user_ids) x 6): does the user perform each activity in [window_start, window_start + 7 days)?"""
    end = window_start + pd.Timedelta(days=WINDOW_DAYS)
    out = {}
    for k in LABELS:
        e = ev[k]
        lo, hi = e.t.searchsorted(window_start, "left"), e.t.searchsorted(end, "left")    # e is sorted by time
        out[k] = np.isin(user_ids, e.u.values[lo:hi]).astype(np.int8)
    return pd.DataFrame(out, index=user_ids)


def window_coverage(alive, window_start):
    """Share of the hours of the label window in which the push stream was being collected (see data.hourly_alive)."""
    h0 = int((window_start - DAY0) / pd.Timedelta(hours=1))
    return float(alive[h0:h0 + 24 * WINDOW_DAYS].mean())
