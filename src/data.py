"""Load the 8 relational tables, build one (user, time) event stream per target activity, measure data coverage."""
import numpy as np
import pandas as pd
from .config import DATA_DIR, LABELS, DAY0, NB, PUSH_ALIVE_MIN, DATA_START

TABLES = ["Users", "Repositories", "PullRequests", "Issues", "IssueComments", "PRReviews", "PRReviewComments", "Pushes"]
ROW_TIME = {"Users": "first_seen_at", "Repositories": "first_seen_at", "PullRequests": "first_seen_at", "Issues": "first_seen_at",
            "IssueComments": "created_at", "PRReviews": "first_seen_at", "PRReviewComments": "created_at", "Pushes": "created_at"}


def load_tables(columns=None):
    columns = columns or {}
    T = {n: pd.read_parquet(DATA_DIR / f"{n}.parquet", columns=columns.get(n)) for n in TABLES}
    first = min(T[n][c].min() for n, c in ROW_TIME.items())
    if first < DATA_START:
        print(f"WARNING: the tables contain rows from {first}, before {DATA_START} (UTC): the submitted model was built from the GH Archive hours "
              f"of 2026-06-01 .. 2026-11-21 only, so a re-run with older raw data does not give exactly the same result.", flush=True)
    return T


def later(a, b):
    """Element-wise later one of two time arrays of equal length."""
    a, b = pd.Series(np.asarray(a)), pd.Series(np.asarray(b))
    return a.where(a >= b, b).values


def event_streams(T, visible=False):
    """One DataFrame per target activity with columns u (Users row id), repo_id and t (UTC), sorted by t.

    visible=False: t is the time the user performed the action (PR opened, issue opened, review submitted, ...), which is what the
                   labels are defined on.  Rows whose opening time is unknown (null) are not events.
    visible=True:  t is the time from which the event EXISTS in the database: the later of the event time and the row time of its own row
                   (first_seen_at; for comments created_at).  This is the time to use for features: a row that is first seen only later (for
                   example an issue that was opened inside a gap of GH Archive and shows up with a later comment) must not be used before it
                   appears, otherwise the training features contain information that the test features cannot have.
    Events whose real time lies before the study period (DAY0) are dropped in both cases.
    """
    pr, iss, ic, rv, rc, ps = (T[n] for n in ["PullRequests", "Issues", "IssueComments", "PRReviews", "PRReviewComments", "Pushes"])
    p_ = pr.dropna(subset=["author_id", "created_at"])
    i_ = iss[~iss.is_pull_request].dropna(subset=["author_id"])
    spec = {   # activity -> (table, user column, event-time column, own row-time column)
        "open_pr": (p_, "author_id", "created_at", "first_seen_at"), "open_issue": (i_, "author_id", "created_at", "first_seen_at"),
        "issue_comment": (ic, "author_id", "created_at", "created_at"), "pr_review": (rv, "reviewer_id", "submitted_at", "first_seen_at"),
        "pr_review_comment": (rc, "author_id", "created_at", "created_at"), "push": (ps, "user_id", "created_at", "created_at")}
    ev = {}
    for k, (df, uc, tc, rc_) in spec.items():
        t = df[tc].values
        e = pd.DataFrame({"u": df[uc].values.astype("int64"), "repo_id": df.repo_id.values.astype("int64"), "t0": t,
                          "t": later(t, df[rc_].values) if visible else t})
        ev[k] = e[e.t0 >= DAY0].drop(columns="t0").sort_values("t", kind="stable").reset_index(drop=True)
    return ev


def release_time(T):
    """End of the data: the end of the hour of the newest event of any table (GH Archive files are hourly: the hour file 2026-11-21-11.json.gz ends at
    2026-11-21 12:00 UTC, its newest event is at 11:59:xx)."""
    latest = max(T[n][c].max() for n, c in ROW_TIME.items())
    return latest.floor("h") + pd.Timedelta(hours=1)


def hourly_alive(push_events):
    """alive[h] = 1 if the push stream was collected in hour h (hours counted from DAY0), else 0.

    The cohort pushes 100-400 times per hour when GH Archive is complete, so an hour with fewer than PUSH_ALIVE_MIN pushes is a gap
    or a heavily thinned hour, not a quiet one (the push stream is almost empty from 10/27 on, and on 8/19-8/20, 9/1-9/3, 10/17)."""
    h = ((push_events.t - DAY0) // pd.Timedelta(hours=1)).values.astype(np.int64)
    ok = (h >= 0) & (h < NB * 24)
    return (np.bincount(h[ok], minlength=NB * 24) >= PUSH_ALIVE_MIN).astype(np.float32)
