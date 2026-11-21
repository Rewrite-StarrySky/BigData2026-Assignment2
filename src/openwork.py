"""Work-in-progress features (Q2): items (PRs / issues / threads) a user is attached to through foreign keys that are still open.

An item counts as open at cutoff c unless its closed_at (known only later) is before c, so nothing from the future is used.  An item
exists from the time its row is in the database (the later of created_at and first_seen_at), not from its created_at.
"""
import numpy as np
import pandas as pd
from .config import DAY0
from .data import later


class OpenWork:
    def __init__(self, daily, T):
        cidx = daily.cidx
        pr, iss, ic, rv, rc = (T[n] for n in ["PullRequests", "Issues", "IssueComments", "PRReviews", "PRReviewComments"])
        INF = pd.Timestamp("2100-01-01")
        pr_close = pr.closed_at.fillna(INF).values
        iss_close = iss.closed_at.fillna(INF).values
        ci = lambda a: cidx[np.asarray(a, dtype="int64")]
        # (a) issues I opened                                      Issues.author_id -> Users
        i = iss[~iss.is_pull_request & iss.created_at.notna()]
        self.my_issue = dict(u=ci(i.author_id), t=later(i.created_at.values, i.first_seen_at.values), close=i.closed_at.fillna(INF).values)
        # (b) PRs I opened                                         PullRequests.author_id -> Users
        p = pr.dropna(subset=["author_id", "created_at"])
        self.my_pr = dict(u=ci(p.author_id), t=later(p.created_at.values, p.first_seen_at.values), close=p.closed_at.fillna(INF).values)
        # (c) discussion threads I commented in                    IssueComments.issue_id -> Issues (closed_at of the thread)
        self.thread = dict(u=ci(ic.author_id), key=ic.issue_id.values, t=ic.created_at.values, close=iss_close[ic.issue_id.values])
        # (d) PRs I reviewed or review-commented                   PRReviews / PRReviewComments.pr_id -> PullRequests (closed_at of the PR)
        u = np.concatenate([rv.reviewer_id.values, rc.author_id.values]); k = np.concatenate([rv.pr_id.values, rc.pr_id.values])
        t = np.concatenate([later(rv.submitted_at.values, rv.first_seen_at.values), rc.created_at.values]); o = np.argsort(t, kind="stable")
        self.reviewed = dict(u=ci(u[o]), key=k[o], t=t[o], close=pr_close[k[o]])
        # (e) issues assigned to me                                Issues.assignee_id -> Users
        a = iss.dropna(subset=["assignee_id"])
        self.assigned = dict(u=ci(a.assignee_id), t=a.first_seen_at.values, close=a.closed_at.fillna(INF).values)
        self.n = len(daily.cohort_ids)

    def _count(self, d, c, lo=None, unique=False):
        m = (d["t"] < np.datetime64(c)) & (d["close"] >= np.datetime64(c)) & (d["u"] >= 0)
        if lo is not None:
            m &= d["t"] >= np.datetime64(lo)
        u = d["u"][m]
        if unique:                                           # one count per (user, thread)
            key = d["u"][m].astype(np.int64) * 10_000_000 + d["key"][m]
            u = np.unique(key) // 10_000_000
        return np.bincount(u, minlength=self.n).astype(np.float32)

    def features(self, rows, b):
        c = DAY0 + pd.Timedelta(days=b)
        f = {}
        f["ow_my_open_issues"] = self._count(self.my_issue, c)
        f["ow_my_open_issues_recent28"] = self._count(self.my_issue, c, c - pd.Timedelta(days=28))
        f["ow_my_open_prs"] = self._count(self.my_pr, c)
        f["ow_my_open_prs_recent28"] = self._count(self.my_pr, c, c - pd.Timedelta(days=28))
        f["ow_open_threads_28d"] = self._count(self.thread, c, c - pd.Timedelta(days=28), unique=True)
        f["ow_open_threads_7d"] = self._count(self.thread, c, c - pd.Timedelta(days=7), unique=True)
        f["ow_open_reviewed_prs_28d"] = self._count(self.reviewed, c, c - pd.Timedelta(days=28), unique=True)
        f["ow_open_reviewed_prs_7d"] = self._count(self.reviewed, c, c - pd.Timedelta(days=7), unique=True)
        f["ow_assigned_open"] = self._count(self.assigned, c)
        d = self.my_pr                                       # age (days) of my newest open PR
        m = (d["t"] < np.datetime64(c)) & (d["close"] >= np.datetime64(c)) & (d["u"] >= 0)
        age = np.full(self.n, 999, np.float32)
        newest = pd.Series((np.datetime64(c) - d["t"][m]) / np.timedelta64(1, "D")).groupby(d["u"][m]).min()
        age[newest.index.values] = newest.values.astype(np.float32); f["ow_newest_open_pr_age"] = age
        return pd.DataFrame(f).iloc[rows].reset_index(drop=True)
