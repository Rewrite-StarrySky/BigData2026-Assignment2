"""Multi-hop features (Q2): user -> repositories the user works in -> what the OTHER cohort users do in those repositories.

Two hops over foreign keys:  Users <- (author / pusher / reviewer) - events - repo_id -> Repositories <- repo_id - events - other users.
"""
import numpy as np
import pandas as pd
import scipy.sparse as sp
from .config import LABELS, DAY0
from .features import MIN_COVER

AFFINITY_DAYS = 56          # a user "belongs" to the repositories he/she was active in during the last 56 days
WINDOWS = [3, 28]           # how recent the repository activity of the OTHER users is
PUSH = LABELS.index("push")


class RepoFeatures:
    def __init__(self, daily, ev, n_repos):
        self.daily = daily
        self.n_cohort, self.n_repos = len(daily.cohort_ids), n_repos
        r, rp, b, k = [], [], [], []
        for ki, name in enumerate(LABELS):
            e = ev[name]
            row = daily.cidx[e.u.values]; ok = row >= 0
            r.append(row[ok]); rp.append(e.repo_id.values[ok])
            b.append(((e.t - DAY0) // pd.Timedelta(days=1)).values[ok]); k.append(np.full(ok.sum(), ki))
        r, rp, b, k = map(np.concatenate, (r, rp, b, k))
        o = np.argsort(b, kind="stable")
        self.r, self.rp, self.b, self.k = r[o], rp[o], b[o], k[o]

    def _mat(self, sl, kinds=None):
        m = np.ones(len(self.r[sl]), bool) if kinds is None else np.isin(self.k[sl], kinds)
        return sp.csr_matrix((np.ones(m.sum(), np.float32), (self.r[sl][m], self.rp[sl][m])), shape=(self.n_cohort, self.n_repos))

    def features(self, rows, b):
        d = self.daily
        lo = np.searchsorted(self.b, b - AFFINITY_DAYS, "left"); hi = np.searchsorted(self.b, b, "left")
        A = self._mat(slice(lo, hi))                                                # user x repo event counts (56 days)
        Aw = A.copy(); Aw.data = np.log1p(Aw.data)
        rs = np.asarray(Aw.sum(1)).ravel(); inv = np.where(rs > 0, 1 / np.maximum(rs, 1e-9), 0)
        An = sp.diags(inv) @ Aw                                                    # rows sum to 1: affinity weights
        f = {"repo_n_repos56": np.diff(A.indptr).astype(np.float32),
             "repo_top_share": np.asarray(A.max(1).todense()).ravel() / np.maximum(np.asarray(A.sum(1)).ravel(), 1)}
        collab = [i for i, n in enumerate(LABELS) if n != "push"]
        for w in WINDOWS:
            sl = slice(np.searchsorted(self.b, b - w, "left"), hi)
            e, n = d.SE["push"][b] - d.SE["push"][b - w], w                        # collected share of the push stream in the window
            push_scale = 1.0 if e >= n - 1e-6 else (np.nan if e < MIN_COVER * n else n / e)
            Mc = self._mat(sl, collab)                                              # collaboration activity of everybody
            f[f"repo_users_in_my_repos_{w}d"] = An @ np.asarray((Mc > 0).sum(0)).ravel().astype(np.float32)        # distinct users active
            f[f"repo_events_in_my_repos_{w}d"] = An @ np.asarray(Mc.sum(0)).ravel().astype(np.float32)
            for ki, name in enumerate(LABELS):
                M = self._mat(sl, [ki])
                rv = np.asarray(M.sum(0)).ravel().astype(np.float32)               # repo-level count of this activity by everybody
                own = np.asarray(An.multiply(M).sum(1)).ravel()                    # the user's own contribution, removed again
                v = (An @ rv) - own
                f[f"repo_{name}_in_my_repos_{w}d"] = v * push_scale if ki == PUSH else v
        return pd.DataFrame(f).iloc[rows].reset_index(drop=True)
