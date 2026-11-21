"""The relational database as a temporal heterogeneous graph (Q2 / Q3 / Q4).

Node  = one row of one of the 8 tables (node type = table).
Edge  = one foreign key value (child row -> parent row); every foreign key is used in both directions, so 20 foreign keys give 40 relations.
Time  = a snapshot at cutoff c contains exactly the rows whose row time is before c.  The course script sorts every table by its row time and uses
        the row number as primary key, so the visible rows of a table are simply its first n(c) rows, and the visible edges are the edges whose two
        end rows are both visible (edges are pre-sorted by that time, so they are a prefix too).  Nothing is copied or re-indexed per cutoff.
Features of a node = the columns of its row that can be used (flags, counts, text lengths; ids and masked text carry no content) plus
        * a relative-time encoding of the age of the row at the cutoff (TIME ENCODING, see time_encoding()),
        * the columns that are known only later (merged_at, closed_at, state_reason, dismissed_at), used only if their time is before the cutoff.
"""
import numpy as np
import pandas as pd
import torch
from .config import DAY0
from .data import load_tables, ROW_TIME, TABLES
from .graphs import FKS

TAUS = (1.0, 2.0, 3.0, 7.0, 14.0, 28.0, 56.0, 90.0)        # time scales (days) of the time encoding: the same scales as the count windows of the feature baseline
N_TIME = 2 * len(TAUS) + 1


def days(t):
    """Timestamps -> days since DAY0 as float64 (NaT -> +inf, i.e. 'never / not yet')."""
    t = pd.to_datetime(pd.Series(np.asarray(t)))
    d = ((t - DAY0) / pd.Timedelta(days=1)).astype("float64").values
    return np.where(np.isnan(d), np.inf, d)


def time_encoding(age, on=True):
    """Relative-time encoding of an age in days (>= 0): for every scale tau a step 1[age <= tau] and a decay exp(-age / tau), plus log(1 + age).

    Summed over the event rows of a user by the message passing, the steps are exactly the event counts of the last tau days and the decays are
    exponentially decayed counts; their maximum over the rows encodes the time since the most recent event.  on=False returns zeros (ablation
    'GNN without time encoding': the model then only knows THAT a row exists before the cutoff, not WHEN)."""
    age = age.clamp(min=0).to(torch.float32)
    if not on:
        return torch.zeros(age.shape[0], N_TIME, device=age.device)
    tau = torch.tensor(TAUS, device=age.device, dtype=torch.float32)
    a = age[:, None]
    return torch.cat([(a <= tau).float(), torch.exp(-a / tau), torch.log1p(a) / 5.3], 1)          # log(1 + 200 days) = 5.3


def _len(s, scale):
    """log(1 + length of a masked text) / scale; missing text -> 0 (the masking keeps the length of every text)."""
    return (np.log1p(s.fillna("").str.len().values.astype("float32")) / scale).astype("float32")


def _b(s):
    return s.fillna(False).astype(bool).values.astype("float32")


class GraphData:
    """All tables as tensors on `device`; snapshot(c) gives the node features and the edge lists of the rows before the cutoff c."""

    def __init__(self, T=None, device="cpu"):
        T = T or load_tables()
        self.device = device
        U, R, PR, IS, IC, RV, RC, PS = (T[n] for n in TABLES)
        self.row_time = {n: days(T[n][ROW_TIME[n]].values) for n in TABLES}
        assert all((np.diff(v) >= 0).all() for v in self.row_time.values()), "tables must be sorted by row time (course script)"
        st = {}                                                                                   # static (cutoff-independent) features per table
        st["Users"] = np.stack([_b(U.is_bot), _b(U.is_cohort)], 1)
        st["Repositories"] = np.stack([R.org_id.notna().values.astype("float32")], 1)
        st["PullRequests"] = np.stack([_b(PR.draft), PR.draft.notna().values.astype("float32"), _b(PR.is_cross_repo), PR.author_id.notna().values.astype("float32"),
                                       PR.created_at.notna().values.astype("float32"), _len(PR.title, 5.0), _len(PR.head_ref, 4.0)], 1)
        st["Issues"] = np.stack([_b(IS.is_pull_request), np.log1p(IS.num_labels.fillna(0).values.astype("float32")) / 3.0, _b(IS.locked), IS.assignee_id.notna().values.astype("float32"),
                                 IS.milestone_title.notna().values.astype("float32"), IS.issue_type.notna().values.astype("float32"), IS.created_via_app.notna().values.astype("float32"),
                                 _len(IS.title, 5.0), _len(IS.body, 9.0)], 1)
        st["IssueComments"] = np.stack([_b(IC.is_pull_request), _len(IC.body, 9.0), IC.via_app.notna().values.astype("float32")], 1)
        states = ["approved", "commented", "changes_requested"]
        st["PRReviews"] = np.stack([(RV.state == s).values.astype("float32") for s in states] + [(~RV.state.isin(states)).values.astype("float32"), _len(RV.body, 9.0)], 1)
        st["PRReviewComments"] = np.stack([_len(RC.body, 9.0), (RC.subject_type == "line").values.astype("float32"), RC.pull_request_review_id.notna().values.astype("float32"),
                                           RC.in_reply_to_id.notna().values.astype("float32"), _len(RC.diff_hunk, 9.0), _len(RC.path, 5.0)], 1)
        st["Pushes"] = np.stack([_len(PS.branch, 4.0), (PS.ref_type == "tag").values.astype("float32")], 1)
        self.static = {n: torch.tensor(v, dtype=torch.float32, device=device) for n, v in st.items()}
        self.t_row = {n: torch.tensor(v, dtype=torch.float64, device=device) for n, v in self.row_time.items()}
        # columns that become known only later: their time, used only if it is before the cutoff
        tt = lambda a: torch.tensor(days(a), dtype=torch.float64, device=device)
        self.t_created = {"PullRequests": tt(PR.created_at.values), "Issues": tt(IS.created_at.values)}   # when the item was opened (can be long before the row time)
        self.t_closed = {"PullRequests": tt(PR.closed_at.values), "Issues": tt(IS.closed_at.values)}
        self.t_merged = tt(PR.merged_at.values)
        self.t_dismissed = tt(RV.dismissed_at.values)
        reasons = ["completed", "not_planned", "reopened", "duplicate"]
        self.state_reason = torch.tensor(np.stack([(IS.state_reason == r).fillna(False).values.astype("float32") for r in reasons], 1), device=device)
        # edges: every foreign key, sorted by the time from which both end rows exist (merged_by_id: additionally the merge time)
        self.rel, self.edges = [], {}
        for child, col, parent in FKS:
            v = T[child][col]; ok = v.notna().values
            c_idx = np.flatnonzero(ok); p_idx = v.values[ok].astype("int64")
            et = np.maximum(self.row_time[child][c_idx], self.row_time[parent][p_idx])
            if col == "merged_by_id":
                et = np.maximum(et, days(PR.merged_at.values)[c_idx])
            o = np.argsort(et, kind="stable")
            name = f"{child}.{col}"
            self.edges[name] = (torch.tensor(c_idx[o], device=device), torch.tensor(p_idx[o], device=device), et[o])
            self.rel += [(name, child, parent), ("rev_" + name, parent, child)]
        self.in_dim = {"Users": 2 + N_TIME, "Repositories": 1 + N_TIME, "PullRequests": 7 + 2 + 3 * N_TIME, "Issues": 9 + 1 + 4 + 3 * N_TIME, "IssueComments": 3 + N_TIME,
                       "PRReviews": 5 + 1 + N_TIME, "PRReviewComments": 6 + N_TIME, "Pushes": 2 + N_TIME}

    def n_visible(self, c):
        return {n: int(np.searchsorted(self.row_time[n], c, "left")) for n in TABLES}

    def snapshot(self, cutoff, time_enc=True):
        """(x, edges, n): node features, edge lists (relation -> (source rows, destination rows)) and row counts of the database as it is before `cutoff`."""
        c = float((pd.Timestamp(cutoff) - DAY0) / pd.Timedelta(days=1))
        n = self.n_visible(c); x = {}
        te = lambda t: time_encoding(c - t, time_enc)
        for name in TABLES:
            k = n[name]; parts = [self.static[name][:k], te(self.t_row[name][:k])]
            if name in ("PullRequests", "Issues"):
                closed = self.t_closed[name][:k] < c                                             # closed_at is used only if it is before the cutoff
                created = self.t_created[name][:k]; known = torch.isfinite(created)
                parts += [closed.float()[:, None], te(torch.where(closed, self.t_closed[name][:k], torch.full_like(created, c))) * closed.float()[:, None],
                          te(torch.where(known, created, torch.full_like(created, c))) * known.float()[:, None]]
            if name == "PullRequests":
                parts.append((self.t_merged[:k] < c).float()[:, None])                           # merged_at likewise
            if name == "Issues":
                parts.append(self.state_reason[:k] * closed.float()[:, None])                    # state_reason belongs to the closing
            if name == "PRReviews":
                parts.append((self.t_dismissed[:k] < c).float()[:, None])
            x[name] = torch.cat(parts, 1)
        edges = {}
        for name, (ci, pi, et) in self.edges.items():
            m = int(np.searchsorted(et, c, "left"))
            edges[name] = (ci[:m], pi[:m]); edges["rev_" + name] = (pi[:m], ci[:m])
        return x, edges, n
