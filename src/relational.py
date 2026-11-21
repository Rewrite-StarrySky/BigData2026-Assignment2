"""Relational features (Q2): events that reach a user THROUGH foreign keys, i.e. what other rows / users do to the user's rows.

Each relational event is a real row with its own timestamp (a review, a comment, a merge ...) attributed to the user it happens
TO, and goes into the same users x days matrices as the target activities; counting rows before a cutoff is leak-free.

Time of an event = the time from which it EXISTS in the database: the later of its own time and the row time of the row it is attributed
through (the PR / issue whose author receives it).  A PR or issue that is first seen only later (e.g. an issue opened inside a gap of GH
Archive that shows up with a later comment) is therefore not used before it appears (see data.event_streams(visible=True)).
"""
import numpy as np
import pandas as pd
from .data import later

REL_WINDOWS = [1, 3, 7, 14, 28]
REL_NAMES = ["recv_review", "recv_review_comment", "recv_issue_comment", "thread_reply_issue", "thread_reply_pr", "my_pr_merged", "my_pr_closed",
             "my_issue_closed", "assigned_issue", "i_merged_pr", "push_side_branch"]          # the streams built by relational_events(), in order


def _pr_author(T):
    """(author, visible_from) of every PR: PullRequests.author_id, filled from the PR's conversation issue (Issues.pull_request_id) when the PR
    row has none; visible_from = first time the PR row (and, if filled, the conversation issue) is in the database."""
    pr, iss = T["PullRequests"], T["Issues"]
    m = iss[iss.is_pull_request & iss.pull_request_id.notna()].drop_duplicates("pull_request_id").set_index("pull_request_id")
    fill_a, fill_t = pr.pr_id.map(m.author_id).values, pr.pr_id.map(m.first_seen_at).values
    own = pr.author_id.notna().values
    author = np.where(own, pr.author_id.values, fill_a)
    vis = pr.first_seen_at.values.copy()
    need = ~own & pd.notna(fill_t)
    vis[need] = later(pr.first_seen_at.values[need], fill_t[need])
    return author.astype(float), vis


def relational_events(T):
    """List of (name, Users row id, time) events that happen TO a user because of other rows (or lifecycle changes of the user's items)."""
    pr, iss, ic, rv, rc, ps = (T[n] for n in ["PullRequests", "Issues", "IssueComments", "PRReviews", "PRReviewComments", "Pushes"])
    pr_author, pr_vis = _pr_author(T)
    rv_t = later(rv.submitted_at.values, rv.first_seen_at.values)                      # a review exists from its first_seen_at on
    out = []
    # 1) reviews / review comments written by others on a PR I authored   (PRReviews.pr_id -> PullRequests.author_id)
    r = rv.assign(owner=pr_author[rv.pr_id.values], tv=later(rv_t, pr_vis[rv.pr_id.values])).dropna(subset=["owner"])
    r = r[r.owner != r.reviewer_id]; out.append(("recv_review", r.owner.values, r.tv.values))
    c = rc.assign(owner=pr_author[rc.pr_id.values], tv=later(rc.created_at.values, pr_vis[rc.pr_id.values])).dropna(subset=["owner"])
    c = c[c.owner != c.author_id]; out.append(("recv_review_comment", c.owner.values, c.tv.values))
    # 2) comments written by others on an issue / PR conversation I authored   (IssueComments.issue_id -> Issues.author_id)
    i = ic.assign(owner=iss.author_id.values[ic.issue_id.values], tv=later(ic.created_at.values, iss.first_seen_at.values[ic.issue_id.values]))
    i = i[i.owner != i.author_id]
    out.append(("recv_issue_comment", i.owner.values, i.tv.values))
    # 3) replies in threads I took part in: the issue author and every earlier commenter get +1 per later comment of someone else
    first = ic.groupby(["issue_id", "author_id"]).created_at.min().reset_index().rename(columns={"author_id": "u", "created_at": "joined"})
    auth = pd.DataFrame({"issue_id": iss.issue_id.values, "u": iss.author_id.values, "joined": iss.created_at.values})
    part = pd.concat([first, auth]).groupby(["issue_id", "u"]).joined.min().reset_index()
    m = ic[["issue_id", "author_id", "created_at"]].merge(part, on="issue_id")
    m = m[(m.u != m.author_id) & (m.joined < m.created_at)]
    out.append(("thread_reply_issue", m.u.values, later(m.created_at.values, iss.first_seen_at.values[m.issue_id.values])))
    ev = pd.concat([pd.DataFrame({"pr_id": rv.pr_id.values, "a": rv.reviewer_id.values, "t": rv_t}),
                    rc.rename(columns={"author_id": "a", "created_at": "t"})[["pr_id", "a", "t"]]])
    pa = pd.DataFrame({"pr_id": np.arange(len(pr)), "u": pr_author}).dropna()
    first = ev.groupby(["pr_id", "a"]).t.min().reset_index().rename(columns={"a": "u", "t": "joined"})
    part = pd.concat([first, pa.assign(joined=pd.Timestamp("1970-01-01"))]).groupby(["pr_id", "u"]).joined.min().reset_index()
    m = ev.merge(part, on="pr_id"); m = m[(m.u != m.a) & (m.joined < m.t)]
    out.append(("thread_reply_pr", m.u.values, later(m.t.values, pr_vis[m.pr_id.values])))
    # 4) lifecycle of my own items: PR merged / closed, issue closed.  merged_at / closed_at are known only later, but every
    #    event carries its own time and is counted only when that time is before the cutoff.
    p = pr.assign(owner=pr_author, vis=pr_vis).dropna(subset=["owner"])
    mg = p.dropna(subset=["merged_at"]); out.append(("my_pr_merged", mg.owner.values, later(mg.merged_at.values, mg.vis.values)))
    cl = p.dropna(subset=["closed_at"]); out.append(("my_pr_closed", cl.owner.values, later(cl.closed_at.values, cl.vis.values)))
    ci = iss[~iss.is_pull_request].dropna(subset=["closed_at"]); out.append(("my_issue_closed", ci.author_id.values, later(ci.closed_at.values, ci.first_seen_at.values)))
    # 5) issues assigned to me (assignee known from the first event of the issue)
    asg = iss.dropna(subset=["assignee_id"]); out.append(("assigned_issue", asg.assignee_id.values, asg.first_seen_at.values))
    # 6) maintainer activity: PRs I merged
    mm = pr.dropna(subset=["merged_by_id", "merged_at"]); out.append(("i_merged_pr", mm.merged_by_id.values, later(mm.merged_at.values, mm.first_seen_at.values)))
    # 7) pushes to a side branch (anything but main / master): side-branch work precedes PRs
    other = ps[~ps.branch.isin(["main", "master"])]; out.append(("push_side_branch", other.user_id.values, other.created_at.values))
    return out


def add_relational_streams(daily, T):
    """Put the relational events into the same per-user daily matrices as the target activities."""
    names = []
    for name, u, t in relational_events(T):
        daily.add_stream(name, u, t, daily.push_cover if name == "push_side_branch" else None)     # derived from pushes: same gaps
        names.append(name)
    assert names == REL_NAMES, names
    daily.rel_names = names
    return names


def relational_features(daily, rows, b):
    f = {}
    for k in daily.rel_names:
        for w in REL_WINDOWS:
            f[f"{k}_n{w}d"] = daily.window(k, rows, b, w)
        f[f"{k}_days_since_last"] = daily.days_since_last(k, rows, b)
    # PRs I opened inside the data window and that are not closed yet
    f["open_prs_now"] = np.clip((daily.S["open_pr"][rows, b] - daily.S["my_pr_closed"][rows, b]).astype(np.float32), 0, None)
    return pd.DataFrame(f)
