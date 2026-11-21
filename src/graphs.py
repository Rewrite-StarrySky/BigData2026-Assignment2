"""Q1: schema graph (8 tables, foreign keys as edges) as a Graphviz DOT file."""
import numpy as np
import pandas as pd
from .data import load_tables

# table -> (primary key, time column, fill colour, edge colour used for foreign keys that point INTO the table)
TABLES = {
    "Users": ("user_id", "first_seen_at", "#cfe2fb", "#2a66c4"), "Repositories": ("repo_id", "first_seen_at", "#c9f0e3", "#1f9d73"),
    "PullRequests": ("pr_id", "first_seen_at", "#fbd9c9", "#d9622b"), "Issues": ("issue_id", "first_seen_at", "#fbeab8", "#b8860b"),
    "IssueComments": ("comment_id", "created_at", "#f7d3e1", "#c2457f"), "PRReviews": ("review_id", "first_seen_at", "#cfe9cf", "#3a9a3a"),
    "PRReviewComments": ("comment_id", "created_at", "#dcd6f6", "#6f5bd0"), "Pushes": ("push_id", "created_at", "#f6cfcf", "#c0392b"),
}
# every foreign key of the data release (data/db/SCHEMA.md): (child table, column, parent table)
FKS = [
    ("PullRequests", "repo_id", "Repositories"), ("PullRequests", "author_id", "Users"), ("PullRequests", "merged_by_id", "Users"),
    ("Issues", "repo_id", "Repositories"), ("Issues", "author_id", "Users"), ("Issues", "assignee_id", "Users"), ("Issues", "pull_request_id", "PullRequests"),
    ("IssueComments", "issue_id", "Issues"), ("IssueComments", "repo_id", "Repositories"), ("IssueComments", "author_id", "Users"),
    ("PRReviews", "pr_id", "PullRequests"), ("PRReviews", "repo_id", "Repositories"), ("PRReviews", "reviewer_id", "Users"),
    ("PRReviewComments", "pr_id", "PullRequests"), ("PRReviewComments", "repo_id", "Repositories"), ("PRReviewComments", "author_id", "Users"),
    ("PRReviewComments", "pull_request_review_id", "PRReviews"), ("PRReviewComments", "in_reply_to_id", "PRReviewComments"),
    ("Pushes", "user_id", "Users"), ("Pushes", "repo_id", "Repositories"),
]
USER_COLS = {"PullRequests": ["author_id", "merged_by_id"], "Issues": ["author_id", "assignee_id"], "IssueComments": ["author_id"],
             "PRReviews": ["reviewer_id"], "PRReviewComments": ["author_id"], "Pushes": ["user_id"]}


def schema_dot():
    """Schema graph: one node per table (PK, FKs and time column listed), one edge per foreign key (FK column -> referenced PK)."""
    fk_cols = {t: [c for (tt, c, _) in FKS if tt == t] for t in TABLES}
    lines = ['digraph schema {', ' graph [rankdir=LR, nodesep=0.35, ranksep=1.3, fontname="Helvetica", bgcolor="white", pad=0.2];',
             ' node [shape=plain, fontname="Helvetica", fontsize=12]; edge [fontname="Helvetica", fontsize=10, arrowsize=0.8];']
    for t, (pk, tc, col, _) in TABLES.items():
        rows = [f'<TR><TD BGCOLOR="{col}" ALIGN="CENTER"><B>{t}</B></TD></TR>', f'<TR><TD ALIGN="LEFT" PORT="pk"><B>PK</B> {pk}</TD></TR>']
        rows += [f'<TR><TD ALIGN="LEFT" PORT="{c}">FK {c}</TD></TR>' for c in fk_cols[t]]
        rows += [f'<TR><TD ALIGN="LEFT"><I>time: {tc}</I></TD></TR>']
        lines.append(f' {t} [label=<<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" CELLPADDING="4" COLOR="#52514e">{"".join(rows)}</TABLE>>];')
    for ch, c, pa in FKS:
        ec = TABLES[pa][3]
        if ch != pa:
            lines.append(f' {ch}:{c}:e -> {pa}:pk:w [label=" {c}", color="{ec}", fontcolor="{ec}"];')
        else:
            lines.append(f' {ch}:{c}:e -> {pa}:pk:e [label=" {c}", color="{ec}", fontcolor="{ec}", constraint=false];')
    lines.append('}')
    return "\n".join(lines)
