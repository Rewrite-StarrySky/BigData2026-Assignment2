"""Everything derived from one data release: event streams, daily matrices, feature blocks and the cached (user, cutoff) samples.

A sample is a (user, cutoff) pair: features use only data before the cutoff, the label is "does the user perform the activity in
[cutoff + gap, cutoff + gap + 7 days)".  The users of a cutoff are the cohort users that a list published `LIST_AGE` before the
window start would contain (>= 5 records in the 90 days before the list time), exactly like the official test list.
"""
import time
import numpy as np
import pandas as pd
from .config import *
from .data import load_tables, event_streams, release_time
from .labels import eligible_users, window_labels, window_coverage
from .features import Daily
from .relational import add_relational_streams, relational_features
from .repo_features import RepoFeatures
from .openwork import OpenWork

BLOCKS = ("temporal", "relational", "openwork", "repo")


class Context:
    """All tables / streams / matrices of the data release found in DATA_DIR."""

    def __init__(self, adjust=True, verbose=True):
        t0 = time.time()
        self.adjust = adjust
        self.T = load_tables()
        self.ev = event_streams(self.T)                                  # event times: labels and eligibility
        self.ev_vis = event_streams(self.T, visible=True)               # times from which an event exists in the database: features
        self.end = min(release_time(self.T), DATA_CAP)                  # end of the data that may be used (see config.DATA_CAP)
        self.release_id = f"{self.end:%Y%m%d%H}"
        self.daily = Daily(self.T, self.ev_vis, adjust=adjust)
        add_relational_streams(self.daily, self.T)
        self.openwork = OpenWork(self.daily, self.T)
        self.repo = RepoFeatures(self.daily, self.ev_vis, len(self.T["Repositories"]))
        self.github_id = self.T["Users"].github_user_id.values            # Users row id -> GitHub user id
        if verbose:
            print(f"data release {self.end} (UTC): {self.daily.n} cohort users; context built in {time.time() - t0:.0f} s", flush=True)

    def blocks(self, rows, b, which=BLOCKS):
        """Feature blocks of the cohort rows `rows` at daily bin b (= cutoff)."""
        out = {}
        if "temporal" in which:
            out["temporal"] = self.daily.temporal_features(rows, b)
        if "relational" in which:
            out["relational"] = relational_features(self.daily, rows, b)
        if "openwork" in which:
            out["openwork"] = self.openwork.features(rows, b)
        if "repo" in which:
            out["repo"] = self.repo.features(rows, b)
        return {k: v.astype(np.float32) for k, v in out.items()}


def training_cutoffs(end, gap_hours):
    """Daily cutoffs (12:00 UTC) whose label window [c + gap, c + gap + 7 d) lies completely inside the data."""
    last = end - pd.Timedelta(days=WINDOW_DAYS) - pd.Timedelta(hours=gap_hours)
    return list(pd.date_range(FIRST_CUTOFF, last, freq="D"))


def _tag(ctx, gap_hours):
    return f"{ctx.release_id}_g{gap_hours}"


def build_samples(ctx, gap_hours=FINAL_GAP_HOURS, which=BLOCKS, verbose=True):
    """Build (or load) the training samples of all cutoffs; cached as parquet files keyed by release time, gap and feature variant."""
    tag, ftag = _tag(ctx, gap_hours), f"{_tag(ctx, gap_hours)}_{'adj' if ctx.adjust else 'raw'}"
    meta_p, y_p = CACHE_DIR / f"{tag}_meta.parquet", CACHE_DIR / f"{tag}_y.parquet"
    need = [w for w in which if not (CACHE_DIR / f"{ftag}_{w}.parquet").exists()]
    if meta_p.exists() and y_p.exists() and not need:
        return load_samples(ctx, gap_hours, which)
    t0 = time.time()
    meta, ys, blocks = [], [], {w: [] for w in which}
    for c in training_cutoffs(ctx.end, gap_hours):
        start = c + pd.Timedelta(hours=gap_hours)
        elig = eligible_users(ctx.ev, ctx.daily.cohort_ids, start - LIST_AGE)
        elig = elig[np.argsort(ctx.github_id[elig], kind="stable")]       # order by the anonymised GitHub id: row ids depend on the period that was processed, this order does not
        rows, b = ctx.daily.cidx[elig], ctx.daily.bin_of(c)
        meta.append(pd.DataFrame({"user_row": elig, "cutoff": c, "push_cover": window_coverage(ctx.daily.alive, start)}))
        ys.append(window_labels(ctx.ev, elig, start).reset_index(drop=True))
        for k, v in ctx.blocks(rows, b, which).items():
            blocks[k].append(v)
    M, Y = pd.concat(meta, ignore_index=True), pd.concat(ys, ignore_index=True)
    M.to_parquet(meta_p); Y.to_parquet(y_p)
    for k, v in blocks.items():
        pd.concat(v, ignore_index=True).to_parquet(CACHE_DIR / f"{ftag}_{k}.parquet")
    if verbose:
        print(f"built {len(M):,} samples of {M.cutoff.nunique()} cutoffs in {time.time() - t0:.0f} s -> {CACHE_DIR}", flush=True)
    return load_samples(ctx, gap_hours, which)


def load_samples(ctx, gap_hours, which=BLOCKS):
    tag, ftag = _tag(ctx, gap_hours), f"{_tag(ctx, gap_hours)}_{'adj' if ctx.adjust else 'raw'}"
    M, Y = pd.read_parquet(CACHE_DIR / f"{tag}_meta.parquet"), pd.read_parquet(CACHE_DIR / f"{tag}_y.parquet")
    return {w: pd.read_parquet(CACHE_DIR / f"{ftag}_{w}.parquet") for w in which}, Y, M


def test_features(ctx, github_ids, as_of, which=BLOCKS):
    """Features of the given GitHub user ids at cutoff `as_of` (rows of the result follow the order of github_ids)."""
    row_of = pd.Series(np.arange(len(ctx.github_id)), index=ctx.github_id)
    user_rows = row_of.loc[np.asarray(github_ids)].values
    return ctx.blocks(ctx.daily.cidx[user_rows], ctx.daily.bin_of(as_of), which)
