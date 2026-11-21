"""Leak check: are the features of a cutoff really computed from data before the cutoff only?

    # 1. features at the cutoff from the full tables (ASG2_DB_DIR = the usual tables)
    python -m scripts.leak_check --cutoff "2026-10-03 12:00" --out cache/leak_full.parquet
    # 2. rebuild the tables with the course script from the raw hours BEFORE the cutoff only, and compute the same features from them
    python course_repo/scripts/preprocess.py --raw raw_until_cutoff --out preprocessed_until_cutoff
    ASG2_DB_DIR=preprocessed_until_cutoff/db python -m scripts.leak_check --cutoff "2026-10-03 12:00" --out cache/leak_trunc.parquet
    # 3. compare
    python -m scripts.leak_check --compare cache/leak_full.parquet cache/leak_trunc.parquet

If nothing from the future leaks into the features, both files are identical (for the users that exist in both).  Result -> results/leak_check.json
"""
import argparse, json
import numpy as np
import pandas as pd
from src.config import *
from src.pipeline import Context, build_samples, test_features
from src.final_model import feature_sets


def compute(cutoff, out):
    cutoff = pd.Timestamp(cutoff)
    ctx, ctx_raw = Context(adjust=True), Context(adjust=False, verbose=False)
    ids = pd.read_csv(TEST_INPUT).UserID.values
    ids = ids[np.isin(ids, ctx.github_id)]                                             # test users that exist in these tables
    which = ("temporal",) + tuple(FINAL_BLOCKS)
    X = feature_sets(test_features(ctx, ids, cutoff, which), test_features(ctx_raw, ids, cutoff, ("temporal",)), np.repeat(cutoff.to_datetime64(), len(ids)), ctx.daily)
    pd.concat({b: x.set_index(pd.Index(ids, name="UserID")) for b, x in X.items()}, axis=1).to_parquet(out)
    print(f"wrote {out}: {len(ids)} users, {sum(x.shape[1] for x in X.values())} feature columns, data ends {ctx.end}")


def compare(a, b, cutoff=None):
    A, B = pd.read_parquet(a), pd.read_parquet(b)
    common = A.index.intersection(B.index); A, B = A.loc[common].astype(float), B.loc[common].astype(float)
    d = (A - B).abs().fillna(0).values; nan_mismatch = (A.isna().values != B.isna().values)
    ctx_cols = np.array(["glob_" in c[1] for c in A.columns])                              # cohort-level context: one value per cutoff, shifts when a single event differs
    user_cols = ~ctx_cols
    bad = ((d > 1e-5) | nan_mismatch)[:, user_cols]
    res = dict(cutoff=str(cutoff), users_compared=int(len(common)), feature_columns=int(A.shape[1]),
               user_level_columns=int(user_cols.sum()), users_with_any_difference=int(bad.any(1).sum()), user_level_columns_with_a_difference=int(bad.any(0).sum()),
               identical_user_level_values_share=float(1 - bad.mean()), max_abs_difference_of_cohort_context_columns=float(d[:, ctx_cols].max()),
               columns_that_differ=sorted({c[1] for c, k in zip(A.columns[user_cols], bad.any(0)) if k})[:30])
    print(json.dumps(res, indent=1))
    json.dump(res, open(RESULTS_DIR / "leak_check.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", default="2026-10-03 12:00"); ap.add_argument("--out"); ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    compare(*a.compare, cutoff=a.cutoff) if a.compare else compute(a.cutoff, a.out)
