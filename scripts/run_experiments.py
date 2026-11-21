"""Ablation and hyper-parameter experiments (rolling-origin evaluation); one JSON file per experiment is written to results/runs/.

    python -m scripts.run_experiments --suite ablation            # Q3 temporal ablation + Q2 relational ablation
    python -m scripts.run_experiments --suite hparams             # model / training-scheme variants of the final feature set
    python -m scripts.run_experiments --suite ablation --shard 0/2   # run half of the configurations (two processes in parallel)

Protocol: 7 origins T = Wednesdays 12:00 UTC (9/9 ... 10/21, the weekday of the final cutoff 11/11).  For each origin the models
are trained only on cutoffs <= T - 8 days (the 7-day label windows of all training samples end before T) and predict the
users of cutoff T for the week that starts 4 h later.  Origins whose push label window was collected < 80 % are marked
"push_contaminated" in the summary (their push labels are too low because of collection gaps; see results/data_gaps.json).
"""
import argparse, json, time
import numpy as np
import pandas as pd
from src.config import *
from src.pipeline import Context, load_samples
from src.features import global_context
from src.final_model import feature_sets
from src.model import rolling_eval

FOLDS = [pd.Timestamp(s) + pd.Timedelta(hours=12) for s in ("2026-09-19", "2026-09-26", "2026-10-03", "2026-10-10", "2026-10-17", "2026-10-24", "2026-10-31")]   # Saturdays, the weekday of the final cutoff 11/21
OUT_DIR = RESULTS_DIR / "runs"                      # one JSON file per experiment (safe for parallel shards)
OUT_DIR.mkdir(exist_ok=True)
STREAMS = LABELS + ["total"]


def temporal_subsets(XT):
    """Cumulative feature sets of the temporal ablation (Q3)."""
    counts = [c for c in XT.columns if "_n" in c and c.endswith("d") and "week" not in c]
    recency = [c for c in XT.columns if any(p in c for p in ("days_since_last", "ewma", "active_days"))]
    return {"T0 counts of the last 7 days only": XT[[f"{k}_n7d" for k in STREAMS]],
            "T1 + multi-window counts (1-90 days)": XT[counts],
            "T2 + recency (days since last, EWMA, active days)": XT[counts + recency],
            "T3 + weekly lags and trend (= temporal block)": XT}


def build_runs(ctx_adj, ctx_raw):
    Ba, Y, M = load_samples(ctx_adj, FINAL_GAP_HOURS)
    Br, _, _ = load_samples(ctx_raw, FINAL_GAP_HOURS, ("temporal",))
    all_ok = pd.DataFrame({k: np.ones(len(M), bool) for k in LABELS})
    push_ok = all_ok.copy(); push_ok["push"] = (M.push_cover >= PUSH_COVER_MIN).values
    glob_r, glob_a = global_context(Br["temporal"], M.cutoff.values), global_context(Ba["temporal"], M.cutoff.values)
    Xa = pd.concat([Ba["temporal"], glob_a], axis=1)                                   # final temporal feature set (adjusted + global)
    runs = {}                                                                          # name -> (callable returning X, valid mask, kwargs)
    for name, X in temporal_subsets(Br["temporal"]).items():
        runs[name] = (lambda X=X: X, all_ok, {})
    runs["T4 + global activity-volume context"] = (lambda: pd.concat([Br["temporal"], glob_r], axis=1), all_ok, {})
    runs["T5a + coverage-adjusted features (push gaps)"] = (lambda: Xa, all_ok, {})
    runs["T5b + drop push labels of partly collected weeks"] = (lambda: Xa, push_ok, {})
    glob_c = global_context(Ba["temporal"], M.cutoff.values, ctx_adj.daily)           # + push-collection coverage of the last 3 / 7 / 28 days
    Xc = pd.concat([Ba["temporal"], glob_c], axis=1)
    runs["T5c + push-collection coverage as context (keep all labels)"] = (lambda: Xc, all_ok, {})
    blocks = {"R1 FK lifecycle": ["relational"], "R2 work in progress": ["openwork"], "R3 repo multi-hop": ["repo"],
              "R4 R1+R2": ["relational", "openwork"], "R5 R1+R2+R3 (all blocks)": ["relational", "openwork", "repo"]}
    for name, bl in blocks.items():
        runs[f"{name}"] = (lambda bl=bl: pd.concat([Xc] + [Ba[b] for b in bl], axis=1), all_ok, {})
    # --- the plain temporal base (T4: no coverage adjustment, no push-coverage context) with all relational blocks, for all six activities
    FS = feature_sets(Ba, Br, M.cutoff.values, ctx_adj.daily)                           # {"raw": T4 base + blocks, "cover": T5c base + blocks}
    runs["R6 T4 base (plain counts + activity-volume context) + all blocks"] = (lambda: FS["raw"], all_ok, {})
    # --- model / training-scheme variants, on the final feature set: base per activity as in config.FINAL_BASE, blocks as in config.FINAL_BLOCKS
    Xall = lambda: FS
    hb = dict(base_of=FINAL_BASE)
    H = {"H0 reference (depth 6, 300 trees, lr 0.05, stride 2)": {},
         "H1 depth 4, 400 trees": dict(params=dict(max_depth=4, n_estimators=400)),
         "H2 depth 8, 250 trees": dict(params=dict(max_depth=8, n_estimators=250)),
         "H3 lr 0.03, 600 trees": dict(params=dict(learning_rate=0.03, n_estimators=600)),
         "H4 stronger regularisation": dict(params=dict(min_child_weight=100, colsample_bytree=0.5, reg_lambda=20)),
         "H5 every daily cutoff (stride 1)": dict(stride=1),
         "H6 recency weights, half-life 30 d": dict(half_life=30),
         "H7 train on the last 120 days only": dict(recent_days=120),
         "H8 average of 3 seeds": dict(seeds=3)}
    for name, kw in H.items():
        runs[name] = (Xall, all_ok, {**kw, **hb})
    # --- TA announcement: an activity with missing data in the test week is NOT scored.  Push therefore only counts when the test
    #     week is fully collected, so the push model is trained on fully collected weeks only (push labels of the other windows dropped).
    for th in (0.8, 0.9):
        ok = all_ok.copy(); ok["push"] = (M.push_cover >= th).values
        runs[f"R4m{int(th * 100)} final features, push labels of windows collected < {th:.0%} dropped"] = (Xall, ok, hb)
    ok = all_ok.copy(); ok["push"] = (M.push_cover >= FINAL_PUSH_MIN_COVER).values
    runs[f"H8m final model (final features, push labels of windows collected < {FINAL_PUSH_MIN_COVER:.0%} dropped, {FINAL_SEEDS} seeds)"] = (Xall, ok, dict(seeds=FINAL_SEEDS, **hb))
    return runs, Y, M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["ablation", "hparams", "all"], default="all")
    ap.add_argument("--shard", default="0/1", help="i/n: run every n-th configuration starting at i")
    ap.add_argument("--only", default=None, help="comma separated run ids (e.g. T5c,R1)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    ctx_adj, ctx_raw = Context(adjust=True), Context(adjust=False, verbose=False)
    runs, Y, M = build_runs(ctx_adj, ctx_raw)
    names = [n for n in runs if (a.suite == "all" or (a.suite == "hparams") == n.startswith("H"))]
    if a.only:
        names = [n for n in runs if n.split()[0] in a.only.split(",")]
    i, n = map(int, a.shard.split("/")); names = names[i::n]
    for name in names:
        out = OUT_DIR / f"{name.split()[0]}.json"
        if out.exists() and not a.force:
            print("skip (done):", name); continue
        mk, valid, kw = runs[name]
        t0 = time.time()
        res = rolling_eval(mk(), Y, M, FOLDS, valid=valid, label=name, **kw)
        meta = dict(release=ctx_adj.release_id, push_cover_min=PUSH_COVER_MIN, push_labels_kept_share=float(valid["push"].mean()),
                    push_cover_of_fold={str(f): float(M.loc[M.cutoff == f, "push_cover"].iloc[0]) for f in FOLDS})
        json.dump(dict(name=name, meta=meta, folds=res), open(out, "w"))
        print(f"   ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
