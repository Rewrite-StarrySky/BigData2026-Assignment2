"""Rolling-origin evaluation of the graph model (src/gnn.py) and its ablations  ->  results/runs/G*.json

    python -m scripts.run_gnn                      # all variants
    python -m scripts.run_gnn --only G0,G1         # some variants
    python -m scripts.run_gnn --only GF            # the final configuration (seeds averaged); also stores its validation predictions in cache/

Same protocol and file format as scripts/run_experiments.py: 7 origins T (Saturdays 12:00 UTC); for each origin the model is trained from scratch
on the snapshots of the daily cutoffs <= T - 8 days (every second day), then predicts the users of cutoff T for the week that starts 4 h later.
"""
import argparse, json, time
import numpy as np
import pandas as pd
import torch
from src.config import *
from src.pipeline import Context, load_samples
from src.graph import GraphData
from src.gnn import GNN, fit, predict, make_samples, coverage, torch_device
from src.model import fold_metrics
from scripts.run_experiments import FOLDS, OUT_DIR

KNOCKOUT = ["Repositories", "PullRequests", "Issues", "IssueComments", "PRReviews", "PRReviewComments", "Pushes"]     # tables removed one at a time (run GF)

# name -> (model / training overrides, push label mask, seeds)
VARIANTS = {
    "G0 graph model: 3 layers, time encoding, cohort context, sum + max": ({}, FINAL_PUSH_MIN_COVER, 1),
    "G1 without time encoding (rows before the cutoff, but not when)": (dict(time_enc=False), FINAL_PUSH_MIN_COVER, 1),
    "G2 without cohort context in the readout": (dict(context=False), FINAL_PUSH_MIN_COVER, 1),
    "G3 sum aggregation only (no max)": (dict(max_agg=False), FINAL_PUSH_MIN_COVER, 1),
    "G4 1 layer: a user only sees its own rows": (dict(layers=1), FINAL_PUSH_MIN_COVER, 1),
    "G5 2 layers: + rows of others on my PRs / issues": (dict(layers=2), FINAL_PUSH_MIN_COVER, 1),
    "G6 4 layers": (dict(layers=4), FINAL_PUSH_MIN_COVER, 1),
    "G7 push labels of all windows (no mask)": ({}, None, 1),
    "G8 width 128": (dict(d=128), FINAL_PUSH_MIN_COVER, 1),
    "G9 60 epochs": (dict(epochs=60), FINAL_PUSH_MIN_COVER, 1),
    f"GF final graph model (G0, {FINAL_SEEDS} seeds averaged)": ({}, FINAL_PUSH_MIN_COVER, FINAL_SEEDS),
}


def rolling_eval_gnn(G, daily, Y, M, folds, cfg, push_min_cover, seeds, lag_days=8, stride=FINAL_STRIDE, label="", save=False):
    cuts = np.sort(M.cutoff.unique()); t_first = pd.Timestamp(cuts[0]); out = []
    for T in folds:
        train = [pd.Timestamp(c) for c in cuts if pd.Timestamp(c) <= T - pd.Timedelta(days=lag_days) and (pd.Timestamp(c) - t_first).days % stride == 0]
        samples = make_samples(daily, Y, M, train, push_min_cover)
        te = (M.cutoff == T).values
        models = [fit(G, samples, cfg, seed=s) for s in range(seeds)]
        rows, cov = M.user_row[te].values, coverage(daily, T)
        P = np.mean([predict(G, m, T, rows, cov, cfg) for m in models], axis=0)
        out.append(fold_metrics(T, P, Y[te].values))
        if save:
            np.savez_compressed(CACHE_DIR / f"gnn_fold_predictions_{str(T)[:10]}.npz", P=P, y=Y[te].values)
            # knock-out analysis: the same trained models predict from a snapshot in which the rows of one table are disconnected
            out[-1]["knockout"] = {t: fold_metrics(T, np.mean([predict(G, m, T, rows, cov, cfg, drop=(t,)) for m in models], axis=0), Y[te].values) for t in KNOCKOUT}
        print(f"   {T:%m-%d}: {len(train)} training cutoffs, exp {out[-1]['exp']:.4f}", flush=True)
    r = np.array([[f["exp"], f["oracle"]] for f in out])
    print(f"{label:58s} exp {r[:, 0].mean():.4f} | oracle {r[:, 1].mean():.4f} | per fold exp: " + " ".join(f"{v:.3f}" for v in r[:, 0]), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="comma separated run ids (e.g. G0,G1)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    ctx = Context(adjust=True)
    _, Y, M = load_samples(ctx, FINAL_GAP_HOURS, ("temporal",))
    G = GraphData(ctx.T, torch_device())
    print(f"graph: {sum(len(v) for v in G.row_time.values()):,} nodes, {sum(len(e[2]) for e in G.edges.values()):,} foreign-key edges, {len(G.rel)} relations, on {G.device}", flush=True)
    for name, (cfg, mask, seeds) in VARIANTS.items():
        rid = name.split()[0]
        if a.only and rid not in a.only.split(","):
            continue
        out = OUT_DIR / f"{rid}.json"
        if out.exists() and not a.force:
            print("skip (done):", name); continue
        t0 = time.time()
        res = rolling_eval_gnn(G, ctx.daily, Y, M, FOLDS, cfg, mask, seeds, label=name, save=(rid == "GF"))
        kept = float((M.push_cover >= mask).mean()) if mask else 1.0
        meta = dict(release=ctx.release_id, push_cover_min=PUSH_COVER_MIN, push_labels_kept_share=kept, config={**GNN, **cfg}, seeds=seeds,
                    push_cover_of_fold={str(f): float(M.loc[M.cutoff == f, "push_cover"].iloc[0]) for f in FOLDS})
        json.dump(dict(name=name, meta=meta, folds=res), open(out, "w"))
        print(f"   ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
