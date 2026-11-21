"""Leak check of the graph model: does a prediction at a cutoff depend on anything after the cutoff?

    # 1. train one model on the full tables (cutoffs <= cutoff - 8 days) and store its weights and its predictions at the cutoff
    python -m scripts.leak_check_gnn --train --weights cache/leak_gnn.pt --out cache/leak_gnn_full.parquet
    # 2. rebuild the tables with the course script from the raw hours BEFORE the cutoff only; predict with the SAME weights from those tables
    python course_repo/scripts/preprocess.py --raw raw_until_cutoff --out preprocessed_until_cutoff
    ASG2_DB_DIR=preprocessed_until_cutoff/db python -m scripts.leak_check_gnn --weights cache/leak_gnn.pt --out cache/leak_gnn_trunc.parquet
    # 3. compare
    python -m scripts.leak_check_gnn --compare cache/leak_gnn_full.parquet cache/leak_gnn_trunc.parquet

If the snapshot of the full tables at the cutoff contains nothing from the future, both predictions are identical (for the test users that exist in both
sets of tables).  Result -> results/leak_check_gnn.json
"""
import argparse, json
import numpy as np
import pandas as pd
import torch
from src.config import *
from src.pipeline import Context, load_samples
from src.graph import GraphData
from src.gnn import GNN, HeteroTemporalGNN, fit, predict, make_samples, coverage, torch_device
from src.metrics import best_cutoffs


def run(cutoff, weights, out, train):
    cutoff = pd.Timestamp(cutoff)
    ctx = Context(adjust=True); G = GraphData(ctx.T, torch_device()); cfg = {**GNN, **FINAL_GNN}
    if train:
        _, Y, M = load_samples(ctx, FINAL_GAP_HOURS, ("temporal",))
        cuts = [pd.Timestamp(c) for c in np.sort(M.cutoff.unique())]
        tr = [c for c in cuts if c <= cutoff - pd.Timedelta(days=8) and (c - cuts[0]).days % FINAL_STRIDE == 0]
        model = fit(G, make_samples(ctx.daily, Y, M, tr, FINAL_PUSH_MIN_COVER), cfg, seed=0)
        torch.save(dict(config=cfg, state_dict={k: v.cpu() for k, v in model.state_dict().items()}), weights)
    else:
        ck = torch.load(weights, map_location=G.device); cfg = ck["config"]
        model = HeteroTemporalGNN(G.in_dim, G.rel, **cfg).to(G.device); model.load_state_dict(ck["state_dict"])
    ids = pd.read_csv(TEST_INPUT).UserID.values
    row_of = pd.Series(np.arange(len(ctx.github_id)), index=ctx.github_id)
    ids = ids[np.isin(ids, ctx.github_id)]; rows = row_of.loc[ids].values
    keep = rows < G.n_visible(float((cutoff - DAY0) / pd.Timedelta(days=1)))["Users"]            # users that have a row before the cutoff
    P = predict(G, model, cutoff, rows[keep], coverage(ctx.daily, cutoff), cfg)
    pd.DataFrame(P, columns=LABELS, index=pd.Index(ids[keep], name="UserID")).to_parquet(out)
    print(f"wrote {out}: {int(keep.sum())} users, data ends {ctx.end}")


def compare(a, b, cutoff):
    A, B = pd.read_parquet(a), pd.read_parquet(b)
    common = A.index.intersection(B.index); A, B = A.loc[common], B.loc[common]
    d = (A - B).abs().values
    pa, pb = best_cutoffs(A.values)[0], best_cutoffs(B.values)[0]
    res = dict(cutoff=str(cutoff), users_compared=int(len(common)), max_abs_probability_difference=float(d.max()), mean_abs_probability_difference=float(d.mean()),
               users_with_a_difference_above_0_001=int((d > 1e-3).any(1).sum()), users_with_a_difference_above_0_01=int((d > 1e-2).any(1).sum()),
               identical_01_predictions_share=float((pa == pb).mean()), users_with_a_different_01_prediction=int((pa != pb).any(1).sum()))
    print(json.dumps(res, indent=1))
    json.dump(res, open(RESULTS_DIR / "leak_check_gnn.json", "w"), indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", default="2026-10-03 12:00"); ap.add_argument("--weights", default=str(CACHE_DIR / "leak_gnn.pt")); ap.add_argument("--out")
    ap.add_argument("--train", action="store_true"); ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    compare(*a.compare, cutoff=a.cutoff) if a.compare else run(a.cutoff, a.weights, a.out, a.train)
