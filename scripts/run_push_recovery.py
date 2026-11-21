"""Push after a collection outage  ->  results/push_recovery.json

    python -m scripts.run_push_recovery            # graph model (push F1 of the 6-task model) and feature baseline (push-only models)

The push stream of GH Archive is down from 10/27 on, but it was also down on 8/28 - 9/3 and came back on 9/5.  If it comes back during the test week the
model must predict push well although the last weeks before the cutoff show (almost) no pushes.  The two origins 9/5 and 9/12 are exactly this situation
(push window collected 84.5 % / 100 %; the snapshot of 9/5 is cut right after the outage).  Compared: training with the push labels of all windows vs. only of
the windows collected >= FINAL_PUSH_MIN_COVER; for the feature baseline also the temporal base ('raw' = plain counts, 'cover' = coverage-adjusted counts +
push-collection context).  Same protocol as the other experiments: train on cutoffs <= origin - 8 days; k chosen without labels.
"""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from src.config import *
from src.pipeline import Context, load_samples
from src.final_model import feature_sets
from src.model import fit_predict
from src.metrics import best_cutoffs, evaluate

ORIGINS = [pd.Timestamp("2026-09-05 12:00"), pd.Timestamp("2026-09-12 12:00")]


def main():
    ctx, ctx_raw = Context(adjust=True, verbose=False), Context(adjust=False, verbose=False)
    B, Y, M = load_samples(ctx, FINAL_GAP_HOURS, ("temporal",) + tuple(FINAL_BLOCKS))
    B_raw, _, _ = load_samples(ctx_raw, FINAL_GAP_HOURS, ("temporal",))
    X = feature_sets(B, B_raw, M.cutoff.values, ctx.daily)
    t_first = M.cutoff.min(); res = []
    for T in ORIGINS:
        tr = (M.cutoff <= T - pd.Timedelta(days=8)).values & (((M.cutoff - t_first).dt.days % FINAL_STRIDE) == 0).values
        te = (M.cutoff == T).values; y = Y[te].values[:, 5]
        for base in ("raw", "cover"):
            for masked in (False, True):
                valid = pd.DataFrame({k: np.ones(len(M), bool) for k in LABELS})
                if masked:
                    valid["push"] = (M.push_cover >= FINAL_PUSH_MIN_COVER).values
                P = fit_predict({b: x[tr] for b, x in X.items()}, Y[tr].reset_index(drop=True), {b: x[te] for b, x in X.items()}, FINAL_PARAMS, None,
                                valid[tr].reset_index(drop=True), FINAL_SEEDS, {k: base for k in LABELS}, acts=["push"])[:, 5:6]
                pred, ks = best_cutoffs(P)
                f1 = evaluate(pred, y[:, None])["per_label"][0]
                res.append(dict(origin=str(T), base=base, labels="windows collected >= %.0f%%" % (100 * FINAL_PUSH_MIN_COVER) if masked else "all windows",
                                n=int(te.sum()), true_rate=float(y.mean()), mean_prob=float(P.mean()), flagged=float(pred.mean()), f1=float(f1),
                                ap=float(average_precision_score(y, P[:, 0])), train_samples=int(tr.sum()),
                                push_cover_label_week=float(M.loc[te, "push_cover"].iloc[0]), push_cover_last_7_days=float(ctx.daily.push_cover[ctx.daily.bin_of(T) - 7:ctx.daily.bin_of(T)].mean())))
                print(res[-1], flush=True)
    # ---- the graph model: one 6-task model per setting, only its push output is scored here
    from src.graph import GraphData
    from src.gnn import GNN, fit, predict, make_samples, coverage, torch_device
    G = GraphData(ctx.T, torch_device()); cfg = {**GNN, **FINAL_GNN}; gres = []
    cuts = [pd.Timestamp(c) for c in np.sort(M.cutoff.unique())]
    for T in ORIGINS:
        train = [c for c in cuts if c <= T - pd.Timedelta(days=8) and (c - cuts[0]).days % FINAL_STRIDE == 0]
        te = (M.cutoff == T).values; y = Y[te].values[:, 5]
        for masked in (False, True):
            S = make_samples(ctx.daily, Y, M, train, FINAL_PUSH_MIN_COVER if masked else None)
            P = np.mean([predict(G, fit(G, S, cfg, seed=s), T, M.user_row[te].values, coverage(ctx.daily, T), cfg) for s in range(FINAL_SEEDS)], axis=0)[:, 5:6]
            pred, _ = best_cutoffs(P)
            gres.append(dict(origin=str(T), labels="windows collected >= %.0f%%" % (100 * FINAL_PUSH_MIN_COVER) if masked else "all windows", n=int(te.sum()),
                             true_rate=float(y.mean()), mean_prob=float(P.mean()), flagged=float(pred.mean()), f1=float(evaluate(pred, y[:, None])["per_label"][0]),
                             ap=float(average_precision_score(y, P[:, 0])), train_cutoffs=len(train), push_windows_masked=int(sum(s[3][5] == 0 for s in S)),
                             push_cover_last_7_days=float(ctx.daily.push_cover[ctx.daily.bin_of(T) - 7:ctx.daily.bin_of(T)].mean())))
            print("gnn", gres[-1], flush=True)
    json.dump(dict(xgb=res, gnn=gres), open(RESULTS_DIR / "push_recovery.json", "w"), indent=1)


if __name__ == "__main__":
    main()
