"""Final prediction with the graph model (src/gnn.py): train on every labelled cutoff of the data in DATA_DIR, predict the official test users, write the CSV.

    python -m scripts.predict                      # train (GPU: ~20 min, deterministic) + predict + format check; also writes results/final_gnn_weights.pt
    python -m scripts.predict --weights results/final_gnn_weights.pt      # no training: load the weights of the submitted model and predict (CPU is enough)

The prediction cutoff is 2026-11-21 12:00 UTC (config.FINAL_CUTOFF) or, while the data does not reach it yet, the newest 12:00 UTC (--as-of sets it).
The label windows of the training cutoffs start `gap` hours after the cutoff, gap = (11/22 00:00 Taiwan time) - cutoff: 4 h for the final cutoff, more for
earlier ones, so the model is always trained for the horizon it is used for.  FINAL_SEEDS models with different seeds are averaged.
"""
import argparse, json, os, subprocess, sys, time
import numpy as np
import pandas as pd
from src.config import *
from src.pipeline import Context, build_samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"output_{STUDENT_ID}.csv")
    ap.add_argument("--gap-hours", type=int, default=None, help="default: hours from the cutoff to the start of the target window")
    ap.add_argument("--push-min-cover", type=float, default=FINAL_PUSH_MIN_COVER,
                    help="mask the push labels of training windows in which less than this share of the hours was collected (default: config.FINAL_PUSH_MIN_COVER; 0 = keep all)")
    ap.add_argument("--as-of", default=None, help="prediction cutoff, a 12:00 UTC time (default: the newest 12:00 UTC not after the end of the data, at most config.FINAL_CUTOFF)")
    ap.add_argument("--device", choices=["cpu", "cuda"], default=None, help="default: GPU if available")
    ap.add_argument("--weights", default=None, help="load these trained weights instead of training (e.g. results/final_gnn_weights.pt)")
    ap.add_argument("--save-weights", default=str(GNN_WEIGHTS), help="where the trained weights are written ('' = do not save)")
    ap.add_argument("--tag", default="", help="suffix of the report file name (for experiments)")
    a = ap.parse_args()
    if a.device:
        os.environ["ASG2_DEVICE"] = a.device
    import torch
    from src.graph import GraphData
    from src.gnn import GNN, HeteroTemporalGNN, fit, predict, make_samples, coverage, torch_device
    from src.metrics import best_cutoffs, expected_score
    t0 = time.time()
    ctx = Context(adjust=True)
    as_of = pd.Timestamp(a.as_of) if a.as_of else min(FINAL_CUTOFF, (ctx.end - pd.Timedelta(hours=12)).floor("D") + pd.Timedelta(hours=12))
    assert as_of <= ctx.end, "the cutoff lies after the end of the data"
    gap = a.gap_hours if a.gap_hours is not None else int((TARGET_WINDOW_START - as_of) / pd.Timedelta(hours=1))
    assert gap >= 0, "the cutoff lies after the start of the target window"
    test = pd.read_csv(TEST_INPUT)
    G = GraphData(ctx.T, torch_device())
    print(f"cutoff {as_of} UTC (data ends {ctx.end}) | target window starts {TARGET_WINDOW_START} UTC | gap {gap} h | {len(test)} test users | "
          f"graph: {sum(len(v) for v in G.row_time.values()):,} rows, {sum(len(e[2]) for e in G.edges.values()):,} foreign-key edges | torch {torch.__version__} on {G.device}", flush=True)
    cfg = {**GNN, **FINAL_GNN}
    row_of = pd.Series(np.arange(len(ctx.github_id)), index=ctx.github_id)
    known = np.isin(test.UserID.values, ctx.github_id)                                  # a test user without any row before the cutoff gets probability 0
    rows = row_of.loc[test.UserID.values[known]].values
    n_vis = G.n_visible(float((as_of - DAY0) / pd.Timedelta(days=1)))["Users"]
    seen = rows < n_vis                                                                # ... and so does a user whose first row appears after the cutoff
    if a.weights:
        ck = torch.load(a.weights, map_location=G.device)
        cfg = ck["config"]; models = []
        for sd in ck["state_dicts"]:
            m = HeteroTemporalGNN(G.in_dim, G.rel, **cfg).to(G.device); m.load_state_dict(sd); models.append(m)
        n_train = ck["n_train_cutoffs"]
        print(f"loaded {len(models)} models from {a.weights} (trained on {n_train} cutoffs up to {ck['last_train_cutoff']}, cutoff of the prediction {ck['as_of']})", flush=True)
    else:
        _, Y, M = build_samples(ctx, gap, ("temporal",))                                # the users and labels of every training cutoff
        cuts = [pd.Timestamp(c) for c in np.sort(M.cutoff.unique())]
        train = [c for c in cuts if (c - cuts[0]).days % FINAL_STRIDE == 0]
        samples = make_samples(ctx.daily, Y, M, train, a.push_min_cover)
        n_masked = sum(int(s[3][LABELS.index("push")] == 0) for s in samples)
        print(f"training: {len(train)} cutoffs ({train[0]:%Y-%m-%d} .. {train[-1]:%Y-%m-%d}), {sum(len(s[1]) for s in samples):,} (user, cutoff) samples, "
              f"push labels of {n_masked} cutoffs masked (window collected < {a.push_min_cover or 0:.0%}); {FINAL_SEEDS} seeds x {cfg['epochs']} epochs", flush=True)
        models = []
        for s in range(FINAL_SEEDS):
            models.append(fit(G, samples, cfg, seed=s, verbose=True)); print(f"   seed {s} done ({time.time() - t0:.0f} s)", flush=True)
        n_train = len(train)
        if a.save_weights:
            torch.save(dict(config=cfg, state_dicts=[{k: v.cpu() for k, v in m.state_dict().items()} for m in models], n_train_cutoffs=n_train,
                            last_train_cutoff=str(train[-1]), as_of=str(as_of), gap_hours=gap), a.save_weights)
            print(f"wrote {a.save_weights}", flush=True)
    P = np.zeros((len(test), len(LABELS)))
    idx = np.flatnonzero(known)[seen]
    P[idx] = np.mean([predict(G, m, as_of, rows[seen], coverage(ctx.daily, as_of), cfg) for m in models], axis=0)
    pred, ks = best_cutoffs(P)

    out = pd.DataFrame(pred, columns=LABELS).astype(int); out.insert(0, "UserID", test.UserID.values)
    out.to_csv(a.out, index=False)
    s, ma, mi = expected_score(P, pred)
    rep = dict(model="graph model (src/gnn.py)", cutoff=str(as_of), release=str(as_of), gap_hours=gap, n_test=len(test), test_users_without_rows_before_cutoff=int(len(test) - len(idx)),
               n_train_cutoffs=n_train, config=cfg, seeds=len(models), weights_loaded=bool(a.weights), device=G.device, torch=torch.__version__,
               flagged_share=dict(zip(LABELS, (pred.mean(0)).round(4).tolist())), mean_probability=dict(zip(LABELS, P.mean(0).round(4).tolist())),
               expected_score=dict(score=s, macro=ma, micro=mi), push_min_cover=a.push_min_cover,
               push_collection_last_7_bins=[round(float(x), 2) for x in ctx.daily.push_cover[ctx.daily.bin_of(as_of) - 7:ctx.daily.bin_of(as_of)]], seconds=round(time.time() - t0))
    json.dump(rep, open(RESULTS_DIR / f"prediction_report_{ctx.release_id}_g{gap}{a.tag}.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))
    print(f"wrote {a.out}; running the official format check:", flush=True)
    subprocess.run([sys.executable, str(EVALUATE_PY), "--pred", a.out, "--input", str(TEST_INPUT)], check=True)


if __name__ == "__main__":
    main()
