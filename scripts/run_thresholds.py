"""Threshold policies on the validation predictions of the final graph model (same rolling-origin folds as run_gnn.py)  ->  results/thresholds.json

    python -m scripts.run_gnn --only GF        # stores the validation predictions of the final graph model in cache/
    python -m scripts.run_thresholds           # graph model   -> results/thresholds.json
    python -m scripts.run_thresholds --model xgb     # feature baseline (trains it per origin) -> results/thresholds_xgb.json, cache/xgb_fold_predictions_*.npz

Policies: how many users to flag per activity.
  expected   label-free choice that maximises the expected Score under the model's own probabilities (used for the real test)
  per_label  the same but every activity optimised on its own F1 only (no coordinate ascent on the combined Score)
  robust     expected Score averaged over three assumptions about the level of the next week (probabilities x 0.67 / x 1 / x 1.49)
  p>=0.5     flag everybody whose probability is at least 0.5
  p>=0.3     flag everybody whose probability is at least 0.3
  five       like `expected`, but the five non-push activities are optimised for the Score WITHOUT push (the case in which push is not scored)
  oracle     top-k chosen with the true labels (upper bound, not available for the real test)

Every policy is scored (a) over all six activities, (b) without push, (c) under the TA rule (push counts only if the push window of the
origin was fully collected; see src/scenarios.py).
"""
import argparse, json
import numpy as np
import pandas as pd
from src.config import *
from src.pipeline import Context, load_samples
from src.model import fit_predict
from src.metrics import evaluate, best_cutoffs, best_cutoffs_robust
from src.scenarios import scores
from src.final_model import feature_sets
from scripts.run_experiments import FOLDS


def per_label_f1_choice(P):
    """Top-k per activity with k = argmax of the expected F1 of that activity alone."""
    n, L = P.shape
    order = np.argsort(-P, axis=0); gain = np.cumsum(np.take_along_axis(P, order, axis=0), axis=0); total = gain[-1]
    ks = np.arange(1, n + 1); pred = np.zeros((n, L), np.int8)
    for j in range(L):
        k = int(np.argmax(2 * gain[:, j] / (ks + total[j])) + 1); pred[order[:k, j], j] = 1
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["gnn", "xgb"], default="gnn", help="gnn: validation predictions of the final graph model (stored by `run_gnn --only GF`); "
                                                                          "xgb: train the feature baseline per origin")
    a = ap.parse_args()
    ctx = Context(adjust=True)
    if a.model == "xgb":
        ctx_raw = Context(adjust=False, verbose=False)
        B, Y, M = load_samples(ctx, FINAL_GAP_HOURS, ("temporal",) + tuple(FINAL_BLOCKS))
        B_raw, _, _ = load_samples(ctx_raw, FINAL_GAP_HOURS, ("temporal",))
        X = feature_sets(B, B_raw, M.cutoff.values, ctx.daily)
        valid = pd.DataFrame({k: np.ones(len(M), bool) for k in LABELS})
        if FINAL_PUSH_MIN_COVER:
            valid["push"] = (M.push_cover >= FINAL_PUSH_MIN_COVER).values            # same training labels as the baseline of predict_xgb.py
    else:
        _, Y, M = load_samples(ctx, FINAL_GAP_HOURS, ("temporal",))
    t_first = M.cutoff.min(); res = {}
    for T in FOLDS:
        te = (M.cutoff == T).values
        y = Y[te].values
        if a.model == "xgb":
            tr = (M.cutoff <= T - pd.Timedelta(days=8)).values & (((M.cutoff - t_first).dt.days % FINAL_STRIDE) == 0).values
            w = None if not FINAL_HALF_LIFE else 0.5 ** ((T - M.cutoff[tr]).dt.days.values / FINAL_HALF_LIFE)
            P = fit_predict({b: x[tr] for b, x in X.items()}, Y[tr].reset_index(drop=True), {b: x[te] for b, x in X.items()}, FINAL_PARAMS, w, valid[tr].reset_index(drop=True), FINAL_SEEDS, FINAL_BASE)
            np.savez_compressed(CACHE_DIR / f"xgb_fold_predictions_{str(T)[:10]}.npz", P=P, y=y)
        else:
            z = np.load(CACHE_DIR / f"gnn_fold_predictions_{str(T)[:10]}.npz"); P = z["P"]
            assert (z["y"] == y).all()
        pred6 = best_cutoffs(P)[0]
        pols = {"expected": pred6, "per_label": per_label_f1_choice(P), "robust": best_cutoffs_robust(P)[0], "p>=0.5": (P >= 0.5).astype(int), "p>=0.3": (P >= 0.3).astype(int),
                "five": np.column_stack([best_cutoffs(P[:, :5])[0], pred6[:, 5]]), "oracle": best_cutoffs(P, y)[0]}
        n = int(te.sum()); cov = float(M.loc[te, "push_cover"].iloc[0]); clean = cov >= PUSH_CLEAN_MIN
        res[str(T)] = {}
        for k, v in pols.items():
            ev = evaluate(v, y); sc = scores(n, ev["per_label"], v.mean(0), y.mean(0), clean)
            res[str(T)][k] = dict(score=ev["score"], score5=sc["five"], score_ta=sc["ta"], f1=ev["per_label"], flag_share=v.mean(0).tolist())
        res[str(T)].update(n=n, push_cover=cov, true_rate=y.mean(0).tolist(), mean_prob=P.mean(0).tolist())
        print(str(T)[:10], {k: round(v["score"], 4) for k, v in res[str(T)].items() if isinstance(v, dict) and "score" in v}, flush=True)
    name = "thresholds.json" if a.model == "gnn" else "thresholds_xgb.json"
    json.dump(dict(model=a.model, push_min_cover=FINAL_PUSH_MIN_COVER, push_clean_min=PUSH_CLEAN_MIN, folds=res), open(RESULTS_DIR / name, "w"), indent=1)


if __name__ == "__main__":
    main()
