"""Graph model vs. feature baseline vs. their average, on the validation origins  ->  results/model_comparison.json

    python -m scripts.run_gnn --only GF ; python -m scripts.run_thresholds --model xgb ; python -m scripts.compare_models

Uses the stored validation predictions of both models (cache/gnn_fold_predictions_*.npz, cache/xgb_fold_predictions_*.npz).  Scores: all six activities
("six"), the five non-push activities ("five") and the TAs' rule ("ta": six if the push week of the origin was >= PUSH_CLEAN_MIN collected, else five);
k is always chosen without labels.  Also: how similar the two models are (rank correlation of their probabilities, overlap of the flagged users).
"""
import json
import numpy as np
from scipy.stats import spearmanr
from src.config import *
from src.pipeline import Context, load_samples
from src.metrics import best_cutoffs, evaluate
from src.scenarios import scores
from scripts.run_experiments import FOLDS


def main():
    ctx = Context(verbose=False)
    _, Y, M = load_samples(ctx, FINAL_GAP_HOURS, ("temporal",))
    rows = []
    for T in FOLDS:
        te = (M.cutoff == T).values; y = Y[te].values; n = int(te.sum())
        clean = float(M.loc[te, "push_cover"].iloc[0]) >= PUSH_CLEAN_MIN
        P = {"gnn": np.load(CACHE_DIR / f"gnn_fold_predictions_{str(T)[:10]}.npz")["P"], "xgb": np.load(CACHE_DIR / f"xgb_fold_predictions_{str(T)[:10]}.npz")["P"]}
        P["average"] = (P["gnn"] + P["xgb"]) / 2
        r = dict(fold=str(T), n=n, push_clean=bool(clean)); pred = {}
        for k, p in P.items():
            pred[k] = best_cutoffs(p)[0]; ev = evaluate(pred[k], y)
            r[k] = {**scores(n, ev["per_label"], pred[k].mean(0), y.mean(0), clean), "f1": ev["per_label"]}
        r["spearman"] = [float(spearmanr(P["gnn"][:, j], P["xgb"][:, j]).statistic) for j in range(len(LABELS))]
        r["flag_jaccard"] = [float((pred["gnn"][:, j] & pred["xgb"][:, j]).sum() / max(1, (pred["gnn"][:, j] | pred["xgb"][:, j]).sum())) for j in range(len(LABELS))]
        rows.append(r)
    mean = {k: {m: float(np.mean([r[k][m] for r in rows])) for m in ("six", "five", "ta")} for k in ("gnn", "xgb", "average")}
    d = np.array([r["gnn"]["ta"] - r["xgb"]["ta"] for r in rows]); da = np.array([r["average"]["ta"] - r["gnn"]["ta"] for r in rows])
    out = dict(folds=rows, mean=mean, gnn_minus_xgb_ta=dict(mean=float(d.mean()), se=float(d.std(ddof=1) / np.sqrt(len(d))), better=int((d > 0).sum())),
               average_minus_gnn_ta=dict(mean=float(da.mean()), se=float(da.std(ddof=1) / np.sqrt(len(da))), better=int((da > 0).sum())),
               spearman_mean=dict(zip(LABELS, np.mean([r["spearman"] for r in rows], 0).round(4).tolist())),
               flag_jaccard_mean=dict(zip(LABELS, np.mean([r["flag_jaccard"] for r in rows], 0).round(4).tolist())))
    json.dump(out, open(RESULTS_DIR / "model_comparison.json", "w"), indent=1)
    print(json.dumps({k: out[k] for k in out if k != "folds"}, indent=1))


if __name__ == "__main__":
    main()
