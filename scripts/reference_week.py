"""The reference week of the course README (2026/09/27 - 10/04 Taiwan time), scored with the official evaluate.py  ->  results/reference_week.json

    python -m scripts.run_gnn --only GF                  # stores the validation predictions of the final graph model (cache/gnn_fold_predictions_*.npz)
    python -m scripts.run_thresholds --model xgb         # the same for the feature baseline (cache/xgb_fold_predictions_*.npz)
    python -m scripts.reference_week

The week is the validation origin 2026-09-26 12:00 UTC (the label week starts 4 h later = 09/27 00:00 Taiwan time).  Both models were trained on
cutoffs <= origin - 8 days only.  Written files: results/reference_week_pred.csv (graph model) / _pred_xgb.csv / _copy.csv / _truth.csv (same format as the
submission) and the output of
    python course_repo/scripts/evaluate.py --pred results/reference_week_pred.csv --truth results/reference_week_truth.csv
The users are the cohort users chosen by the list rule of this repository for that week, NOT the (unpublished) users of the TAs' table, so the comparison with
the reference scores of the course README is indicative only; "copy last week" is computed on our users to show how close the two settings are.
"""
import json, subprocess, sys
import numpy as np
import pandas as pd
from src.config import *
from src.pipeline import Context, load_samples
from src.metrics import best_cutoffs

ORIGIN = pd.Timestamp("2026-09-26 12:00")
TA_README = {"Random": (19.77, 22.74, 21.26), "Predict all 0": (0.0, 0.0, 0.0), "Copy last week": (43.01, 56.20, 49.61),
             "GNN without time encoding": (22.80, 48.61, 35.70), "GNN with time encoding": (50.90, 58.74, 54.82)}      # course README, "Reference scores" (macro, micro, Score in %)


def run_eval(pred, truth):
    out = subprocess.run([sys.executable, str(EVALUATE_PY), "--pred", str(pred), "--truth", str(truth)], check=True, capture_output=True, text=True).stdout
    g = lambda key: float([l for l in out.splitlines() if l.startswith(key)][0].split()[-1].rstrip("%"))
    return out, dict(macro=g("macro-F1"), micro=g("micro-F1"), score=g("Score"))


def main():
    ctx = Context(verbose=False)
    B, Y, M = load_samples(ctx, FINAL_GAP_HOURS, ("temporal",))
    te = (M.cutoff == ORIGIN).values
    z = np.load(CACHE_DIR / f"gnn_fold_predictions_{str(ORIGIN)[:10]}.npz"); P, y = z["P"], z["y"]
    assert (y == Y[te].values).all()
    ids = ctx.github_id[M.user_row[te].values]
    X = B["temporal"][te]
    frames = {"pred": best_cutoffs(P)[0], "copy": np.stack([(X[f"{k}_n7d"].values > 0) for k in LABELS], 1), "truth": y}
    xf = CACHE_DIR / f"xgb_fold_predictions_{str(ORIGIN)[:10]}.npz"
    if xf.exists():
        frames["pred_xgb"] = best_cutoffs(np.load(xf)["P"])[0]
    for name, a in frames.items():
        d = pd.DataFrame(np.asarray(a).astype(int), columns=LABELS); d.insert(0, "UserID", ids); d.to_csv(RESULTS_DIR / f"reference_week_{name}.csv", index=False)
    truth = RESULTS_DIR / "reference_week_truth.csv"
    text, model = run_eval(RESULTS_DIR / "reference_week_pred.csv", truth)
    _, copy = run_eval(RESULTS_DIR / "reference_week_copy.csv", truth)
    res = dict(origin=str(ORIGIN), week_taiwan="2026/09/27 00:00 - 10/04 00:00", n_users=int(te.sum()), evaluate_py_output=text, final_model=model, copy_last_week_on_our_users=copy,
               course_readme_reference={k: dict(macro=v[0], micro=v[1], score=v[2]) for k, v in TA_README.items()}, true_rate=dict(zip(LABELS, y.mean(0).round(4).tolist())))
    if "pred_xgb" in frames:
        res["feature_baseline_xgb"] = run_eval(RESULTS_DIR / "reference_week_pred_xgb.csv", truth)[1]
    json.dump(res, open(RESULTS_DIR / "reference_week.json", "w"), indent=1)
    print(text); print(json.dumps({k: res[k] for k in ("n_users", "final_model", "copy_last_week_on_our_users", "feature_baseline_xgb") if k in res}, indent=1))


if __name__ == "__main__":
    main()
