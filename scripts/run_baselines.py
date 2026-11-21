"""Baselines of the course README on the same rolling-origin folds  ->  results/baselines.json

    python -m scripts.run_baselines

copy_last_week  flag the user for an activity if the user did it in the 7 days before the cutoff (the 'Copy last week' baseline of the course README)
all_zero        predict 0 everywhere
Scores are computed like the final model: (macro-F1 + micro-F1)/2 over the 6 activities ("six"), over the 5 non-push activities ("five") and under the
TAs' rule ("ta": six if the push week of the origin was >= PUSH_CLEAN_MIN collected, else five).
"""
import json
import numpy as np
from src.config import *
from src.pipeline import Context, load_samples
from src.metrics import score_from_counts
from scripts.run_experiments import FOLDS


def counts(pred, y):
    return ((pred == 1) & (y == 1)).sum(0), ((pred == 1) & (y == 0)).sum(0), ((pred == 0) & (y == 1)).sum(0)


def main():
    ctx = Context(verbose=False)
    B, Y, M = load_samples(ctx, FINAL_GAP_HOURS, ("temporal",))
    X = B["temporal"]
    out = {"copy_last_week": [], "all_zero": []}
    for f in FOLDS:
        te = (M.cutoff == f).values; y = Y[te].values
        cov = float(M.loc[te, "push_cover"].iloc[0])
        preds = {"copy_last_week": np.stack([(X.loc[te, f"{k}_n7d"].values > 0) for k in LABELS], 1).astype(int), "all_zero": np.zeros_like(y)}
        for name, p in preds.items():
            tp, fp, fn = counts(p, y)
            six, five = score_from_counts(tp, fp, fn)[0], score_from_counts(tp[:5], fp[:5], fn[:5])[0]
            out[name].append(dict(fold=str(f), six=six, five=five, ta=six if cov >= PUSH_CLEAN_MIN else five, push_cover=cov))
    summary = {k: {m: float(np.mean([r[m] for r in v])) for m in ("six", "five", "ta")} for k, v in out.items()}
    json.dump(dict(folds=out, mean=summary), open(RESULTS_DIR / "baselines.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
