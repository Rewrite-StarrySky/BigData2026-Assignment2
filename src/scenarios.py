"""Scoring rule announced by the TAs: an activity with missing data in the test week is NOT scored; the Score (macro-F1 + micro-F1)/2
is computed over the remaining activities.

Only push has visible gaps in the data (days without any push; see results/data_analysis.json), so there are two cases:
  six   push is fully collected in the target week  -> Score over all 6 activities
  five  push has missing data in the target week    -> Score over the 5 other activities (push predictions are ignored)
For validation an origin counts as "push fully collected" if >= PUSH_CLEAN_MIN of the hours of its label week have pushes; its Score under
the TA rule ("ta") is then `six`, otherwise `five`.

The per-activity counts are rebuilt EXACTLY from the stored F1, flagged share and true rate:  TP = F1 * (P + K) / 2  with P = positives and
K = flagged users (identity F1 = 2 TP / (P + K)); scripts/scenario_scores.py checks that this reproduces the stored Score of every fold.
"""
import numpy as np
from .config import PUSH_CLEAN_MIN
from .metrics import score_from_counts


def counts(n, f1, flag_share, true_rate):
    """(tp, fp, fn) per activity of one evaluation (n users) from per-activity F1, flagged share and true positive rate."""
    P = np.asarray(true_rate, float) * n
    K = np.asarray(flag_share, float) * n
    tp = np.asarray(f1, float) * (P + K) / 2
    return tp, K - tp, P - tp


def scores(n, f1, flag_share, true_rate, push_clean):
    """dict(six, five, ta) for one evaluation; `push_clean` = push fully collected in the label week (decides which of the two counts)."""
    tp, fp, fn = counts(n, f1, flag_share, true_rate)
    six = score_from_counts(tp, fp, fn)[0]
    five = score_from_counts(tp[:5], fp[:5], fn[:5])[0]
    return dict(six=six, five=five, ta=six if push_clean else five)


def run_scores(run):
    """Per-origin scores of one run of results/runs/*.json -> dict of arrays (six, five, ta, clean)."""
    cov = run["meta"]["push_cover_of_fold"]
    rows = [scores(f["n"], f["f1_exp"], f["flag_frac"], f["true_rate"], cov[f["fold"]] >= PUSH_CLEAN_MIN) for f in run["folds"]]
    out = {k: np.array([r[k] for r in rows]) for k in ("six", "five", "ta")}
    out["clean"] = np.array([cov[f["fold"]] >= PUSH_CLEAN_MIN for f in run["folds"]])
    out["reconstruction_error"] = float(max(abs(r["six"] - f["exp"]) for r, f in zip(rows, run["folds"])))
    return out
