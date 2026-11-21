"""One gradient-boosted tree model per target activity, and the rolling-origin evaluation used for every experiment."""
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score
from .config import LABELS
from .metrics import evaluate, best_cutoffs

XGB = dict(n_estimators=300, learning_rate=0.05, max_depth=6, min_child_weight=20, subsample=0.8, colsample_bytree=0.8,
           tree_method="hist", reg_lambda=5.0, verbosity=0, random_state=0)
_DEVICE = None


def device():
    """'cuda' when xgboost can use the GPU, else 'cpu'.  The environment variable ASG2_DEVICE=cpu|cuda forces one of them."""
    global _DEVICE
    if _DEVICE is None and os.environ.get("ASG2_DEVICE") in ("cpu", "cuda"):
        _DEVICE = os.environ["ASG2_DEVICE"]
    if _DEVICE is None:
        try:
            xgb.XGBClassifier(n_estimators=1, tree_method="hist", device="cuda").fit(np.random.rand(20, 2), np.arange(20) % 2)
            _DEVICE = "cuda"
        except Exception:
            _DEVICE = "cpu"
    return _DEVICE


def _rows(X, mask):
    """Rows `mask` of a DataFrame or of every DataFrame of a dict."""
    return {b: x[mask] for b, x in X.items()} if isinstance(X, dict) else X[mask]


def fit_predict(Xtr, Ytr, Xte, params=None, weights=None, valid=None, seeds=1, base_of=None, acts=None):
    """Train one classifier per activity on (Xtr, Ytr[k]) and return the probabilities (len(Xte) x 6).

    Xtr / Xte: one DataFrame (same features for all activities) or a dict {feature base: DataFrame}; then `base_of` maps every activity to its base.
    valid: optional boolean DataFrame (rows of Xtr x activities); a False entry drops that training row for that activity
    (used for push labels whose window was not fully collected).  seeds > 1 averages models with different random seeds.
    acts: only train these activities (the other columns of the result stay 0)."""
    n_te = len(next(iter(Xte.values()))) if isinstance(Xte, dict) else len(Xte)
    P = np.zeros((n_te, len(LABELS)))
    for j, k in enumerate(LABELS):
        if acts is not None and k not in acts:
            continue
        xtr, xte = (Xtr[base_of[k]], Xte[base_of[k]]) if isinstance(Xtr, dict) else (Xtr, Xte)
        keep = np.ones(len(xtr), bool) if valid is None else valid[k].values
        for s in range(seeds):
            m = xgb.XGBClassifier(**{**XGB, "device": device(), **(params or {}), "random_state": s})
            m.fit(xtr[keep], Ytr[k].values[keep], sample_weight=None if weights is None else weights[keep])
            P[:, j] += m.predict_proba(xte)[:, 1] / seeds
    return P


def fold_metrics(T, P, y):
    """Scores of the probabilities P (n x 6) of one origin T against the labels y: label-free top-k choice ("exp") and oracle top-k choice."""
    pe, ke = best_cutoffs(P)
    po, _ = best_cutoffs(P, y)
    e, o = evaluate(pe, y), evaluate(po, y)
    return dict(fold=str(T), n=int(len(y)), exp=e["score"], oracle=o["score"], macro_exp=e["macro"], micro_exp=e["micro"],
                f1_exp=e["per_label"], flag_frac=(ke / len(y)).tolist(), true_rate=y.mean(0).tolist(), mean_prob=P.mean(0).tolist(),
                ap=[float(average_precision_score(y[:, j], P[:, j])) if y[:, j].any() else float("nan") for j in range(len(LABELS))])


def rolling_eval(X, Y, M, folds, valid=None, lag_days=8, stride=2, half_life=None, recent_days=None, params=None, seeds=1, label="", verbose=True, base_of=None):
    """Rolling-origin evaluation.  For every origin T (a cutoff): train on the daily cutoffs <= T - lag_days (every `stride`-th),
    so that the label windows of all training samples end before T, then predict the users of cutoff T and score them with
    (a) label-free top-k choice  (b) oracle top-k choice.  Returns one dict per fold."""
    t_first = M.cutoff.min()
    out = []
    for T in folds:
        tr = (M.cutoff <= T - pd.Timedelta(days=lag_days)).values & (((M.cutoff - t_first).dt.days % stride) == 0).values
        if recent_days:
            tr &= (M.cutoff > T - pd.Timedelta(days=recent_days)).values
        w = None if not half_life else 0.5 ** ((T - M.cutoff[tr]).dt.days.values / half_life)
        te = (M.cutoff == T).values
        P = fit_predict(_rows(X, tr), Y[tr].reset_index(drop=True), _rows(X, te), params, w, None if valid is None else valid[tr].reset_index(drop=True), seeds, base_of)
        out.append(fold_metrics(T, P, Y[te].values))
    if verbose and label:
        r = np.array([[f["exp"], f["oracle"]] for f in out])
        print(f"{label:58s} exp {r[:, 0].mean():.4f} | oracle {r[:, 1].mean():.4f} | per fold exp: " + " ".join(f"{v:.3f}" for v in r[:, 0]), flush=True)
    return out
