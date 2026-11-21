"""Score = (macro-F1 + micro-F1) / 2 (identical to evaluation/evaluate.py) and the choice of how many users to flag per activity."""
import numpy as np


def _f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) > 0 else 1.0


def score_from_counts(tp, fp, fn):
    """tp / fp / fn: arrays with one entry per activity (real or expected counts) -> (score, macro-F1, micro-F1)."""
    macro = np.mean([_f1(a, b, c) for a, b, c in zip(tp, fp, fn)])
    micro = _f1(tp.sum(), fp.sum(), fn.sum())
    return float((macro + micro) / 2), float(macro), float(micro)


def evaluate(pred, truth):
    """pred, truth: (n_users, 6) 0/1 arrays."""
    pred, truth = np.asarray(pred), np.asarray(truth)
    tp = ((pred == 1) & (truth == 1)).sum(0); fp = ((pred == 1) & (truth == 0)).sum(0); fn = ((pred == 0) & (truth == 1)).sum(0)
    s, ma, mi = score_from_counts(tp, fp, fn)
    return dict(score=s, macro=ma, micro=mi, per_label=[float(_f1(a, b, c)) for a, b, c in zip(tp, fp, fn)])


def best_cutoffs(prob, truth=None, rounds=3, grid=None):
    """Choose how many users to flag per activity (top-k by probability) so that the Score is maximal.

    truth is None   -> label-free: expected counts under the model's own probabilities (TP = sum of p over the flagged users,
                       FN = sum of p over the others).  This is what is used for the real test.
    truth given     -> real counts (oracle; only used to measure how much a perfect choice of k would add).
    Returns (boolean prediction matrix, k per activity).
    """
    n, L = prob.shape
    order = np.argsort(-prob, axis=0)
    p_sorted = np.take_along_axis(prob, order, axis=0)
    y_sorted = None if truth is None else np.take_along_axis(np.asarray(truth, float), order, axis=0)
    gain = np.cumsum(p_sorted if truth is None else y_sorted, axis=0)             # tp(k) for k = 1..n flagged
    total_pos = gain[-1]
    ks = np.arange(1, n + 1)
    k_star = np.array([max(1, int(np.argmax(2 * gain[:, j] / (ks + total_pos[j])) + 1)) for j in range(L)])   # per-activity F1 optimum
    cand = grid if grid is not None else np.unique(np.linspace(0, n, 400).astype(int))

    def total(ks_):
        tp = np.array([gain[k - 1, j] if k > 0 else 0.0 for j, k in enumerate(ks_)])
        return score_from_counts(tp, ks_ - tp, total_pos - tp)[0]

    cur = k_star.copy()
    for _ in range(rounds):                                                        # coordinate ascent on the combined score
        for j in range(L):
            best, bk = -1, cur[j]
            for k in cand:
                trial = cur.copy(); trial[j] = k; sc = total(trial)
                if sc > best:
                    best, bk = sc, k
            cur[j] = bk
    pred = np.zeros((n, L), dtype=np.int8)
    for j in range(L):
        pred[order[:cur[j], j], j] = 1
    return pred, cur


def expected_score(prob, pred):
    """Label-free estimate of the Score of a 0/1 prediction: real counts replaced by expectations under the probabilities."""
    prob, pred = np.asarray(prob), np.asarray(pred)
    tp = (prob * pred).sum(0)
    return score_from_counts(tp, pred.sum(0) - tp, prob.sum(0) - tp)


def best_cutoffs_robust(prob, mults=(0.67, 1.0, 1.49), weights=(0.25, 0.5, 0.25), rounds=3):
    """Like best_cutoffs(prob) but the expected Score is averaged over several assumptions about the LEVEL of the next week:
    the probabilities are multiplied by `mults` (clipped at 1) with the given weights, because the realised positive rate of a
    week is often far from the mean predicted probability (regime shifts).  The ranking of the users is not changed."""
    n, L = prob.shape
    order = np.argsort(-prob, axis=0)
    ps = np.take_along_axis(prob, order, axis=0)
    gains = [np.cumsum(np.clip(m * ps, 0, 1), axis=0) for m in mults]
    totals = [np.clip(m * prob, 0, 1).sum(0) for m in mults]
    cand = np.unique(np.linspace(0, n, 400).astype(int))

    def total(ks_):
        sc = 0.0
        for w, g, tot in zip(weights, gains, totals):
            tp = np.array([g[k - 1, j] if k > 0 else 0.0 for j, k in enumerate(ks_)])
            sc += w * score_from_counts(tp, ks_ - tp, tot - tp)[0]
        return sc

    cur = best_cutoffs(prob)[1].copy()
    for _ in range(rounds):
        for j in range(L):
            best, bk = -1, cur[j]
            for k in cand:
                trial = cur.copy(); trial[j] = k; sc = total(trial)
                if sc > best:
                    best, bk = sc, k
            cur[j] = bk
    pred = np.zeros((n, L), dtype=np.int8)
    for j in range(L):
        pred[order[:cur[j], j], j] = 1
    return pred, cur
