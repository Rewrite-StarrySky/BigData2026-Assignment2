"""Which features do the final models use?  Share of the total split gain per feature group and activity  ->  results/feature_importance.json

    python -m scripts.feature_importance

One XGBoost model per activity (seed 0), trained like the final model on the validation data set (gap 4 h, every second daily cutoff, push labels of poorly
collected windows dropped).  Groups: temporal features of the SAME activity as the target, temporal features of the OTHER activities (incl. the 'total' stream),
cohort context, and the three relational blocks.
"""
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from src.config import *
from src.pipeline import Context, load_samples
from src.final_model import feature_sets
from src.model import XGB, device


def main():
    ctx, ctx_raw = Context(adjust=True, verbose=False), Context(adjust=False, verbose=False)
    B, Y, M = load_samples(ctx, FINAL_GAP_HOURS, ("temporal",) + tuple(FINAL_BLOCKS))
    B_raw, _, _ = load_samples(ctx_raw, FINAL_GAP_HOURS, ("temporal",))
    X = feature_sets(B, B_raw, M.cutoff.values, ctx.daily)
    block_of = {c: {"relational": "FK lifecycle", "openwork": "work in progress", "repo": "repository multi-hop"}[b] for b in FINAL_BLOCKS for c in B[b].columns}
    tr = (((M.cutoff - M.cutoff.min()).dt.days % FINAL_STRIDE) == 0).values
    out, top = {}, {}
    for k in LABELS:
        x = X[FINAL_BASE[k]]; keep = tr & ((M.push_cover >= FINAL_PUSH_MIN_COVER).values if (k == "push" and FINAL_PUSH_MIN_COVER) else True)
        m = xgb.XGBClassifier(**{**XGB, "device": device(), **FINAL_PARAMS, "random_state": 0}).fit(x[keep], Y[k].values[keep])
        gain = pd.Series(m.get_booster().get_score(importance_type="total_gain")).reindex(x.columns).fillna(0.0)

        def group(c):
            if c in block_of:
                return block_of[c]
            if c.startswith("glob_"):
                return "cohort context"
            return "temporal: same activity" if c.startswith(k + "_") else "temporal: other activities"
        g = gain.groupby([group(c) for c in gain.index]).sum() / gain.sum()
        out[k] = {n: round(float(v), 4) for n, v in g.items()}
        top[k] = [(c, round(float(v / gain.sum()), 4)) for c, v in gain.sort_values(ascending=False).head(5).items()]
        print(k, out[k], top[k][:3], flush=True)
    json.dump(dict(share_of_total_gain=out, top5_features=top, n_train=int(tr.sum())), open(RESULTS_DIR / "feature_importance.json", "w"), indent=1)


if __name__ == "__main__":
    main()
