"""Feature-engineering baseline (XGBoost on hand-made features), kept for comparison with the graph model: train on every labelled cutoff, predict the
official test users, write a CSV.  The submitted file is made by scripts/predict.py (graph model), NOT by this script.

    python -m scripts.predict_xgb [--out baseline_xgb.csv] [--gap-hours H]

The prediction cutoff is 2026-11-21 12:00 UTC (config.FINAL_CUTOFF) or, while the data does not reach it yet, the newest 12:00 UTC.  The label
windows of the training samples start `gap` hours after their cutoff, where gap = (11/22 00:00 Taiwan time) - cutoff: 4 h for the final cutoff.
"""
import argparse, json, os, subprocess, sys, time
import numpy as np
import pandas as pd
from src.config import *
from src.pipeline import Context, build_samples, test_features
from src.final_model import feature_sets
from src.model import fit_predict, device
from src.metrics import best_cutoffs, expected_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="baseline_xgb.csv")
    ap.add_argument("--gap-hours", type=int, default=None, help="default: hours from the release time to the start of the target window")
    ap.add_argument("--push-min-cover", type=float, default=FINAL_PUSH_MIN_COVER,
                    help="drop the push labels of training windows in which less than this share of the hours was collected (default: config.FINAL_PUSH_MIN_COVER)")
    ap.add_argument("--as-of", default=None, help="prediction cutoff, a 12:00 UTC time (default: the newest 12:00 UTC not after the end of the data, at most config.FINAL_CUTOFF)")
    ap.add_argument("--device", choices=["cpu", "cuda"], default=None, help="xgboost device (default: GPU if available; the submitted file was made with cpu, which is the most portable)")
    ap.add_argument("--push-base", choices=["raw", "cover"], default=None, help="temporal feature base of the push model (default: config.FINAL_BASE; for the stress test)")
    ap.add_argument("--tag", default="", help="suffix of the report file name (for experiments)")
    a = ap.parse_args()
    if a.device:
        os.environ["ASG2_DEVICE"] = a.device
    t0 = time.time()
    ctx, ctx_raw = Context(adjust=True), Context(adjust=False, verbose=False)         # coverage-adjusted / plain temporal features (src/final_model.py)
    as_of = pd.Timestamp(a.as_of) if a.as_of else min(FINAL_CUTOFF, (ctx.end - pd.Timedelta(hours=12)).floor("D") + pd.Timedelta(hours=12))
    assert as_of <= ctx.end, "the cutoff lies after the end of the data"
    gap = a.gap_hours if a.gap_hours is not None else int((TARGET_WINDOW_START - as_of) / pd.Timedelta(hours=1))
    assert gap >= 0, "the data release is newer than the start of the target window"
    test = pd.read_csv(TEST_INPUT)
    print(f"release {as_of} UTC (data ends {ctx.end}) | target window starts {TARGET_WINDOW_START} UTC | gap {gap} h | {len(test)} test users | xgboost on {device()}", flush=True)

    base_of = {**FINAL_BASE, **({"push": a.push_base} if a.push_base else {})}
    which = ("temporal",) + tuple(FINAL_BLOCKS)
    B, Y, M = build_samples(ctx, gap, which)
    B_raw, _, _ = build_samples(ctx_raw, gap, ("temporal",))
    Xtr = feature_sets(B, B_raw, M.cutoff.values, ctx.daily)
    tr = (((M.cutoff - M.cutoff.min()).dt.days % FINAL_STRIDE) == 0).values
    w = None if not FINAL_HALF_LIFE else 0.5 ** ((as_of - M.cutoff[tr]).dt.days.values / FINAL_HALF_LIFE)
    valid = pd.DataFrame({k: np.ones(len(M), bool) for k in LABELS})
    if a.push_min_cover:
        valid["push"] = (M.push_cover >= a.push_min_cover).values
    print(f"training samples: {int(tr.sum()):,} of {M.cutoff[tr].nunique()} cutoffs ({M.cutoff.min():%Y-%m-%d} .. {M.cutoff.max():%Y-%m-%d}), "
          f"{int((~valid.push[tr]).sum()):,} push labels dropped (window collected < {a.push_min_cover or 0:.0%})", flush=True)

    Bt = test_features(ctx, test.UserID.values, as_of, which)
    Bt_raw = test_features(ctx_raw, test.UserID.values, as_of, ("temporal",))
    Xte = feature_sets(Bt, Bt_raw, np.repeat(as_of.to_datetime64(), len(test)), ctx.daily)
    P = fit_predict({b: x[tr] for b, x in Xtr.items()}, Y[tr].reset_index(drop=True), Xte, FINAL_PARAMS, w, valid[tr].reset_index(drop=True), FINAL_SEEDS, base_of)
    pred, ks = best_cutoffs(P)

    out = pd.DataFrame(pred, columns=LABELS).astype(int); out.insert(0, "UserID", test.UserID.values)
    out.to_csv(a.out, index=False)
    s, ma, mi = expected_score(P, pred)
    rep = dict(release=str(as_of), gap_hours=gap, n_test=len(test), n_train=int(tr.sum()), flagged_share=dict(zip(LABELS, (pred.mean(0)).round(4).tolist())),
               mean_probability=dict(zip(LABELS, P.mean(0).round(4).tolist())), expected_score=dict(score=s, macro=ma, micro=mi),
               push_collection_last_7_bins=[round(float(x), 2) for x in ctx.daily.push_cover[ctx.daily.bin_of(as_of) - 7:ctx.daily.bin_of(as_of)]],
               seconds=round(time.time() - t0))
    rep["push_min_cover"] = a.push_min_cover; rep["feature_base"] = base_of
    json.dump(rep, open(RESULTS_DIR / f"prediction_report_xgb_{ctx.release_id}_g{gap}{a.tag}.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))
    print(f"wrote {a.out}; running the official format check:", flush=True)
    subprocess.run([sys.executable, str(EVALUATE_PY), "--pred", a.out, "--input", str(TEST_INPUT)], check=True)


if __name__ == "__main__":
    main()
