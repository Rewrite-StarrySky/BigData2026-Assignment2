"""Build the cached (user, cutoff) training samples of the data release in DATA_DIR.

    python -m scripts.build_features [--gap-hours 4] [--raw]

--gap-hours  hours between the data release and the start of the label window (4 for the final run: release 11/11 20:00 Taiwan
             time -> window start 11/12 00:00 Taiwan time).  predict.py chooses it automatically.
--raw        also build the temporal block WITHOUT the collection-coverage adjustment (needed for the ablation only).
"""
import argparse
from src.config import FINAL_GAP_HOURS
from src.pipeline import Context, build_samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap-hours", type=int, default=FINAL_GAP_HOURS)
    ap.add_argument("--raw", action="store_true")
    a = ap.parse_args()
    ctx = Context(adjust=True)
    build_samples(ctx, a.gap_hours)
    if a.raw:
        ctx = Context(adjust=False)
        build_samples(ctx, a.gap_hours, which=("temporal",))


if __name__ == "__main__":
    main()
