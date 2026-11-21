"""Paths and the task definition.  All times are UTC like the data (Taiwan time = UTC + 8 h)."""
import os
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("ASG2_DATA_ROOT", ROOT.parent / "assignment2_v2"))   # clone of the course repository (scripts/, submission/)
DATA_DIR = Path(os.environ.get("ASG2_DB_DIR", ROOT.parent / "preprocessed" / "db"))  # output of scripts/preprocess.py (--out <folder>/db)
TEST_INPUT = DATA_ROOT / "submission" / "test_input.csv"
EVALUATE_PY = DATA_ROOT / "scripts" / "evaluate.py"
CACHE_DIR = Path(os.environ.get("ASG2_CACHE_DIR", ROOT / "cache"))
RESULTS_DIR = ROOT / "results"
for _d in (CACHE_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

STUDENT_ID = "315581033"
LABELS = ["open_pr", "open_issue", "issue_comment", "pr_review", "pr_review_comment", "push"]

# ---- the task -------------------------------------------------------------------------------------------------------
WINDOW_DAYS = 7
TARGET_WINDOW_START = pd.Timestamp("2026-11-21 16:00")     # 2026-11-22 00:00 Taiwan time
DATA_START = pd.Timestamp("2026-06-01 00:00")              # first GH Archive hour used (events before DAY0 are ignored; a warning is printed if the tables contain older rows)
DATA_CAP = pd.Timestamp("2026-11-21 12:00")                # data at or after this time is ignored: the cutoff (the last hour file needed is 2026-11-21-11.json.gz).
#   With the cap, a re-run after the test week (more raw data) gives the same training set as the submitted run.
FINAL_CUTOFF = pd.Timestamp("2026-11-21 12:00")            # last daily cutoff before the target window (gap 4 h)
TEST_LIST_TIME = pd.Timestamp("2026-11-04 12:00")          # time the test users were selected (found by search: Jaccard 0.9996 with test_input.csv)
LIST_AGE = TARGET_WINDOW_START - TEST_LIST_TIME            # 17 d 4 h: age of the user list when the target window starts
ELIGIBLE_MIN_RECORDS = 3                                   # test users: >= 3 activity records ...
ELIGIBLE_LOOKBACK_DAYS = 90                                # ... in the 90 days before the list time
FINAL_GAP_HOURS = 4                                        # final release (11/11 20:00 Taiwan) -> window start (11/12 00:00 Taiwan)

# ---- time axis of the daily activity matrices ------------------------------------------------------------------------
DAY0 = pd.Timestamp("2026-05-31 12:00")                    # bin 0 = [5/31 12:00, 6/1 12:00); bins start at 12:00 UTC (first data hour: 6/1 00:00)
NB = 200                                                   # number of daily bins (covers the whole project period)
FIRST_CUTOFF = pd.Timestamp("2026-08-01 12:00")             # first training cutoff (data starts 6/1; the 90-day windows of the first cutoffs are cut at the data start)
PUSH_ALIVE_MIN = 10                                        # an hour counts as "push collected" if the cohort has >= this many pushes in it (normal hours: 100-400)

# ---- the final model: the graph model of src/gnn.py (chosen with scripts/run_gnn.py, see results/) -------------------------------------------------
FINAL_GNN = {}                                             # overrides of the default hyper-parameters src.gnn.GNN (run GF of scripts/run_gnn.py)
GNN_WEIGHTS = RESULTS_DIR / "final_gnn_weights.pt"         # trained weights of the submitted model (written by scripts/predict.py, loaded with --weights)

# ---- the feature-engineering baseline (XGBoost; chosen with scripts/run_experiments.py, see results/) ---------------------------------------------
PUSH_COVER_MIN = 0.8                                       # a week is 'poorly collected' for push if < 80 % of its hours have pushes (T5b, R4m80, final model)
PUSH_CLEAN_MIN = 0.9                                       # validation: push counts for an origin only if >= 90 % of the hours of its label week have pushes (see src/scenarios.py)
FINAL_PUSH_MIN_COVER = PUSH_COVER_MIN                      # push labels of training windows collected < this share are dropped (None = keep all labels).
#   TA announcement: an activity with missing data in the test week is not scored, so push only counts when the test week is fully
#   collected -> the push model is trained on fully collected weeks only (run R4m80 / H8m in results/runs/).
FINAL_BLOCKS = ("relational", "openwork", "repo")                  # feature blocks used besides the temporal block + context (ablation R4)
FINAL_BASE = {**{k: "raw" for k in LABELS[:5]}, "push": "cover"}   # temporal feature base of every activity (src/final_model.py): 'raw' = plain counts + platform-activity context (T4),
#   'cover' = collection-coverage adjusted counts + push-collection coverage context (T5c).  Chosen by the validation (see results/summary.md): raw for the five
#   collaboration activities, cover for push (the only stream with long outages: it must still predict when the push data is missing in the last weeks).
FINAL_STRIDE = 2                                           # train on every 2nd daily cutoff
FINAL_HALF_LIFE = None                                     # recency weights of the training samples (days); None = equal weights
FINAL_PARAMS = {}                                          # xgboost overrides (see src/model.XGB)
FINAL_SEEDS = 3                                            # models averaged per activity (seeds 0,1,2): same Score as one model (H8), less random variation of a single run
