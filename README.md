# BigData2026 – Assignment 2: Predicting GitHub User Activity

Student ID: 315581033

For every test user (`submission/test_input.csv` of the course repository, 4,465 users) predict whether the user performs each of the 6 target
activities (`open_pr`, `open_issue`, `issue_comment`, `pr_review`, `pr_review_comment`, `push`) on any GitHub repository during
**2026-11-22 00:00 – 11-29 00:00 (Taiwan time)** (= 2026-11-21 16:00 – 11-28 16:00 UTC).  Score = (macro-F1 + micro-F1) / 2.

## Data Source

The data used in this assignment is sourced from [GH Archive](https://www.gharchive.org), a publicly available archive of GitHub events
([GH Archive on GitHub](https://github.com/igrigorik/gharchive.org)).  This work is for educational use within the course *Big Data Analytics
Techniques and Applications* (Fall 2026).  Only the 8 tables that the course script `scripts/preprocess.py` builds from the GH Archive hour files
(cohort of 9,000 users, anonymised) are used; no other data (other event types, GitHub API, other websites) is collected.

**Scoring rule announced by the TAs.**  Some hours are missing or incomplete in GH Archive itself and cannot be back-filled.  All six activities are still
submitted, but an activity with missing data in the test week is not scored: the Score is computed over the remaining activities.  Consequences for this
solution: the push labels of poorly collected weeks are masked during training, and the validation reports the Score with and without push (`src/scenarios.py`).

## Method in short

**The model is a temporal heterogeneous graph neural network that works directly on the relational database** (`src/graph.py`, `src/gnn.py`, pure PyTorch):

* **Relationships across tables.**  Every row of the 8 tables is a node (node type = table), every foreign-key value is an edge; each of the 20 foreign keys is
  used in both directions (40 relations).  In every layer a node combines its own state with, for every relation, the sum (log-compressed) and the max of the
  states of its neighbours.  3 layers: a user node sees its own rows, the rows attached to them (e.g. reviews of others on its pull requests) and what other users do in
  the same repositories.  No hand-made joins or aggregates.
* **Time.**  The graph of a cutoff contains only the rows whose row time is before the cutoff (the tables are sorted by row time, so the snapshot is a prefix of every
  table).  Every row carries a relative-time encoding of its age at the cutoff (steps and exponential decays for 1 … 90 days); summed by the message passing these give
  the multi-scale activity counts, their maximum the recency.  Columns that are known only later (`merged_at`, `closed_at`, `state_reason`, `dismissed_at`) are used only
  if their time is before the cutoff.  The readout adds the mean state of all users of the cutoff (the activity level of the whole platform).
* **Training.**  One step = the whole snapshot of one daily cutoff (12:00 UTC, every second day); the users of a cutoff are chosen with a rule that reproduces the official
  test list; label = "≥ 1 event in the 7 days that start 4 h after the cutoff"; binary cross-entropy over the six activities, one network for all of them
  (1,119,366 parameters, 3 seeds averaged).  **Push labels of windows in which less than 80 % of the hours were collected are masked.**
  Validation is a rolling origin over time, never a random split.  The number of users flagged per activity is chosen without labels so that the expected Score is maximal (`src/metrics.py`).

A **feature-engineering baseline** (XGBoost on hand-made temporal and relational features, `scripts/predict_xgb.py`, `scripts/run_experiments.py`) is kept for comparison only.

## Repository layout

```
src/        graph.py            the database as a temporal heterogeneous graph: nodes, edges, snapshot at a cutoff, time encoding
            gnn.py              the graph neural network, training, prediction
            config.py, data.py, labels.py, pipeline.py, metrics.py, scenarios.py     task definition, tables, labels / user list, samples, Score, scoring scenarios
            features.py, relational.py, openwork.py, repo_features.py, final_model.py, model.py     the feature baseline (hand-made features + XGBoost)
            graphs.py           schema graph (Graphviz DOT)
scripts/    download_gharchive.sh  download the GH Archive hour files of a period
            predict.py             FINAL MODEL: train the graph model + predict -> output_315581033.csv, results/final_gnn_weights.pt  (--weights reloads the weights)
            run_gnn.py             rolling-origin validation of the graph model, its ablations and the table knock-out   -> results/runs/G*.json
            run_thresholds.py      threshold policies                                       -> results/thresholds.json
            reference_week.py      the reference week of the course README, scored with evaluate.py -> results/reference_week.json, reference_week_*.csv
            compare_models.py      graph model vs. feature baseline, origin by origin       -> results/model_comparison.json
            run_push_recovery.py   push after the outage of 8/28 - 9/3                       -> results/push_recovery.json
            leak_check_gnn.py      predictions at a cutoff from full vs. truncated tables   -> results/leak_check_gnn.json
            analyze_data.py        task definition, weekly rates, push gaps, graph size     -> results/data_analysis.json
            predict_xgb.py, run_experiments.py, run_baselines.py, feature_importance.py, leak_check.py, build_features.py     the feature baseline and the "copy last week" baseline
            make_figures.py, make_charts.py     schema graph and charts -> figures/
results/    every number of the report, as JSON, summary.md, and the weights of the submitted model (final_gnn_weights.pt)
figures/    schema graph, data observations
```

## How to reproduce

```bash
# 0. environment (Python 3.12).  PyTorch: install the build that fits your CUDA version first if the default wheel does not (https://pytorch.org); the CPU build is enough for step 4a
pip install -r requirements.txt

# 1. the course repository: preprocess.py, evaluate.py, submission/test_input.csv
git clone --depth 1 https://github.com/nycubigdata-2026fall/BigData2026-Assignment2.git course_repo

# 2. the GH Archive hour files 2026-06-01 00:00 UTC ... 2026-11-21 12:00 UTC (about 70 GB, a few hours)
bash scripts/download_gharchive.sh raw_gharchive 2026-06-01 "2026-11-21 12"

# 3. the 8 tables, with the course script (about 5 minutes)
python course_repo/scripts/preprocess.py --raw raw_gharchive --out preprocessed

# 4a. reproduce the submitted file from the stored weights of the submitted model (no training; a CPU is enough, about a minute)
ASG2_DATA_ROOT=course_repo ASG2_DB_DIR=preprocessed/db python -m scripts.predict --weights results/final_gnn_weights.pt --out output_315581033.csv

# 4b. or train from scratch and predict (GPU with about 16 GB memory: about 11 minutes; on a CPU it takes many hours)
ASG2_DATA_ROOT=course_repo ASG2_DB_DIR=preprocessed/db python -m scripts.predict --out output_315581033.csv
```

* **Period.**  The submitted model was built from the hour files of **2026-06-01 00:00 UTC to 2026-11-21 12:00 UTC** and the prediction cutoff
  **2026-11-21 12:00 UTC** (`FINAL_CUTOFF`; the target week starts 4 h later).  Data at or after `DATA_CAP` = 2026-11-21 12:00 UTC is ignored, so a re-run
  with more raw hours (for example after the test week) gives the same snapshots and the same training set; a warning is printed if the tables contain rows from before 2026-06-01.
  GH Archive itself has no file for 06-24 08h, 08-28 02h, 10-04 01h, 10-21 06h (UTC).
* **Determinism.**  The network does not depend on the row numbers of the tables (a graph has no node order), the seeds are fixed and the deterministic CUDA algorithms are switched on:
  two training runs on the same machine give identical weights.  On other hardware or library versions the last digits of the weights can differ, which can move a few
  borderline users; the stored weights (`results/final_gnn_weights.pt`) reproduce the submitted predictions without training (4a).
* `predict.py` takes the cutoff `2026-11-21 12:00 UTC` (or, while the data does not reach it yet, the newest 12:00 UTC; `--as-of` sets it) and trains for the horizon
  "cutoff → 2026-11-22 00:00 Taiwan time" (4 h for the final cutoff).  `--push-min-cover 0` keeps all push labels (default: 0.8).

Optional, to reproduce the report (results/ already contains the outputs):

```bash
python -m scripts.build_features --raw                      # users / labels of every validation cutoff (+ the features of the baseline)
python -m scripts.run_gnn                                   # graph model: 7 rolling origins, ablations G0-G9, final setting GF (about 20 minutes per run and seed on one GPU)
python -m scripts.run_experiments --suite all               # feature baseline: ablations and variants
python -m scripts.run_thresholds && python -m scripts.run_thresholds --model xgb && python -m scripts.run_baselines
python -m scripts.reference_week && python -m scripts.compare_models && python -m scripts.run_push_recovery
python -m scripts.analyze_data && python -m scripts.make_figures && python -m scripts.make_charts      # data observations, figures (DOT -> SVG with `dot -Tsvg` or figures/render.mjs)
# leak checks: see the docstrings of scripts/leak_check_gnn.py and scripts/leak_check.py
```

## Notes on the data (found during the analysis, details in `results/data_analysis.json`)

* The official test list is reproduced almost exactly by the rule "cohort users with ≥ 3 activity records (six event streams, event time) in the 90 days
  before 2026-11-04 12:00 UTC" (Jaccard index 0.9996: 2 of 4,465 users missing, 0 extra).  The rule and the list time were found by a grid
  search (`analyze_data.py`), they are not given by the TAs.  The users of every training cutoff are chosen with the same rule, shifted to that cutoff.
* The tables built from 2026-11-11 11:00 UTC back to 2026-06-01 are a graph of 688,343 rows (nodes) and 1,296,250 foreign-key values (edges);
  7,744 of the 9,000 cohort users appear.
* GH Archive is thinned and has gaps.  The push stream has 15 days without any push since 2026-06-01 and is almost empty from 2026-10-27 on; "collected" =
  the cohort has at least 10 pushes in the hour (`data.hourly_alive`).  The weekly positive rates of the other activities also move a lot (open_pr by a factor of
  3 between the highest and the lowest week since September).
* Rows can appear in the tables later than the event they describe: of the cohort-authored events since June, 3,584 of 20,999 issues,
  362 of 23,464 pull requests and 34 of 16,113 reviews are first seen more than 24 hours after the event
  (GH Archive gaps; the row shows up with a later comment).  The snapshot of a cutoff therefore uses the row time; the labels use the event time.
* Only the events of the 9,000 cohort users are collected, so what other users do to a cohort user's items is rarely visible: since June, 47 reviews, 30 review comments and
  67 issue comments reached a cohort author from another cohort user (`relational_events` in `results/data_analysis.json`).  This limits what multi-hop relations can add.

## Leak check: a prediction uses only data before its cutoff

`scripts/leak_check_gnn.py` rebuilds the tables with the course script from the raw hours **before** the cutoff 2026-10-03 12:00 UTC only and predicts the 4,454 test users that
exist in those tables with the **same trained weights** from the truncated tables and from the full tables.  Largest difference of a probability: 0.0270
(mean 0.00003); 0 users get a different 0/1 output (100.00% of all outputs identical).
The remaining differences come from a few rows that differ between the two table builds (a few PR rows have an author / creation time in the truncated tables but not in the full tables, and a few issue closings
are only revealed by the full tables, which the course rule allows: `closed_at` may be used if it is earlier than the prediction time).  The same check for the hand-made features of the baseline
(`scripts/leak_check.py`): identical for 4,447 of 4,454 users.

## Validation results

Rolling origin over 7 Saturdays (12:00 UTC, 2026-09-19 ... 10-31, the weekday of the final cutoff): for every origin the model is trained from scratch only on cutoffs ≤ origin − 8 days and
predicts the week that starts 4 h after the origin; the number of users flagged per activity is chosen without labels.  Every number below is generated by the scripts from `results/`
(`results/summary.md` has all tables).

**Score** = Score under the TAs' rule: for an origin whose push week was ≥ 90 % collected (2 of the 7 origins) all six activities count, for the others only the five
non-push activities; mean over the 7 origins.  **w/o push** = the five activities on all 7 origins, **with push** = the six activities on the 2 origins with a complete push week.
Changes are paired over the same 7 origins (± standard error).

**Final model** (`GF`): temporal heterogeneous GNN, 3 layers, width 64, 1,119,366 parameters, 30 epochs, 3 seeds averaged, push labels of training windows
collected < 80 % masked, label-free choice of k.  **Score 0.4395** (0.4135 w/o push, 0.4945 with push).
Feature baseline (XGBoost, `H8m`): 0.4382; graph model minus baseline +0.0013 ± 0.0073 (5/7 origins better).  The course baseline "copy last week": 0.3947.
Differences below ~0.005 are within the origin-to-origin noise; the label-free choice of k loses 0.031 to an oracle.

Main findings of the ablations: without the time encoding -0.0265 ± 0.0109 (1/7 origins better); without the cohort context -0.0139 ± 0.0044 (1/7 origins better); 1 layer instead of 3 -0.0085 ± 0.0029 (1/7 origins better).

### The reference week of the course README (2026/09/27 – 10/04, Taiwan time), scored with the official `evaluate.py`

This week is the validation origin 2026-09-26 (models trained on cutoffs ≤ origin − 8 days only, k chosen without labels; `scripts/reference_week.py`).
The users of the README table are not published; ours are the 6,085 cohort users chosen by the list rule for that week, so the comparison is indicative only
("copy last week" on our users is close to the README value).

| method | macro-F1 % | micro-F1 % | Score % |
|---|---:|---:|---:|
| Random (course README) | 19.77 | 22.74 | 21.26 |
| Predict all 0 (course README) | 0.00 | 0.00 | 0.00 |
| Copy last week (course README) | 43.01 | 56.20 | 49.61 |
| GNN without time encoding (course README) | 22.80 | 48.61 | 35.70 |
| GNN with time encoding (course README) | 50.90 | 58.74 | 54.82 |
| Copy last week, our users | 43.54 | 56.26 | 49.90 |
| Feature baseline (XGBoost), our users | 51.76 | 60.54 | 56.15 |
| **Graph model (this repository), our users** | 51.82 | 60.56 | 56.19 |

```
$ python course_repo/scripts/evaluate.py --pred results/reference_week_pred.csv --truth results/reference_week_truth.csv
Format check: no errors (6085 users)
  open_pr            F1  48.50%
  open_issue         F1  47.56%
  issue_comment      F1  46.31%
  pr_review          F1  49.48%
  pr_review_comment  F1  46.10%
  push               F1  72.99%
macro-F1: 51.82%
micro-F1: 60.56%
Score:    56.19%
```

### Graph model: ablations and variants

G0 = reference setting (1 seed); GF = final model (G0 with the seeds averaged).  Changes are paired against G0.

| run | Score (TA rule) | change vs. reference (± s.e.) | origins better | w/o push | with push | six activities, all origins | oracle-k (six, all origins) |
|---|---:|---:|---:|---:|---:|---:|---:|
| G0 graph model: 3 layers, time encoding, cohort context, sum + max | 0.4404 | reference |  | 0.4148 | 0.4954 | 0.4696 | 0.5122 |
| G1 without time encoding (rows before the cutoff, but not when) | 0.4139 | -0.0265 ± 0.0109 | 1/7 | 0.3919 | 0.4911 | 0.4390 | 0.4806 |
| G2 without cohort context in the readout | 0.4265 | -0.0139 ± 0.0044 | 1/7 | 0.3996 | 0.4842 | 0.4588 | 0.5119 |
| G3 sum aggregation only (no max) | 0.4374 | -0.0030 ± 0.0027 | 3/7 | 0.4112 | 0.4855 | 0.4671 | 0.5131 |
| G4 1 layer: a user only sees its own rows | 0.4319 | -0.0085 ± 0.0029 | 1/7 | 0.4047 | 0.4766 | 0.4627 | 0.5116 |
| G5 2 layers: + rows of others on my PRs / issues | 0.4399 | -0.0005 ± 0.0023 | 4/7 | 0.4135 | 0.4878 | 0.4687 | 0.5146 |
| G6 4 layers | 0.4394 | -0.0009 ± 0.0022 | 3/7 | 0.4134 | 0.4956 | 0.4701 | 0.5129 |
| G7 push labels of all windows (no mask) | 0.4386 | -0.0018 ± 0.0010 | 1/7 | 0.4127 | 0.4926 | 0.4717 | 0.5124 |
| G8 width 128 | 0.4376 | -0.0028 ± 0.0042 | 3/7 | 0.4106 | 0.5055 | 0.4702 | 0.5114 |
| G9 60 epochs | 0.4239 | -0.0165 ± 0.0083 | 2/7 | 0.3972 | 0.5059 | 0.4611 | 0.4948 |
| GF final graph model (G0, 3 seeds averaged) | 0.4395 | -0.0008 ± 0.0019 | 4/7 | 0.4135 | 0.4945 | 0.4693 | 0.5146 |

### Graph model vs. feature baseline, origin by origin

| origin | activities scored | graph model | feature baseline | average of both |
|---|---:|---:|---:|---:|
| 2026-09-19 | 6 | 0.4272 | 0.4641 | 0.4467 |
| 2026-09-26 | 6 | 0.5619 | 0.5615 | 0.5685 |
| 2026-10-03 | 5 | 0.4521 | 0.4558 | 0.4601 |
| 2026-10-10 | 5 | 0.4217 | 0.4168 | 0.4263 |
| 2026-10-17 | 5 | 0.4070 | 0.4010 | 0.4122 |
| 2026-10-24 | 5 | 0.3833 | 0.3699 | 0.3818 |
| 2026-10-31 | 5 | 0.4235 | 0.3983 | 0.4198 |
| mean | | 0.4395 | 0.4382 | 0.4451 |

Graph model minus feature baseline: +0.0013 ± 0.0073 (5/7 origins better).  Average of both minus graph model: +0.0055 ± 0.0028.  Spearman correlation of the probabilities per activity: {'open_pr': 0.9571, 'open_issue': 0.93, 'issue_comment': 0.941, 'pr_review': 0.8927, 'pr_review_comment': 0.8963, 'push': 0.9571}; Jaccard overlap of the flagged users: {'open_pr': 0.66, 'open_issue': 0.6462, 'issue_comment': 0.6226, 'pr_review': 0.6396, 'pr_review_comment': 0.6269, 'push': 0.7773}.

### Which tables does the graph model need?  Knock-out of one table at prediction time

The trained final models predict every validation week again from a snapshot in which all foreign-key edges of one table are removed (no retraining).
Change of the F1 of every activity in points (mean of the 7 origins) and of the Score:

| table removed | open_pr | open_issue | issue_comment | pr_review | pr_review_comment | push | Score |
|---|---:|---:|---:|---:|---:|---:|---:|
| Repositories | -0.5 | -0.3 | -0.1 | +0.9 | +0.4 | -0.4 | +0.0016 |
| PullRequests | -12.8 | -1.2 | -1.6 | -0.8 | -1.6 | +0.5 | -0.0359 |
| Issues | +0.0 | -28.1 | -1.4 | +0.3 | -0.4 | +0.1 | -0.0432 |
| IssueComments | -1.0 | -0.8 | -5.4 | +0.2 | +0.0 | +0.0 | -0.0109 |
| PRReviews | -0.1 | -0.7 | +0.0 | -6.1 | -0.5 | -0.0 | -0.0116 |
| PRReviewComments | -0.1 | +0.4 | +0.3 | -0.4 | -2.1 | +0.0 | -0.0010 |
| Pushes | -3.2 | -2.5 | -6.2 | -3.4 | -5.6 | -7.5 | -0.0534 |

### Final model, per origin

| origin | push week collected | Score, 6 activities | Score, 5 activities (w/o push) | Score (TA rule) |
|---|---:|---:|---:|---:|
| 2026-09-19 | 0.952 | 0.4272 | 0.3304 | 0.4272 |
| 2026-09-26 | 1.000 | 0.5619 | 0.4765 | 0.5619 |
| 2026-10-03 | 0.744 | 0.5419 | 0.4521 | 0.4521 |
| 2026-10-10 | 0.589 | 0.5201 | 0.4217 | 0.4217 |
| 2026-10-17 | 0.685 | 0.4963 | 0.4070 | 0.4070 |
| 2026-10-24 | 0.250 | 0.4041 | 0.3833 | 0.3833 |
| 2026-10-31 | 0.113 | 0.3334 | 0.4235 | 0.4235 |

### Final model, per activity

| activity | true rate | flagged | AP | F1 |
|---|---:|---:|---:|---:|
| open_pr | 0.072 | 0.118 | 0.438 | 0.433 |
| open_issue | 0.043 | 0.055 | 0.457 | 0.452 |
| issue_comment | 0.050 | 0.093 | 0.416 | 0.400 |
| pr_review | 0.052 | 0.092 | 0.428 | 0.412 |
| pr_review_comment | 0.030 | 0.069 | 0.381 | 0.362 |
| push | 0.395 | 0.415 | 0.787 | 0.710 |

### How many users to flag (final model)

| policy | Score (TA rule) | w/o push | with push |
|---|---:|---:|---:|
| expected | 0.4395 | 0.4135 | 0.4945 |
| five | 0.4356 | 0.4095 | 0.4913 |
| per_label | 0.4354 | 0.4097 | 0.4892 |
| robust | 0.4382 | 0.4118 | 0.4933 |
| p>=0.5 | 0.3763 | 0.3520 | 0.4823 |
| p>=0.3 | 0.4384 | 0.4114 | 0.4959 |
| oracle | 0.4706 | 0.4429 | 0.5623 |

### Baselines

| baseline | Score (TA rule) | w/o push | six activities, all origins |
|---|---:|---:|---:|
| copy_last_week | 0.3947 | 0.3649 | 0.4432 |
| all_zero | 0.0000 | 0.0000 | 0.0000 |

### Push after a collection outage

The push stream of GH Archive is almost empty from 2026-10-27 on (it was also down on 8/28 – 9/3).  The TAs do not score an activity with missing data in the test week,
so there are two cases: push is still missing in the test week (then only the five other activities count) or the data is back (then push counts, and 24%–42%
of the users push in a fully collected week since August).  The push labels of poorly collected training windows are therefore masked, and the push head also sees how much of the
last 3 / 7 / 28 days was collected.  Two checks:

* the real outage → recovery of 8/28 – 9/5 (origins 2026-09-05 and 09-12):

| origin | model | push labels used for training | true rate | flagged | F1 | AP | push collected in the last 7 days before the origin |
|---|---|---|---:|---:|---:|---:|---:|
| 2026-09-05 | graph model | all windows | 0.217 | 0.231 | 0.560 | 0.596 | 0.23 |
| 2026-09-05 | graph model | windows collected >= 80% | 0.217 | 0.303 | 0.571 | 0.596 | 0.23 |
| 2026-09-12 | graph model | all windows | 0.340 | 0.246 | 0.614 | 0.731 | 0.83 |
| 2026-09-12 | graph model | windows collected >= 80% | 0.340 | 0.218 | 0.584 | 0.714 | 0.83 |
| 2026-09-05 | feature baseline, push-only, raw base | all windows | 0.217 | 0.118 | 0.475 | 0.603 | 0.23 |
| 2026-09-05 | feature baseline, push-only, raw base | windows collected >= 80% | 0.217 | 0.226 | 0.560 | 0.603 | 0.23 |
| 2026-09-05 | feature baseline, push-only, cover base | all windows | 0.217 | 0.160 | 0.528 | 0.612 | 0.23 |
| 2026-09-05 | feature baseline, push-only, cover base | windows collected >= 80% | 0.217 | 0.246 | 0.566 | 0.610 | 0.23 |
| 2026-09-12 | feature baseline, push-only, raw base | all windows | 0.340 | 0.233 | 0.611 | 0.735 | 0.83 |
| 2026-09-12 | feature baseline, push-only, raw base | windows collected >= 80% | 0.340 | 0.241 | 0.621 | 0.744 | 0.83 |
| 2026-09-12 | feature baseline, push-only, cover base | all windows | 0.340 | 0.225 | 0.600 | 0.739 | 0.83 |
| 2026-09-12 | feature baseline, push-only, cover base | windows collected >= 80% | 0.340 | 0.241 | 0.620 | 0.745 | 0.83 |

* the interim cutoff (no push for two weeks): what the graph model predicts for the test users with and without the mask (the validation origins flag 35%–56% of the users for push):

| push labels used for training | mean predicted push probability | share of users flagged for push |
|---|---:|---:|
| all windows | 0.085 | 0.093 |
| windows collected >= 80 % (final) | 0.401 | 0.531 |

## Feature baseline (XGBoost on hand-made features): ablations

Kept for comparison: the same ideas (multi-scale counts, platform context, relations along foreign keys) written as hand-made features.

### Temporal ablation

| run | features | Score (TA rule) | change vs. reference (± s.e.) | origins better | w/o push | with push | six activities, all origins | oracle-k (six, all origins) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T0 counts of the last 7 days only | 7 | 0.4047 | reference |  | 0.3737 | 0.4908 | 0.4493 | 0.4716 |
| T1 + multi-window counts (1-90 days) | 56 | 0.4272 | +0.0225 ± 0.0060 | 6/7 | 0.4042 | 0.4987 | 0.4667 | 0.5016 |
| T2 + recency (days since last, EWMA, active days) | 84 | 0.4248 | -0.0025 ± 0.0019 | 3/7 | 0.4009 | 0.4928 | 0.4647 | 0.5004 |
| T3 + weekly lags and trend (= temporal block) | 147 | 0.4173 | -0.0075 ± 0.0055 | 2/7 | 0.3947 | 0.4994 | 0.4583 | 0.4968 |
| T4 + global activity-volume context | 159 | 0.4360 | +0.0188 ± 0.0064 | 7/7 | 0.4114 | 0.5108 | 0.4800 | 0.5107 |
| T5a + coverage-adjusted features (push gaps) | 159 | 0.4287 | -0.0074 ± 0.0034 | 2/7 | 0.4049 | 0.5028 | 0.4673 | 0.5098 |
| T5b + drop push labels of partly collected weeks | 159 | 0.4298 | +0.0011 ± 0.0005 | 6/7 | 0.4060 | 0.5030 | 0.4650 | 0.5096 |
| T5c + push-collection coverage as context (keep all labels) | 162 | 0.4193 | -0.0094 ± 0.0063 | 2/7 | 0.3946 | 0.5017 | 0.4587 | 0.5090 |

### Relational ablation, all on top of T5c

| run | features | Score (TA rule) | change vs. reference (± s.e.) | origins better | w/o push | with push | six activities, all origins | oracle-k (six, all origins) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T5c + push-collection coverage as context (keep all labels) | 162 | 0.4193 | reference |  | 0.3946 | 0.5017 | 0.4587 | 0.5090 |
| R1 FK lifecycle | 229 | 0.4203 | +0.0010 ± 0.0011 | 6/7 | 0.3957 | 0.5035 | 0.4589 | 0.5094 |
| R2 work in progress | 172 | 0.4199 | +0.0006 ± 0.0021 | 4/7 | 0.3954 | 0.5043 | 0.4589 | 0.5104 |
| R3 repo multi-hop | 180 | 0.4189 | -0.0004 ± 0.0018 | 4/7 | 0.3951 | 0.5041 | 0.4581 | 0.5093 |
| R4 R1+R2 | 239 | 0.4204 | +0.0011 ± 0.0021 | 5/7 | 0.3953 | 0.5036 | 0.4591 | 0.5100 |
| R5 R1+R2+R3 (all blocks) | 257 | 0.4255 | +0.0062 ± 0.0027 | 6/7 | 0.4006 | 0.5062 | 0.4644 | 0.5133 |
| R6 T4 base (plain counts + activity-volume context) + all blocks | 254 | 0.4350 | +0.0095 ± 0.0070 | 5/7 | 0.4106 | 0.5134 | 0.4781 | 0.5121 |

### Variants

H0-H8 use all push labels; R4m80 / R4m90 / H8m drop the push labels of training windows collected < 80 % / 90 % (H8m = the baseline used in the comparisons: 3 seeds).

| run | features | Score (TA rule) | change vs. reference (± s.e.) | origins better | w/o push | with push | six activities, all origins | oracle-k (six, all origins) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| H0 reference (depth 6, 300 trees, lr 0.05, stride 2) | 254/257 | 0.4349 | reference |  | 0.4107 | 0.5120 | 0.4746 | 0.5124 |
| H1 depth 4, 400 trees | 254/257 | 0.4388 | +0.0039 ± 0.0022 | 5/7 | 0.4148 | 0.5087 | 0.4769 | 0.5149 |
| H2 depth 8, 250 trees | 254/257 | 0.4345 | -0.0004 ± 0.0016 | 3/7 | 0.4106 | 0.5137 | 0.4746 | 0.5133 |
| H3 lr 0.03, 600 trees | 254/257 | 0.4392 | +0.0043 ± 0.0020 | 6/7 | 0.4148 | 0.5163 | 0.4774 | 0.5125 |
| H4 stronger regularisation | 254/257 | 0.4258 | -0.0091 ± 0.0063 | 3/7 | 0.4010 | 0.5108 | 0.4671 | 0.5150 |
| H5 every daily cutoff (stride 1) | 254/257 | 0.4337 | -0.0012 ± 0.0031 | 3/7 | 0.4069 | 0.5105 | 0.4754 | 0.5120 |
| H6 recency weights, half-life 30 d | 254/257 | 0.4356 | +0.0007 ± 0.0008 | 5/7 | 0.4116 | 0.5132 | 0.4766 | 0.5118 |
| H8 average of 3 seeds | 254/257 | 0.4388 | +0.0039 ± 0.0009 | 7/7 | 0.4149 | 0.5142 | 0.4774 | 0.5138 |
| R4m80 final features, push labels of windows collected < 80% dropped | 254/257 | 0.4343 | -0.0006 ± 0.0008 | 2/7 | 0.4109 | 0.5105 | 0.4734 | 0.5123 |
| R4m90 final features, push labels of windows collected < 90% dropped | 254/257 | 0.4369 | +0.0020 ± 0.0032 | 1/7 | 0.4110 | 0.5201 | 0.4767 | 0.5125 |
| H8m final model (final features, push labels of windows collected < 80% dropped, 3 seeds) | 254/257 | 0.4382 | -0.0006 ± 0.0006 | 2/7 | 0.4148 | 0.5128 | 0.4766 | 0.5140 |
