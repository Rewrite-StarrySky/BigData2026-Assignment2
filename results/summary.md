# Experiment summary

Every run is evaluated on 7 rolling origins (Saturdays 12:00 UTC, 9/19 ... 10/31): training only on cutoffs <= origin - 8 days, prediction of the week that starts 4 h after the origin, number of users flagged per activity chosen without labels ("oracle-k" = chosen with the true labels, upper bound).

Scoring rule announced by the TAs: an activity with missing data in the test week is not scored; Score = (macro-F1 + micro-F1)/2 over the remaining activities. An origin counts as 'push fully collected' if >= 90% of the hours of its label week have pushes (2 of 7 origins).

* **Score (TA rule)**: mean over the 7 origins of the Score on all 6 activities (push week fully collected) or on the 5 other activities (otherwise)
* **w/o push**: 5 activities, all 7 origins;  **with push**: 6 activities, the 2 origins with a fully collected push week
* **six activities, all origins**: all six activities on every origin (push gaps included in the labels)

## Graph model: ablations and variants (scripts/run_gnn.py)

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

## Graph model vs. feature baseline (XGBoost), origin by origin (Score, TA rule)

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

## Graph model: knock-out of one table at prediction time (change of the F1 in points, mean of the 7 origins; Score = TA rule)

| table removed | open_pr | open_issue | issue_comment | pr_review | pr_review_comment | push | Score |
|---|---:|---:|---:|---:|---:|---:|---:|
| Repositories | -0.5 | -0.3 | -0.1 | +0.9 | +0.4 | -0.4 | +0.0016 |
| PullRequests | -12.8 | -1.2 | -1.6 | -0.8 | -1.6 | +0.5 | -0.0359 |
| Issues | +0.0 | -28.1 | -1.4 | +0.3 | -0.4 | +0.1 | -0.0432 |
| IssueComments | -1.0 | -0.8 | -5.4 | +0.2 | +0.0 | +0.0 | -0.0109 |
| PRReviews | -0.1 | -0.7 | +0.0 | -6.1 | -0.5 | -0.0 | -0.0116 |
| PRReviewComments | -0.1 | +0.4 | +0.3 | -0.4 | -2.1 | +0.0 | -0.0010 |
| Pushes | -3.2 | -2.5 | -6.2 | -3.4 | -5.6 | -7.5 | -0.0534 |

## Threshold policies (final graph model)

| policy | Score (TA rule) | w/o push | with push |
|---|---:|---:|---:|
| expected | 0.4395 | 0.4135 | 0.4945 |
| five | 0.4356 | 0.4095 | 0.4913 |
| per_label | 0.4354 | 0.4097 | 0.4892 |
| robust | 0.4382 | 0.4118 | 0.4933 |
| p>=0.5 | 0.3763 | 0.3520 | 0.4823 |
| p>=0.3 | 0.4384 | 0.4114 | 0.4959 |
| oracle | 0.4706 | 0.4429 | 0.5623 |

## Baselines (course README), same origins

| baseline | Score (TA rule) | w/o push | six activities, all origins |
|---|---:|---:|---:|
| copy_last_week | 0.3947 | 0.3649 | 0.4432 |
| all_zero | 0.0000 | 0.0000 | 0.0000 |

## Final model, per origin

| origin | push week collected | Score, 6 activities | Score, 5 activities (w/o push) | Score (TA rule) |
|---|---:|---:|---:|---:|
| 2026-09-19 | 0.952 | 0.4272 | 0.3304 | 0.4272 |
| 2026-09-26 | 1.000 | 0.5619 | 0.4765 | 0.5619 |
| 2026-10-03 | 0.744 | 0.5419 | 0.4521 | 0.4521 |
| 2026-10-10 | 0.589 | 0.5201 | 0.4217 | 0.4217 |
| 2026-10-17 | 0.685 | 0.4963 | 0.4070 | 0.4070 |
| 2026-10-24 | 0.250 | 0.4041 | 0.3833 | 0.3833 |
| 2026-10-31 | 0.113 | 0.3334 | 0.4235 | 0.4235 |

## Final model, per activity (mean of the 7 origins, label-free k; push: the 2 origins with a fully collected push week)

| activity | true rate | flagged | AP | F1 |
|---|---:|---:|---:|---:|
| open_pr | 0.072 | 0.118 | 0.438 | 0.433 |
| open_issue | 0.043 | 0.055 | 0.457 | 0.452 |
| issue_comment | 0.050 | 0.093 | 0.416 | 0.400 |
| pr_review | 0.052 | 0.092 | 0.428 | 0.412 |
| pr_review_comment | 0.030 | 0.069 | 0.381 | 0.362 |
| push | 0.395 | 0.415 | 0.787 | 0.710 |

## Push stress test: interim cutoff 2026-11-10 12:00 UTC (the push stream has almost no data since 10/27), graph model

| push labels used for training | mean predicted push probability | share of users flagged for push |
|---|---:|---:|
| all windows | 0.085 | 0.093 |
| windows collected >= 80 % (final) | 0.401 | 0.531 |

## Push after an outage: origins 2026-09-05 and 2026-09-12 (push stream down 8/28 - 9/3, back on 9/5)

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

# Feature baseline (XGBoost on hand-made features, scripts/run_experiments.py)

## Temporal ablation of the feature baseline

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

## Relational ablation of the feature baseline, all on top of T5c

| run | features | Score (TA rule) | change vs. reference (± s.e.) | origins better | w/o push | with push | six activities, all origins | oracle-k (six, all origins) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| T5c + push-collection coverage as context (keep all labels) | 162 | 0.4193 | reference |  | 0.3946 | 0.5017 | 0.4587 | 0.5090 |
| R1 FK lifecycle | 229 | 0.4203 | +0.0010 ± 0.0011 | 6/7 | 0.3957 | 0.5035 | 0.4589 | 0.5094 |
| R2 work in progress | 172 | 0.4199 | +0.0006 ± 0.0021 | 4/7 | 0.3954 | 0.5043 | 0.4589 | 0.5104 |
| R3 repo multi-hop | 180 | 0.4189 | -0.0004 ± 0.0018 | 4/7 | 0.3951 | 0.5041 | 0.4581 | 0.5093 |
| R4 R1+R2 | 239 | 0.4204 | +0.0011 ± 0.0021 | 5/7 | 0.3953 | 0.5036 | 0.4591 | 0.5100 |
| R5 R1+R2+R3 (all blocks) | 257 | 0.4255 | +0.0062 ± 0.0027 | 6/7 | 0.4006 | 0.5062 | 0.4644 | 0.5133 |
| R6 T4 base (plain counts + activity-volume context) + all blocks | 254 | 0.4350 | +0.0095 ± 0.0070 | 5/7 | 0.4106 | 0.5134 | 0.4781 | 0.5121 |

## Variants of the feature baseline

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

