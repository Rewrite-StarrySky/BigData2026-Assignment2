"""Data observations used in the report  ->  results/data_analysis.json

    python -m scripts.analyze_data

1. the test list is reproduced by the rule "cohort users with >= ELIGIBLE_MIN_RECORDS activity records in the 90 days before the list time"
   (the rule and the list time were found by a grid search over list time / look-back / minimum records, see `search_rule`)
2. data period and size of the preprocessed tables
3. weekly positive rates of the eligible population and the share of every label window in which pushes were collected
4. collection gaps of the push stream (hours with fewer than PUSH_ALIVE_MIN pushes) and daily event counts of all six activities
5. rows that appear in the database later than the event they describe;  6. number of events that reach a cohort user through foreign keys
"""
import json, os
from pathlib import Path
import numpy as np
import pandas as pd
from src.config import *
from src.data import load_tables, event_streams, release_time, TABLES, ROW_TIME
from src.labels import eligible_users, window_labels, window_coverage
from src.features import Daily


def search_rule(ev, U, test):
    """Grid search: which (list time, look-back, minimum records) makes the rule closest to the official list (Jaccard index)?"""
    cohort = U.loc[U.is_cohort, "user_id"].values
    want = set(U.set_index("github_user_id").user_id.loc[test.UserID].astype(int))
    allev = pd.concat([e[["u", "t"]] for e in ev.values()])
    best = []
    for L in pd.date_range("2026-10-25 00:00", "2026-11-08 00:00", freq="12h"):
        for days in (28, 60, 90):
            for k in (1, 3, 5, 10):
                m = allev[(allev.t > L - pd.Timedelta(days=days)) & (allev.t <= L)]
                c = m.u.value_counts(); got = set(c[c >= k].index) & set(cohort)
                best.append((len(got & want) / len(got | want), str(L), days, k, len(got)))
    best.sort(reverse=True)
    return [dict(jaccard=round(j, 5), list_time=L, lookback_days=d, min_records=k, n=n) for j, L, d, k, n in best[:5]]


def main():
    T = load_tables(); ev = event_streams(T); U = T["Users"]
    D = Daily(T, ev); end = release_time(T)
    cohort = U.loc[U.is_cohort, "user_id"].values
    g2r = U.set_index("github_user_id").user_id
    out = {}

    test = pd.read_csv(TEST_INPUT)
    want = set(g2r.loc[test.UserID].astype(int)); got = set(eligible_users(ev, cohort, TEST_LIST_TIME))
    out["test_list"] = dict(n_official=len(test), n_rebuilt=len(got), identical=bool(got == want), missing=len(want - got), extra=len(got - want),
                            jaccard=round(len(got & want) / len(got | want), 5), all_test_users_are_cohort=bool(want <= set(cohort)),
                            rule=f">= {ELIGIBLE_MIN_RECORDS} records in the {ELIGIBLE_LOOKBACK_DAYS} days before {TEST_LIST_TIME} UTC", search_top5=search_rule(ev, U, test))

    raw = Path(os.environ.get("ASG2_RAW_DIR", ROOT.parent / "raw_gharchive"))
    have = {f.name for f in raw.glob("*.json.gz")} if raw.exists() else set()
    missing = [f"{t:%m-%d} {t.hour:02d}h" for t in pd.date_range("2026-06-01", end - pd.Timedelta(hours=1), freq="h") if f"{t:%Y-%m-%d}-{t.hour}.json.gz" not in have] if have else []
    out["data"] = dict(missing_hour_files=missing, first_event_hour="2026-06-01 00:00 UTC (first downloaded GH Archive file)", end=str(end),
                       rows={n: int(len(T[n])) for n in TABLES}, cohort_users_seen=int(U.is_cohort.sum()))

    weekly = []
    for c in pd.date_range(FIRST_CUTOFF, end, freq="D"):
        if c.dayofweek != FINAL_CUTOFF.dayofweek:
            continue
        start = c + pd.Timedelta(hours=FINAL_GAP_HOURS)
        if start + pd.Timedelta(days=WINDOW_DAYS) > end:
            continue
        elig = eligible_users(ev, cohort, start - LIST_AGE)
        L = window_labels(ev, elig, start)
        weekly.append(dict(cutoff=str(c), n=int(len(elig)), push_window_coverage=round(window_coverage(D.alive, start), 3), **L.mean().round(4).to_dict()))
    out["weekly_rates"] = weekly

    alive = pd.Series(D.alive, index=DAY0 + pd.to_timedelta(np.arange(NB * 24), unit="h"))
    hours_per_day = alive.groupby(alive.index.floor("D")).sum()
    hours_per_day = hours_per_day[(hours_per_day.index >= "2026-06-01") & (hours_per_day.index < end.floor("D"))]
    out["push_gaps"] = dict(note=f"UTC calendar days; value = hours (of 24) in which the cohort has at least {PUSH_ALIVE_MIN} pushes",
                            days=int(len(hours_per_day)), zero_days=[str(d.date()) for d in hours_per_day.index[hours_per_day == 0]],
                            days_below_20_hours=int((hours_per_day < 20).sum()), daily_hours={str(d.date()): int(v) for d, v in hours_per_day.items()})
    cnt = {k: ev[k].t.dt.floor("D").value_counts() for k in LABELS}
    out["daily_counts"] = {k: {str(d.date()): int(v) for d, v in s.sort_index().loc["2026-06-01":].items()} for k, s in cnt.items()}
    # 5 -- rows that appear in the database later than the event they describe (why features use the time a row exists, see data.event_streams) ----
    coh_ids = set(cohort.tolist()); late = {}
    for k, df, uc, tc in [("open_pr", T["PullRequests"], "author_id", "created_at"), ("open_issue", T["Issues"][~T["Issues"].is_pull_request], "author_id", "created_at"),
                          ("pr_review", T["PRReviews"], "reviewer_id", "submitted_at")]:
        d = df[df[uc].isin(coh_ids) & (df[tc] >= DAY0)]; lag = (d.first_seen_at - d[tc]).dt.total_seconds() / 3600
        late[k] = dict(events=int(len(d)), seen_later_than_1h=int((lag > 1).sum()), seen_later_than_24h=int((lag > 24).sum()))
    out["late_rows"] = late
    # 6 -- how many events reach a cohort user THROUGH foreign keys from other users (relational streams, whole period) --------------------------------
    from src.relational import relational_events
    cohort_rows = set(cohort.tolist())
    out["relational_events"] = {name: int(np.isin(np.asarray(u)[~pd.isna(np.asarray(u, dtype=float))].astype(np.int64), cohort).sum()) for name, u, t in relational_events(T)}
    # 7 -- size of the graph the model works on: rows = nodes, non-null foreign-key values = edges (each used in both directions)
    from src.graphs import FKS
    out["graph"] = dict(nodes=int(sum(len(T[n]) for n in TABLES)), foreign_keys=len(FKS), relations=2 * len(FKS),
                        foreign_key_values=int(sum(T[c][col].notna().sum() for c, col, _ in FKS)),
                        per_foreign_key={f"{c}.{col}": int(T[c][col].notna().sum()) for c, col, _ in FKS})
    json.dump(out, open(RESULTS_DIR / "data_analysis.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("daily_counts", "weekly_rates", "push_gaps")}, indent=1)[:3000])
    print(pd.DataFrame(weekly).to_string(index=False))


if __name__ == "__main__":
    main()
