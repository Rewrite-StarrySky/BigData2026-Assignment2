"""Charts of the report, drawn from results/*.json (nothing is typed in by hand)  ->  figures/*.svg

    python -m scripts.make_charts
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from src.config import *

FIG = ROOT / "figures"
# validated categorical palette (light surface), fixed order; text always uses the ink tokens
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8985", "#e4e3df", "#fcfcfb"
plt.rcParams.update({"font.family": ["Liberation Sans", "Arial", "DejaVu Sans"], "svg.fonttype": "none", "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.axisbelow": True, "font.size": 9})
NAMES = {"open_pr": "open_pr", "open_issue": "open_issue", "issue_comment": "issue_comment", "pr_review": "pr_review", "pr_review_comment": "pr_review_comment", "push": "push"}


def data_observations():
    A = json.load(open(RESULTS_DIR / "data_analysis.json"))
    W = pd.DataFrame(A["weekly_rates"]); W["cutoff"] = pd.to_datetime(W.cutoff)
    hours = pd.Series(A["push_gaps"]["daily_hours"]); hours.index = pd.to_datetime(hours.index)
    fig = plt.figure(figsize=(10.0, 3.55))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.12, 1], height_ratios=[1, 1], hspace=0.34, wspace=0.17, left=0.058, right=0.995, top=0.93, bottom=0.115)
    top, bot, right = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[:, 1])

    collab = [("open_pr", BLUE), ("open_issue", ORANGE), ("issue_comment", AQUA), ("pr_review", MAGENTA), ("pr_review_comment", VIOLET)]
    for k, c in collab:
        top.plot(W.cutoff, W[k] * 100, color=c, lw=1.6, solid_capstyle="round", label=k)
    top.set_ylabel("% of users", fontsize=8.5); top.set_xticklabels([]); top.set_title("Weekly positive rate of the eligible users: five collaboration activities", loc="left", fontsize=9.2, color=INK, pad=4)
    top.legend(ncol=5, frameon=False, fontsize=7.6, loc="upper left", handlelength=1.2, columnspacing=0.9, borderaxespad=0.0)
    top.set_ylim(0, W[[k for k, _ in collab]].max().max() * 100 * 1.32)
    bot.plot(W.cutoff, W.push * 100, color=GREEN, lw=1.9, solid_capstyle="round")
    low = W[W.push_window_coverage < PUSH_CLEAN_MIN]
    for _, r in low.iterrows():                                                      # weeks whose push label window was collected < 90 % (push would not be scored there)
        bot.axvspan(r.cutoff, r.cutoff + pd.Timedelta(days=7), color=ORANGE, alpha=0.16, lw=0)
    bot.scatter(low.cutoff, low.push * 100, s=22, color=ORANGE, zorder=3, edgecolor=SURFACE, linewidth=1.2)
    bot.set_ylabel("% of users", fontsize=8.5); bot.set_title("push", loc="left", fontsize=9.2, color=INK, pad=4)
    bot.legend(handles=[Line2D([0], [0], color=GREEN, lw=1.9), Patch(color=ORANGE, alpha=0.4, lw=0)], labels=["weekly rate", "push not fully collected (< 90 %)"],
               frameon=False, fontsize=7.4, loc="upper right", borderaxespad=0.1)
    for a in (top, bot):
        a.set_xlim(W.cutoff.min() - pd.Timedelta(days=3), W.cutoff.max() + pd.Timedelta(days=9))
        if a is bot: a.set_ylim(0, W.push.max() * 100 * 1.25)
        a.tick_params(length=0, labelsize=8)
    bot.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b")); bot.xaxis.set_major_locator(matplotlib.dates.MonthLocator())
    bot.set_xlabel("week of the cutoff (Saturday); label = the 7 days that start 4 h later", fontsize=7.6, color=MUTED)

    ok = hours >= 0.8 * 24
    right.bar(hours.index[ok], hours[ok], width=0.8, color=BLUE, label="≥ 80 % of the hours with ≥ 10 pushes")
    right.bar(hours.index[~ok], hours[~ok], width=0.8, color=ORANGE, label="< 80 %: collection gap")
    right.set_ylim(0, 34); right.set_yticks([0, 6, 12, 18, 24]); right.set_ylabel("hours of the UTC day with ≥ 10 pushes", fontsize=8.5)
    right.set_title("Push collection per day (hours with at least 10 pushes of the cohort)", loc="left", fontsize=9.2, color=INK, pad=4)
    right.legend(frameon=False, fontsize=7.6, loc="upper left", ncol=2, borderaxespad=0.0, columnspacing=1.2)
    right.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %d")); right.xaxis.set_major_locator(matplotlib.dates.MonthLocator())
    right.set_xlim(hours.index.min() - pd.Timedelta(days=1), hours.index.max() + pd.Timedelta(days=8)); right.tick_params(length=0, labelsize=8)
    zero = hours[hours == 0]
    runs, cur = [], []                                                              # longest run of consecutive days without any push
    for d in zero.index:
        if cur and (d - cur[-1]).days == 1:
            cur.append(d)
        else:
            cur = [d]; runs.append(cur)
    longest = max(runs, key=len)
    mid = longest[len(longest) // 2]
    right.plot([mid, mid], [2.2, 27.5], color=MUTED, lw=0.9)                          # leader line through the empty columns of the gap
    right.text(mid + pd.Timedelta(days=4), 29.0, f"{len(longest)} days in a row without any push ({longest[0]:%m/%d} - {longest[-1]:%m/%d})", fontsize=8, color=INK2, ha="right", va="center")
    fig.savefig(FIG / "data_observations.svg"); plt.close(fig)
    return dict(zero_days=len(zero), low_weeks=len(low))


if __name__ == "__main__":
    print(data_observations())
