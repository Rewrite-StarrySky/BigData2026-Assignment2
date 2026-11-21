"""The feature sets of the final model.

Both start from the same blocks (FK lifecycle / work in progress / repository multi-hop, built on the coverage-adjusted daily matrices) but differ in the
temporal part ("base"):

  raw    plain window counts + cohort-level activity context (no collection-coverage adjustment)                         -> run T4 of the ablation
  cover  collection-coverage adjusted counts + cohort context incl. the push-collection coverage of the last 3 / 7 / 28 days -> run T5c

config.FINAL_BASE says which base every activity uses.
"""
import pandas as pd
from .config import FINAL_BLOCKS, FINAL_BASE
from .features import global_context


def feature_sets(B_adj, B_raw, cutoffs, daily_adj):
    """dict base name -> feature DataFrame for the rows of the sample blocks B_adj (adjusted context) / B_raw (un-adjusted temporal block)."""
    out = {}
    blocks = [B_adj[b] for b in FINAL_BLOCKS]
    if "cover" in FINAL_BASE.values():
        out["cover"] = pd.concat([B_adj["temporal"], global_context(B_adj["temporal"], cutoffs, daily_adj)] + blocks, axis=1)
    if "raw" in FINAL_BASE.values():
        out["raw"] = pd.concat([B_raw["temporal"], global_context(B_raw["temporal"], cutoffs)] + blocks, axis=1)
    return out
