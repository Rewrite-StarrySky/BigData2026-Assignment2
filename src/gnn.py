"""Temporal heterogeneous graph neural network on the relational database (the model of this solution).

Input: the snapshot of the 8 tables before a cutoff (src/graph.py).  Every row is a node, every foreign key an edge (both directions).
    encoder   one linear layer per table: row features (+ relative-time encoding) -> d
    L layers  for every node: its own state + for every relation the SUM (log-compressed) and the MAX of the states of its neighbours
              (sum = "how many rows, how recent" -> counts; max = "the most recent / strongest row" -> recency), LayerNorm, ReLU, residual
    readout   state of the user node  +  mean state of all users of the cutoff (cohort context: the activity level of the whole platform)
    heads     5 logits (open_pr, open_issue, issue_comment, pr_review, pr_review_comment)  and a push logit that also sees the push-collection
              coverage of the last 3 / 7 / 28 days (the push stream of GH Archive has outages)
One model predicts the six activities.  A training step is one cutoff: the whole snapshot is passed through the network once and the loss is the
binary cross-entropy of all users of that cutoff (push labels of poorly collected windows are masked).
"""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")       # needed by the deterministic CUDA algorithms (must be set before CUDA starts)
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import LABELS
from .data import TABLES

GNN = dict(d=64, layers=3, dropout=0.1, lr=2e-3, weight_decay=1e-5, epochs=30, time_enc=True, context=True, max_agg=True)      # default hyper-parameters


def torch_device():
    want = os.environ.get("ASG2_DEVICE")
    if want in ("cpu", "cuda"):
        return want
    return "cuda" if torch.cuda.is_available() else "cpu"


class HeteroTemporalGNN(nn.Module):
    def __init__(self, in_dim, rel, d=64, layers=3, dropout=0.1, n_cov=3, context=True, max_agg=True, **_):
        super().__init__()
        self.rel, self.d, self.L, self.p, self.context, self.max_agg = rel, d, layers, dropout, context, max_agg
        self.enc = nn.ModuleDict({t: nn.Linear(in_dim[t], d) for t in TABLES})
        k = 2 if max_agg else 1
        self.self_lin = nn.ModuleList([nn.ModuleDict({t: nn.Linear(d, d) for t in TABLES}) for _ in range(layers)])
        self.rel_lin = nn.ModuleList([nn.ModuleDict({name.replace(".", "__"): nn.Linear(k * d, d, bias=False) for name, _, _ in rel}) for _ in range(layers)])
        self.norm = nn.ModuleList([nn.ModuleDict({t: nn.LayerNorm(d) for t in TABLES}) for _ in range(layers)])
        r = d * (2 if context else 1)
        self.head = nn.Sequential(nn.Linear(r, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, len(LABELS) - 1))
        self.head_push = nn.Sequential(nn.Linear(r + n_cov, d), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d, 1))

    def needed(self):
        """Node types whose state is needed after layer l so that the user states of the last layer are correct (saves work in the last layers)."""
        need = [{"Users"}]
        for _ in range(self.L - 1):
            need.insert(0, need[0] | {s for _, s, d in self.rel if d in need[0]})
        return [[t for t in TABLES if t in n] for n in need]                                     # fixed order: the order of a Python set differs between processes,
        #                                                                                          and the dropout masks are drawn in the order the node types are visited

    def forward(self, x, edges, user_rows, cov):
        h = {t: F.relu(self.enc[t](x[t])) for t in TABLES}                                       # states are >= 0, so a sum over neighbours is count-like
        for l, need in enumerate(self.needed()):
            out = {t: self.self_lin[l][t](h[t]) for t in need}
            for name, s, dst in self.rel:
                if dst not in need:
                    continue
                si, di = edges[name]
                if si.numel() == 0:
                    continue
                m = h[s][si]; n = h[dst].shape[0]
                agg = torch.log1p(torch.zeros(n, self.d, device=m.device).index_add_(0, di, m))
                if self.max_agg:
                    mx = torch.zeros(n, self.d, device=m.device).scatter_reduce_(0, di[:, None].expand(-1, self.d), m, "amax", include_self=True)
                    agg = torch.cat([agg, mx], 1)
                out[dst] = out[dst] + self.rel_lin[l][name.replace(".", "__")](agg)
            h = {**h, **{t: h[t] + F.dropout(F.relu(self.norm[l][t](out[t])), self.p, self.training) for t in need}}
        z = h["Users"][user_rows]
        if self.context:
            z = torch.cat([z, z.mean(0, keepdim=True).expand_as(z)], 1)                          # cohort context: mean state of the users of this cutoff
        return torch.cat([self.head(z), self.head_push(torch.cat([z, cov.expand(z.shape[0], -1)], 1))], 1)


def fit(G, samples, cfg=None, seed=0, verbose=False):
    """Train on `samples` = list of (cutoff, user rows, labels (n, 6) float array, mask (6,) of the activities whose labels are used, coverage (3,))."""
    cfg = {**GNN, **(cfg or {})}
    if os.environ.get("ASG2_SMOKE"):                                                            # smoke test of the scripts only: 2 cutoffs, 1 epoch
        cfg, samples = {**cfg, "epochs": 1}, samples[:2]
    torch.use_deterministic_algorithms(True, warn_only=True)                                    # two runs on the same machine give identical weights
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    model = HeteroTemporalGNN(G.in_dim, G.rel, **cfg).to(G.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps = cfg["epochs"] * len(samples)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg["lr"], total_steps=steps, pct_start=0.1)
    dev = G.device
    prep = [(c, torch.tensor(np.asarray(r), device=dev), torch.tensor(np.asarray(y, dtype="float32"), device=dev),
             torch.tensor(np.asarray(m, dtype="float32"), device=dev), torch.tensor(np.asarray(cov, dtype="float32"), device=dev)[None, :]) for c, r, y, m, cov in samples]
    for ep in range(cfg["epochs"]):
        model.train(); tot = 0.0
        for i in rng.permutation(len(prep)):
            c, rows, y, mask, cov = prep[i]
            x, edges, _ = G.snapshot(c, cfg["time_enc"])
            loss = (F.binary_cross_entropy_with_logits(model(x, edges, rows, cov), y, reduction="none").mean(0) * mask).sum() / mask.sum()
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); tot += loss.item()
        if verbose and (ep % 5 == 4 or ep == cfg["epochs"] - 1):
            print(f"      epoch {ep + 1:3d}  loss {tot / len(prep):.4f}", flush=True)
    return model


@torch.no_grad()
def predict(G, model, cutoff, user_rows, cov, cfg=None, drop=()):
    """Probabilities (n users x 6) at `cutoff`.  drop: tables whose rows are disconnected from the graph (all their foreign-key edges removed) - used
    for the knock-out analysis 'which tables does the model need?'."""
    cfg = {**GNN, **(cfg or {})}
    model.eval()
    x, edges, _ = G.snapshot(cutoff, cfg["time_enc"])
    if drop:
        empty = torch.zeros(0, dtype=torch.long, device=G.device)
        edges = {name: ((empty, empty) if (s in drop or d in drop) else edges[name]) for name, s, d in G.rel}
    rows = torch.tensor(np.asarray(user_rows), device=G.device)
    return torch.sigmoid(model(x, edges, rows, torch.tensor(np.asarray(cov, dtype="float32"), device=G.device)[None, :])).double().cpu().numpy()


def coverage(daily, cutoff):
    """Share of the last 3 / 7 / 28 days before the cutoff in which the push stream was collected (input of the push head)."""
    b = daily.bin_of(pd.Timestamp(cutoff))
    return [float(daily.push_cover[max(b - w, 0):b].mean()) for w in (3, 7, 28)]


def make_samples(daily, Y, M, cutoffs, push_min_cover):
    """Training samples of the given cutoffs from the cached (user, cutoff) table M and the labels Y."""
    out = []
    for c in cutoffs:
        te = (M.cutoff == c).values
        mask = np.ones(len(LABELS), "float32")
        if push_min_cover and float(M.loc[te, "push_cover"].iloc[0]) < push_min_cover:
            mask[LABELS.index("push")] = 0.0                                                     # push label week poorly collected: not used
        out.append((c, M.user_row[te].values, Y[te].values, mask, coverage(daily, c)))
    return out
