"""Reference statistics from real pedestrians (ETH/UCY) and the Human-Likeness Score.

HLS = 100 * (1 - mean_k JSD(p_sim^k || p_real^k)), k in {speed, |acceleration|,
turning rate, nearest-neighbour distance}, base-2 Jensen-Shannon divergence on
fixed histograms. Both real and simulated data are sampled at 0.4 s and use the
same filters (moving agents, speed > 0.2 m/s in two consecutive steps).
"""
from __future__ import annotations

import glob
import json
import math
import os

import numpy as np

BINS = {"speed": np.linspace(0, 3, 31), "acc": np.linspace(0, 4, 41),
        "turn": np.linspace(0, math.pi / 0.4, 41), "nn": np.linspace(0, 6, 31)}
DT = 0.4


def _hist(x, b):
    idx = np.searchsorted(b, x, side="left")   # matches torch.bucketize(right=False)
    h = np.bincount(idx, minlength=len(b) + 1)[: len(b) + 1].astype(np.float64)
    return h


def real_reference(root="data/raw", cache="results/eth_ucy_reference.json"):
    if os.path.exists(cache):
        return json.load(open(cache))
    files = sorted(set(glob.glob(os.path.join(root, "*", "test", "*.txt"))))
    H = {k: np.zeros(len(b) + 1) for k, b in BINS.items()}
    grp_sum, grp_n = 0.0, 0
    for f in files:
        d = np.loadtxt(f)
        frames = np.unique(d[:, 0])
        step = np.median(np.diff(frames))
        pos = {}
        for fr in frames:
            rows = d[d[:, 0] == fr]
            pos[fr] = {int(r[1]): r[2:4] for r in rows}
        pair_cnt = {}
        for i in range(2, len(frames)):
            f0, f1, f2 = frames[i - 2], frames[i - 1], frames[i]
            if f2 - f1 != step or f1 - f0 != step:
                pair_cnt = {}
                continue
            ids = [p for p in pos[f2] if p in pos[f1] and p in pos[f0]]
            if not ids:
                continue
            P2 = np.array([pos[f2][p] for p in ids]); P1 = np.array([pos[f1][p] for p in ids]); P0 = np.array([pos[f0][p] for p in ids])
            v1, v0 = (P2 - P1) / DT, (P1 - P0) / DT
            sp, sp0 = np.linalg.norm(v1, axis=1), np.linalg.norm(v0, axis=1)
            allp = np.array(list(pos[f2].values()))
            dist = np.linalg.norm(P2[:, None] - allp[None], axis=-1)
            dist[dist < 1e-6] = np.inf
            nn = dist.min(1)
            sel = (sp > 0.2) & (sp0 > 0.2)
            acc = np.linalg.norm(v1 - v0, axis=1) / DT
            turn = np.abs((np.arctan2(v1[:, 1], v1[:, 0]) - np.arctan2(v0[:, 1], v0[:, 0]) + np.pi) % (2 * np.pi) - np.pi) / DT
            for k, x in (("speed", sp), ("acc", acc), ("turn", turn), ("nn", np.minimum(nn, 5.99))):
                H[k] += _hist(x[sel], BINS[k])
            # groups: within 2 m, |dv| < 0.3 m/s, both moving (> 0.3 m/s), persisting >= 10 steps
            mv = sp > 0.3
            dd = np.linalg.norm(P2[:, None] - P2[None], axis=-1)
            dv = np.linalg.norm(v1[:, None] - v1[None], axis=-1)
            cond = (dd < 2.0) & (dv < 0.3) & mv[:, None] & mv[None] & ~np.eye(len(ids), dtype=bool)
            new = {}
            for a, b in zip(*np.nonzero(cond)):
                key = (ids[a], ids[b])
                new[key] = pair_cnt.get(key, 0) + 1
            pair_cnt = new
            if mv.sum() >= 3:
                members = {k[0] for k, c in pair_cnt.items() if c >= 10}
                grp_sum += len([p for j, p in enumerate(ids) if mv[j] and p in members]) / mv.sum()
                grp_n += 1
    ref = {"hist": {k: (v / v.sum()).tolist() for k, v in H.items()}, "group_fraction": grp_sum / max(1, grp_n),
           "files": [os.path.basename(f) for f in files],
           "mean_speed": float((H["speed"][1:-1] * (BINS["speed"][1:] - 0.05)).sum() / H["speed"][1:-1].sum())}
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    json.dump(ref, open(cache, "w"))
    return ref


def jsd(p, q):
    p = np.asarray(p, float) + 1e-12; q = np.asarray(q, float) + 1e-12
    p /= p.sum(); q /= q.sum()
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * np.log2(a / b)).sum()
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def human_likeness(sim_hist, ref):
    parts = {k: 1 - jsd(sim_hist[k], ref["hist"][k]) for k in BINS}
    return 100 * float(np.mean(list(parts.values()))), {k: 100 * v for k, v in parts.items()}


if __name__ == "__main__":
    r = real_reference()
    print("group fraction (ETH/UCY):", r["group_fraction"], "mean speed", r["mean_speed"])
