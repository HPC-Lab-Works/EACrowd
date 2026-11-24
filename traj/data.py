"""ETH/UCY data pipeline (Social-GAN leave-one-scene-out protocol).

Raw files: <frame> <ped_id> <x> <y> in metres, 2.5 FPS after sub-sampling
(consecutive frame ids differ by 10 for ETH/UCY, 0.4 s per step).
A sample is a 20-step window (8 observed, 12 predicted) containing every
pedestrian that is present in all 20 steps (the protocol of Gupta et al., 2018,
used by Social-STGCNN, AgentFormer, MID, EqMotion, LED, ...).
"""
from __future__ import annotations

import glob
import math
import os
from dataclasses import dataclass

import numpy as np
import torch

OBS_LEN, PRED_LEN = 8, 12
SEQ_LEN = OBS_LEN + PRED_LEN
SCENES = ["eth", "hotel", "univ", "zara1", "zara2"]


def read_file(path: str) -> np.ndarray:
    data = np.loadtxt(path, dtype=np.float64)
    return data[np.lexsort((data[:, 1], data[:, 0]))]


@dataclass
class Scene:
    traj: np.ndarray        # [N, 20, 2] absolute positions (m)
    file_id: int
    start_frame: float
    ped_ids: np.ndarray     # [N]


def windows_from_file(path: str, file_id: int, skip: int = 1, min_ped: int = 1):
    data = read_file(path)
    frames = np.unique(data[:, 0])
    by_frame = {f: data[data[:, 0] == f] for f in frames}
    scenes = []
    for idx in range(0, len(frames) - SEQ_LEN + 1, skip):
        fr = frames[idx: idx + SEQ_LEN]
        chunk = np.concatenate([by_frame[f] for f in fr], 0)
        peds = np.unique(chunk[:, 1])
        trajs, ids = [], []
        for p in peds:
            pc = chunk[chunk[:, 1] == p]
            if len(pc) != SEQ_LEN:
                continue
            if not np.allclose(pc[:, 0], fr):
                continue
            trajs.append(pc[:, 2:4])
            ids.append(p)
        if len(trajs) >= min_ped:
            scenes.append(Scene(np.stack(trajs).astype(np.float32), file_id, float(fr[0]), np.array(ids)))
    return scenes


def load_split(root: str, scene: str, split: str, skip: int = 1):
    files = sorted(glob.glob(os.path.join(root, scene, split, "*.txt")))
    if not files:
        raise FileNotFoundError(f"no files for {scene}/{split} under {root}")
    out = []
    for fid, f in enumerate(files):
        out += windows_from_file(f, fid + (0 if split == "test" else 100 * (split == "val")), skip=skip)
    return out


# --------------------------------------------------------------------------
# Hand-crafted, rotation-invariant descriptor of the observed history, used as
# the key of the Retrieval-Augmented Memory Graph (RAMG). Expressed in the
# heading frame of the agent at the last observed step.
# --------------------------------------------------------------------------
def heading_rotation(obs: np.ndarray) -> np.ndarray:
    """obs [N, 8, 2] -> rotation matrices [N, 2, 2] mapping world -> heading frame."""
    v = obs[:, -1] - obs[:, -3]
    ang = np.arctan2(v[:, 1], v[:, 0])
    still = np.linalg.norm(v, axis=-1) < 0.05
    ang[still] = 0.0
    c, s = np.cos(-ang), np.sin(-ang)
    return np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2).astype(np.float32)


def memory_key(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    R = heading_rotation(obs)
    rel = obs - obs[:, -1:]
    rel_h = np.einsum("nij,ntj->nti", R, rel)
    speed = np.linalg.norm(obs[:, -1] - obs[:, -2], axis=-1, keepdims=True)
    key = np.concatenate([rel_h.reshape(len(obs), -1), 4.0 * speed], -1)
    return key.astype(np.float32), R


class TrajDataset(torch.utils.data.Dataset):
    def __init__(self, scenes, max_agents: int = 64):
        self.scenes = []
        # very crowded windows are split into chunks of max_agents (rare; univ)
        for s in scenes:
            n = len(s.traj)
            for a in range(0, n, max_agents):
                sl = slice(a, min(n, a + max_agents))
                self.scenes.append(Scene(s.traj[sl], s.file_id, s.start_frame, s.ped_ids[sl]))
        # flat agent index -> (scene, row) for the memory bank
        self.flat = [(i, j) for i, s in enumerate(self.scenes) for j in range(len(s.traj))]
        self.offsets = np.cumsum([0] + [len(s.traj) for s in self.scenes])
        self.retrieval = None  # [num_agents, k] indices into a memory bank

    def __len__(self):
        return len(self.scenes)

    def all_agent_arrays(self):
        traj = np.concatenate([s.traj for s in self.scenes], 0)
        meta = np.concatenate([np.stack([np.full(len(s.traj), s.file_id), np.full(len(s.traj), s.start_frame), s.ped_ids], -1)
                               for s in self.scenes], 0)
        return traj, meta

    def __getitem__(self, i):
        s = self.scenes[i]
        r = None
        if self.retrieval is not None:
            r = self.retrieval[self.offsets[i]: self.offsets[i + 1]]
        return torch.from_numpy(s.traj), (None if r is None else torch.from_numpy(r))


def collate(batch, augment: bool = False):
    nmax = max(b[0].shape[0] for b in batch)
    B = len(batch)
    traj = torch.zeros(B, nmax, SEQ_LEN, 2)
    mask = torch.zeros(B, nmax, dtype=torch.bool)
    ret = None
    if batch[0][1] is not None:
        k = batch[0][1].shape[1]
        ret = torch.zeros(B, nmax, k, dtype=torch.long)
    for i, (t, r) in enumerate(batch):
        n = t.shape[0]
        tt = t
        if augment:
            th = torch.rand(()) * 2 * math.pi
            R = torch.tensor([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]], dtype=torch.float32)
            tt = (t - t[:, OBS_LEN - 1:OBS_LEN].mean(0, keepdim=True)) @ R.T
        traj[i, :n] = tt
        mask[i, :n] = True
        if ret is not None:
            ret[i, :n] = r
    return traj, mask, ret


@torch.no_grad()
def build_memory_bank(train_ds: TrajDataset, k: int, device="cpu", chunk: int = 4096, beta: float = 0.1):
    """Builds the RAMG for a split.

    Nodes  : training agents; key = rotation-invariant history descriptor,
             value = future in the heading frame.
    Edges  : cosine similarity > tau (k-NN graph, Alg. 2 of the paper).
    Importance I = alpha*R + beta*S + gamma*T (Eq. 6); on a static corpus the
             recency term R is constant, S = mean similarity to graph neighbours,
             T = number of co-present agents (interaction significance).
    Returns dict with keys/values/importance and self-retrieval indices for the
    training set (same-pedestrian, same-file entries masked to avoid leakage).
    """
    traj, meta = train_ds.all_agent_arrays()
    key, R = memory_key(traj[:, :OBS_LEN])
    fut = traj[:, OBS_LEN:] - traj[:, OBS_LEN - 1:OBS_LEN]
    val = np.einsum("nij,ntj->nti", R, fut)
    keys = torch.from_numpy(key).to(device)
    keys_n = torch.nn.functional.normalize(keys, dim=-1)
    n = len(keys)
    co_present = np.concatenate([np.full(len(s.traj), len(s.traj)) for s in train_ds.scenes]).astype(np.float32)
    meta_t = torch.from_numpy(meta).to(device)
    def masked_sim(a):
        q = keys_n[a:a + chunk]
        sim = q @ keys_n.T
        m = meta_t[a:a + chunk]
        same = (m[:, None, 0] == meta_t[None, :, 0]) & (m[:, None, 2] == meta_t[None, :, 2]) & \
               ((m[:, None, 1] - meta_t[None, :, 1]).abs() < 200)
        return sim.masked_fill(same, -2.0)

    # pass 1: semantic-relevance statistic S (mean similarity to graph neighbours)
    S = torch.zeros(n, device=device)
    for a in range(0, n, chunk):
        S[a:a + chunk] = masked_sim(a).topk(k, dim=-1).values.clamp(min=0).mean(-1)
    T = torch.from_numpy(np.log1p(co_present)).to(device)
    T = T / T.max()
    importance = 0.6 * S + 0.4 * T
    # pass 2: importance-weighted retrieval (identical scoring rule to test time)
    idx_all = torch.zeros(n, k, dtype=torch.long, device=device)
    for a in range(0, n, chunk):
        idx_all[a:a + chunk] = (masked_sim(a) + beta * importance[None]).topk(k, dim=-1).indices
    return {
        "keys": keys_n.cpu(), "values": torch.from_numpy(val.astype(np.float32)),
        "importance": importance.cpu(), "meta": meta, "train_retrieval": idx_all.cpu().numpy(),
    }


@torch.no_grad()
def retrieve(bank, ds: TrajDataset, k: int, device="cpu", chunk: int = 4096, beta: float = 0.1):
    traj, _ = ds.all_agent_arrays()
    key, _ = memory_key(traj[:, :OBS_LEN])
    q = torch.nn.functional.normalize(torch.from_numpy(key).to(device), dim=-1)
    K = bank["keys"].to(device)
    imp = bank["importance"].to(device)
    out = []
    for a in range(0, len(q), chunk):
        sc = q[a:a + chunk] @ K.T + beta * imp[None]
        out.append(sc.topk(k, dim=-1).indices.cpu())
    return torch.cat(out).numpy()
