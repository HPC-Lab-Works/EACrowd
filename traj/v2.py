"""NSRF-T v2: learned-intention memory retrieval and leapfrog diffusion refinement.

Two changes relative to NSRF-T (models.NSRFT):

1. Intention memory (in the spirit of MemoNet, CVPR 2022). The RAMG key is no longer a
   hand-crafted history descriptor but a learned embedding of the encoder state, trained
   with an auxiliary loss to predict the agent's end point (its intention) in its own
   heading frame. Keys of the whole training set are re-encoded every `mem_refresh`
   epochs and retrieval is recomputed with the same leakage mask as before. The end
   points of the k retrieved futures act as intention anchors for the first k decoder
   modes.

2. Leapfrog diffusion refinement (in the spirit of LED, CVPR 2023). The K-mode decoder is
   kept as an initialiser. A conditional denoiser is trained on ground-truth futures;
   at inference each initial mode is noised to an intermediate step t_start (instead of
   pure noise at T) and denoised back with a few DDIM steps.
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import OBS_LEN, PRED_LEN, collate
from .models import NSRFT, mlp, relation_features, DT

SCALE = 0.3  # metres -> diffusion units


def heading_R(obs):
    v = obs[..., -1, :] - obs[..., -3, :]
    ang = torch.atan2(v[..., 1], v[..., 0])
    ang = torch.where(v.norm(dim=-1) < 0.05, torch.zeros_like(ang), ang)
    c, s = torch.cos(-ang), torch.sin(-ang)
    return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)


def time_embed(t, dim=32):
    half = dim // 2
    f = torch.exp(-math.log(1000.0) * torch.arange(half, device=t.device) / half)
    a = t.float()[..., None] * f
    return torch.cat([a.sin(), a.cos()], -1)


class NSRFT2(NSRFT):
    def __init__(self, d=128, heads=4, K=20, mem_k=10, intent=True, refine=True, T=50, t_start=10,
                 ddim_steps=5, d_key=64, **kw):
        super().__init__(d=d, heads=heads, K=K, mem_k=mem_k, **kw)
        self.v2, self.intent, self.refine = True, intent, refine
        self.mem_k, self.T, self.t_start, self.ddim_steps = mem_k, T, t_start, ddim_steps
        self.key_proj = mlp(d, d, d_key)
        self.int_head = mlp(d_key, d, 2)
        if intent:
            self.anchor = mlp(3, d, d)
            self.anchor_gate = nn.Parameter(torch.zeros(1))   # starts closed: anchors are used only if they help
        if refine:
            self.den = mlp(PRED_LEN * 2 + d + 32, 2 * d, PRED_LEN * 2, n=4)
            s = torch.arange(T + 1, dtype=torch.float32) / T
            ab = torch.cos((s + 0.008) / 1.008 * math.pi / 2) ** 2
            self.register_buffer("ab", (ab / ab[0]).clamp(min=1e-5))
        self._ctx = None

    # ---------------------------------------------------------------- memory keys
    def keys(self, obs, mask):
        h, _ = self.encode(obs, mask)
        return F.normalize(self.key_proj(h), dim=-1), h

    # ---------------------------------------------------------------- forward
    def forward(self, obs, mask, mem_vals=None, mem_keys=None):
        B, N = obs.shape[:2]
        key, h = self.keys(obs, mask)
        vel = torch.cat([torch.zeros_like(obs[:, :, :1]), obs[:, :, 1:] - obs[:, :, :-1]], 2)
        ctx, s = h, torch.zeros_like(h)
        if self.use_social:
            e = relation_features(obs, mask)
            s = h
            for layer in self.sgt:
                s = layer(s, e, mask)
            ctx = ctx + s
        m = torch.zeros_like(h)
        sim = None
        if self.use_memory and mem_vals is not None:
            k = mem_vals.shape[2]
            sim = (key[:, :, None] * mem_keys).sum(-1)                      # learned relevance
            tok = self.mem_in(torch.cat([mem_vals.flatten(3), sim[..., None]], -1)).view(B * N, k, -1)
            m, _ = self.mem_att(h.reshape(B * N, 1, -1), tok, tok)
            m = m.view(B, N, -1)
            ctx = ctx + m
        pad = None
        if self.use_emotion:
            sp = vel.norm(dim=-1) / DT
            stats = torch.stack([sp.mean(-1), sp.std(-1), sp[..., -1], (sp[..., 1:] - sp[..., :-1]).abs().mean(-1)], -1)
            toks = torch.stack([self.pers(stats), m, s, h], 2).view(B * N, 4, -1) + self.type_emb
            z = self.met(toks).mean(1).view(B, N, -1)
            pad = torch.tanh(self.pad_head(z))
            ctx = ctx + z + self.pad_in(pad)
        modes = self.modes.expand(B, N, -1, -1)
        if self.intent and mem_vals is not None:
            k = min(mem_vals.shape[2], self.K)
            anc = self.anchor(torch.cat([mem_vals[:, :, :k, -1], sim[:, :, :k, None]], -1))   # [B,N,k,d]
            modes = torch.cat([modes[:, :, :k] + torch.tanh(self.anchor_gate) * anc, modes[:, :, k:]], 2)
        q = torch.cat([ctx[:, :, None].expand(-1, -1, self.K, -1), modes], -1)
        pred = self.dec(q).view(B, N, self.K, PRED_LEN, 2).cumsum(-2)
        logits = self.score(q).squeeze(-1)
        self._ctx, self._key = ctx, key
        if self.refine and not self.training:
            pred = self.ddim_refine(pred, ctx)
        return pred, logits, pad

    # ---------------------------------------------------------------- diffusion
    def denoise(self, x, t, ctx):
        return self.den(torch.cat([x, ctx, time_embed(t)], -1))

    def ddim_refine(self, init, ctx):
        """Leapfrog start: noise each initial mode to t_start, then DDIM back to 0. The denoiser
        predicts noise (eps), so for small t_start the output stays close to its mode and the
        diversity of the K initial modes is kept."""
        if self.t_start <= 0:
            return init
        B, N, K = init.shape[:3]
        x0 = init.flatten(3) * SCALE
        c = ctx[:, :, None].expand(-1, -1, K, -1)
        ts = np.linspace(self.t_start, 0, self.ddim_steps + 1).round().astype(int)
        ts = np.unique(ts)[::-1]
        t0 = int(ts[0])
        x = self.ab[t0].sqrt() * x0 + (1 - self.ab[t0]).sqrt() * torch.randn_like(x0)
        for a, b in zip(ts[:-1], ts[1:]):
            tt = torch.full(x.shape[:-1], int(a), device=x.device)
            eps = self.denoise(x, tt, c)
            x0p = (x - (1 - self.ab[a]).sqrt() * eps) / self.ab[a].sqrt()
            x = self.ab[b].sqrt() * x0p + (1 - self.ab[b]).sqrt() * eps
        return (x / SCALE).view(B, N, K, PRED_LEN, 2)

    def aux_losses(self, obs, fut, mask):
        """Intention loss on the memory key and the denoising loss (called after forward)."""
        R = heading_R(obs)
        end_h = torch.einsum("bnij,bnj->bni", R, fut[:, :, -1])
        key, ctx = self._key.float(), self._ctx.float()      # forward may have run under bf16 autocast
        l_int = F.smooth_l1_loss(self.int_head(key)[mask], end_h[mask])
        loss = 0.5 * l_int
        if self.refine:
            x0 = fut.flatten(2) * SCALE
            t = torch.randint(1, self.T + 1, x0.shape[:2], device=x0.device)
            ab = self.ab[t][..., None]
            eps = torch.randn_like(x0)
            xt = ab.sqrt() * x0 + (1 - ab).sqrt() * eps
            l_eps = (self.denoise(xt, t, ctx) - eps).pow(2).mean(-1)
            loss = loss + l_eps[mask].mean()
        return loss


# -------------------------------------------------------------------- learned retrieval
@torch.no_grad()
def encode_keys(model, ds, device, bs=64):
    model.eval()
    out = []
    for a in range(0, len(ds.scenes), bs):
        batch = [(torch.from_numpy(s.traj), None) for s in ds.scenes[a:a + bs]]
        traj, mask, _ = collate(batch, False)
        traj, mask = traj.to(device), mask.to(device)
        k, _ = model.keys(traj[:, :, :OBS_LEN], mask)
        out.append(k[mask].float())
    return torch.cat(out)


@torch.no_grad()
def retrieve_learned(q, qmeta, keys, kmeta, importance, k, beta=0.1, chunk=4096):
    """Top-k by learned cosine + beta * importance; entries of the same pedestrian in the same
    file within 200 frames are masked (no self-retrieval) when qmeta is given."""
    out = []
    for a in range(0, len(q), chunk):
        sc = q[a:a + chunk] @ keys.T + beta * importance[None]
        if qmeta is not None:
            m = qmeta[a:a + chunk]
            same = (m[:, None, 0] == kmeta[None, :, 0]) & (m[:, None, 2] == kmeta[None, :, 2]) & \
                   ((m[:, None, 1] - kmeta[None, :, 1]).abs() < 200)
            sc = sc.masked_fill(same, -1e4)
        out.append(sc.topk(k, -1).indices.cpu())
    return torch.cat(out).numpy()


def refresh_memory(model, bank, tr, va, te, k, device, learned=True):
    """Re-encodes training keys with the current model and recomputes retrieval. The score is
    0.5 * hand-crafted history similarity + 0.5 * learned intention similarity (+ importance);
    with learned=False (warm-up) only the hand-crafted part is used."""
    from .data import memory_key
    kt = encode_keys(model, tr, device)
    bank["lkeys_dev"] = kt
    meta = torch.from_numpy(bank["meta"]).to(device)
    imp = bank["importance"].to(device)
    w = 0.5 ** 0.5 if learned else 0.0

    def hyb(ds, lk):
        traj, _ = ds.all_agent_arrays()
        hk = F.normalize(torch.from_numpy(memory_key(traj[:, :OBS_LEN])[0]).to(device), dim=-1)
        return torch.cat([hk * (1.0 if not learned else w), lk * w], -1)

    bank_k = hyb(tr, kt)
    tr.retrieval = retrieve_learned(bank_k, meta, bank_k, meta, imp, k)
    for ds in (va, te):
        ds.retrieval = retrieve_learned(hyb(ds, encode_keys(model, ds, device)), None, bank_k, meta, imp, k)
    model.train()
