"""NeuroSocial-RAGFormer trajectory module (NSRF-T) and learned baselines.

Mapping of the paper's modules onto the forecasting task:
  * Temporal encoder  -> per-agent motion embedding (DTEG agent-node features)
  * RAMG              -> retrieval of k similar historical futures from the
                         training-set memory graph, fused by cross-attention
  * MET               -> 4-token transformer fusing [personality, memory,
                         social, context] into a latent affective state
                         (3-d pleasure/arousal/dominance-like head, Eq. 7-10).
                         NOTE: unsupervised on ETH/UCY (no emotion labels).
  * SGT               -> graph transformer over co-present agents with
                         relation features e_ij = [F_ij, T_ij, I_ij] (Eq. 12-15)
  * NSRE              -> differentiable symbolic constraints (speed, acceleration,
                         collision) used as training penalties and as the
                         validity check that re-ranks modes at inference (Eq. 17-18)
  * Decoder           -> K mode queries, winner-takes-all training
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import OBS_LEN, PRED_LEN

DT = 0.4


def mlp(i, h, o, n=2, drop=0.0):
    layers, d = [], i
    for _ in range(n - 1):
        layers += [nn.Linear(d, h), nn.GELU(), nn.Dropout(drop)]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


def relation_features(obs, mask):
    """obs [B,N,8,2] -> edge features [B,N,N,9]: relative pose/velocity at t_obs plus
    the relationship attributes of Eq. (12): friendship F (co-movement), trust T
    (proximity stability), influence I (heading alignment weighted by proximity)."""
    pos = obs[:, :, -1]
    vel = (obs[:, :, -1] - obs[:, :, -2]) / DT
    dp = pos[:, None] - pos[:, :, None]                          # j - i
    dv = vel[:, None] - vel[:, :, None]
    dist = dp.norm(dim=-1, keepdim=True)
    d_all = (obs[:, None] - obs[:, :, None]).norm(dim=-1)        # [B,N,N,8]
    v_all = obs[:, :, 1:] - obs[:, :, :-1]
    vdiff = (v_all[:, None] - v_all[:, :, None]).norm(dim=-1) / DT
    Fr = torch.exp(-vdiff.mean(-1)) * torch.exp(-d_all.mean(-1) / 2.0)       # co-movement
    Tr = torch.exp(-d_all.std(-1) * 2.0)                                     # stable proximity
    hi = F.normalize(vel, dim=-1)
    In = (hi[:, None] * hi[:, :, None]).sum(-1) * torch.exp(-dist.squeeze(-1) / 3.0)
    e = torch.cat([dp, dv, dist, torch.log1p(dist), Fr[..., None], Tr[..., None], In[..., None]], -1)
    m = mask[:, None] & mask[:, :, None]
    return e * m[..., None]


class SGTLayer(nn.Module):
    """Graph transformer layer with relation-aware attention bias (Eq. 14-15)."""

    def __init__(self, d, heads, edim=9, drop=0.1):
        super().__init__()
        self.h, self.dk = heads, d // heads
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        self.ebias = mlp(edim, d, heads)
        self.eval_ = mlp(edim, d, d)
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ff = mlp(d, 4 * d, d, drop=drop)
        self.drop = nn.Dropout(drop)

    def forward(self, x, e, mask):
        B, N, D = x.shape
        y = self.n1(x)
        q = self.q(y).view(B, N, self.h, self.dk).transpose(1, 2)
        k = self.k(y).view(B, N, self.h, self.dk).transpose(1, 2)
        v = self.v(y).view(B, N, self.h, self.dk).transpose(1, 2)
        att = q @ k.transpose(-1, -2) / math.sqrt(self.dk) + self.ebias(e).permute(0, 3, 1, 2)
        att = att.masked_fill(~mask[:, None, None, :], -1e9).softmax(-1)
        ev = self.eval_(e).view(B, N, N, self.h, self.dk).permute(0, 3, 1, 2, 4)  # [B,h,N,N,dk]
        out = att @ v + (att[..., None] * ev).sum(-2)
        out = out.transpose(1, 2).reshape(B, N, D)
        x = x + self.drop(self.o(out))
        return x + self.drop(self.ff(self.n2(x)))


class NSRFT(nn.Module):
    def __init__(self, d=128, heads=4, enc_layers=2, sgt_layers=2, K=20, use_memory=True,
                 use_social=True, use_emotion=True, mem_k=10, drop=0.1):
        super().__init__()
        self.K, self.d = K, d
        self.use_memory, self.use_social, self.use_emotion = use_memory, use_social, use_emotion
        self.inp = nn.Linear(6, d)
        self.pos = nn.Parameter(torch.randn(1, OBS_LEN, d) * 0.02)
        el = nn.TransformerEncoderLayer(d, heads, 4 * d, drop, batch_first=True, norm_first=True)
        self.temporal = nn.TransformerEncoder(el, enc_layers)
        if use_social:
            self.sgt = nn.ModuleList([SGTLayer(d, heads, drop=drop) for _ in range(sgt_layers)])
        if use_memory:
            self.mem_in = mlp(PRED_LEN * 2 + 1, d, d)
            self.mem_att = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        if use_emotion:
            self.pers = mlp(4, d, d)               # personality from motion statistics
            self.type_emb = nn.Parameter(torch.randn(1, 4, d) * 0.02)
            ml = nn.TransformerEncoderLayer(d, heads, 2 * d, drop, batch_first=True, norm_first=True)
            self.met = nn.TransformerEncoder(ml, 1)
            self.pad_head = nn.Linear(d, 3)
            self.pad_in = nn.Linear(3, d)
        self.modes = nn.Parameter(torch.randn(K, d) * 0.1)
        self.dec = mlp(2 * d, 2 * d, PRED_LEN * 2, n=3)
        self.score = mlp(2 * d, d, 1)

    def encode(self, obs, mask):
        B, N = obs.shape[:2]
        rel = obs - obs[:, :, -1:]
        vel = torch.cat([torch.zeros_like(obs[:, :, :1]), obs[:, :, 1:] - obs[:, :, :-1]], 2)
        acc = torch.cat([torch.zeros_like(vel[:, :, :1]), vel[:, :, 1:] - vel[:, :, :-1]], 2)
        x = self.inp(torch.cat([rel, vel / DT, acc / DT], -1)).view(B * N, OBS_LEN, -1) + self.pos
        h = self.temporal(x)[:, -1].view(B, N, -1)
        return h, vel

    def forward(self, obs, mask, mem_vals=None, mem_sim=None):
        """obs [B,N,8,2] absolute; mem_vals [B,N,k,12,2] retrieved futures already
        rotated into the world frame; returns pred [B,N,K,12,2] displacement from last
        obs position, logits [B,N,K], pad [B,N,3] or None."""
        B, N = obs.shape[:2]
        h, vel = self.encode(obs, mask)
        ctx = h
        s = torch.zeros_like(h)
        if self.use_social:
            e = relation_features(obs, mask)
            s = h
            for layer in self.sgt:
                s = layer(s, e, mask)
            ctx = ctx + s
        m = torch.zeros_like(h)
        if self.use_memory and mem_vals is not None:
            k = mem_vals.shape[2]
            tok = self.mem_in(torch.cat([mem_vals.flatten(3), mem_sim[..., None]], -1)).view(B * N, k, -1)
            m, _ = self.mem_att(h.view(B * N, 1, -1), tok, tok)
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
        q = torch.cat([ctx[:, :, None].expand(-1, -1, self.K, -1), self.modes.expand(B, N, -1, -1)], -1)
        steps = self.dec(q).view(B, N, self.K, PRED_LEN, 2)
        pred = steps.cumsum(-2)
        logits = self.score(q).squeeze(-1)
        return pred, logits, pad


class LSTMWTA(nn.Module):
    """Vanilla LSTM encoder with the same K-mode decoder (no social, no memory)."""

    def __init__(self, d=128, K=20):
        super().__init__()
        self.K = K
        self.inp = nn.Linear(4, d)
        self.lstm = nn.LSTM(d, d, batch_first=True)
        self.modes = nn.Parameter(torch.randn(K, d) * 0.1)
        self.dec = mlp(2 * d, 2 * d, PRED_LEN * 2, n=3)
        self.score = mlp(2 * d, d, 1)

    def forward(self, obs, mask, *_):
        B, N = obs.shape[:2]
        rel = obs - obs[:, :, -1:]
        vel = torch.cat([torch.zeros_like(obs[:, :, :1]), obs[:, :, 1:] - obs[:, :, :-1]], 2) / DT
        _, (h, _) = self.lstm(self.inp(torch.cat([rel, vel], -1)).view(B * N, OBS_LEN, -1))
        h = h[-1].view(B, N, -1)
        q = torch.cat([h[:, :, None].expand(-1, -1, self.K, -1), self.modes.expand(B, N, -1, -1)], -1)
        return self.dec(q).view(B, N, self.K, PRED_LEN, 2).cumsum(-2), self.score(q).squeeze(-1), None


# ----------------------------------------------------------------------------
# Neuro-Symbolic Reasoning Engine for trajectories
# ----------------------------------------------------------------------------
V_MAX, A_MAX, R_COLL = 3.0, 4.0, 0.2   # m/s, m/s^2, m


def nsre_violation(pred, obs, mask):
    """Soft rule-violation scores per mode [B,N,K] (0 = all rules satisfied).
    Rule 1: speed <= V_MAX. Rule 2: |acc| <= A_MAX. Rule 3: no collision with the
    most likely future of any other agent (evaluated by the caller via `others`)."""
    full = torch.cat([torch.zeros_like(pred[..., :1, :]), pred], -2)
    full = full + obs[:, :, None, -1:, :]
    v0 = (obs[:, :, -1] - obs[:, :, -2])[:, :, None, None]
    vel = torch.cat([v0.expand(-1, -1, pred.shape[2], -1, -1), full[..., 1:, :] - full[..., :-1, :]], -2) / DT
    sp = vel.norm(dim=-1)
    acc = (vel[..., 1:, :] - vel[..., :-1, :]).norm(dim=-1) / DT
    return F.relu(sp - V_MAX).mean(-1) + 0.25 * F.relu(acc - A_MAX).mean(-1)


def collision_violation(pred, logits, obs, mask, stride: int = 1):
    """Rule 3: each mode of agent i against the top-scored mode of every other agent j.
    `stride` sub-samples prediction steps (training-time speed-up)."""
    pred = pred[..., stride - 1::stride, :]
    B, N, K, T, _ = pred.shape
    absf = pred + obs[:, :, None, -1:, :]
    best = absf.gather(2, logits.argmax(-1)[:, :, None, None, None].expand(-1, -1, 1, T, 2)).squeeze(2)  # [B,N,T,2]
    d = (absf[:, :, None] - best[:, None, :, None]).norm(dim=-1)        # [B,N,N,K,T]
    eye = torch.eye(N, dtype=torch.bool, device=pred.device)[None, :, :, None, None]
    pm = (mask[:, :, None] & mask[:, None, :])[..., None, None] & ~eye
    v = F.relu(R_COLL - d) * pm
    return v.sum(2).mean(-1)                                          # [B,N,K]
