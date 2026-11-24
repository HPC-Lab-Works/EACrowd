"""Learned modules of NeuroSocial-RAGFormer used inside the simulator.

MET  : Multimodal Emotion Transformer (Eq. 7-10). Four tokens - personality,
       retrieved memory, social influence, contextual event - fused by
       multi-head self-attention; outputs the next PAD state and an emotion
       embedding. Trained by (i) distillation towards an ALMA-style appraisal
       teacher with trust-weighted contagion (keeps PAD interpretable) and
       (ii) end-to-end PPO gradients.
SGT  : Social Graph Transformer (Eq. 12-15): graph attention over the k nearest
       agents with relation features [F, T, I] (friendship, trust, influence).
RAMG : projection of the graph-retrieved memory summary (retrieval itself is
       non-parametric, see core.retrieve_memory).
Actor/critic: MAPPO with parameter sharing and a centralised critic.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

N_ACTIONS = 7
ACTIONS = ["continue_task", "socialize_follow", "share_info", "avoid_crowd",
           "evacuate_nearest", "evacuate_least_congested", "assist_friend"]


def mlp(i, h, o, n=2):
    layers, d = [], i
    for _ in range(n - 1):
        layers += [nn.Linear(d, h), nn.Tanh()]
        d = h
    layers.append(nn.Linear(d, o))
    return nn.Sequential(*layers)


class MET(nn.Module):
    def __init__(self, d_pers=5, d_mem=23, d_soc=5, d_ctx=9, d=32, heads=4):
        super().__init__()
        self.proj = nn.ModuleList([nn.Linear(x, d) for x in (d_pers, d_mem, d_soc, d_ctx)])
        self.type_emb = nn.Parameter(torch.randn(1, 4, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, 0.0, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, 1, enable_nested_tensor=False)
        self.pad = nn.Linear(d, 3)

    def forward(self, pers, mem, soc, ctx):
        toks = torch.stack([p(x) for p, x in zip(self.proj, (pers, mem, soc, ctx))], 1) + self.type_emb
        z = self.enc(toks).mean(1)
        return torch.tanh(self.pad(z)), z


class SGT(nn.Module):
    """Graph attention (Velickovic et al., 2018) over neighbour sets, Eq. (14)-(15)."""

    def __init__(self, d_self, d_edge, d=32, heads=2):
        super().__init__()
        self.h, self.d = heads, d
        self.Ws = nn.Linear(d_self, d * heads)
        self.Wn = nn.Linear(d_edge, d * heads)
        self.a = nn.Parameter(torch.randn(heads, 2 * d) * 0.1)
        self.out = nn.Linear(d * heads, d)

    def forward(self, x_self, x_nb, nb_mask):
        B, K, _ = x_nb.shape
        hs = self.Ws(x_self).view(B, 1, self.h, self.d)
        hn = self.Wn(x_nb).view(B, K, self.h, self.d)
        e = F.leaky_relu((torch.cat([hs.expand(-1, K, -1, -1), hn], -1) * self.a).sum(-1), 0.2)   # [B,K,h]
        e = e.masked_fill(~nb_mask[..., None], -1e9)
        alpha = e.softmax(1) * nb_mask[..., None]
        agg = (alpha[..., None] * hn).sum(1).reshape(B, -1)
        return F.elu(self.out(agg)), alpha


class ActorCritic(nn.Module):
    def __init__(self, d_obs, d_glob, use_met=True, use_sgt=True, use_mem=True, d=32, d_edge=11, d_mem=23):
        super().__init__()
        self.use_met, self.use_sgt, self.use_mem = use_met, use_sgt, use_mem
        d_in = d_obs
        if use_met:
            self.met = MET(d=d)
            d_in += d + 3
        if use_sgt:
            self.sgt = SGT(d_obs, d_edge, d=d)
            d_in += d
        if use_mem:
            self.mem = mlp(d_mem, d, d)
            d_in += d
        self.actor = mlp(d_in, 128, N_ACTIONS, n=3)
        self.critic = mlp(d_in + d_glob, 128, 1, n=3)

    def features(self, b):
        parts = [b["obs"]]
        pad = None
        if self.use_met:
            pad, z = self.met(b["pers"], b["mem"], b["soc"], b["ctx"])
            parts += [z, pad]
        if self.use_sgt:
            s, _ = self.sgt(b["obs"], b["nb"], b["nb_mask"])
            parts.append(s)
        if self.use_mem:
            parts.append(self.mem(b["mem"]))
        return torch.cat(parts, -1), pad

    def forward(self, b):
        f, pad = self.features(b)
        logits = self.actor(f).masked_fill(~b["amask"], -1e9)
        v = self.critic(torch.cat([f, b["glob"]], -1)).squeeze(-1)
        return logits, v, pad
