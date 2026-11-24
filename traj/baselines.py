"""Non-learned forecasting baselines evaluated under the identical protocol.

* CV      : constant velocity (Schoeller et al., RA-L 2020)
* CV-S    : constant velocity with angular/speed sampling for best-of-K
* SFM     : Social Force Model roll-out (Helbing & Molnar 1995; parameters of
            Helbing, Farkas & Vicsek 2000 rescaled to unit mass), joint scene
            simulation with sampled desired velocities for best-of-K
"""
from __future__ import annotations

import math

import torch

from .data import PRED_LEN
from .models import DT


def cv_predict(obs, K=20, sigma_deg=25.0, speed_sigma=0.15, sample=True, gen=None):
    v = obs[:, :, -1] - obs[:, :, -2]                                   # m / step
    B, N = v.shape[:2]
    t = torch.arange(1, PRED_LEN + 1, device=obs.device, dtype=obs.dtype)
    if not sample:
        return (v[:, :, None, None] * t[None, None, None, :, None]).expand(B, N, K, PRED_LEN, 2).clone()
    ang = torch.randn(B, N, K, generator=gen, device=obs.device) * math.radians(sigma_deg)
    ang[:, :, 0] = 0
    sc = 1 + torch.randn(B, N, K, generator=gen, device=obs.device) * speed_sigma
    sc[:, :, 0] = 1
    c, s = torch.cos(ang), torch.sin(ang)
    vx = (c * v[:, :, None, 0] - s * v[:, :, None, 1]) * sc
    vy = (s * v[:, :, None, 0] + c * v[:, :, None, 1]) * sc
    vv = torch.stack([vx, vy], -1)
    return vv[:, :, :, None] * t[None, None, None, :, None]


def sfm_predict(obs, mask, K=20, sigma_deg=20.0, speed_sigma=0.15, A=2.1, Br=0.3, radius=0.6,
                tau=0.5, substeps=4, gen=None):
    """Roll out K joint samples of the Social Force Model from the last observation."""
    Bsz, N = obs.shape[:2]
    v0 = (obs[:, :, -1] - obs[:, :, -2]) / DT                                     # m/s
    ang = torch.randn(Bsz, N, K, generator=gen, device=obs.device) * math.radians(sigma_deg)
    sc = 1 + torch.randn(Bsz, N, K, generator=gen, device=obs.device) * speed_sigma
    ang[:, :, 0] = 0
    sc[:, :, 0] = 1
    c, s = torch.cos(ang), torch.sin(ang)
    vd = torch.stack([(c * v0[:, :, None, 0] - s * v0[:, :, None, 1]) * sc,
                      (s * v0[:, :, None, 0] + c * v0[:, :, None, 1]) * sc], -1)  # desired velocity [B,N,K,2]
    p = obs[:, :, None, -1].expand(-1, -1, K, -1).clone()
    v = v0[:, :, None].expand(-1, -1, K, -1).clone()
    out = []
    pm = (mask[:, :, None] & mask[:, None, :]) & ~torch.eye(N, dtype=torch.bool, device=obs.device)[None]
    h = DT / substeps
    for _ in range(PRED_LEN):
        for _ in range(substeps):
            d = p[:, :, None] - p[:, None, :]                       # i - j  [B,N,N,K,2]
            dist = d.norm(dim=-1, keepdim=True).clamp(min=1e-3)
            f_rep = A * torch.exp((radius - dist) / Br) * d / dist
            f_rep = (f_rep * pm[..., None, None]).sum(2)
            acc = (vd - v) / tau + f_rep
            v = v + h * acc
            spd = v.norm(dim=-1, keepdim=True)
            v = v * (spd.clamp(max=2.5) / spd.clamp(min=1e-6))
            p = p + h * v
        out.append(p)
    pred = torch.stack(out, 3) - obs[:, :, None, None, -1]
    return pred


def straight_goal(obs, goal, T=PRED_LEN):
    """Grid/A*-style shortest path in open space: straight line at uniform speed."""
    a = torch.arange(1, T + 1, device=obs.device, dtype=obs.dtype) / T
    start = obs[:, :, -1]
    return (goal - start)[:, :, None] * a[None, None, :, None]


def sfm_goal(obs, mask, goal, A=2.1, Br=0.3, radius=0.6, tau=0.5, substeps=4):
    """Goal-directed SFM: desired speed = remaining distance / horizon."""
    B, N = obs.shape[:2]
    p = obs[:, :, -1].clone()
    v = (obs[:, :, -1] - obs[:, :, -2]) / DT
    horizon = PRED_LEN * DT
    pm = (mask[:, :, None] & mask[:, None, :]) & ~torch.eye(N, dtype=torch.bool, device=obs.device)[None]
    out, h = [], DT / substeps
    for step in range(PRED_LEN):
        for sub in range(substeps):
            t_left = max(horizon - (step * substeps + sub) * h, h)
            to_g = goal - p
            vd = to_g / t_left
            d = p[:, :, None] - p[:, None]
            dist = d.norm(dim=-1, keepdim=True).clamp(min=1e-3)
            f = (A * torch.exp((radius - dist) / Br) * d / dist * pm[..., None]).sum(2)
            v = v + h * ((vd - v) / tau + f)
            p = p + h * v
        out.append(p)
    return torch.stack(out, 2) - obs[:, :, -1:]


def discrete_frechet(P, Q):
    """Batched discrete Frechet distance. P, Q [M, T, 2] -> [M]."""
    M, T, _ = P.shape
    d = (P[:, :, None] - Q[:, None]).norm(dim=-1)        # [M,T,T]
    ca = torch.full((M, T, T), float("inf"), device=P.device, dtype=P.dtype)
    ca[:, 0, 0] = d[:, 0, 0]
    for i in range(1, T):
        ca[:, i, 0] = torch.maximum(ca[:, i - 1, 0], d[:, i, 0])
        ca[:, 0, i] = torch.maximum(ca[:, 0, i - 1], d[:, 0, i])
    for i in range(1, T):
        for j in range(1, T):
            prev = torch.minimum(torch.minimum(ca[:, i - 1, j], ca[:, i - 1, j - 1]), ca[:, i, j - 1])
            ca[:, i, j] = torch.maximum(prev, d[:, i, j])
    return ca[:, -1, -1]
