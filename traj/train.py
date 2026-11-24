"""Train and evaluate NSRF-T, its ablations and the baselines on one ETH/UCY split.

Example (one GPU):
  python -m traj.train --scene eth --model nsrf --epochs 100 --d 256 --seed 0
Non-learned baselines:
  python -m traj.train --scene eth --model cv
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from .baselines import cv_predict, discrete_frechet, sfm_goal, sfm_predict, straight_goal
from .data import OBS_LEN, TrajDataset, build_memory_bank, collate, load_split, retrieve
from .models import LSTMWTA, NSRFT, collision_violation, nsre_violation
from .v2 import NSRFT2, refresh_memory

MODELS = {
    "nsrf": dict(),
    "nsrf_no_memory": dict(use_memory=False),
    "nsrf_no_social": dict(use_social=False),
    "nsrf_no_emotion": dict(use_emotion=False),
    "nsrf_no_nsre": dict(),               # same network, trained/evaluated without NSRE
    "transformer": dict(use_memory=False, use_social=False, use_emotion=False),
    "nsrf_v2": dict(v2=True),                               # learned-intention RAMG + diffusion refinement
    "nsrf_v2_no_refine": dict(v2=True, refine=False),       # learned-intention RAMG only
    "nsrf_v2_keys_only": dict(v2=True, refine=False, intent=False),  # learned keys, no anchors, no refinement
}


def heading_R_torch(obs):
    v = obs[..., -1, :] - obs[..., -3, :]
    ang = torch.atan2(v[..., 1], v[..., 0])
    ang = torch.where(v.norm(dim=-1) < 0.05, torch.zeros_like(ang), ang)
    c, s = torch.cos(-ang), torch.sin(-ang)
    return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)


def memory_inputs(obs, ret, bank):
    """Gathers retrieved futures, rotates them into the query's world frame."""
    vals = bank["values_dev"][ret]                       # [B,N,k,12,2] heading frame of memory agent
    R = heading_R_torch(obs)                             # world -> heading of query
    vals_w = torch.einsum("bnji,bnktj->bnkti", R, vals)  # R^T @ v
    rel = obs - obs[:, :, -1:]
    rel_h = torch.einsum("bnij,bntj->bnti", R, rel)
    sp = (obs[:, :, -1] - obs[:, :, -2]).norm(dim=-1, keepdim=True)
    qk = F.normalize(torch.cat([rel_h.flatten(2), 4.0 * sp], -1), dim=-1)
    sim = (qk[:, :, None] * bank["keys_dev"][ret]).sum(-1)
    return vals_w, sim


def run_model(model, obs, mask, ret, bank):
    if getattr(model, "v2", False):
        if ret is None:
            return model(obs, mask)
        mv, _ = memory_inputs(obs, ret, bank)
        return model(obs, mask, mv, bank["lkeys_dev"][ret])
    if getattr(model, "use_memory", False) and ret is not None:
        mv, ms = memory_inputs(obs, ret, bank)
        return model(obs, mask, mv, ms)
    return model(obs, mask)


def metrics_from_pred(pred, scores, fut, mask, obs):
    err = (pred - fut[:, :, None]).norm(dim=-1)                      # [B,N,K,12]
    ade, fde = err.mean(-1), err[..., -1]
    minade, minfde = ade.min(-1).values, fde.min(-1).values
    top = scores.argmax(-1, keepdim=True)
    ade1, fde1 = ade.gather(-1, top).squeeze(-1), fde.gather(-1, top).squeeze(-1)
    # collision rate of the top-1 joint prediction (pairs closer than 0.2 m at any step)
    T = pred.shape[3]
    best = pred.gather(2, top[..., None, None].expand(-1, -1, 1, T, 2)).squeeze(2) + obs[:, :, -1:]
    d = (best[:, :, None] - best[:, None]).norm(dim=-1).min(-1).values
    N = mask.shape[1]
    pm = (mask[:, :, None] & mask[:, None]) & ~torch.eye(N, dtype=torch.bool, device=mask.device)[None]
    coll = ((d < 0.2) & pm).any(-1)
    m = mask
    return {k: v[m].double().sum().item() for k, v in
            dict(minADE=minade, minFDE=minfde, ADE1=ade1, FDE1=fde1, COL1=coll.float()).items()}, m.sum().item()


def goal_conditioned(pred_modes, fut, mask):
    """Goal-conditioned navigation realism: every method knows start and goal (the true
    final position); learned models pick the mode whose end point is closest to the goal.
    Returns sums of discrete Frechet distance, path length, tortuosity and the GT path
    length/tortuosity over valid agents."""
    goal = fut[:, :, -1]
    if pred_modes.dim() == 5:
        j = (pred_modes[..., -1, :] - goal[:, :, None]).norm(dim=-1).argmin(-1)
        path = pred_modes.gather(2, j[..., None, None, None].expand(-1, -1, 1, *pred_modes.shape[3:])).squeeze(2)
        # endpoint conditioning: warp the selected mode linearly so that it ends exactly at the
        # goal (all baselines also receive the exact goal)
        ramp = torch.arange(1, path.shape[2] + 1, device=path.device, dtype=path.dtype) / path.shape[2]
        path = path + (goal - path[:, :, -1])[:, :, None] * ramp[None, None, :, None]
    else:
        path = pred_modes
    z = torch.zeros_like(path[:, :, :1])
    P = torch.cat([z, path], 2)[mask]
    Q = torch.cat([z, fut], 2)[mask]
    fr = discrete_frechet(P, Q)
    def plen(X):
        return (X[:, 1:] - X[:, :-1]).norm(dim=-1).sum(-1)
    def tort(X):
        return plen(X) / (X[:, -1] - X[:, 0]).norm(dim=-1).clamp(min=0.5)
    moving = (Q[:, -1] - Q[:, 0]).norm(dim=-1) > 1.0       # tortuosity only for agents that travel > 1 m
    return {"GC_Frechet": fr.sum().item(), "GC_PathLen": plen(P).sum().item(), "GC_PathLen_GT": plen(Q).sum().item(),
            "GC_Tort": tort(P)[moving].sum().item(), "GC_Tort_GT": tort(Q)[moving].sum().item(),
            "GC_moving": moving.sum().item()}


@torch.no_grad()
def evaluate(model_name, model, loader, bank, device, K, nsre_rerank=True, seed=0):
    tot, n = {}, 0
    gen = torch.Generator(device=device).manual_seed(seed)
    if model is not None:
        model.eval()
    for traj, mask, ret in loader:
        traj, mask = traj.to(device), mask.to(device)
        ret = None if ret is None else ret.to(device)
        obs, fut = traj[:, :, :OBS_LEN], traj[:, :, OBS_LEN:] - traj[:, :, OBS_LEN - 1:OBS_LEN]
        if model_name == "cv":
            pred = cv_predict(obs, K, gen=gen)
            scores = torch.zeros(pred.shape[:3], device=device); scores[..., 0] = 1
        elif model_name == "cv_det":
            pred = cv_predict(obs, K, sample=False)
            scores = torch.zeros(pred.shape[:3], device=device); scores[..., 0] = 1
        elif model_name == "sfm":
            pred = sfm_predict(obs, mask, K, gen=gen)
            scores = torch.zeros(pred.shape[:3], device=device); scores[..., 0] = 1
        else:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                pred, logits, _ = run_model(model, obs, mask, ret, bank)
            pred, logits = pred.float(), logits.float()
            scores = logits
            if nsre_rerank:
                viol = nsre_violation(pred, obs, mask) + collision_violation(pred, logits, obs, mask)
                sv = (viol < 1e-3).float()                             # Eq. (17)
                scores = logits.log_softmax(-1) + torch.log(sv * 0.99 + 0.01)   # Eq. (18): invalid modes demoted
        r, c = metrics_from_pred(pred, scores, fut, mask, obs)
        if model_name == "cv_det":
            r.update(goal_conditioned(straight_goal(obs, traj[:, :, -1]), fut, mask))
        elif model_name == "sfm":
            r.update(goal_conditioned(sfm_goal(obs, mask, traj[:, :, -1]), fut, mask))
        elif model_name != "cv":
            r.update(goal_conditioned(pred, fut, mask))
        for k, v in r.items():
            tot[k] = tot.get(k, 0.0) + v
        n += c
    res = {k: v / n for k, v in tot.items() if not k.startswith("GC_Tort") and k != "GC_moving"}
    if "GC_moving" in tot:
        res["GC_Tort"] = tot["GC_Tort"] / tot["GC_moving"]
        res["GC_Tort_GT"] = tot["GC_Tort_GT"] / tot["GC_moving"]
    return res | {"agents": n}


def train(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    t0 = time.time()
    torch.set_num_threads(args.threads)
    tr = TrajDataset(load_split(args.data, args.scene, "train", skip=args.train_skip))
    va = TrajDataset(load_split(args.data, args.scene, "val"))
    te = TrajDataset(load_split(args.data, args.scene, "test"))
    is_v2 = MODELS.get(args.model, {}).get("v2", False)
    needs_mem = args.model in ("nsrf", "nsrf_no_social", "nsrf_no_emotion", "nsrf_no_nsre") or is_v2
    bank = None
    if needs_mem:
        bank = build_memory_bank(tr, args.mem_k, device=device)
        tr.retrieval = bank["train_retrieval"]
        va.retrieval = retrieve(bank, va, args.mem_k, device=device)
        te.retrieval = retrieve(bank, te, args.mem_k, device=device)
        bank["values_dev"], bank["keys_dev"] = bank["values"].to(device), bank["keys"].to(device)
    mk = lambda ds, sh, aug: torch.utils.data.DataLoader(ds, batch_size=args.bs, shuffle=sh, num_workers=args.workers,
                                                         collate_fn=lambda b: collate(b, aug), drop_last=False)
    te_loader, va_loader = mk(te, False, False), mk(va, False, False)
    out = {"scene": args.scene, "model": args.model, "seed": args.seed, "config": vars(args)}
    if args.model in ("cv", "cv_det", "sfm"):
        out["test"] = evaluate(args.model, None, te_loader, None, device, args.K, seed=args.seed)
        out["time_s"] = time.time() - t0
        return out
    if args.model == "lstm":
        model = LSTMWTA(args.d, args.K)
    elif is_v2:
        kw = {k: v for k, v in MODELS[args.model].items() if k != "v2"}
        model = NSRFT2(d=args.d, heads=args.heads, K=args.K, mem_k=args.mem_k, t_start=args.t_start, **kw)
    else:
        model = NSRFT(d=args.d, heads=args.heads, K=args.K, mem_k=args.mem_k, **MODELS[args.model])
    model.to(device)
    if is_v2:
        refresh_memory(model, bank, tr, va, te, args.mem_k, device)
    out["params"] = sum(p.numel() for p in model.parameters())
    use_nsre = args.model not in ("nsrf_no_nsre", "lstm", "transformer")
    if args.eval_only:
        model.load_state_dict(torch.load(args.eval_only, map_location=device))
        if is_v2:
            refresh_memory(model, bank, tr, va, te, args.mem_k, device)
        out["test"] = evaluate(args.model, model, te_loader, bank, device, args.K, nsre_rerank=use_nsre)
        return out
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    tr_loader = mk(tr, True, True)
    total = args.epochs * len(tr_loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=total, pct_start=0.05)
    use_nsre = args.model not in ("nsrf_no_nsre", "lstm", "transformer")
    best, best_state, hist = 1e9, None, []
    for ep in range(args.epochs):
        model.train()
        tl, nb = 0.0, 0
        for traj, mask, ret in tr_loader:
            traj, mask = traj.to(device), mask.to(device)
            ret = None if ret is None else ret.to(device)
            obs, fut = traj[:, :, :OBS_LEN], traj[:, :, OBS_LEN:] - traj[:, :, OBS_LEN - 1:OBS_LEN]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                pred, logits, pad = run_model(model, obs, mask, ret, bank)
            pred, logits = pred.float(), logits.float()
            err = (pred - fut[:, :, None]).norm(dim=-1)
            ade, fde = err.mean(-1), err[..., -1]
            j = (ade + fde).argmin(-1)
            reg = (ade + fde).gather(-1, j[..., None]).squeeze(-1)
            cls = F.cross_entropy(logits[mask], j[mask])
            loss = reg[mask].mean() + 0.5 * cls
            if use_nsre:
                v = nsre_violation(pred, obs, mask) + collision_violation(pred, logits.detach(), obs, mask, stride=2)
                loss = loss + args.nsre_w * v[mask].mean()
            if is_v2:
                loss = loss + model.aux_losses(obs, fut, mask)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tl += loss.item(); nb += 1
        if is_v2 and (ep + 1) % args.mem_refresh == 0:
            refresh_memory(model, bank, tr, va, te, args.mem_k, device)
        if (ep + 1) % args.eval_every == 0 or ep == args.epochs - 1:
            if is_v2:
                model.t_start = 0          # checkpoints are selected on the initialiser; t_start is tuned afterwards
            v = evaluate(args.model, model, va_loader, bank, device, args.K, nsre_rerank=use_nsre)
            hist.append({"epoch": ep + 1, "train_loss": tl / nb, **{f"val_{k}": x for k, x in v.items()}})
            print(json.dumps(hist[-1]), flush=True)
            if v["minADE"] + v["minFDE"] < best:
                best = v["minADE"] + v["minFDE"]
                best_state = {k: x.detach().clone() for k, x in model.state_dict().items()}
    model.load_state_dict(best_state)
    if is_v2:
        refresh_memory(model, bank, tr, va, te, args.mem_k, device)
        if model.refine:   # choose the leapfrog start step on the validation split (never on test)
            grid = {}
            for ts in (0, 1, 2, 4, 6, 10):   # 0 = keep the initialiser's modes
                model.t_start = ts
                torch.manual_seed(args.seed)
                v = evaluate(args.model, model, va_loader, bank, device, args.K, nsre_rerank=use_nsre)
                grid[ts] = v["minADE"] + v["minFDE"]
            model.t_start = min(grid, key=grid.get)
            out["t_start_val_grid"], out["t_start"] = grid, model.t_start
            print("t_start grid (val minADE+minFDE):", grid, flush=True)
    out["history"] = hist
    out["test"] = evaluate(args.model, model, te_loader, bank, device, args.K, nsre_rerank=use_nsre)
    # inference latency per scene batch
    model.eval()
    traj, mask, ret = next(iter(te_loader))
    traj, mask = traj.to(device), mask.to(device); ret = None if ret is None else ret.to(device)
    with torch.no_grad():
        for _ in range(3):
            run_model(model, traj[:, :, :OBS_LEN], mask, ret, bank)
        if device.type == "cuda": torch.cuda.synchronize()
        t1 = time.time()
        for _ in range(10):
            run_model(model, traj[:, :, :OBS_LEN], mask, ret, bank)
        if device.type == "cuda": torch.cuda.synchronize()
    out["latency_ms_per_batch"] = (time.time() - t1) / 10 * 1000
    if device.type == "cuda":
        out["peak_gpu_mem_GB"] = torch.cuda.max_memory_allocated() / 2 ** 30
    out["time_s"] = time.time() - t0
    if args.save_ckpt:
        os.makedirs(os.path.dirname(args.save_ckpt) or ".", exist_ok=True)
        torch.save(model.state_dict(), args.save_ckpt)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/raw")
    p.add_argument("--scene", default="eth")
    p.add_argument("--model", default="nsrf", choices=list(MODELS) + ["lstm", "cv", "cv_det", "sfm"])
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--d", type=int, default=256)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--K", type=int, default=20)
    p.add_argument("--mem_k", type=int, default=10)
    p.add_argument("--nsre_w", type=float, default=0.1)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--train_skip", type=int, default=1, help="stride between training windows (1 = standard protocol)")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", default="results/traj")
    p.add_argument("--save_ckpt", default="")
    p.add_argument("--mem_refresh", type=int, default=2, help="epochs between re-encoding learned memory keys (v2)")
    p.add_argument("--t_start", type=int, default=10, help="leapfrog start step of the diffusion refinement (v2)")
    p.add_argument("--eval_only", default="", help="checkpoint to evaluate without training")
    args = p.parse_args()
    res = train(args)
    os.makedirs(args.out, exist_ok=True)
    fn = os.path.join(args.out, f"{args.scene}_{args.model}_s{args.seed}.json")
    with open(fn, "w") as f:
        json.dump(res, f, indent=1)
    print("TEST", args.scene, args.model, json.dumps(res["test"]))


if __name__ == "__main__":
    main()
