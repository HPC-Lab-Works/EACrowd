"""Multi-Agent PPO (parameter sharing, centralised critic; Yu et al., 2022) for the
high-level planner of NeuroSocial-RAGFormer, with an auxiliary MET distillation loss.

python -m sim.mappo --method nsrf --episodes 1000 --n_agents 500 --steps 3000   (DGX)
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch

from .config import METHODS, Scenario
from .core import Sim
from .nets import ActorCritic


def make_net(method, sim):
    idx = sim.active.nonzero().squeeze(-1)[:4]
    _, dist = sim.pairwise()
    b, _ = sim.build_obs(idx, dist)
    return ActorCritic(b["obs"].shape[-1], b["glob"].shape[-1], use_met=method.emotion == "met",
                       use_sgt=method.social == "dynamic", use_mem=method.memory == "graph")


def cat_b(bs):
    return {k: torch.cat([b[k] for b in bs], 0) for k in bs[0]}


def finish_episode(sim, net, gamma, lam):
    sim.flush_pending()
    buf = sim.buffer
    T, N = len(buf), sim.N
    dev = sim.dev
    r = torch.zeros(T, N, device=dev); v = torch.zeros(T + 1, N, device=dev)
    valid = torch.zeros(T, N, dtype=torch.bool, device=dev); done = torch.zeros(T, N, dtype=torch.bool, device=dev)
    for t, p in enumerate(buf):
        r[t, p["idx"]] = p["r"]; v[t, p["idx"]] = p["v"]; valid[t, p["idx"]] = True; done[t, p["idx"]] = p["done"]
    # bootstrap with the critic at the final state for agents still in the building
    idx = sim.active.nonzero().squeeze(-1)
    if len(idx):
        _, dist = sim.pairwise()
        b, _ = sim.build_obs(idx, dist)
        with torch.no_grad():
            _, vl, _ = net(b)
        v[T, idx] = vl
    adv = torch.zeros(T, N, device=dev)
    last = torch.zeros(N, device=dev)
    for t in reversed(range(T)):
        nv = torch.where(done[t], torch.zeros_like(v[t + 1]), v[t + 1])
        nonterm = (~done[t]).float()
        delta = r[t] + gamma * nv - v[t]
        last = delta + gamma * lam * nonterm * last
        last = last * valid[t]
        adv[t] = last
    ret = adv + v[:T]
    samples = {"b": cat_b([p["b"] for p in buf]), "a": torch.cat([p["a"] for p in buf]),
               "logp": torch.cat([p["logp"] for p in buf]), "pad_t": torch.cat([p["pad_t"] for p in buf]),
               "adv": torch.cat([adv[t, p["idx"]] for t, p in enumerate(buf)]),
               "ret": torch.cat([ret[t, p["idx"]] for t, p in enumerate(buf)])}
    ep_ret = (r.sum(0) / valid.sum(0).clamp(min=1)).mean().item()
    return samples, ep_ret


def ppo_update(net, opt, S, epochs=4, mb=8192, clip=0.2, vcoef=0.5, ent=0.01, met_coef=1.0):
    n = len(S["a"])
    adv = (S["adv"] - S["adv"].mean()) / (S["adv"].std() + 1e-8)
    logs = []
    for _ in range(epochs):
        perm = torch.randperm(n, device=S["a"].device)
        for i in range(0, n, mb):
            j = perm[i:i + mb]
            b = {k: x[j] for k, x in S["b"].items()}
            logits, v, pad = net(b)
            d = torch.distributions.Categorical(logits=logits)
            ratio = torch.exp(d.log_prob(S["a"][j]) - S["logp"][j])
            pl = -torch.min(ratio * adv[j], ratio.clamp(1 - clip, 1 + clip) * adv[j]).mean()
            vl = (v - S["ret"][j]).pow(2).mean()
            loss = pl + vcoef * vl - ent * d.entropy().mean()
            if pad is not None:
                loss = loss + met_coef * (pad - S["pad_t"][j]).pow(2).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()
            logs.append([pl.item(), vl.item(), d.entropy().mean().item()])
    return np.mean(logs, 0).tolist()


def train(args):
    torch.manual_seed(args.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    method = METHODS[args.method]
    sc = Scenario(n_agents=args.n_agents, steps=args.steps)
    probe = Sim(method, sc, seed=0, device=dev)
    net = make_net(method, probe).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    log = []
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    for ep in range(args.episodes):
        net.train()
        sim = Sim(method, sc, seed=10_000 + ep + 1000 * args.seed, device=dev, policy=net)
        sim.buffer, sim._pending = [], None
        summ = sim.run()
        S, ep_ret = finish_episode(sim, net, args.gamma, args.lam)
        stats = ppo_update(net, opt, S, epochs=args.ppo_epochs, mb=args.mb)
        rec = {"episode": ep, "return": ep_ret, "pl_vl_ent": stats, "evac_t95": summ["evac_t95_s"],
               "evac_frac": summ["evac_fraction"], "info_cov": summ["info_coverage_at_em"],
               "tasks": summ["tasks_per_agent_hour"], "coll": summ["collision_rate_normal"],
               "samples": len(S["a"]), "elapsed_s": time.time() - t0}
        log.append(rec)
        print(json.dumps(rec), flush=True)
    path = os.path.join(args.out, f"{args.method}_s{args.seed}.pt")
    torch.save({"state": net.state_dict(), "method": args.method, "args": vars(args)}, path)
    json.dump(log, open(os.path.join(args.out, f"{args.method}_s{args.seed}_log.json"), "w"))
    return path


def load_policy(path, method, sim):
    ck = torch.load(path, map_location=sim.dev)
    net = make_net(method, sim).to(sim.dev)
    net.load_state_dict(ck["state"])
    net.eval()
    return net


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--method", default="nsrf")
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--n_agents", type=int, default=500)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--ppo_epochs", type=int, default=4)
    p.add_argument("--mb", type=int, default=8192)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--out", default="checkpoints/sim")
    train(p.parse_args())
