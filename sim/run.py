"""Evaluation driver for the hospital digital-twin benchmark.

python -m sim.run eval   --methods sfm astar ga ebp nsrf --seeds 0 1 2 --n_agents 500 --steps 3000
python -m sim.run scale  --methods sfm nsrf --sizes 100 500 1000 2000
python -m sim.run days   --methods ga nsrf --days 3
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import time

import numpy as np
import torch

from .config import METHODS, Scenario
from .core import Sim
from .metrics import human_likeness, real_reference
from .mappo import load_policy

PPO_CKPT = {"nsrf": "nsrf", "nsrf_no_ramg": "nsrf_no_ramg", "nsrf_no_met": "nsrf_no_met",
            "nsrf_no_sgt": "nsrf_no_sgt", "nsrf_no_nsre": "nsrf_no_nsre", "nsrf_no_ppo": "nsrf",
            "nsrf_v2": "nsrf_v2", "nsrf_v2_task_only": "nsrf_v2_task_only", "nsrf_v2_no_escort": "nsrf_v2_no_escort",
            "nsrf_v2_no_exitbal": "nsrf_v2_no_exitbal", "nsrf_v3": "nsrf_v3"}


def attach_policy(sim, key, ckpt_dir, train_seed=0):
    if key not in PPO_CKPT:
        return sim
    path = os.path.join(ckpt_dir, f"{PPO_CKPT[key]}_s{train_seed}.pt")
    if not os.path.exists(path):
        print(f"[warn] no checkpoint {path}; {key} runs with its rule fallback")
        return sim
    src_method = METHODS[PPO_CKPT[key]]
    sim.policy = load_policy(path, src_method, sim)
    return sim


LLM_HOOK = None


def evaluate(args):
    global LLM_HOOK
    if args.llm:
        from .llm_hook import LLMHook
        LLM_HOOK = LLMHook(args.llm, n_agents=args.llm_agents)
        args.out = args.out + "_llm"
    ref = real_reference(args.data)
    dev = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    os.makedirs(args.out, exist_ok=True)
    for key in args.methods:
        for seed in args.seeds:
            fn = os.path.join(args.out, f"{key}_N{args.n_agents}_s{seed}.json")
            if os.path.exists(fn) and not args.force:
                continue
            sc = Scenario(n_agents=args.n_agents, steps=args.steps)
            method = METHODS[key]
            if args.llm:
                from dataclasses import replace as _r
                method = _r(method, llm=True, name=method.name + " + LLM")
            sim = Sim(method, sc, seed=seed, device=dev, llm=LLM_HOOK)
            attach_policy(sim, key, args.ckpt)
            res = sim.run()
            res["HLS"], res["HLS_parts"] = human_likeness(res["hist"], ref)
            res["group_fraction_real"] = ref["group_fraction"]
            res["group_fraction_abs_err"] = abs(res["group_fraction"] - ref["group_fraction"])
            res["key"], res["seed"] = key, seed
            res["ms_per_step"] = 1000 * res["wall_time_s"] / sc.steps
            json.dump(res, open(fn, "w"))
            print(key, seed, {k: res[k] for k in ("HLS", "collision_rate_normal", "evac_t95_s", "info_t50_s",
                                                   "group_fraction", "ms_per_step")}, flush=True)


def scale(args):
    dev = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    os.makedirs(args.out, exist_ok=True)
    rows = []
    for key in args.methods:
        for n in args.sizes:
            sc = Scenario(n_agents=n, steps=args.steps, announce_frac=0.0, emergency_frac=2.0)
            sim = Sim(METHODS[key], sc, seed=0, device=dev)
            attach_policy(sim, key, args.ckpt)
            for _ in range(10):
                sim.step()                                   # warm-up
            if dev == "cuda":
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            t0 = time.time()
            for _ in range(args.steps):
                sim.step()
            if dev == "cuda":
                torch.cuda.synchronize()
            el = time.time() - t0
            mem = torch.cuda.max_memory_allocated() / 2 ** 30 if dev == "cuda" else \
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20
            row = {"method": key, "N": n, "ms_per_step": 1000 * el / args.steps, "steps_per_s": args.steps / el,
                   "realtime_factor": args.steps * sc.dt / el, "peak_mem_GB": mem, "device": dev}
            rows.append(row)
            print(json.dumps(row), flush=True)
    json.dump(rows, open(os.path.join(args.out, f"scalability_{dev}.json"), "w"), indent=1)


def topk_ties(sim, k=5):
    F = sim.F()
    top = F.topk(k, -1)
    return [set(i[v > 0.3].tolist()) for i, v in zip(top.indices.cpu(), top.values.cpu())]


def days(args):
    """Scenario 5: multi-day continuity. Memory graph and social graph persist across
    days; we report the Relationship Stability Index (mean Jaccard of each agent's top-5
    ties between consecutive days) and long-term memory retention."""
    dev = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    os.makedirs(args.out, exist_ok=True)
    out = {}
    for key in args.methods:
        for seed in args.seeds:
            persist, prev, rows = None, None, []
            for d in range(args.days):
                sc = Scenario(n_agents=args.n_agents, steps=args.steps, emergency_frac=2.0)  # no emergency on routine days
                sim = Sim(METHODS[key], sc, seed=seed, device=dev, persist=persist)   # same population every day
                sim.g.manual_seed(seed * 100 + d)                                      # different daily dynamics
                attach_policy(sim, key, args.ckpt)
                sim.run()
                ties = topk_ties(sim)
                n_ties = float(np.mean([len(t) for t in ties]))
                rsi = None
                if prev is not None:
                    js = [len(a & b) / len(a | b) for a, b in zip(prev, ties) if len(a | b) > 0]
                    rsi = float(np.mean(js)) if js else 0.0
                old = (sim.mem_valid & (sim.mem_t < 0)).float().sum(-1).mean().item()
                rows.append({"day": d, "mean_top5_ties": n_ties, "RSI_vs_prev_day": rsi,
                             "memories_from_previous_days": old,
                             "memory_slots_used": sim.mem_valid.float().sum(-1).mean().item()})
                prev, persist = ties, sim.persist_state()
                print(key, seed, rows[-1], flush=True)
            out[f"{key}_s{seed}"] = rows
    json.dump(out, open(os.path.join(args.out, "multiday.json"), "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["eval", "scale", "days"])
    p.add_argument("--methods", nargs="+", default=["sfm", "astar", "ga", "ebp", "nsrf"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--sizes", nargs="+", type=int, default=[100, 500, 1000, 2000])
    p.add_argument("--n_agents", type=int, default=500)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--days", type=int, default=3)
    p.add_argument("--data", default="data/raw")
    p.add_argument("--ckpt", default="checkpoints/sim")
    p.add_argument("--out", default="results/sim")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--llm", default="", help="HF model id for the optional LLM reasoning hook (GPU)")
    p.add_argument("--llm_agents", type=int, default=32)
    a = p.parse_args()
    {"eval": evaluate, "scale": scale, "days": days}[a.mode](a)
