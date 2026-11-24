"""Vectorised (PyTorch, CPU or CUDA) hospital crowd simulator shared by all methods.

One step = 0.4 s (same sampling as ETH/UCY so kinematic statistics are comparable).
Method-specific behaviour is switched by `config.Method` flags; everything else
(map, population, schedules, events, metrics) is identical across methods.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

from .config import ROLE_COUNTS_500, ROLES, Method, Scenario
from .nets import N_ACTIONS
from .world import EXITS, RES, Z, ZONES, build_world

NZ = len(ZONES)
EV_TYPES = ["task_done", "long_wait", "crowd", "alarm", "info", "social"]
EV_SIG = torch.tensor([0.2, 0.3, 0.4, 1.0, 0.6, 0.5])
D_MEM = NZ + len(EV_TYPES) + 3
MEM_SLOTS = 32

# ------------------------------------------------------------------ schedules
PREF = {  # destination preferences over zones 0..11 (corridor never a goal)
    "doctor": {4: .12, 5: .12, 6: .12, 7: .12, 8: .22, 9: .15, 10: .10, 11: .05},
    "nurse": {2: .12, 3: .12, 8: .22, 9: .22, 4: .04, 5: .04, 6: .04, 7: .04, 1: .08, 11: .08},
    "visitor": {2: .2, 3: .2, 9: .25, 11: .2, 1: .15},
    "admin": {10: .6, 1: .3, 11: .1},
}
PATIENT_T = {1: {2: .5, 3: .5}, 2: {4: .175, 5: .175, 6: .175, 7: .175, 11: .15, 1: .15},
             3: {4: .175, 5: .175, 6: .175, 7: .175, 11: .15, 1: .15},
             4: {2: .25, 3: .25, 11: .3, 1: .2}, 5: {2: .25, 3: .25, 11: .3, 1: .2},
             6: {2: .25, 3: .25, 11: .3, 1: .2}, 7: {2: .25, 3: .25, 11: .3, 1: .2},
             11: {2: .4, 3: .4, 1: .2}}
DWELL = {"doctor": 150, "nurse": 90, "visitor": 120, "admin": 200}
PATIENT_DWELL = {1: 40, 2: 150, 3: 150, 4: 120, 5: 120, 6: 120, 7: 120, 11: 90}


def _trans_tensor():
    T = torch.zeros(len(ROLES), NZ, NZ, device='cpu')
    for r, role in enumerate(ROLES):
        for z in range(NZ):
            if role == "patient":
                row = PATIENT_T.get(z, {1: 1.0})
            else:
                row = dict(PREF[role]); row.pop(z, None)
            for k, v in row.items():
                T[r, z, k] = v
        T[r] /= T[r].sum(-1, keepdim=True).clamp(min=1e-9)
    return T


class Sim:
    def __init__(self, method: Method, sc: Scenario, seed: int = 0, device="cpu", policy=None, world=None,
                 llm=None, persist: dict | None = None):
        self.m, self.sc, self.dev = method, sc, torch.device(device)
        self.g = torch.Generator(device=self.dev).manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.world = world or build_world()
        w = self.world
        tt = lambda a, dt=torch.float32: torch.as_tensor(a, dtype=dt, device=self.dev)
        self.walls, self.zone_map = tt(w.walls, torch.bool), tt(w.zone, torch.long)
        self.dist_f, self.dirs = tt(w.dist), tt(w.dirs)
        self.wall_d, self.wall_g = tt(w.wall_dist), tt(w.wall_grad)
        self.H, self.W = w.shape
        self.rects = torch.zeros(NZ, 4, device=self.dev)
        for name, r in w.rects.items():
            self.rects[Z[name]] = tt(r)
        self.rects[Z["exit_main"]] = tt((0.2, 1.2, 23.5, 26.5))
        self.rects[Z["exit_east"]] = tt((78.8, 79.8, 23.5, 26.5))
        self.area = ((self.rects[:, 1] - self.rects[:, 0]) * (self.rects[:, 3] - self.rects[:, 2])).clamp(min=1)
        self.exit_c = torch.stack([self.rects[EXITS, :2].mean(-1), self.rects[EXITS, 2:].mean(-1)], -1)
        self.hazard = tt(w.hazard_center)
        self.trans = _trans_tensor().to(self.dev)
        self.policy, self.llm = policy, llm
        self.t = 0
        self._init_agents(persist)
        self._init_metrics()

    # ------------------------------------------------------------------ helpers
    def rand(self, *s):
        return torch.rand(*s, generator=self.g, device=self.dev)

    def randn(self, *s):
        return torch.randn(*s, generator=self.g, device=self.dev)

    def cells(self, p):
        ix = (p[:, 0] / RES).long().clamp(0, self.W - 1)
        iy = (p[:, 1] / RES).long().clamp(0, self.H - 1)
        return iy, ix

    def sample_point(self, zones):
        r = self.rects[zones]
        u = self.rand(len(zones), 2)
        margin = 0.8
        x = r[:, 0] + margin + u[:, 0] * (r[:, 1] - r[:, 0] - 2 * margin).clamp(min=0.1)
        y = r[:, 2] + margin + u[:, 1] * (r[:, 3] - r[:, 2] - 2 * margin).clamp(min=0.1)
        return torch.stack([x, y], -1)

    def sample_dwell(self, idx, zones):
        mean = torch.zeros(len(idx), device=self.dev)
        for r, role in enumerate(ROLES):
            sel = self.role[idx] == r
            if role == "patient":
                pd = torch.tensor([PATIENT_DWELL.get(z, 60) for z in range(NZ)], device=self.dev, dtype=torch.float32)
                mean[sel] = pd[zones[sel]]
            else:
                mean[sel] = DWELL[role]
        mean = mean * self.sc.dwell_scale
        d = -mean * torch.log(self.rand(len(idx)).clamp(min=1e-6))
        return (d.clamp(min=10.0) / self.sc.dt).long()

    # ------------------------------------------------------------------ init
    def _init_agents(self, persist):
        N, dev = self.sc.n_agents, self.dev
        counts = {r: max(1, round(c * N / 500)) for r, c in ROLE_COUNTS_500.items()}
        counts["patient"] += N - sum(counts.values())
        self.N = N
        self.role = torch.cat([torch.full((c,), i, dtype=torch.long, device=dev) for i, c in enumerate(counts.values())]).to(dev)
        self.ocean = self.rand(N, 5)                       # O, C, E, A, Neuroticism  ~ U(0,1)
        O, C, E, A, Nn = self.ocean.unbind(-1)
        S = 1 - Nn                                          # emotional stability
        c = lambda x: 2 * x - 1                             # [0,1] -> [-1,1]
        # Mehrabian (1996) temperament mapping as used in ALMA (Gebhard, 2005)
        P0 = 0.21 * c(E) + 0.59 * c(A) + 0.19 * c(S)
        A0 = 0.15 * c(O) + 0.30 * c(A) - 0.57 * c(S)
        D0 = 0.25 * c(O) + 0.17 * c(C) + 0.60 * c(E) - 0.32 * c(A)
        self.pad_base = torch.stack([P0, A0, D0], -1).clamp(-1, 1) * 0.6
        self.pad = self.pad_base.clone()
        self.pad_teacher = self.pad_base.clone()
        if self.m.emotion == "none":
            self.pad_base.zero_(); self.pad.zero_(); self.pad_teacher.zero_()
        self.v_pref = (1.3 + 0.2 * self.randn(N)).clamp(0.8, 1.8)
        self.v_pref[self.role == ROLES.index("patient")] *= 0.85
        self.radius = torch.full((N,), 0.25, device=dev)
        # family ties: each visitor accompanies one patient
        pat = (self.role == ROLES.index("patient")).nonzero().squeeze(-1)
        vis = (self.role == ROLES.index("visitor")).nonzero().squeeze(-1)
        self.family = torch.full((N,), -1, dtype=torch.long, device=dev)
        partner = pat[torch.randperm(len(pat), generator=self.g, device=dev)[:len(vis)]]
        self.family[vis] = partner
        self.family[partner] = vis
        # initial goal zone & position
        goal = torch.zeros(N, dtype=torch.long, device=dev)
        for r, role in enumerate(ROLES):
            sel = (self.role == r).nonzero().squeeze(-1)
            if role == "patient":
                pz = torch.tensor([0, .3, .2, .2, .05, .05, .05, .05, 0, 0, 0, .1, 0, 0], device=dev)
            else:
                pz = torch.zeros(NZ, device=dev)
                for k, v in PREF[role].items():
                    pz[k] = v
            goal[sel] = torch.multinomial(pz / pz.sum(), len(sel), replacement=True, generator=self.g)
        goal[vis] = goal[partner]
        self.goal = goal
        self.pos = self.sample_point(goal)
        self.pos[vis] = self.pos[partner] + 0.7 * torch.randn(len(vis), 2, generator=self.g, device=dev).clamp(-1, 1)
        self.pos = self._fix_inside(self.pos)
        self.target = self.pos.clone()
        self.vel = torch.zeros(N, 2, device=dev)
        self.at_goal = torch.ones(N, dtype=torch.bool, device=dev)
        self.dwell = (self.rand(N) * self.sample_dwell(torch.arange(N, device=dev), goal).float()).long()
        self.active = torch.ones(N, dtype=torch.bool, device=dev)
        self.em_known = torch.zeros(N, dtype=torch.bool, device=dev)
        self.informed = torch.zeros(N, dtype=torch.bool, device=dev)
        self.exit_step = torch.full((N,), -1, dtype=torch.long, device=dev)
        self.exit_used = torch.full((N,), -1, dtype=torch.long, device=dev)
        self.wait_time = torch.zeros(N, device=dev)
        self.follow = torch.full((N,), -1, dtype=torch.long, device=dev)
        self.evac_exit = torch.full((N,), -1, dtype=torch.long, device=dev)
        self.action = torch.zeros(N, dtype=torch.long, device=dev)
        self.share_boost = torch.zeros(N, device=dev)
        self.last_crowd_mem = torch.full((N,), -10 ** 6, dtype=torch.long, device=dev)
        self.waited_logged = torch.zeros(N, dtype=torch.bool, device=dev)
        self.helped = torch.zeros(N, dtype=torch.bool, device=dev)
        self.task_pending = torch.zeros(N, dtype=torch.bool, device=dev)  # a scheduled task awaits completion
        # social graph (Eq. 11-12): raw accumulators -> F, T, I in [0, 1)
        self.Fr = torch.zeros(N, N, device=dev); self.Tr = torch.zeros(N, N, device=dev); self.Ir = torch.zeros(N, N, device=dev)
        if self.m.social in ("static", "dynamic"):
            has = self.family >= 0
            i = has.nonzero().squeeze(-1)
            v0 = -math.log(1 - 0.8)
            self.Fr[i, self.family[i]] = v0; self.Tr[i, self.family[i]] = v0; self.Ir[i, self.family[i]] = v0
        # memory graph (RAMG) / memory stream
        self.mem_key = torch.zeros(N, MEM_SLOTS, D_MEM, device=dev)
        self.mem_t = torch.zeros(N, MEM_SLOTS, device=dev)
        self.mem_sig = torch.zeros(N, MEM_SLOTS, device=dev)
        self.mem_valid = torch.zeros(N, MEM_SLOTS, dtype=torch.bool, device=dev)
        if persist:  # multi-day continuity (Scenario 5)
            for k, v in persist.items():
                getattr(self, k).copy_(v)
        self.r_acc = torch.zeros(N, device=dev)
        self.prev_goal_dist = self.geo_dist(self.goal, self.pos)

    def _fix_inside(self, p):
        iy, ix = self.cells(p)
        bad = self.walls[iy, ix]
        if bad.any():
            p[bad] = self.sample_point(self.goal[bad])
        return p

    def F(self):
        return 1 - torch.exp(-self.Fr)

    def T_(self):
        return 1 - torch.exp(-self.Tr)

    def I_(self):
        return 1 - torch.exp(-self.Ir)

    def geo_dist(self, zones, p):
        iy, ix = self.cells(p)
        d = self.dist_f[zones, iy, ix]
        return torch.where(torch.isfinite(d), d, torch.full_like(d, 200.0))

    # ------------------------------------------------------------------ metrics
    def _init_metrics(self):
        self.bins = {"speed": torch.linspace(0, 3, 31, device=self.dev), "acc": torch.linspace(0, 4, 41, device=self.dev),
                     "turn": torch.linspace(0, math.pi / 0.4, 41, device=self.dev), "nn": torch.linspace(0, 6, 31, device=self.dev)}
        self.hist = {k: torch.zeros(len(b) + 1, device=self.dev) for k, b in self.bins.items()}
        for k in self.bins:
            self.bins[k] = self.bins[k].to(self.dev)
        rec = torch.randperm(self.N, generator=self.g, device=self.dev)[: self.sc.record_agents]
        self.rec_mask = torch.zeros(self.N, dtype=torch.bool, device=self.dev); self.rec_mask[rec] = True
        self.prev_v = torch.zeros(self.N, 2, device=self.dev)
        self.pair_cnt = torch.zeros(self.N, self.N, dtype=torch.int16, device=self.dev)
        self.stats = {"group_frac_sum": 0.0, "group_frac_n": 0, "coll_steps": 0.0, "agent_steps": 0.0,
                      "coll_steps_evac": 0.0, "agent_steps_evac": 0.0, "tasks": 0, "tasks_nv": 0, "agent_steps_nv": 0.0, "trip_len": 0.0,
                      "trip_geo": 0.0, "trip_time": 0.0, "trips": 0, "info_t50": None, "info_cov_at_em": None,
                      "max_exit_density": 0.0, "pad_normal": torch.zeros(3, device=self.dev), "pad_normal_n": 0,
                      "pad_em": torch.zeros(3, device=self.dev), "pad_em_n": 0, "helpers": 0,
                      "contagion": None, "ties_at_em": None}
        self.trip_len = torch.zeros(self.N, device=self.dev)
        self.trip_start = self.pos.clone()
        self.trip_t0 = torch.zeros(self.N, device=self.dev)
        self.trip_geo0 = torch.zeros(self.N, device=self.dev)

    @property
    def t_announce(self):
        return int(self.sc.announce_frac * self.sc.steps)

    @property
    def t_em(self):
        return int(self.sc.emergency_frac * self.sc.steps)

    # ------------------------------------------------------------------ memory
    def mem_insert(self, mask, etype, zone):
        if self.m.memory == "none" or not mask.any():
            return
        idx = mask.nonzero().squeeze(-1)
        key = torch.zeros(len(idx), D_MEM, device=self.dev)
        key[torch.arange(len(idx), device=self.dev), zone[idx]] = 1.0
        key[:, NZ + etype] = 1.0
        key[:, NZ + len(EV_TYPES):] = self.pad[idx]
        rec = torch.exp(-(self.t - self.mem_t[idx]) * self.sc.dt / 600.0)
        imp = torch.where(self.mem_valid[idx], 0.5 * rec + 0.5 * self.mem_sig[idx], torch.full_like(rec, -1.0))
        slot = imp.argmin(-1)
        # RAMG (Alg. 2): reinforce edges to similar memories -> stored as significance boost
        if self.m.memory == "graph":
            sim = torch.nn.functional.cosine_similarity(self.mem_key[idx], key[:, None], dim=-1)
            linked = (sim > 0.8) & self.mem_valid[idx]
            self.mem_sig[idx] = torch.where(linked, (self.mem_sig[idx] + 0.05).clamp(max=1.0), self.mem_sig[idx])
        self.mem_key[idx, slot] = key
        self.mem_t[idx, slot] = float(self.t)
        self.mem_sig[idx, slot] = EV_SIG.to(self.dev)[etype]
        self.mem_valid[idx, slot] = True

    def retrieve_memory(self, idx, k=10):
        """Eq. (5)-(6): score = a*recency + b*relevance + g*significance. The RAMG variant
        expands the top-3 seeds to graph neighbours (cosine > tau) before top-k pooling;
        the memory-stream variant (Generative Agents) uses the flat score only."""
        if self.m.memory == "none":
            return torch.zeros(len(idx), D_MEM, device=self.dev)
        q = torch.zeros(len(idx), D_MEM, device=self.dev)
        q[torch.arange(len(idx), device=self.dev), self.goal[idx]] = 1.0
        q[:, NZ + len(EV_TYPES):] = self.pad[idx]
        keys, valid = self.mem_key[idx], self.mem_valid[idx]
        rel = torch.nn.functional.cosine_similarity(keys, q[:, None], dim=-1).clamp(min=0)
        rec = torch.exp(-(self.t - self.mem_t[idx]) * self.sc.dt / 600.0)
        score = 0.3 * rec + 0.4 * rel + 0.3 * self.mem_sig[idx]
        score = score.masked_fill(~valid, -1e9)
        if self.m.memory == "graph":
            seeds = score.topk(3, -1).indices
            sk = keys.gather(1, seeds[..., None].expand(-1, -1, D_MEM))
            adj = torch.nn.functional.cosine_similarity(keys[:, :, None], sk[:, None], dim=-1)  # [n,M,3]
            ss = score.gather(1, seeds).clamp(min=0)
            boost = ((adj > 0.8).float() * ss[:, None]).max(-1).values
            score = torch.where(valid, score + 0.5 * boost, score)
        top = score.topk(k, -1)
        wts = torch.softmax(top.values * 5, -1) * (top.values > -1e8)
        return (wts[..., None] * keys.gather(1, top.indices[..., None].expand(-1, -1, D_MEM))).sum(1)

    # ------------------------------------------------------------------ goals
    def choose_next_goal(self, idx):
        cur = self.goal[idx]
        probs = self.trans[self.role[idx], cur]
        nxt = torch.multinomial(probs, 1, generator=self.g).squeeze(-1)
        if self.m.density_routing:
            occ = torch.bincount(self.goal[self.active], minlength=NZ).float() / self.area
            is_w = (nxt == 2) | (nxt == 3)
            if is_w.any():
                unhappy = 1 + (-self.pad[idx, 0]).clamp(min=0)
                cost = torch.stack([occ[2], occ[3]])[None] * unhappy[:, None] * 3.0
                pick = torch.multinomial(torch.softmax(-cost * 4, -1), 1, generator=self.g).squeeze(-1)
                nxt = torch.where(is_w, 2 + pick, nxt)
        return nxt

    def set_goal(self, idx, zones):
        if len(idx) == 0:
            return
        self.goal[idx] = zones
        self.target[idx] = self.sample_point(zones)
        self.at_goal[idx] = False
        self.waited_logged[idx] = False
        self.trip_start[idx] = self.pos[idx]
        self.trip_len[idx] = 0
        self.trip_t0[idx] = float(self.t)
        self.trip_geo0[idx] = self.geo_dist(zones, self.pos[idx])
        self.prev_goal_dist[idx] = self.trip_geo0[idx]

    # ------------------------------------------------------------------ physics
    def pairwise(self):
        d = self.pos[:, None] - self.pos[None]               # i - j
        dist = d.norm(dim=-1)
        act = self.active
        valid = act[:, None] & act[None] & ~torch.eye(self.N, dtype=torch.bool, device=self.dev)
        dist = dist.masked_fill(~valid, 1e4)
        return d, dist

    def desired_velocity(self):
        iy, ix = self.cells(self.pos)
        zone_now = self.zone_map[iy, ix]
        nav_zone = self.goal.clone()
        nav_pt = self.target.clone()
        lead = self.follow.clone()
        if self.m.escort:
            fam = self.family
            esc = (lead < 0) & (fam >= 0) & (self.role == ROLES.index("visitor")) & (self.evac_exit < 0)
            esc &= self.active[fam.clamp(min=0)]
            lead = torch.where(esc, fam, lead)
        self._lead = lead
        fol = lead >= 0
        if fol.any():
            f = lead[fol]
            fiy, fix = self.cells(self.pos[f])
            nav_zone[fol] = self.zone_map[fiy, fix].clamp(min=0)
            nav_pt[fol] = self.pos[f]
        evac = self.evac_exit >= 0
        nav_zone[evac] = self.evac_exit[evac]
        nav_pt[evac] = self.exit_c[self.evac_exit[evac] - EXITS[0]]
        same = (zone_now == nav_zone) | ((nav_zone == 0) & (zone_now >= 0))
        field = self.dirs[nav_zone, iy, ix]
        direct = nav_pt - self.pos
        dd = direct.norm(dim=-1, keepdim=True)
        dirv = torch.where(same[:, None], direct / dd.clamp(min=1e-6), field)
        speed = self.v_pref.clone()
        if self.m.emotion != "none":
            speed = speed * (1 + 0.3 * self.pad[:, 1].clamp(min=0))
        speed = torch.where(evac, speed * 1.3, speed)
        # stop when arrived / dwelling / close to followed friend
        stop = self.at_goal & ~evac & ~fol
        close_f = fol & (dd.squeeze(-1) < 1.2)
        slow = same & (dd.squeeze(-1) < 1.0) & ~evac
        vdes = dirv * speed[:, None]
        vdes = torch.where(slow[:, None], direct.clamp(-1, 1) * speed[:, None] * 0.8, vdes)
        if close_f.any():
            vdes[close_f] = self.vel[lead[close_f]]          # velocity alignment with the companion
        if self.m.escort:
            # side-by-side walking (Moussaid et al., 2010): keep a lateral offset next to the companion
            # and match its velocity; proportional catch-up toward that slot
            near = fol & (dd.squeeze(-1) < 4.0)
            if near.any():
                li = lead[near]
                vl = 0.5 * (self.vel[li] + vdes[li])          # companion's motion intention, not only its velocity
                e = vl / vl.norm(dim=-1, keepdim=True).clamp(min=1e-3)
                perp = torch.stack([-e[:, 1], e[:, 0]], -1)
                side = torch.sign(((self.pos[near] - self.pos[li]) * perp).sum(-1, keepdim=True) + 1e-6)
                slot = self.pos[li] + 0.75 * side * perp
                v = vl + 0.8 * (slot - self.pos[near])
                vmax = (1.5 * self.v_pref[near])[:, None]
                v = v * (torch.minimum(v.norm(dim=-1, keepdim=True), vmax) / v.norm(dim=-1, keepdim=True).clamp(min=1e-6))
                still = vl.norm(dim=-1) < 0.2
                v = torch.where((still & ((self.pos[near] - self.pos[li]).norm(dim=-1) < 1.2))[:, None], torch.zeros_like(v), v)
                vdes[near] = v
        vdes = torch.where(stop[:, None], torch.zeros_like(vdes), vdes)
        return vdes, zone_now, dirv

    def forces(self, vdes, dirv, d, dist):
        r_i = self.radius
        if self.m.emotion != "none":
            r_i = r_i * (1 + 0.4 * (-self.pad[:, 0]).clamp(min=0))
        rij = r_i[:, None] + self.radius[None]
        n = d / dist[..., None].clamp(min=1e-3)
        near = dist < 3.0
        if self.m.motion == "sfm":
            e = dirv
            cosphi = -(n * e[:, None]).sum(-1)
            lam = 0.35
            w = lam + (1 - lam) * (1 + cosphi) / 2
            mag = 2.0 * torch.exp((rij - dist) / 0.3) * w
        else:   # A*: reactive short-range avoidance only
            mag = 1.0 * torch.exp((rij - dist) / 0.1)
        lead = getattr(self, "_lead", None)
        if self.m.escort and lead is not None and (lead >= 0).any():
            # companions keep only a weak social repulsion between them (physical contact force stays)
            i = (lead >= 0).nonzero().squeeze(-1)
            mag[i, lead[i]] *= 0.2
            mag[lead[i], i] *= 0.2
        mag = mag + 10.0 * (rij - dist).clamp(min=0)
        f = (mag * near)[..., None] * n
        f = f.sum(1)
        iy, ix = self.cells(self.pos)
        wd = self.wall_d[iy, ix]
        f = f + (3.0 * torch.exp((0.25 - wd) / 0.15))[:, None] * self.wall_g[iy, ix]
        if self.m.groups:
            lead = getattr(self, "_lead", self.follow)
            fol = lead >= 0
            if fol.any():
                to = self.pos[lead[fol]] - self.pos[fol]
                dd = to.norm(dim=-1, keepdim=True)
                f[fol] += 0.5 * (dd - 1.0).clamp(min=0) * to / dd.clamp(min=1e-3)
        return f

    def move(self, vdes, dirv, h):
        d, dist = self.pairwise()
        f = self.forces(vdes, dirv, d, dist)
        tau = torch.full((self.N, 1), 0.5, device=self.dev)
        if self.m.escort and getattr(self, "_lead", None) is not None:
            tau[self._lead >= 0] = 0.3                     # companions adapt faster to each other
        v = self.vel + h * ((vdes - self.vel) / tau + f)
        vmax = (1.6 * self.v_pref).clamp(max=2.6)
        sp = v.norm(dim=-1, keepdim=True)
        v = v * (torch.minimum(sp, vmax[:, None]) / sp.clamp(min=1e-6))
        v = v * self.active[:, None]
        newp = self.pos + h * v
        iy, ix = self.cells(newp)
        bad = self.walls[iy, ix]
        if bad.any():
            px = torch.stack([newp[:, 0], self.pos[:, 1]], -1)
            py = torch.stack([self.pos[:, 0], newp[:, 1]], -1)
            okx = ~self.walls[self.cells(px)]
            oky = ~self.walls[self.cells(py)]
            alt = torch.where(okx[:, None], px, torch.where(oky[:, None], py, self.pos))
            v = torch.where(bad[:, None], torch.where(okx[:, None], v * torch.tensor([1., 0.], device=self.dev),
                                                      torch.where(oky[:, None], v * torch.tensor([0., 1.], device=self.dev),
                                                                  torch.zeros_like(v))), v)
            newp = torch.where(bad[:, None], alt, newp)
        moved = (newp - self.pos).norm(dim=-1)
        self.pos, self.vel = newp, v
        return moved

    # ------------------------------------------------------------------ main step
    def step(self):
        sc, dt, t = self.sc, self.sc.dt, self.t
        if t % sc.decision_every == 0:
            self.decide()
        if t == self.t_announce:
            cand = ((self.role == ROLES.index("nurse")) | (self.role == ROLES.index("admin"))).nonzero().squeeze(-1)
            pick = cand[torch.randperm(len(cand), generator=self.g, device=self.dev)[:sc.n_seed_informed]]
            self.informed[pick] = True
            seed_mask = torch.zeros_like(self.informed)
            seed_mask[pick] = True
            self.mem_insert(seed_mask, 4, self.goal)
        vdes, zone_now, dirv = self.desired_velocity()
        moved = self.move(vdes, dirv, dt / 2)
        moved = moved + self.move(vdes, dirv, dt / 2)
        d, dist = self.pairwise()
        iy, ix = self.cells(self.pos)
        zone_now = self.zone_map[iy, ix]
        act = self.active
        dens = (dist < 1.0).sum(-1).float() / math.pi
        # ---------------- arrivals / dwelling
        prog = self.prev_goal_dist - self.geo_dist(self.goal, self.pos)
        self.prev_goal_dist = self.geo_dist(self.goal, self.pos)
        moving_trip = act & ~self.at_goal & (self.evac_exit < 0)
        self.r_acc += 0.05 * prog * moving_trip
        self.trip_len += moved * moving_trip
        arrive = moving_trip & (zone_now == self.goal) & ((self.pos - self.target).norm(dim=-1) < 0.6) & (self.follow < 0)
        if arrive.any():
            idx = arrive.nonzero().squeeze(-1)
            self.at_goal[idx] = True
            reloc = ~self.task_pending[idx]               # arrival that does not complete a scheduled task
            self.task_pending[idx] = False
            keep = reloc & (self.dwell[idx] > 0)          # a relocation keeps the remaining dwell time
            self.dwell[idx] = torch.where(keep, self.dwell[idx], self.sample_dwell(idx, self.goal[idx]))
            ti = idx[~reloc]                              # scheduled tasks only (no relocation reward hacking)
            self.stats["tasks"] += len(ti)
            self.stats["tasks_nv"] += int((self.role[ti] != ROLES.index("visitor")).sum())
            if t < self.t_em and len(ti):
                geo = self.trip_geo0[ti]
                ok = geo > 2.0
                self.stats["trip_len"] += self.trip_len[ti][ok].sum().item()
                self.stats["trip_geo"] += geo[ok].sum().item()
                self.stats["trip_time"] += ((t - self.trip_t0[ti][ok]) * dt).sum().item()
                self.stats["trips"] += int(ok.sum())
            self.r_acc[ti] += self.m.task_reward
            if self.m.emotion != "none":
                self.pad_teacher[ti, 0] += 0.05
            tm = torch.zeros_like(arrive)
            tm[ti] = True
            self.mem_insert(tm, 0, self.goal)
        dw = act & self.at_goal & (self.evac_exit < 0)
        self.dwell = torch.where(dw, self.dwell - 1, self.dwell)
        waiting = dw & ((self.goal == 2) | (self.goal == 3)) & (self.role == ROLES.index("patient"))
        self.wait_time = torch.where(waiting, self.wait_time + dt, torch.zeros_like(self.wait_time))
        long_wait = waiting & (self.wait_time > 120)
        newlw = long_wait & ~self.waited_logged
        self.waited_logged |= newlw
        self.mem_insert(newlw, 1, self.goal)
        crowd = act & (dens > 1.0)
        newc = crowd & (t - self.last_crowd_mem > 75)
        self.last_crowd_mem = torch.where(newc, torch.full_like(self.last_crowd_mem, t), self.last_crowd_mem)
        self.mem_insert(newc, 2, zone_now.clamp(min=0))
        # rule-policy agents (no planner) pick next goal as soon as dwell ends
        if self.m.policy == "rules" or self.policy is None:
            done = dw & (self.dwell <= 0) & (self.follow < 0)
            self._next_task(done.nonzero().squeeze(-1))
        # ---------------- emergency
        if t >= self.t_em:
            dh = (self.pos - self.hazard).norm(dim=-1)
            detect = act & ((dh < 12.0) | (zone_now == Z["emergency"]))
            if t * dt >= self.t_em * dt + sc.alarm_delay_s:
                detect = act.clone()
            newly = detect & ~self.em_known
            self.em_known |= detect
            self.mem_insert(newly, 3, zone_now.clamp(min=0))
            ex = act & self.em_known & ((zone_now == EXITS[0]) | (zone_now == EXITS[1]))
            if ex.any():
                self.active[ex] = False
                self.exit_step[ex] = t
                self.exit_used[ex] = zone_now[ex]
                self.r_acc[ex] += 10.0
                self.pos[ex] = torch.tensor([-1000.0, -1000.0], device=self.dev) - 50 * torch.arange(int(ex.sum()), device=self.dev)[:, None].float()
                self.vel[ex] = 0
            self.r_acc -= 0.05 * (act & self.em_known)
            if self.m.exit_balance:      # crowding near the exits is penalised (crush risk)
                self.r_acc -= 0.02 * ((dens - 2.0).clamp(min=0) * (self.evac_exit >= 0))
            for k, e in enumerate(EXITS):
                cnt = ((self.pos - self.exit_c[k]).norm(dim=-1) < 2.0).sum().item()
                self.stats["max_exit_density"] = max(self.stats["max_exit_density"], cnt / (2 * math.pi))
        # ---------------- information diffusion (announcement + emergency awareness)
        self._diffuse(dist)
        # ---------------- social graph (Alg. 3)
        if self.m.social == "dynamic":
            sp_ = self.vel.norm(dim=-1)
            # a conversation needs proximity, low speed and attention: each agent talks to its
            # nearest neighbour only (pairs are symmetrised), so ties form one partner at a time
            nn = dist.argmin(-1)
            conv = torch.zeros_like(dist, dtype=torch.bool)
            conv[torch.arange(self.N, device=self.dev), nn] = True
            conv = conv | conv.T
            inter = conv & (dist < 2.0) & (sp_[:, None] < 0.4) & (sp_[None] < 0.4)
            S = 1 + 0.25 * (self.pad[:, None, 0] + self.pad[None, :, 0])
            F_before = self.F()
            self.Fr += inter * 0.008 * S * dt
            self.Tr += inter * 0.006 * S * dt * (0.5 + self.ocean[None, :, 3])
            self.Ir += inter * 0.006 * S * dt * (0.5 + self.ocean[None, :, 2])
            decay = 1 - 2e-4 * dt
            self.Fr *= decay; self.Tr *= decay; self.Ir *= decay
            if t % 25 == 0:
                newtie = ((self.F() > 0.3) & (F_before <= 0.3)).any(-1) & act
                self.mem_insert(newtie, 5, zone_now.clamp(min=0))
                self.r_acc += 0.2 * ((self.F() - F_before).clamp(min=0).sum(-1)).clamp(max=1.0)
        # ---------------- emotion teacher (appraisal + contagion)
        if self.m.emotion != "none":
            self._appraise(dist, dens, long_wait, zone_now, dt)
            if self.m.emotion == "rule":
                self.pad = self.pad_teacher.clone()
        # ---------------- metrics
        coll = ((dist < 0.4).any(-1) & act).float()
        self.r_acc -= 0.5 * coll
        if t < self.t_em:
            self.stats["coll_steps"] += coll.sum().item(); self.stats["agent_steps"] += act.sum().item()
            self.stats["agent_steps_nv"] += (act & (self.role != ROLES.index("visitor"))).sum().item()
            self._kinematics(dist, act)
            self.stats["pad_normal"] += self.pad[act].mean(0); self.stats["pad_normal_n"] += 1
        else:
            self.stats["coll_steps_evac"] += coll.sum().item(); self.stats["agent_steps_evac"] += act.sum().item()
            if (act & self.em_known).any():
                self.stats["pad_em"] += self.pad[act & self.em_known].mean(0); self.stats["pad_em_n"] += 1
        if t >= self.t_announce and self.stats["info_t50"] is None and self.informed[self.active].float().mean() >= 0.5:
            self.stats["info_t50"] = (t - self.t_announce) * dt
        if t == self.t_em - 1:
            self.stats["info_cov_at_em"] = self.informed.float().mean().item()
            self._social_snapshot()
        self.t += 1

    def _next_task(self, idx):
        if len(idx) == 0:
            return
        nxt = self.choose_next_goal(idx)
        if self.m.memory != "none" and self.m.policy == "rules":
            # memory-driven crowd avoidance (Generative-Agents style reflection on past experience)
            mem = self.retrieve_memory(idx)
            crowd_mem = mem[:, NZ + 2]
            zone_mem = mem.gather(1, nxt[:, None]).squeeze(-1)
            switch = ((nxt == 2) | (nxt == 3)) & (crowd_mem * zone_mem > 0.05)
            nxt = torch.where(switch, 5 - nxt, nxt)
        if self.m.social == "static":   # plan coordination with family (GA-lite)
            fam = self.family[idx]
            hasf = fam >= 0
            # visitors adopt their patient's current goal
            vis = hasf & (self.role[idx] == ROLES.index("visitor"))
            nxt = torch.where(vis, self.goal[fam.clamp(min=0)], nxt)
        self.set_goal(idx, nxt)
        self.task_pending[idx] = True

    def _diffuse(self, dist):
        act = self.active
        C = (dist < 2.0)
        base = 0.005
        m = self.m.info
        if m == "contact":
            p = torch.full_like(dist, base)
        elif m == "memory":
            share = torch.full((self.N,), base, device=self.dev)
            if self.t % self.sc.decision_every == 0 or not hasattr(self, "_ga_share"):
                self._ga_share = torch.ones(self.N, device=self.dev)
                idx = self.informed.nonzero().squeeze(-1)
                if len(idx):
                    mem = self.retrieve_memory(idx)
                    self._ga_share[idx] = 1 + 2 * (mem[:, NZ + 4] > 0.05).float()
            p = (share * self._ga_share)[:, None].expand(-1, self.N)
        elif m == "emotion":
            k = 1 + self.ocean[:, 2] + self.pad[:, 1].clamp(min=0)
            p = (base * k)[:, None].expand(-1, self.N)
        else:  # trust-weighted sharing (SGT edges)
            p = 0.0025 + 0.015 * self.T_()
            p = p * (1 + 2 * self.share_boost)[:, None]
        p = p * C
        for flag, mult, ev in ((self.informed, 1.0, 4), (self.em_known, 5.0, 3)):
            if flag is self.em_known and self.t < self.t_em:
                continue
            src = (flag & act).float()
            lp = torch.log1p(-(p * mult).clamp(max=0.999) * src[:, None]).sum(0)
            new = act & ~flag & (self.rand(self.N) < 1 - torch.exp(lp))
            if new.any():
                senders = (C & (flag & act)[:, None] & new[None]).float()
                self.r_acc += 0.2 * senders.sum(1) * (flag is self.informed)
                flag |= new
                iy, ix = self.cells(self.pos)
                self.mem_insert(new, ev, self.zone_map[iy, ix].clamp(min=0))
                if self.m.emotion != "none":
                    self.pad_teacher[new, 1] += 0.05

    def _appraise(self, dist, dens, long_wait, zone_now, dt):
        P = self.pad_teacher
        neuro = 0.5 + self.ocean[:, 4]
        aware = self.em_known.float()
        dh = (self.pos - self.hazard).norm(dim=-1)
        hz = torch.exp(-dh / 8.0) * aware
        crowd = (dens > 1.0).float()
        lw = long_wait.float()
        if self.m.social != "none":
            near_friend = ((dist < 2.0) & (self.F() > 0.3)).any(-1).float()
        else:
            near_friend = torch.zeros_like(lw)
        tau = 180.0
        dP = (self.pad_base[:, 0] - P[:, 0]) / tau + (-0.004 * lw - 0.006 * crowd + 0.004 * near_friend - 0.03 * hz - 0.004 * aware) * neuro
        dA = (self.pad_base[:, 1] - P[:, 1]) / tau + (0.003 * lw + 0.005 * crowd + 0.05 * hz + 0.01 * aware) * neuro
        dD = (self.pad_base[:, 2] - P[:, 2]) / tau + (-0.02 * hz - 0.003 * crowd)
        P = P + dt * torch.stack([dP, dA, dD], -1)
        if self.m.social == "dynamic":
            # Eq. (13): SI_i = sum_j T_ij I_ij E_j over neighbours within 5 m (emotional contagion)
            Wt = self.T_() * self.I_() * (dist < 5.0)
            s = Wt.sum(-1, keepdim=True)
            SI = (Wt @ self.pad) / s.clamp(min=1e-6)
            P = P + dt * 0.02 * (s > 1e-3).float() * (SI - P)
        self.pad_teacher = P.clamp(-1, 1)

    def _kinematics(self, dist, act):
        sp = self.vel.norm(dim=-1)
        sel = act & self.rec_mask & (sp > 0.2) & (self.prev_v.norm(dim=-1) > 0.2) & (self.t > 25)
        if sel.any():
            dt = self.sc.dt
            acc = ((self.vel - self.prev_v).norm(dim=-1) / dt)[sel]
            a1 = torch.atan2(self.vel[:, 1], self.vel[:, 0]); a0 = torch.atan2(self.prev_v[:, 1], self.prev_v[:, 0])
            turn = (torch.remainder(a1 - a0 + math.pi, 2 * math.pi) - math.pi).abs()[sel] / dt
            nn_d = dist.min(-1).values[sel].clamp(max=5.99)
            for k, x in (("speed", sp[sel]), ("acc", acc), ("turn", turn), ("nn", nn_d)):
                self.hist[k] += torch.bincount(torch.bucketize(x, self.bins[k]), minlength=len(self.bins[k]) + 1)[: len(self.bins[k]) + 1].float()
        # group detection: pairs within 2 m moving with |dv| < 0.3 m/s for >= 4 s
        mv = act & (sp > 0.3)
        dv = (self.vel[:, None] - self.vel[None]).norm(dim=-1)
        cond = (dist < 2.0) & (dv < 0.3) & mv[:, None] & mv[None]
        self.pair_cnt = torch.where(cond, (self.pair_cnt + 1).clamp(max=1000), torch.zeros_like(self.pair_cnt))
        if mv.sum() >= 20:
            member = (self.pair_cnt >= 10).any(-1) & mv
            self.stats["group_frac_sum"] += (member.sum() / mv.sum()).item()
            self.stats["group_frac_n"] += 1
        self.prev_v = self.vel.clone()

    def _social_snapshot(self):
        if self.m.social == "none":
            self.stats["ties_at_em"] = 0.0
            self.stats["contagion"] = None
            return
        F = self.F()
        ties = (F > 0.3)
        self.stats["ties_at_em"] = ties.sum().item() / self.N
        if self.m.emotion != "none" and ties.sum() > 10:
            i, j = ties.nonzero(as_tuple=True)
            a = self.pad[:, 1]
            r_tie = torch.corrcoef(torch.stack([a[i], a[j]]))[0, 1].item()
            perm = torch.randperm(self.N, generator=self.g, device=self.dev)
            r_rand = torch.corrcoef(torch.stack([a, a[perm]]))[0, 1].item()
            self.stats["contagion"] = r_tie - r_rand

    # ------------------------------------------------------------------ decisions
    def friend_info(self, idx, dist):
        F = self.F()[idx]
        near = dist[idx] < 15.0
        Fm = F * near
        best = Fm.max(-1)
        trusted = (self.T_()[idx] > 0.3) & near
        Pn = torch.where(trusted, self.pad[None, :, 0].expand(len(idx), -1), torch.full_like(F, 9.0))
        distress = Pn.min(-1)
        return best.values, best.indices, distress.values, distress.indices

    def build_obs(self, idx, dist):
        n = len(idx)
        iy, ix = self.cells(self.pos[idx])
        zn = self.zone_map[iy, ix].clamp(min=0)
        oh = lambda x, k: torch.nn.functional.one_hot(x, k).float()
        dens = ((dist[idx] < 1.0).sum(-1).float() / math.pi)
        occ = torch.bincount(self.goal[self.active], minlength=NZ).float() / self.area
        exit_cnt = torch.stack([((self.pos - c).norm(dim=-1) < 10).sum() for c in self.exit_c]).float() / 50
        exd = torch.stack([self.geo_dist(torch.full((n,), e, device=self.dev, dtype=torch.long), self.pos[idx]) for e in EXITS], -1) / 100
        bf, bfi, fp, fpi = self.friend_info(idx, dist)
        bfd = (self.pos[bfi] - self.pos[idx]).norm(dim=-1).clamp(max=15) / 15
        maxT = (self.T_()[idx] * (dist[idx] < 15)).max(-1).values
        obs = torch.cat([oh(zn, NZ), oh(self.goal[idx], NZ), oh(self.role[idx], len(ROLES)), dens[:, None],
                         occ[self.goal[idx]][:, None], exit_cnt[None].expand(n, -1), exd,
                         torch.stack([self.em_known[idx].float(), self.informed[idx].float(), self.at_goal[idx].float(),
                                      maxT, fp.clamp(max=1), bfd, torch.full((n,), self.t / self.sc.steps, device=self.dev)], -1)], -1)
        mem = self.retrieve_memory(idx)
        Wt = self.T_()[idx] * self.I_()[idx] * (dist[idx] < 5.0)
        s = Wt.sum(-1, keepdim=True)
        SI = (Wt @ self.pad) / s.clamp(min=1e-6)
        soc = torch.cat([SI, s, ((self.F()[idx] > 0.3) & (dist[idx] < 5)).sum(-1, keepdim=True).float()], -1)
        dh = (self.pos[idx] - self.hazard).norm(dim=-1)
        ctx = torch.cat([dens[:, None], (torch.exp(-dh / 8) * self.em_known[idx])[:, None],
                         (self.wait_time[idx] > 120).float()[:, None], self.informed[idx].float()[:, None],
                         self.em_known[idx].float()[:, None], self.at_goal[idx].float()[:, None], self.pad[idx]], -1)
        k = 8
        dn = dist[idx].clone()
        dn[torch.arange(n, device=self.dev), idx] = 1e4
        top = dn.topk(k, -1, largest=False)
        nbm = top.values < 8.0
        j = top.indices
        rel = self.pos[j] - self.pos[idx][:, None]
        nb = torch.cat([rel / 8, top.values[..., None].clamp(max=8) / 8,
                        self.F()[idx[:, None], j][..., None], self.T_()[idx[:, None], j][..., None],
                        self.I_()[idx[:, None], j][..., None], self.pad[j], self.informed[j][..., None].float(),
                        self.em_known[j][..., None].float()], -1)
        # NSRE action mask (Eq. 17): symbolic preconditions of each high-level action
        am = torch.ones(n, N_ACTIONS, dtype=torch.bool, device=self.dev)
        if self.m.nsre:
            em = self.em_known[idx]
            has_friend = bf > 0.3
            distressed = (fp < -0.3) & (fp < 9)
            am[:, 0] = ~em
            am[:, 1] = ~em & has_friend
            am[:, 2] = ~em & self.informed[idx]
            am[:, 3] = ~em
            am[:, 4] = em
            am[:, 5] = em
            am[:, 6] = em & distressed & ~self.helped[idx]
            if self.m.exit_balance:
                # rule: IF the queue at the nearest exit makes the other exit cheaper THEN do not pick the nearest
                cost = exd * 100 + self.m.queue_cost * exit_cnt[None] * 50
                swap = em & (exd.argmin(-1) != cost.argmin(-1))
                am[:, 4] &= ~swap
        glob = torch.tensor([1 - self.active.float().mean().item(), self.informed.float().mean().item(),
                             self.em_known.float().mean().item(), dens.mean().item(), self.t / self.sc.steps],
                            device=self.dev)[None].expand(n, -1)
        return {"obs": obs, "pers": self.ocean[idx], "mem": mem, "soc": soc, "ctx": ctx, "nb": nb, "nb_mask": nbm,
                "amask": am, "glob": glob}, (bfi, fpi)

    def rule_actions(self, idx, b, aux):
        em = self.em_known[idx]
        a = torch.zeros(len(idx), dtype=torch.long, device=self.dev)
        if self.m.exit_choice == "familiar":
            ev = torch.full_like(a, 4)  # main entrance (familiar) handled in apply
        else:
            ev = torch.full_like(a, 4)
        a = torch.where(em, ev, a)
        if self.m.nsre:   # NSRE rule base (w/o PPO ablation): assist trusted friend in distress, then evacuate
            a = torch.where(b["amask"][:, 6], torch.full_like(a, 6), a)
            a = torch.where(~em & b["amask"][:, 2] & (self.rand(len(idx)) < 0.3), torch.full_like(a, 2), a)
            a = torch.where(~em & b["amask"][:, 1] & (self.rand(len(idx)) < 0.2), torch.full_like(a, 1), a)
        return a

    def flush_pending(self):
        if getattr(self, "buffer", None) is not None and getattr(self, "_pending", None) is not None:
            p = self._pending
            p["r"] = self.r_acc[p["idx"]].clone()
            p["done"] = ~self.active[p["idx"]]
            self.buffer.append(p)
            self._pending = None
        self.r_acc.zero_()

    def decide(self):
        self.flush_pending()
        idx = self.active.nonzero().squeeze(-1)
        if len(idx) == 0:
            return
        _, dist = self.pairwise()
        need_obs = self.m.policy == "ppo" or self.m.nsre or self.m.emotion == "met" or (self.llm is not None and self.m.llm)
        if not need_obs:
            a = torch.where(self.em_known[idx], 4, 0)
            self.apply_actions(idx, a, None, dist)
            return
        b, aux = self.build_obs(idx, dist)
        self._last = (idx, b)
        pad_new = None
        if self.policy is not None:
            with torch.no_grad():
                logits, v, pad_new = self.policy(b)
            if self.m.policy == "ppo":
                dist_a = torch.distributions.Categorical(logits=logits)
                a = dist_a.sample() if self.policy.training else logits.argmax(-1) if getattr(self, "greedy", False) else dist_a.sample()
                if getattr(self, "buffer", None) is not None:
                    self._pending = {"idx": idx, "b": b, "a": a, "logp": dist_a.log_prob(a), "v": v,
                                     "pad_t": self.pad_teacher[idx].clone()}
            else:
                a = self.rule_actions(idx, b, aux)
        else:
            a = self.rule_actions(idx, b, aux)
        if self.llm is not None and self.m.llm:
            a = self.llm.override(self, idx, b, a)
        if self.m.emotion == "met" and pad_new is not None:
            self.pad[idx] = pad_new
        elif self.m.emotion == "met":
            self.pad[idx] = self.pad_teacher[idx]
        self._last_a = a
        self.apply_actions(idx, a, aux, dist)

    def apply_actions(self, idx, a, aux, dist):
        self.action[idx] = a
        self.share_boost.zero_()
        self.follow[idx] = -1
        # a0 continue task: start next scheduled task when dwelling is over
        done = (a == 0) & self.at_goal[idx] & (self.dwell[idx] <= 0) & (self.evac_exit[idx] < 0)
        self._next_task(idx[done])
        if aux is not None:
            bfi, fpi = aux
            # a1 socialise / follow the strongest tie
            f = a == 1
            if f.any():
                ok = f & (self.F()[idx, bfi] > 0.3)
                self.follow[idx[ok]] = bfi[ok]
                self.at_goal[idx[ok]] = False
            # a2 share information: approach nearest uninformed trusted agent
            s = (a == 2) & self.informed[idx]
            if s.any():
                self.share_boost[idx[s]] = 1.0
            # a3 avoid crowd: switch waiting room or relocate inside the zone
            c = a == 3
            if c.any():
                ci = idx[c]
                w = (self.goal[ci] == 2) | (self.goal[ci] == 3)
                self.set_goal(ci[w], 5 - self.goal[ci[w]])   # a pending task stays pending (counted once)
                other = ci[~w]
                self.target[other] = self.sample_point(self.goal[other])
            # a6 assist friend in distress, then evacuate together
            h = (a == 6)
            if h.any():
                hi = idx[h]
                self.follow[hi] = fpi[h]
                self.evac_exit[hi] = -1
                self.helped[hi] = True
                self.stats["helpers"] += len(hi)
                friend = fpi[h]
                self.em_known[friend] = True
        # a4 / a5 evacuate
        e = ((a == 4) | (a == 5)) & self.em_known[idx]
        if e.any():
            ei = idx[e]
            if self.m.exit_choice == "familiar":
                choice = torch.full((len(ei),), EXITS[0], device=self.dev)
                staff = self.role[ei] <= ROLES.index("nurse")
                if staff.any():
                    gd = torch.stack([self.geo_dist(torch.full((len(ei),), x, device=self.dev), self.pos[ei]) for x in EXITS], -1)
                    choice = torch.where(staff, torch.tensor(EXITS, device=self.dev)[gd.argmin(-1)], choice)
            else:
                gd = torch.stack([self.geo_dist(torch.full((len(ei),), x, device=self.dev), self.pos[ei]) for x in EXITS], -1)
                if (a[e] == 5).any():
                    cnt = torch.stack([((self.pos - c).norm(dim=-1) < 10).sum() for c in self.exit_c]).float()
                    congested = gd + self.m.queue_cost * cnt[None]   # extra route cost per queued person
                    gd = torch.where((a[e] == 5)[:, None], congested, gd)
                choice = torch.tensor(EXITS, device=self.dev)[gd.argmin(-1)]
            self.evac_exit[ei] = choice
            self.at_goal[ei] = False
        # without NSRE an evacuation action in normal time sends agents towards the exit (penalised by task loss)
        bad = ((a == 4) | (a == 5)) & ~self.em_known[idx]
        if bad.any():
            self.set_goal(idx[bad], torch.full((int(bad.sum()),), EXITS[0], device=self.dev))

    # ------------------------------------------------------------------ run
    def run(self, steps=None, progress=False):
        steps = steps or self.sc.steps
        t0 = time.time()
        for _ in range(steps):
            self.step()
        if self.dev.type == "cuda":
            torch.cuda.synchronize()
        self.wall_time = time.time() - t0
        return self.summary()

    def summary(self):
        s, sc = self.stats, self.sc
        hist = {k: (v / v.sum().clamp(min=1)).cpu().numpy() for k, v in self.hist.items()}
        em0 = self.em_known.sum().item()
        n_em = self.N
        ex = self.exit_step[self.exit_step >= 0]
        times = ((ex - self.t_em).float() * sc.dt).sort().values.cpu().numpy()
        frac = len(times) / n_em
        t95 = float(times[int(math.ceil(0.95 * n_em)) - 1]) if frac >= 0.95 else float("nan")
        t50 = float(times[int(math.ceil(0.5 * n_em)) - 1]) if frac >= 0.5 else float("nan")
        east = (self.exit_used == EXITS[1]).sum().item() / max(1, len(times))
        hours = s["agent_steps"] * sc.dt / 3600
        out = {
            "method": self.m.name, "N": self.N, "steps": self.t,
            "hist": {k: v.tolist() for k, v in hist.items()},
            "collision_rate_normal": s["coll_steps"] / max(1, s["agent_steps"]),
            "collision_rate_evac": s["coll_steps_evac"] / max(1, s["agent_steps_evac"]),
            "tasks_per_agent_hour": s["tasks"] / max(1e-9, hours) if hours else 0.0,
            "tasks_per_staff_patient_hour": s["tasks_nv"] / max(1e-9, s["agent_steps_nv"] * sc.dt / 3600),
            "route_ratio": s["trip_len"] / max(1e-9, s["trip_geo"]),
            "mean_trip_len_m": s["trip_len"] / max(1, s["trips"]),
            "mean_trip_time_s": s["trip_time"] / max(1, s["trips"]),
            "group_fraction": s["group_frac_sum"] / max(1, s["group_frac_n"]),
            "info_coverage_at_em": s["info_cov_at_em"], "info_t50_s": s["info_t50"],
            "evac_fraction": frac, "evac_t50_s": t50, "evac_t95_s": t95,
            "evac_mean_s": float(times.mean()) if len(times) else float("nan"),
            "east_exit_share": east, "max_exit_density": s["max_exit_density"],
            "ties_per_agent": s["ties_at_em"], "contagion_index": s["contagion"], "helpers": s["helpers"],
            "pad_normal": (s["pad_normal"] / max(1, s["pad_normal_n"])).tolist(),
            "pad_emergency": (s["pad_em"] / max(1, s["pad_em_n"])).tolist(),
            "wall_time_s": getattr(self, "wall_time", None),
        }
        return out

    def persist_state(self):
        keys = ["Fr", "Tr", "Ir", "mem_key", "mem_sig", "mem_valid", "ocean", "pad_base"]
        out = {k: getattr(self, k).clone() for k in keys}
        out["mem_t"] = self.mem_t - self.t          # timestamps relative to the next day's start
        return out
