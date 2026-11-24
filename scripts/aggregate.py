"""Aggregates benchmark JSONs into Markdown + LaTeX tables and figures.

python scripts/aggregate.py --traj results/traj_local --sim results/sim --out results/report
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
SCENES = ["eth", "hotel", "univ", "zara1", "zara2"]

# Published best-of-20 minADE/minFDE (m). Sources: MGF, NeurIPS 2024, Table 1 (Trajectron++ and MID
# re-evaluated after the np.gradient future-leak fix); LED and MemoNet from LED, CVPR 2023, Table 4.
SOTA = {
    "Social-GAN (CVPR'18)": [(0.87, 1.62), (0.67, 1.37), (0.76, 1.52), (0.35, 0.68), (0.42, 0.84)],
    "Social-STGCNN (CVPR'20)": [(0.64, 1.11), (0.49, 0.85), (0.44, 0.79), (0.34, 0.53), (0.30, 0.48)],
    "Trajectron++ (ECCV'20)*": [(0.61, 1.03), (0.20, 0.28), (0.30, 0.55), (0.24, 0.41), (0.18, 0.32)],
    "PECNet (ECCV'20)": [(0.54, 0.87), (0.18, 0.24), (0.35, 0.60), (0.22, 0.39), (0.17, 0.30)],
    "AgentFormer (ICCV'21)": [(0.45, 0.75), (0.14, 0.22), (0.25, 0.45), (0.18, 0.30), (0.14, 0.24)],
    "MID (CVPR'22)*": [(0.55, 0.88), (0.20, 0.35), (0.30, 0.55), (0.29, 0.51), (0.20, 0.38)],
    "GroupNet (CVPR'22)": [(0.46, 0.73), (0.15, 0.25), (0.26, 0.49), (0.21, 0.39), (0.17, 0.33)],
    "MemoNet (CVPR'22)": [(0.40, 0.61), (0.11, 0.17), (0.24, 0.43), (0.18, 0.32), (0.14, 0.24)],
    "EqMotion (CVPR'23)": [(0.40, 0.61), (0.12, 0.18), (0.23, 0.43), (0.18, 0.32), (0.13, 0.23)],
    "LED (CVPR'23)": [(0.39, 0.58), (0.11, 0.17), (0.26, 0.43), (0.18, 0.26), (0.13, 0.22)],
    "MGF (NeurIPS'24)": [(0.39, 0.59), (0.13, 0.20), (0.21, 0.39), (0.17, 0.29), (0.14, 0.24)],
}

TRAJ_NAMES = {"cv_det": "Constant velocity (det.)", "cv": "Constant velocity + sampling", "sfm": "Social Force Model",
              "lstm": "LSTM (K-mode)", "transformer": "Transformer (no RAMG/SGT/MET)",
              "nsrf_no_memory": "NSRF-T w/o RAMG", "nsrf_no_social": "NSRF-T w/o SGT",
              "nsrf_no_emotion": "NSRF-T w/o MET", "nsrf_no_nsre": "NSRF-T w/o NSRE", "nsrf": "NSRF-T (full)",
              "nsrf_v2_no_refine": "NSRF-T v2 w/o diffusion refinement", "nsrf_v2": "NSRF-T v2 (full)"}
SIM_NAMES = {"sfm": "Social Force Model", "astar": "A* path following", "ga": "Generative-Agents-lite",
             "ebp": "Emotion-Based Planner", "nsrf": "NeuroSocial-RAGFormer", "nsrf_no_ramg": "w/o RAMG",
             "nsrf_no_met": "w/o MET", "nsrf_no_sgt": "w/o SGT", "nsrf_no_nsre": "w/o NSRE",
             "nsrf_no_ppo": "w/o PPO (NSRE rules only)", "nsrf_v2": "NeuroSocial-RAGFormer v2",
             "nsrf_v2_task_only": "v1 + task reward only", "nsrf_v2_no_escort": "v2 w/o escort groups",
             "nsrf_v2_no_exitbal": "v2 w/o exit balancing", "nsrf_v3": "NeuroSocial-RAGFormer v3 (task reward + calibrated exit rule)"}


def ms(xs, f="{:.2f}"):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if not xs:
        return "n/a"
    if len(xs) == 1:
        return f.format(xs[0])
    return (f + " ± " + f).format(np.mean(xs), np.std(xs))


def load_traj(d):
    R = defaultdict(lambda: defaultdict(list))
    for fn in glob.glob(os.path.join(d, "*.json")):
        j = json.load(open(fn))
        t = dict(j["test"]); t["seed"] = j["seed"]
        R[j["model"]][j["scene"]].append(t)
    return R


def traj_tables(R):
    md, tex = [], []
    order = [m for m in TRAJ_NAMES if m in R]
    md.append("| Method | " + " | ".join(s.upper() for s in SCENES) + " | AVG |")
    md.append("|---" * 7 + "|")
    tex.append(r"\begin{tabular}{l" + "c" * 6 + r"}\toprule")
    tex.append("Method & " + " & ".join(s.upper() for s in SCENES) + r" & AVG \\ \midrule")
    for name, rows in SOTA.items():
        a = np.mean([r[0] for r in rows]); f = np.mean([r[1] for r in rows])
        cells = [f"{x:.2f}/{y:.2f}" for x, y in rows] + [f"{a:.2f}/{f:.2f}"]
        md.append(f"| {name} [published] | " + " | ".join(cells) + " |")
        tex.append(f"{name} & " + " & ".join(cells) + r" \\")
    tex.append(r"\midrule")
    for m in order:
        cells, A, F = [], [], []
        for s in SCENES:
            v = R[m].get(s)
            if not v:
                cells.append("–"); continue
            a = np.mean([x["minADE"] for x in v]); f = np.mean([x["minFDE"] for x in v])
            A.append(a); F.append(f)
            cells.append(f"{a:.2f}/{f:.2f}")
        avg = f"{np.mean(A):.2f}/{np.mean(F):.2f}" if len(A) == 5 else "–"
        seeds = sorted({x["seed"] for s in SCENES for x in R[m].get(s, [])})
        if len(A) == 5 and len(seeds) > 1:
            pa = [np.mean([x["minADE"] for s in SCENES for x in R[m][s] if x["seed"] == sd]) for sd in seeds]
            pf = [np.mean([x["minFDE"] for s in SCENES for x in R[m][s] if x["seed"] == sd]) for sd in seeds]
            avg = f"{np.mean(pa):.3f}±{np.std(pa):.3f}/{np.mean(pf):.3f}±{np.std(pf):.3f} (n={len(seeds)})"
        md.append(f"| {TRAJ_NAMES[m]} [ours, this run] | " + " | ".join(cells) + f" | {avg} |")
        tex.append(f"{TRAJ_NAMES[m]} & " + " & ".join(cells) + f" & {avg}" + r" \\")
    tex.append(r"\bottomrule\end{tabular}")
    return "\n".join(md), "\n".join(tex)


def traj_det_table(R):
    md = ["| Method | ADE₁ | FDE₁ | Collision rate, top-1 (%) | GC Fréchet (m) ↓ | GC path length (m) | GC tortuosity | real tortuosity |",
          "|---|---|---|---|---|---|---|---|"]
    tex = [r"\begin{tabular}{lccccccc}\toprule",
           r"Method & ADE$_1$ & FDE$_1$ & Coll. (\%) & Fr\'echet $\downarrow$ & Path (m) & Tort. & Tort.$_{\mathrm{real}}$ \\ \midrule"]
    for m in [m for m in TRAJ_NAMES if m in R]:
        def avg(k):
            vals = [np.mean([x[k] for x in R[m][s]]) for s in SCENES if R[m].get(s) and k in R[m][s][0]]
            return np.mean(vals) if len(vals) == 5 else float("nan")
        if m == "cv":
            continue
        row = [avg("ADE1"), avg("FDE1"), 100 * avg("COL1"), avg("GC_Frechet"), avg("GC_PathLen"), avg("GC_Tort"), avg("GC_Tort_GT")]
        f = lambda x: "–" if math.isnan(x) else f"{x:.3f}" if x < 10 else f"{x:.2f}"
        md.append(f"| {TRAJ_NAMES[m]} | " + " | ".join(f(x) for x in row) + " |")
        tex.append(f"{TRAJ_NAMES[m]} & " + " & ".join(f(x) for x in row) + r" \\")
    tex.append(r"\bottomrule\end{tabular}")
    return "\n".join(md), "\n".join(tex)


SIM_COLS = [("HLS", "Human-likeness (HLS) ↑", "{:.1f}"),
            ("group_fraction_abs_err", "Group-fraction error vs real ↓", "{:.3f}"),
            ("collision_rate_normal", "Collision rate (normal) ↓", "{:.4f}"),
            ("route_ratio", "Route length / geodesic", "{:.2f}"),
            ("tasks_per_agent_hour", "Tasks per agent-hour ↑", "{:.1f}"),
            ("tasks_per_staff_patient_hour", "Tasks per staff/patient-hour ↑", "{:.1f}"),
            ("info_t50_s", "Info T50 (s) ↓", "{:.0f}"),
            ("info_coverage_at_em", "Info coverage ↑", "{:.2f}"),
            ("evac_t95_s", "Evacuation T95 (s) ↓", "{:.0f}"),
            ("evac_mean_s", "Mean evac. time (s) ↓", "{:.0f}"),
            ("collision_rate_evac", "Collision rate (evac.) ↓", "{:.4f}"),
            ("max_exit_density", "Peak exit density (p/m²)", "{:.2f}"),
            ("east_exit_share", "East-exit share", "{:.2f}"),
            ("ties_per_agent", "Ties per agent", "{:.2f}"),
            ("contagion_index", "Contagion index", "{:.2f}"),
            ("ms_per_step", "ms / step", "{:.1f}")]


def load_sim(d, N=500):
    R = defaultdict(list)
    for fn in glob.glob(os.path.join(d, f"*_N{N}_s*.json")):
        j = json.load(open(fn))
        R[j["key"]].append(j)
    return R


def sim_table(R, keys, cols):
    md = ["| Method | " + " | ".join(c[1] for c in cols) + " |", "|---" * (len(cols) + 1) + "|"]
    tex = [r"\begin{tabular}{l" + "c" * len(cols) + r"}\toprule",
           "Method & " + " & ".join(c[1].replace("↑", r"$\uparrow$").replace("↓", r"$\downarrow$").replace("²", r"$^2$") for c in cols) + r" \\ \midrule"]
    for k in keys:
        if k not in R:
            continue
        cells = [ms([r.get(c[0]) for r in R[k]], c[2]) for c in cols]
        md.append(f"| {SIM_NAMES[k]} (n={len(R[k])}) | " + " | ".join(cells) + " |")
        tex.append(f"{SIM_NAMES[k]} & " + " & ".join(c.replace("±", r"$\pm$") for c in cells) + r" \\")
    tex.append(r"\bottomrule\end{tabular}")
    return "\n".join(md), "\n".join(tex)


def figures(R, Rt, out, ref):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sim.metrics import BINS
    keys = [k for k in ["sfm", "astar", "ga", "ebp", "nsrf"] if k in R]
    if keys:
        fig, ax = plt.subplots(1, 4, figsize=(16, 3.4))
        for i, (k, lab) in enumerate([("speed", "speed (m/s)"), ("acc", "|acceleration| (m/s²)"), ("turn", "turning rate (rad/s)"), ("nn", "nearest-neighbour distance (m)")]):
            b = BINS[k]; c = np.r_[b[0], (b[:-1] + b[1:]) / 2, b[-1]]
            ax[i].plot(c, ref["hist"][k], "k-", lw=2.5, label="ETH/UCY (real)")
            for m in keys:
                h = np.mean([r["hist"][k] for r in R[m]], 0)
                ax[i].plot(c, h, lw=1.3, label=SIM_NAMES[m])
            ax[i].set_xlabel(lab); ax[i].set_ylabel("probability")
        ax[0].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(out, "fig_kinematics_vs_real.png"), dpi=200); plt.close(fig)
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
        for i, (c, lab) in enumerate([("HLS", "Human-likeness score ↑"), ("evac_t95_s", "Evacuation T95 (s) ↓"), ("info_t50_s", "Information T50 (s) ↓")]):
            vals = [[r[c] for r in R[m] if r.get(c) is not None and not (isinstance(r[c], float) and math.isnan(r[c]))] for m in keys]
            ax[i].bar(range(len(keys)), [np.mean(v) if v else 0 for v in vals], yerr=[np.std(v) if v else 0 for v in vals], capsize=3,
                      color=["#9aa5b1"] * (len(keys) - 1) + ["#2563eb"])
            ax[i].set_xticks(range(len(keys))); ax[i].set_xticklabels([SIM_NAMES[m].replace(" ", "\n", 1) for m in keys], fontsize=7)
            ax[i].set_title(lab, fontsize=10)
        fig.tight_layout(); fig.savefig(os.path.join(out, "fig_sim_comparison.png"), dpi=200); plt.close(fig)
    if Rt:
        ms_ = [m for m in ["cv", "sfm", "lstm", "transformer", "nsrf", "nsrf_v2"] if m in Rt]
        fig, ax = plt.subplots(figsize=(8, 3.4))
        w = 0.8 / (len(ms_) + 3)
        for i, m in enumerate(ms_):
            ax.bar(np.arange(5) + i * w, [np.mean([x["minADE"] for x in Rt[m][s]]) if Rt[m].get(s) else 0 for s in SCENES], w, label=TRAJ_NAMES[m])
        for j, nm in enumerate(["AgentFormer (ICCV'21)", "LED (CVPR'23)", "MGF (NeurIPS'24)"]):
            ax.bar(np.arange(5) + (len(ms_) + j) * w, [r[0] for r in SOTA[nm]], w, label=nm + " [published]", hatch="//", alpha=0.6)
        ax.set_xticks(np.arange(5) + 0.4); ax.set_xticklabels([s.upper() for s in SCENES]); ax.set_ylabel("minADE$_{20}$ (m)")
        ax.legend(fontsize=6, ncol=2); fig.tight_layout(); fig.savefig(os.path.join(out, "fig_ethucy_minade.png"), dpi=200); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--traj", default="results/traj_local")
    p.add_argument("--sim", default="results/sim")
    p.add_argument("--out", default="results/report")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from sim.metrics import real_reference
    ref = real_reference()
    Rt = load_traj(a.traj)
    parts = []
    if Rt:
        m1, t1 = traj_tables(Rt)
        m2, t2 = traj_det_table(Rt)
        parts += ["## ETH/UCY best-of-20 minADE/minFDE (m)", m1, "", "## ETH/UCY top-1, collisions and goal-conditioned realism (averaged over 5 scenes)", m2, ""]
        open(os.path.join(a.out, "table_ethucy.tex"), "w").write(t1)
        open(os.path.join(a.out, "table_ethucy_det.tex"), "w").write(t2)
    Rs = load_sim(a.sim)
    if Rs:
        main_keys = ["sfm", "astar", "ga", "ebp", "nsrf", "nsrf_v2", "nsrf_v3"]
        abl = ["nsrf", "nsrf_no_ramg", "nsrf_no_met", "nsrf_no_sgt", "nsrf_no_nsre", "nsrf_no_ppo",
               "nsrf_v2_task_only", "nsrf_v2_no_escort", "nsrf_v2_no_exitbal", "nsrf_v2", "nsrf_v3"]
        c1 = [c for c in SIM_COLS if c[0] in ("HLS", "group_fraction_abs_err", "collision_rate_normal", "route_ratio", "tasks_per_agent_hour", "tasks_per_staff_patient_hour", "ms_per_step")]
        c2 = [c for c in SIM_COLS if c[0] in ("info_t50_s", "info_coverage_at_em", "evac_t95_s", "evac_mean_s", "collision_rate_evac", "max_exit_density", "east_exit_share")]
        c3 = [c for c in SIM_COLS if c[0] in ("ties_per_agent", "contagion_index")]
        for title, keys, cols, tag in [("Hospital twin: realism, safety, efficiency", main_keys, c1, "sim_realism"),
                                       ("Hospital twin: information diffusion and evacuation", main_keys, c2, "sim_events"),
                                       ("Hospital twin: social structure", main_keys, c3, "sim_social"),
                                       ("Ablation (realism)", abl, c1, "abl_realism"), ("Ablation (events)", abl, c2, "abl_events")]:
            md, tex = sim_table(Rs, keys, cols)
            parts += [f"## {title}", md, ""]
            open(os.path.join(a.out, f"table_{tag}.tex"), "w").write(tex)
        hp = ["| Method | " + " | ".join(["speed", "acc", "turn", "nn"]) + " |", "|---|---|---|---|---|"]
        for k in main_keys:
            if k in Rs:
                hp.append(f"| {SIM_NAMES[k]} | " + " | ".join(ms([r["HLS_parts"][x] for r in Rs[k]], "{:.1f}") for x in ["speed", "acc", "turn", "nn"]) + " |")
        parts += ["## HLS components (100·(1−JSD) per kinematic feature)", "\n".join(hp), "",
                  f"Real ETH/UCY group fraction: {ref['group_fraction']:.3f}"]
    for fn in glob.glob(os.path.join(a.sim, "scalability_*.json")):
        rows = json.load(open(fn))
        md = ["| Method | N | ms/step | steps/s | real-time factor | peak memory (GB) | device |", "|---|---|---|---|---|---|---|"]
        md += [f"| {SIM_NAMES.get(r['method'], r['method'])} | {r['N']} | {r['ms_per_step']:.1f} | {r['steps_per_s']:.1f} | {r['realtime_factor']:.1f}× | {r['peak_mem_GB']:.2f} | {r['device']} |" for r in rows]
        parts += ["", f"## Scalability ({os.path.basename(fn)})", "\n".join(md)]
    md_fn = os.path.join(a.sim, "multiday.json")
    if os.path.exists(md_fn):
        D = json.load(open(md_fn))
        md = ["| Run | Day | Top-5 ties | RSI vs previous day | Memories carried from earlier days | Slots used |", "|---|---|---|---|---|---|"]
        for k, rows in D.items():
            for r in rows:
                rsi = "–" if r["RSI_vs_prev_day"] is None else f"{r['RSI_vs_prev_day']:.3f}"
                md.append(f"| {k} | {r['day'] + 1} | {r['mean_top5_ties']:.2f} | {rsi} | {r['memories_from_previous_days']:.1f} | {r['memory_slots_used']:.1f} |")
        parts += ["", "## Multi-day continuity (Scenario 5)", "\n".join(md)]
    figures(Rs, Rt, a.out, ref)
    open(os.path.join(a.out, "results.md"), "w").write("\n".join(parts))
    print("\n".join(parts))


if __name__ == "__main__":
    main()
