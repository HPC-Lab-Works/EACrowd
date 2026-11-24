import json, glob, sys, collections, math
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = __import__("os").path.dirname(__import__("os").path.dirname(__import__("os").path.abspath(__file__))); sys.path.insert(0, ROOT)
D = sys.argv[1] if len(sys.argv) > 1 else ROOT + "/results"; OUT = sys.argv[2] if len(sys.argv) > 2 else ROOT + "/results/paper_figures"; __import__("os").makedirs(OUT, exist_ok=True)
BLUE, LBLUE, GREY, LGREY, INK, INK2 = "#2a78d6", "#8db8ec", "#8a8986", "#c9c8c4", "#0b0b0b", "#52514e"
CAT = {"A*": "#eb6834", "SFM": "#1baf7a", "GA": "#eda100", "EBP": "#e87ba4"}
plt.rcParams.update({"font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e1",
                     "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False, "savefig.dpi": 300})
SC = ["eth", "hotel", "univ", "zara1", "zara2"]

# ---------------- 1. floor plan
from sim.world import build_world, ZONES, RES
w = build_world()
fig, ax = plt.subplots(figsize=(6.6, 4.2))
z = w.zone.astype(float); img = np.ones(z.shape + (3,))
pal = {0: (0.93, 0.93, 0.91), 12: (0.16, 0.47, 0.84), 13: (0.16, 0.47, 0.84)}
for zi in range(1, 12): pal[zi] = (0.97, 0.97, 0.96)
for zi, c in pal.items(): img[w.zone == zi] = c
img[w.walls] = (0.2, 0.2, 0.2)
ax.imshow(img, origin="lower", extent=(0, 80, 0, 50))
names = {"reception": "Reception", "waiting_A": "Waiting room A", "waiting_B": "Waiting room B", "consult_1": "Consult. 1",
         "consult_2": "Consult. 2", "consult_3": "Consult. 3", "consult_4": "Consult. 4", "emergency": "Emergency unit",
         "icu": "ICU", "admin": "Administration", "lounge": "Lounge"}
for k, (x0, x1, y0, y1) in w.rects.items():
    ax.text((x0 + x1) / 2, (y0 + y1) / 2 + (5 if k == "emergency" else 0), names[k], ha="center", va="center", fontsize=7, color=INK)
ax.text(40, 25, "Corridor", ha="center", va="center", fontsize=7, color=INK2)
ax.text(2.2, 22.3, "Main exit", fontsize=6.5, color=BLUE, ha="left"); ax.text(77.8, 22.3, "East exit", fontsize=6.5, color=BLUE, ha="right")
ax.plot(*w.hazard_center, marker="*", ms=11, color="#e34948"); ax.text(w.hazard_center[0], w.hazard_center[1] - 3.0, "fire (Scenario 4)", fontsize=6.5, color="#b22e2d", ha="center")
ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)"); ax.grid(False)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_floorplan.png"); plt.close(fig)

# ---------------- 2. ETH/UCY
SOTA = {"AgentFormer": [(0.45, 0.75), (0.14, 0.22), (0.25, 0.45), (0.18, 0.30), (0.14, 0.24)],
        "LED": [(0.39, 0.58), (0.11, 0.17), (0.26, 0.43), (0.18, 0.26), (0.13, 0.22)],
        "MGF": [(0.39, 0.59), (0.13, 0.20), (0.21, 0.39), (0.17, 0.29), (0.14, 0.24)]}
R = collections.defaultdict(lambda: collections.defaultdict(list))
for f in glob.glob(f"{D}/traj_dgx/*.json"):
    j = json.load(open(f)); R[j["model"]][j["scene"]].append(j["test"])
ours = {"LSTM": "lstm", "Transformer": "transformer", "NSRF-T": "nsrf"}
series = [(n, [v for v in SOTA[n]], LGREY, "//") for n in SOTA] + \
         [(n, [(np.mean([t["minADE"] for t in R[m][s]]), np.mean([t["minFDE"] for t in R[m][s]])) for s in SC],
           BLUE if n == "NSRF-T" else GREY, None) for n, m in ours.items()]
fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.6))
w_ = 0.8 / len(series)
for k, lab in enumerate(["minADE$_{20}$ (m)", "minFDE$_{20}$ (m)"]):
    ax = axs[k]
    for i, (n, v, c, h) in enumerate(series):
        ax.bar(np.arange(5) + (i - (len(series) - 1) / 2) * w_, [x[k] for x in v], w_ * 0.9, color=c if h is None else "white",
               edgecolor=c if h is None else GREY, hatch=h, linewidth=0.6, label=n + (" (published)" if h else " (this study)"))
    ax.set_xticks(range(5)); ax.set_xticklabels([s.upper() for s in SC]); ax.set_ylabel(lab)
fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=6.5, bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.16, 1, 1)); fig.savefig(f"{OUT}/fig_ethucy.png"); plt.close(fig)

# ---------------- simulation data
S = collections.defaultdict(list)
for f in glob.glob(f"{D}/sim_dgx/*_N500_s*.json"):
    j = json.load(open(f)); S[j["key"]].append(j)
from sim.metrics import BINS
ref = json.load(open(ROOT + "/results/eth_ucy_reference.json"))

# ---------------- 3. kinematics vs real
lines = [("Real pedestrians (ETH and UCY)", None, INK, 2.0, "-"), ("Social Force Model", "sfm", CAT["SFM"], 1.0, "--"),
         ("A* path following", "astar", CAT["A*"], 1.0, ":"), ("Generative-Agents-lite", "ga", CAT["GA"], 1.0, "-."),
         ("Emotion-Based Planner", "ebp", CAT["EBP"], 1.0, (0, (3, 1, 1, 1))), ("NeuroSocial-RAGFormer", "nsrf_v2_task_only", BLUE, 1.6, "-")]
fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.1))
for i, (k, lab) in enumerate([("speed", "speed (m/s)"), ("acc", "acceleration (m/s$^2$)"), ("turn", "turning rate (rad/s)"), ("nn", "nearest neighbour (m)")]):
    b = BINS[k]; c = np.r_[b[0], (b[:-1] + b[1:]) / 2, b[-1]]
    for name, key, col, lw, ls in lines:
        h = ref["hist"][k] if key is None else np.mean([r["hist"][k] for r in S[key]], 0)
        axs[i].plot(c, h, color=col, lw=lw, ls=ls, label=name)
    axs[i].set_xlabel(lab); axs[i].set_ylabel("probability" if i == 0 else "")
    if k == "acc": axs[i].set_xlim(0, 3)
    if k == "turn": axs[i].set_xlim(0, 3)
fig.legend(*axs[0].get_legend_handles_labels(), loc="lower center", ncol=3, fontsize=6.5, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.14, 1, 1)); fig.savefig(f"{OUT}/fig_kinematics.png"); plt.close(fig)

# ---------------- 4. simulation comparison
M = [("SFM", "sfm"), ("A*", "astar"), ("GA-lite", "ga"), ("EBP", "ebp"), ("NSRF v1", "nsrf"), ("NSRF", "nsrf_v2_task_only")]
cols = [GREY] * 4 + [LBLUE, BLUE]
panels = [("HLS", "Human-likeness score", 1), ("collision_rate_normal", "Collisions, normal (%)", 100),
          ("info_t50_s", "Information T50 (s)", 1), ("evac_t95_s", "Evacuation T95 (s)", 1)]
fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.2))
for ax, (key, lab, sc) in zip(axs, panels):
    mu = [np.mean([r[key] for r in S[m]]) * sc for _, m in M]; sd = [np.std([r[key] for r in S[m]]) * sc for _, m in M]
    if key == "HLS":   # narrow range: point and whisker instead of a truncated bar
        for i in range(len(M)):
            ax.errorbar(i, mu[i], yerr=sd[i], fmt="o", ms=5, color=cols[i], ecolor=INK2, elinewidth=0.7, capsize=2)
        ax.set_xlim(-0.6, len(M) - 0.4)
    else:
        ax.bar(range(len(M)), mu, 0.7, yerr=sd, color=cols, error_kw=dict(lw=0.7, capsize=2, ecolor=INK2))
    ax.set_xticks(range(len(M))); ax.set_xticklabels([n for n, _ in M], rotation=45, ha="right", fontsize=6.5); ax.set_title(lab, fontsize=7.5)
    if key == "HLS": ax.set_ylim(89, 94.5)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_sim_comparison.png"); plt.close(fig)

# ---------------- 5. ablation
A = [("Full (v1)", "nsrf"), ("w/o RAMG", "nsrf_no_ramg"), ("w/o MET", "nsrf_no_met"), ("w/o SGT", "nsrf_no_sgt"),
     ("w/o NSRE", "nsrf_no_nsre"), ("w/o PPO", "nsrf_no_ppo")]
panels = [("HLS", "Human-likeness score", 1), ("tasks_per_agent_hour", "Tasks per agent-hour", 1),
          ("collision_rate_normal", "Collisions, normal (%)", 100), ("max_exit_density", "Peak exit density (persons/m$^2$)", 1)]
fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.2))
for ax, (key, lab, sc) in zip(axs, panels):
    mu = [np.mean([r[key] for r in S[m]]) * sc for _, m in A]; sd = [np.std([r[key] for r in S[m]]) * sc for _, m in A]
    if key == "HLS":
        for i in range(len(A)):
            ax.errorbar(i, mu[i], yerr=sd[i], fmt="o", ms=5, color=BLUE if i == 0 else GREY, ecolor=INK2, elinewidth=0.7, capsize=2)
        ax.set_xlim(-0.6, len(A) - 0.4)
    else:
        ax.bar(range(len(A)), mu, 0.7, yerr=sd, color=[BLUE] + [GREY] * 5, error_kw=dict(lw=0.7, capsize=2, ecolor=INK2))
    ax.set_xticks(range(len(A))); ax.set_xticklabels([n for n, _ in A], rotation=45, ha="right", fontsize=6.5); ax.set_title(lab, fontsize=7.5)
    if key == "HLS": ax.set_ylim(89, 94.5)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_ablation.png"); plt.close(fig)

# ---------------- 6. scalability
rows = json.load(open(f"{D}/sim_dgx/scalability_cuda.json"))
names = {"sfm": ("Social Force Model", CAT["SFM"], "--"), "astar": ("A* path following", CAT["A*"], ":"),
         "ga": ("Generative-Agents-lite", CAT["GA"], "-."), "ebp": ("Emotion-Based Planner", CAT["EBP"], (0, (3, 1, 1, 1))),
         "nsrf_v2_task_only": ("NeuroSocial-RAGFormer", BLUE, "-")}
fig, ax = plt.subplots(figsize=(3.4, 2.4))
for k, (n, c, ls) in names.items():
    rr = sorted([r for r in rows if r["method"] == k], key=lambda r: r["N"])
    ax.plot([r["N"] for r in rr], [r["ms_per_step"] for r in rr], color=c, ls=ls, marker="o", ms=3, lw=1.6 if c == BLUE else 1.0, label=n)
ax.set_xscale("log"); ax.set_xticks([100, 500, 1000, 2000, 5000]); ax.set_xticklabels(["100", "500", "1k", "2k", "5k"])
ax.set_xlabel("number of agents"); ax.set_ylabel("ms per simulation step"); ax.legend(fontsize=6)
fig.tight_layout(); fig.savefig(f"{OUT}/fig_scalability.png"); plt.close(fig)
print("ok")
