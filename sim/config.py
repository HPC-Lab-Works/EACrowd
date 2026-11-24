"""Method and scenario configurations for the hospital digital-twin benchmark."""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Method:
    name: str
    motion: str = "sfm"          # "sfm" social-force steering | "astar" path following + contact only
    emotion: str = "none"        # "none" | "rule" (ALMA-style appraisal) | "met" (Multimodal Emotion Transformer)
    memory: str = "none"         # "none" | "stream" (flat memory stream) | "graph" (RAMG)
    social: str = "none"         # "none" | "static" (fixed ties) | "dynamic" (SGT, evolving F/T/I)
    groups: bool = False         # group cohesion when following a friend
    policy: str = "rules"        # "rules" | "ppo" (multi-agent PPO planner)
    nsre: bool = False           # neuro-symbolic rule layer (action masking + rules)
    info: str = "contact"        # information-sharing rule: contact | memory | emotion | trust
    exit_choice: str = "nearest" # nearest | familiar | policy
    density_routing: bool = False  # emotion-weighted cost (avoid crowded zones when unhappy)
    llm: bool = False            # optional LLM reasoning hook (DGX only, see sim/llm_hook.py)
    task_reward: float = 1.0     # PPO reward for completing a scheduled task
    escort: bool = False         # visitors walk with their patient (group motion with velocity alignment)
    exit_balance: bool = False   # NSRE exit-queue rule + crowding penalty near exits
    queue_cost: float = 2.0      # metres of extra route per person queued within 10 m of an exit


SFM = Method("SFM", motion="sfm")
ASTAR = Method("A*", motion="astar")
GA = Method("GenerativeAgents-lite", motion="sfm", memory="stream", social="static", info="memory",
            exit_choice="familiar")
EBP = Method("Emotion-Based Planner", motion="astar", emotion="rule", info="emotion", density_routing=True)
NSRF = Method("NeuroSocial-RAGFormer", motion="sfm", emotion="met", memory="graph", social="dynamic", groups=True,
              policy="ppo", nsre=True, info="trust", exit_choice="policy", density_routing=True)

# v2: re-weighted task reward, escorted family groups with group force, exit-queue balancing
NSRF_V2 = replace(NSRF, name="NeuroSocial-RAGFormer v2", task_reward=3.0, escort=True, exit_balance=True)

ABLATIONS = {
    "w/o RAMG": replace(NSRF, name="w/o RAMG", memory="none"),
    "w/o MET": replace(NSRF, name="w/o MET", emotion="none", density_routing=False),
    "w/o SGT": replace(NSRF, name="w/o SGT", social="none", groups=False, info="contact"),
    "w/o NSRE": replace(NSRF, name="w/o NSRE", nsre=False),
    "w/o PPO": replace(NSRF, name="w/o PPO (rules)", policy="rules"),
}

METHODS = {"sfm": SFM, "astar": ASTAR, "ga": GA, "ebp": EBP, "nsrf": NSRF, "nsrf_v2": NSRF_V2,
           "nsrf_v2_task_only": replace(NSRF, name="v2: task reward only", task_reward=3.0),
           "nsrf_v2_no_escort": replace(NSRF_V2, name="v2 w/o escort groups", escort=False),
           "nsrf_v2_no_exitbal": replace(NSRF_V2, name="v2 w/o exit balancing", exit_balance=False),
           "nsrf_v3": replace(NSRF, name="NeuroSocial-RAGFormer v3", task_reward=3.0, exit_balance=True, queue_cost=0.5),
           "nsrf_no_ramg": ABLATIONS["w/o RAMG"], "nsrf_no_met": ABLATIONS["w/o MET"],
           "nsrf_no_sgt": ABLATIONS["w/o SGT"], "nsrf_no_nsre": ABLATIONS["w/o NSRE"],
           "nsrf_no_ppo": ABLATIONS["w/o PPO"]}


@dataclass
class Scenario:
    n_agents: int = 500
    steps: int = 3000              # 0.4 s per step -> 20 virtual minutes
    dt: float = 0.4
    announce_frac: float = 0.10    # Scenario 3: announcement time (fraction of episode)
    emergency_frac: float = 0.60   # Scenario 4: emergency time
    alarm_delay_s: float = 60.0    # building-wide alarm after local detection
    n_seed_informed: int = 5
    decision_every: int = 10       # high-level decision interval (steps)
    dwell_scale: float = 1.0
    record_agents: int = 250       # agents whose kinematics feed the human-likeness histograms


ROLE_COUNTS_500 = {"doctor": 50, "nurse": 80, "patient": 250, "visitor": 90, "admin": 30}
ROLES = list(ROLE_COUNTS_500)
