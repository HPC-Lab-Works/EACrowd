#!/usr/bin/env bash
# Step 1 on the DGX: checks that CUDA works with this code and measures run times,
# then prints an estimate of the full single-GPU budget. Takes roughly 5-15 minutes.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs results/gpu_check
python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA not visible inside the container (check --gpus)"
print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__)
PY
echo "== forecasting: 1 epoch of NSRF-T v2 at full width (d=256) on zara1"
python -m traj.train --scene zara1 --model nsrf_v2 --epochs 1 --eval_every 1 --d 256 --heads 8 --bs 32 \
  --workers 4 --out results/gpu_check > logs/gpu_check_traj.log 2>&1
echo "== simulation: 2 PPO episodes, 500 agents, 3000 steps"
python -m sim.mappo --method nsrf_v2_task_only --episodes 2 --n_agents 500 --steps 3000 \
  --out results/gpu_check > logs/gpu_check_sim.log 2>&1
python - <<'PY'
import json, glob
t = json.load(open("results/gpu_check/zara1_nsrf_v2_s0.json"))
ep_traj = t["time_s"]                      # 1 epoch incl. data loading, memory refresh and test evaluation
sim = [json.loads(l) for l in open("logs/gpu_check_sim.log") if l.startswith("{")]
ep_sim = sim[-1]["elapsed_s"] / len(sim)
print(f"forecasting: {ep_traj:.0f} s for one epoch (+evaluation) | simulation: {ep_sim:.0f} s per PPO episode")
print(f"peak GPU memory of one forecasting job: {t.get('peak_gpu_mem_GB', float('nan')):.1f} GB")
def show(name, jobs, units, per_unit):
    h = jobs * units * per_unit / 3600
    print(f"  {name}: {jobs} jobs x {units} -> {h:.1f} GPU-hours if run one at a time")
print("Serial estimates (concurrent jobs on the same GPU usually finish sooner; measure before relying on it):")
show("forecasting, 9 learned models x 5 scenes x 3 seeds, 100 epochs", 135, 100, ep_traj)
show("forecasting, same with 50 epochs", 135, 50, ep_traj)
show("simulation PPO, 6 variants x 1 seed, 1000 episodes", 6, 1000, ep_sim)
show("simulation PPO, 6 variants x 1 seed, 300 episodes", 6, 300, ep_sim)
PY
echo "GPU check finished. Choose TRAJ_EPOCHS / SIM_EPISODES for run_dgx_1gpu.sh from the estimates above."
