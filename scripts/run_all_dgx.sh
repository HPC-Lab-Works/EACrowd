#!/usr/bin/env bash
# Full-budget reproduction on an NVIDIA DGX H200 (8 GPUs). Total wall time ~ 12-20 h.
# Usage:  bash scripts/run_all_dgx.sh            (from the repository root)
set -euo pipefail
cd "$(dirname "$0")/.."
SEEDS="0 1 2"
mkdir -p logs results checkpoints

# ---------------------------------------------------------------- 1. ETH/UCY benchmark
JOBS=logs/jobs_traj.txt; : > $JOBS
for s in eth hotel univ zara1 zara2; do
  for m in cv_det cv sfm; do echo "python -m traj.train --scene $s --model $m --out results/traj_dgx" >> $JOBS; done
  for m in nsrf nsrf_v2 nsrf_v2_no_refine lstm transformer nsrf_no_memory nsrf_no_social nsrf_no_emotion nsrf_no_nsre; do
    for seed in $SEEDS; do
      echo "python -m traj.train --scene $s --model $m --seed $seed --epochs 100 --d 256 --heads 8 --bs 32 --lr 1e-3 \
--eval_every 5 --workers 6 --out results/traj_dgx --save_ckpt checkpoints/traj/${s}_${m}_s${seed}.pt" >> $JOBS
    done
  done
done
python scripts/dgx_launcher.py $JOBS --per_gpu 3

# ---------------------------------------------------------------- 2. hospital digital twin: MAPPO training
JOBS=logs/jobs_sim_train.txt; : > $JOBS
for m in nsrf nsrf_v2 nsrf_v3 nsrf_v2_task_only nsrf_v2_no_escort nsrf_v2_no_exitbal nsrf_no_ramg nsrf_no_met nsrf_no_sgt nsrf_no_nsre; do
  for seed in $SEEDS; do
    echo "python -m sim.mappo --method $m --seed $seed --episodes 1000 --n_agents 500 --steps 3000 --out checkpoints/sim" >> $JOBS
  done
done
python scripts/dgx_launcher.py $JOBS --per_gpu 2

# ---------------------------------------------------------------- 3. evaluation (10 seeds), multi-day, scalability
python -m sim.run eval --methods sfm astar ga ebp nsrf nsrf_v2 nsrf_v3 nsrf_v2_task_only nsrf_v2_no_escort nsrf_v2_no_exitbal nsrf_no_ramg nsrf_no_met nsrf_no_sgt nsrf_no_nsre nsrf_no_ppo \
  --seeds 0 1 2 3 4 5 6 7 8 9 --n_agents 500 --steps 3000 --out results/sim_dgx
python -m sim.run days  --methods ga nsrf nsrf_v3 --seeds 0 1 2 --days 7 --n_agents 500 --steps 3000 --out results/sim_dgx
python -m sim.run scale --methods sfm astar ga ebp nsrf nsrf_v3 --sizes 100 500 1000 2000 5000 --steps 200 --out results/sim_dgx

# ---------------------------------------------------------------- 4. optional: LLM-backed reasoning (needs vllm)
# python -m sim.run eval --methods ga nsrf nsrf_v2 --seeds 0 1 2 --llm Qwen/Qwen2.5-7B-Instruct --llm_agents 32 --out results/sim_dgx

# ---------------------------------------------------------------- 5. tables
python scripts/aggregate.py --traj results/traj_dgx --sim results/sim_dgx --out results/report_dgx
