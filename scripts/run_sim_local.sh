#!/usr/bin/env bash
# Reduced-budget simulation benchmark used for the report (CPU, 1 thread).
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1
mkdir -p logs checkpoints/sim results/sim
for m in nsrf nsrf_no_ramg nsrf_no_met nsrf_no_sgt nsrf_no_nsre; do
  [ -f checkpoints/sim/${m}_s0.pt ] && continue
  python3 -m sim.mappo --method $m --episodes 60 --n_agents 200 --steps 1500 --cpu --mb 4096 \
      --out checkpoints/sim > logs/sim_train_${m}.log 2>&1
done
python3 -m sim.run eval --methods sfm astar ga ebp nsrf nsrf_no_ramg nsrf_no_met nsrf_no_sgt nsrf_no_nsre nsrf_no_ppo \
    --seeds 0 1 2 --n_agents 500 --steps 3000 --cpu --out results/sim > logs/sim_eval.log 2>&1
python3 -m sim.run days --methods ga nsrf --seeds 0 --days 3 --n_agents 500 --steps 1500 --cpu --out results/sim > logs/sim_days.log 2>&1
python3 -m sim.run scale --methods sfm ga ebp nsrf --sizes 100 500 1000 2000 --steps 30 --cpu --out results/sim > logs/sim_scale.log 2>&1
echo DONE > logs/sim_local.done
