#!/usr/bin/env bash
# Full benchmark on the GPUs visible in the container (one or several). Jobs run in priority order so that
# the main tables are complete first; several small jobs share the GPU concurrently (PER_GPU).
#
# Budget knobs (defaults shown):
#   TRAJ_EPOCHS=100  TRAJ_SEEDS="0 1 2"  SIM_EPISODES=1000  SIM_SEEDS="0"  EVAL_SEEDS="0 1 2 3 4 5 6 7 8 9"
#   PER_GPU=4 (concurrent jobs)   WORKERS=4 (data-loader workers per forecasting job)
#   EXTRA=1 also runs the variants that were rejected in the CPU study (escort, exit balancing, v3)
set -uo pipefail
cd "$(dirname "$0")/.."
TRAJ_EPOCHS=${TRAJ_EPOCHS:-100}; TRAJ_SEEDS=${TRAJ_SEEDS:-"0 1 2"}
SIM_EPISODES=${SIM_EPISODES:-1000}; SIM_SEEDS=${SIM_SEEDS:-"0"}; EVAL_SEEDS=${EVAL_SEEDS:-"0 1 2 3 4 5 6 7 8 9"}
PER_GPU=${PER_GPU:-4}; WORKERS=${WORKERS:-4}; EXTRA=${EXTRA:-0}
mkdir -p logs results checkpoints/traj checkpoints/sim
# ---------------------------------------------------------------- 0. GPU selection and memory guard
# Uses every GPU visible in the container (e.g. docker run --gpus '"device=0,3,5,6"'),
# skipping any GPU with less than 20 GB free (another user's process). FORCE=1 keeps them all.
GPUS=""
while IFS=, read -r idx free; do
  idx=$(echo $idx | tr -d ' '); free=$(echo $free | tr -d ' ')
  if [ "$free" -ge 20000 ] || [ "${FORCE:-0}" = "1" ]; then GPUS="$GPUS $idx"; echo "GPU $idx: ${free} MiB free -> used"
  else echo "GPU $idx: ${free} MiB free -> skipped (busy)"; fi
done < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits)
if [ -z "$GPUS" ]; then echo "ERROR: no GPU with 20 GB free; choose other GPUs or set FORCE=1 PER_GPU=1"; exit 1; fi
echo "Running on GPUs:$GPUS with $PER_GPU concurrent jobs each"

# ---------------------------------------------------------------- 1. ETH/UCY forecasting
JOBS=logs/jobs_traj.txt; : > $JOBS
for s in $SCENES; do for m in cv_det cv sfm; do
  [ -f results/traj_dgx/${s}_${m}_s0.json ] || echo "python -m traj.train --scene $s --model $m --workers 0 --out results/traj_dgx" >> $JOBS
done; done
for m in nsrf_v2 nsrf lstm transformer nsrf_no_memory nsrf_no_social nsrf_no_emotion nsrf_no_nsre nsrf_v2_no_refine; do
  for seed in $TRAJ_SEEDS; do for s in $SCENES; do
    [ -f results/traj_dgx/${s}_${m}_s${seed}.json ] && continue
    echo "python -m traj.train --scene $s --model $m --seed $seed --epochs $TRAJ_EPOCHS --d 256 --heads 8 --bs 32 --lr 1e-3 \
--eval_every 5 --workers $WORKERS --out results/traj_dgx --save_ckpt checkpoints/traj/${s}_${m}_s${seed}.pt" >> $JOBS
  done; done
done
python scripts/dgx_launcher.py $JOBS --gpus $GPUS --per_gpu $PER_GPU

# ---------------------------------------------------------------- 2. hospital twin: MAPPO training
JOBS=logs/jobs_sim_train.txt; : > $JOBS
VARIANTS="nsrf_v2_task_only nsrf nsrf_no_ramg nsrf_no_met nsrf_no_sgt nsrf_no_nsre"
[ "$EXTRA" = "1" ] && VARIANTS="$VARIANTS nsrf_v2 nsrf_v3 nsrf_v2_no_escort nsrf_v2_no_exitbal"
for m in $VARIANTS; do for seed in $SIM_SEEDS; do
  [ -f checkpoints/sim/${m}_s${seed}.pt ] || \
  echo "python -m sim.mappo --method $m --seed $seed --episodes $SIM_EPISODES --n_agents 500 --steps 3000 --out checkpoints/sim" >> $JOBS
done; done
python scripts/dgx_launcher.py $JOBS --gpus $GPUS --per_gpu $PER_GPU

# ---------------------------------------------------------------- 3. evaluation, multi-day continuity, scalability
EVAL_METHODS="sfm astar ga ebp $VARIANTS nsrf_no_ppo"
JOBS=logs/jobs_sim_eval.txt; : > $JOBS
for m in $EVAL_METHODS; do
  echo "python -m sim.run eval --methods $m --seeds $EVAL_SEEDS --n_agents 500 --steps 3000 --out results/sim_dgx" >> $JOBS
done
echo "python -m sim.run days --methods ga nsrf_v2_task_only --seeds 0 1 2 --days 7 --n_agents 500 --steps 3000 --out results/sim_dgx" >> $JOBS
python scripts/dgx_launcher.py $JOBS --gpus $GPUS --per_gpu $PER_GPU
# scalability is timed alone so that concurrent jobs do not distort it
python -m sim.run scale --methods sfm astar ga ebp nsrf_v2_task_only --sizes 100 500 1000 2000 5000 --steps 200 --out results/sim_dgx

# ---------------------------------------------------------------- 4. tables and figures
python scripts/aggregate.py --traj results/traj_dgx --sim results/sim_dgx --out results/report_dgx
echo "ALL DONE -> results/report_dgx/results.md"
