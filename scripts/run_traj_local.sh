#!/usr/bin/env bash
# Reduced-budget run used for the numbers in the report (2 shared CPU cores, no GPU):
# d=64, 4 heads, 10 epochs, every 3rd training window (test protocol unchanged), 1 seed.
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1
OUT=results/traj_local
mkdir -p $OUT logs checkpoints
for s in eth hotel univ zara1 zara2; do
  for m in cv_det cv sfm; do
    [ -f $OUT/${s}_${m}_s0.json ] || python3 -m traj.train --scene $s --model $m --workers 0 --threads 1 --out $OUT > logs/traj_${s}_${m}.log 2>&1
  done
done
for m in nsrf transformer lstm nsrf_no_memory nsrf_no_social nsrf_no_emotion nsrf_no_nsre; do
  for s in eth hotel univ zara1 zara2; do
    [ -f $OUT/${s}_${m}_s0.json ] && continue
    python3 -m traj.train --scene $s --model $m --epochs 10 --eval_every 2 --d 64 --heads 4 --bs 32 --train_skip 3 \
      --workers 0 --threads 1 --cpu --out $OUT --save_ckpt checkpoints/${s}_${m}.pt > logs/traj_${s}_${m}.log 2>&1
  done
done
echo DONE > logs/traj_local.done
