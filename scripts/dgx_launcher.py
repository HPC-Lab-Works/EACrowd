"""Minimal multi-GPU job queue for a DGX node: runs each command on the next free GPU.

python scripts/dgx_launcher.py jobs.txt --gpus 0 1 2 3 4 5 6 7 --per_gpu 2
"""
import argparse
import os
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument("jobs")
p.add_argument("--gpus", nargs="+", default=[str(i) for i in range(8)])
p.add_argument("--per_gpu", type=int, default=2, help="concurrent jobs per GPU (small models share well)")
a = p.parse_args()
jobs = [l.strip() for l in open(a.jobs) if l.strip() and not l.startswith("#")]
slots = [g for _ in range(a.per_gpu) for g in a.gpus]   # round-robin over GPUs
running = {}
os.makedirs("logs", exist_ok=True)
while jobs or running:
    for s in list(running):
        if running[s][0].poll() is not None:
            print(f"[done rc={running[s][0].returncode}] {running[s][1]}", flush=True)
            del running[s]
    free = [i for i in range(len(slots)) if i not in running]
    while jobs and free:
        i = free.pop(0)
        cmd = jobs.pop(0)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=slots[i], OMP_NUM_THREADS="4")
        log = open(f"logs/job_{abs(hash(cmd)) % 10**8}.log", "w")
        running[i] = (subprocess.Popen(cmd, shell=True, env=env, stdout=log, stderr=subprocess.STDOUT), cmd)
        print(f"[gpu {slots[i]}] {cmd}", flush=True)
    time.sleep(5)
