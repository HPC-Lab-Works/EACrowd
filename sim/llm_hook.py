"""Optional LLM reasoning hook (DGX only; off by default).

Gives the neural part of the Neuro-Symbolic Reasoning Engine (and the
Generative-Agents baseline) a real language model: for a subset of agents at
each decision step, the LLM reads a short natural-language description of the
agent's situation (role, place, emotion, top retrieved memories, social context)
and proposes one of the seven high-level actions. The NSRE action mask is then
applied, so an LLM proposal that violates a symbolic rule is rejected (Eq. 17-18)
and replaced by the planner's choice.

Usage (example):
    python -m sim.run eval --methods nsrf --llm Qwen/Qwen2.5-7B-Instruct --llm_agents 32
Requires: pip install vllm  (or transformers; vLLM is used when available)
"""
from __future__ import annotations

import re

import torch

from .nets import ACTIONS
from .world import ZONES
from .config import ROLES

EMO = lambda p: ("pleased" if p[0] > 0.2 else "upset" if p[0] < -0.2 else "calm") + \
    (", agitated" if p[1] > 0.3 else "") + (", powerless" if p[2] < -0.3 else "")


class LLMHook:
    def __init__(self, model_name: str, n_agents: int = 32, every: int = 5, max_tokens: int = 8):
        self.n, self.every, self.calls = n_agents, every, 0
        try:
            from vllm import LLM, SamplingParams
            self.llm = LLM(model=model_name, dtype="bfloat16", gpu_memory_utilization=0.35)
            self.sp = SamplingParams(temperature=0.7, max_tokens=max_tokens)
            self.backend = "vllm"
        except Exception:  # transformers fallback
            from transformers import AutoModelForCausalLM, AutoTokenizer
            self.tok = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
            self.backend = "hf"
            self.max_tokens = max_tokens

    def prompt(self, sim, i, mem_vec):
        z = ZONES[int(sim.goal[i])]
        mem_types = ["finished a task", "waited a long time", "was in a crowd", "heard an alarm", "received news", "met a friend"]
        nz = len(ZONES)
        top = mem_vec[nz:nz + 6].topk(2).indices.tolist()
        mem_txt = "; ".join(mem_types[t] for t in top if mem_vec[nz + t] > 0.05) or "nothing notable"
        em = "YES - an emergency has been announced" if bool(sim.em_known[i]) else "no"
        opts = "\n".join(f"{k}. {a.replace('_', ' ')}" for k, a in enumerate(ACTIONS))
        return (f"You are a {ROLES[int(sim.role[i])]} in a hospital, heading to/at the {z}. You feel {EMO(sim.pad[i].tolist())}. "
                f"Recently you {mem_txt}. Emergency: {em}. You {'know' if bool(sim.informed[i]) else 'do not know'} "
                f"the latest announcement.\nChoose ONE action number:\n{opts}\nAnswer with the number only.")

    def generate(self, prompts):
        if self.backend == "vllm":
            outs = self.llm.generate(prompts, self.sp, use_tqdm=False)
            return [o.outputs[0].text for o in outs]
        enc = self.tok(prompts, return_tensors="pt", padding=True).to(self.model.device)
        out = self.model.generate(**enc, max_new_tokens=self.max_tokens, do_sample=True, temperature=0.7)
        return self.tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    def override(self, sim, idx, b, a):
        self.calls += 1
        if self.calls % self.every:
            return a
        sel = torch.randperm(len(idx), device=idx.device)[: self.n]
        prompts = [self.prompt(sim, int(idx[s]), b["mem"][s]) for s in sel]
        texts = self.generate(prompts)
        a = a.clone()
        for s, txt in zip(sel.tolist(), texts):
            m = re.search(r"\d", txt)
            if not m:
                continue
            k = int(m.group())
            if k < len(ACTIONS) and bool(b["amask"][s, k]):     # NSRE verification (Eq. 17)
                a[s] = k
        return a
