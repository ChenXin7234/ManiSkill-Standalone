import argparse
import json
import os
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from model_shared import Agent
from model_shared_adaptor import Agent as SharedAdapterAgent
from model_task_adaptor import Agent as TaskAdapterAgent

from cotrain_prepare import make_official_env, SeedFirstReset

class TaskPolicy(nn.Module):
    """Bind task_index to the official get_action(obs) interface; share the model."""
    def __init__(self, model, task_index):
        super().__init__()
        self.model = model
        self.task_index = task_index

    def get_action(self, obs):
        return self.model.get_action(obs, self.task_index)

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=200001)
    parser.add_argument("--num-eval-episodes", type=int, default=20)
    parser.add_argument("--num-eval-envs", type=int, default=1)
    parser.add_argument("--sim-backend", choices=("physx_cpu", "physx_cuda"), default="physx_cpu")
    parser.add_argument("--capture-video", action=argparse.BooleanOptionalAction, default=False)
    args = parser.parse_args(argv)

    if args.output.exists():
        parser.error("Choose a new output path")
    if not 0 <= args.seed < 2**32:
        parser.error("seed must be in [0, 2**32)")
    if args.num_eval_envs < 1 or args.num_eval_episodes < 1 or args.num_eval_episodes % args.num_eval_envs:
        parser.error("Positive episode/env counts required; episodes must be divisible by env count")
    if os.name == "nt" and args.sim_backend == "physx_cpu" and args.num_eval_envs > 1:
        parser.error("Use num-eval-envs=1 with the official CPU factory on Windows")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)

    # The checkpoint contains the model structure metadata, so no dataset/config is read.
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    entries = checkpoint["entries"]
    settings = checkpoint["settings"]
    if any(args.seed in entry.get("train_reset_seeds", []) for entry in entries):
        raise ValueError("Evaluation seed overlaps a selected training reset seed")

    model_type = checkpoint.get("model_type", "shared")
    if model_type == "shared":
        agent = Agent(entries, SimpleNamespace(**settings)).to(device)
    elif model_type == "shared_adapter":
        agent = SharedAdapterAgent(entries, SimpleNamespace(**settings)).to(device)
    elif model_type == "task_adapter":
        agent = TaskAdapterAgent(entries, SimpleNamespace(**settings)).to(device)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    agent.load_state_dict(checkpoint["ema_agent"], strict=True)
    agent.eval()

    from diffusion_policy.evaluate import evaluate
    results = {}
    for task_index, entry in enumerate(entries):
        print(f"Evaluating {entry['task_id']}: {args.num_eval_episodes} episodes")
        video_dir = (str(args.output.parent / f"{args.output.stem}_videos" / entry["task_id"])
                        if args.capture_video else None)
        env = make_official_env(entry, settings, args, video_dir)
        try:
            metrics = evaluate(
                args.num_eval_episodes,
                TaskPolicy(agent, task_index),
                SeedFirstReset(env, args.seed),
                device,
                args.sim_backend,
            )
            results[entry["task_id"]] = {
                name: float(np.asarray(values).mean())
                for name, values in metrics.items()
            }
        finally:
            env.close()

    result = {
        "model_type": checkpoint["model_type"],
        "checkpoint": str(args.checkpoint.resolve()),
        "completed_updates": checkpoint["completed_updates"],
        "weights": "ema_agent", "eval_seed": args.seed,
        "num_eval_episodes_per_task": args.num_eval_episodes,
        "num_eval_envs": args.num_eval_envs, "sim_backend": args.sim_backend,
        "per_task": results,
        "macro": {
            metric: float(np.mean([task[metric] for task in results.values()]))
            for metric in ("success_once", "success_at_end")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
