import argparse
import copy
import json
import os
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.optim as optim

from torch.utils.data.sampler import RandomSampler, BatchSampler
from torch.utils.data.dataloader import DataLoader
from diffusion_policy.utils import IterationBasedBatchSampler

from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler

from model_shared import Agent
from model_shared_adaptor import Agent as SharedAdapterAgent
from model_task_adaptor import Agent as TaskAdapterAgent

from cotrain_prepare import (
    OfficialWindowDataset, default_settings, read_experiment,
    make_official_env, SeedFirstReset, TaskBoundPolicy,
)

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("shared", "shared_adapter", "task_adapter"), default="shared")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-config", type=Path, help="Optional JSON object of model settings, separate from the data manifest")
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--total-iters", type=int, default=30000)
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size per task")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--save-freq", type=int, default=5000)
    parser.add_argument("--log-freq", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/shared_04"))
    
    parser.add_argument("--eval-freq", type=int, default=5000)
    parser.add_argument("--eval-seed", type=int, default=100001)
    parser.add_argument("--num-eval-episodes", type=int, default=20)
    parser.add_argument("--num-eval-envs", type=int, default=1)
    parser.add_argument("--sim-backend", choices=("physx_cpu", "physx_cuda"), default="physx_cpu")
    args = parser.parse_args(argv)

    for name in ("total_iters", "batch_size", "save_freq", "log_freq",
                 "eval_freq", "num_eval_episodes", "num_eval_envs"):
        if getattr(args, name) <= 0:
            parser.error(f"{name} must be positive")
    if not np.isfinite(args.lr) or args.lr <= 0 or args.warmup_steps < 0:
        parser.error("lr must be finite and positive; warmup-steps must be nonnegative")
    if not 0 <= args.warmup_steps < args.total_iters:
        parser.error("warmup-steps must be nonnegative and smaller than total-iters")
    if not 0 <= args.eval_seed < 2**32:
        parser.error("eval-seed must be in [0, 2**32)")
    if args.num_eval_episodes % args.num_eval_envs:
        parser.error("num-eval-episodes must be divisible by num-eval-envs")
    if os.name == "nt" and args.sim_backend == "physx_cpu" and args.num_eval_envs > 1:
        parser.error("Use num-eval-envs=1 with the official CPU factory on Windows")
    if args.run_dir.exists() and (not args.run_dir.is_dir() or any(args.run_dir.iterdir())):
        raise ValueError(f"Output directory {args.run_dir} is not empty; choose another --run-dir.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    device = torch.device(args.device)

    entries, model_config = read_experiment(args.config, mode=args.model, tasks=args.tasks)
    if args.model_config is not None:
        model_config = json.loads(args.model_config.read_text(encoding="utf-8-sig"))
        if not isinstance(model_config, dict):
            raise ValueError("--model-config must contain a JSON object of model settings")
        allowed = {"obs_horizon", "act_horizon", "pred_horizon", "encoder_dim",
                    "task_embedding_dim", "adapter_rank", "diffusion_step_embed_dim",
                    "unet_dims", "down_dims", "n_groups", "kernel_size",
                    "num_train_timesteps", "num_inference_steps"}
        unknown = sorted(set(model_config) - allowed)
        if unknown:
            raise ValueError(f"Unknown model config: {unknown}")
    settings = default_settings(model_config)
    if any(args.eval_seed in entry.get("train_reset_seeds", []) for entry in entries):
        raise ValueError("Evaluation seed overlaps a selected training reset seed")
    # DataLoaders for tasks
    train_dataloaders = []
    for entry in entries:
        dataset = OfficialWindowDataset(entry, settings, device)
        if len(dataset) < args.batch_size:
            raise ValueError(f"{entry['task_id']}: too few windows; reduce --batch-size.")
        
        # randomly sample dataset for each task
        sampler = RandomSampler(dataset, replacement=True)
        batch_sampler = BatchSampler(sampler, args.batch_size, drop_last=True)
        iter_batch_sampler = IterationBasedBatchSampler(batch_sampler, args.total_iters)
        
        train_dataloader = DataLoader(
            dataset,
            batch_sampler=iter_batch_sampler,
            num_workers=0,
        )
        train_dataloaders.append(train_dataloader)
    train_iterators = [iter(loader) for loader in train_dataloaders]

    if args.model == "shared":
        agent = Agent(entries, SimpleNamespace(**settings)).to(device)
    elif args.model == "shared_adapter":
        agent = SharedAdapterAgent(entries, SimpleNamespace(**settings)).to(device)
    elif args.model == "task_adapter":
        agent = TaskAdapterAgent(entries, SimpleNamespace(**settings)).to(device)
    else:
        raise ValueError(f"Unknown model type: {args.model}")

    optimizer = optim.AdamW(params=agent.parameters(),
            lr=args.lr, betas=(0.95, 0.999), weight_decay=1e-6)
    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.total_iters,
    )
    # Exponential Moving Average
    # accelerates training and improves stability
    # holds a copy of the model weights
    ema = EMAModel(parameters=agent.parameters(), power=0.75)
    ema_agent = copy.deepcopy(agent)

    # Save Checkpoints
    checkpoint_dir = args.run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)

    eval_envs = []
    best_eval_metrics = {}
    last_eval_iteration = None


    def save_checkpoint(tag, update):
        ema.copy_to(ema_agent.parameters())
        torch.save({
            "model_type": args.model,
            "agent": agent.state_dict(),
            "ema_agent": ema_agent.state_dict(),
            "entries": entries,
            "settings": settings,
            "completed_updates": update,
        }, checkpoint_dir / f"checkpoint_{tag}.pt")


    def evaluate_and_save_best(iteration, force=False):
        nonlocal last_eval_iteration
        if last_eval_iteration == iteration:
            return
        if not force and iteration % args.eval_freq != 0:
            return

        ema.copy_to(ema_agent.parameters())
        results = {}
        from diffusion_policy.evaluate import evaluate
        for task_index, (entry, env) in enumerate(zip(entries, eval_envs)):
            metrics = evaluate(
                args.num_eval_episodes,
                TaskBoundPolicy(ema_agent, task_index),
                SeedFirstReset(env, args.eval_seed),
                device,
                args.sim_backend,
            )
            results[entry["task_id"]] = {
                name: float(np.asarray(values).mean())
                for name, values in metrics.items()
            }

        mean_metrics = {
            name: float(np.mean([task[name] for task in results.values()]))
            for name in ("success_once", "success_at_end")
        }
        record = {
            "iteration": iteration, "per_task": results, "mean": mean_metrics,
            "eval_seed": args.eval_seed, "episodes_per_task": args.num_eval_episodes,
        }
        print(json.dumps(record, ensure_ascii=False))
        with (args.run_dir / "validation.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")

        for name, value in mean_metrics.items():
            if name not in best_eval_metrics or value > best_eval_metrics[name]:
                best_eval_metrics[name] = value
                save_checkpoint(f"best_eval_{name}", iteration)
        last_eval_iteration = iteration

    # ---------------------------------------------------------------------------- #
    # Training begins.
    # ---------------------------------------------------------------------------- # 
    try:
        for entry in entries:
            env = make_official_env(entry, settings, args, video_dir=None)
            eval_envs.append(env)
    
        for iteration in range(1, args.total_iters + 1):
            agent.train()
            optimizer.zero_grad(set_to_none=True)
            losses = {}

            for task_index, iterator in enumerate(train_iterators):
                data_batch = next(iterator)
                task_loss = agent.compute_loss(
                    obs_seq=data_batch["observations"],
                    action_seq=data_batch["actions"], task_id=task_index,
                )
                if not torch.isfinite(task_loss):
                    raise RuntimeError(f"Nonfinite loss for {entries[task_index]['task_id']}")
                (task_loss / len(entries)).backward()
                losses[entries[task_index]["task_id"]] = task_loss.detach().item()

            optimizer.step()
            lr_scheduler.step()
            ema.step(agent.parameters())

            if iteration == 1 or iteration % args.log_freq == 0:
                total_loss = sum(losses.values()) / len(entries)
                print(f"iteration={iteration}, loss={total_loss:.6f}, per_task={losses}")
            if iteration % args.save_freq == 0:
                save_checkpoint(str(iteration), iteration)
            evaluate_and_save_best(iteration)

        save_checkpoint("final", args.total_iters)
        evaluate_and_save_best(args.total_iters, force=True)
    finally:
        for env in eval_envs:
            env.close()


if __name__ == "__main__":
    main()



