from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


def read_experiment(path, mode, task=None, tasks=None):
    path = Path(path).resolve()
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    if raw.get("format_version") != 2 or raw.get("maniskill_version") != "3.0.1":
        raise ValueError("Use your existing prepared format-2 experiments/three_tasks.json.")
    order = raw.get("task_order", [])
    if not 1 <= len(order) <= 6 or len(set(order)) != len(order):
        raise ValueError("task_order must contain 1-6 distinct tasks.")
    if mode in ("single", "single_encoder"):
        if tasks:
            raise ValueError("Use --task for a single-task mode.")
        chosen = task or (order[0] if len(order) == 1 else None)
        if chosen not in order:
            raise ValueError(f"--task must be one of {order}")
        names = [chosen]
    else:
        if task:
            raise ValueError("Use --tasks, not --task, for joint training.")
        names = tasks or order
        if not names or len(set(names)) != len(names) or any(n not in order for n in names):
            raise ValueError("--tasks must select distinct configured tasks.")
    entries = []
    for name in names:
        source = raw["tasks"][name]
        spec = source["audit"]["task_spec"]
        entry = {"task_id": name, "obs_dim": spec["obs_dim"],
                 "action_dim": spec["action_dim"], "robot_uid": spec["robot_uid"],
                 "control_mode": spec["control_mode"], "control_freq": spec["control_freq"],
                 "action_contract": spec["action_contract"],
                 "max_episode_steps": source["eval"]["max_episode_steps"],
                 "env_kwargs": source["eval"].get("env_kwargs", {}),
                 "train_reset_seeds": source.get("train_reset_seeds", []),
                 "trajectory_keys": source["trajectory_keys"],
                 "demo_path": source["demo_path"], "metadata_path": source["metadata_path"]}
        for key in ("demo_path", "metadata_path"):
            location = Path(entry[key]).expanduser()
            entry[key] = str(location.resolve() if location.is_absolute() else (path.parent / location).resolve())
        if (spec["task_id"] != name or source["eval"]["env_id"] != name
                or source["eval"]["robot_uids"] != entry["robot_uid"]
                or source["eval"]["control_mode"] != entry["control_mode"]):
            raise ValueError(f"Conflicting evaluation metadata for {name}")
        if type(entry["max_episode_steps"]) is not int or entry["max_episode_steps"] < 1:
            raise ValueError("Each task requires an explicit positive episode time limit.")
        if not entry["trajectory_keys"] or len(set(entry["trajectory_keys"])) != len(entry["trajectory_keys"]):
            raise ValueError("Each task requires distinct selected trajectory_keys.")
        entries.append(entry)
    signatures = {json.dumps({k: e[k] for k in ("robot_uid", "control_mode", "control_freq",
                                               "action_dim", "action_contract")}, sort_keys=True)
                  for e in entries}
    if len(signatures) != 1:
        raise ValueError("Joint training requires identical audited robot/controller/action contracts.")
    return entries, raw.get("model", {})


class OfficialWindowDataset(Dataset):
    """Official GPU-resident windows, with explicit trajectory keys from config.

    The range endpoint and padding deliberately match the supplied train.py,
    including its last-window omission. No uniform-trajectory sampler or state
    normalization is introduced. Read only obs/actions, not unused env_states.
    """
    def __init__(self, entry, settings, device):
        metadata = json.loads(Path(entry["metadata_path"]).read_text(encoding="utf-8-sig"))
        env_info = metadata.get("env_info", {})
        kwargs = env_info.get("env_kwargs", {})
        if env_info.get("env_id", entry["task_id"]) != entry["task_id"]:
            raise ValueError("Dataset env_id differs from config.")
        modes = [kwargs["control_mode"]] if "control_mode" in kwargs else [
            ep["control_mode"] for ep in metadata.get("episodes", []) if "control_mode" in ep]
        if not modes or any(m != entry["control_mode"] for m in modes):
            raise ValueError("Dataset control_mode differs from config or is missing.")
        robot = kwargs.get("robot_uids")
        if robot is not None and robot != entry["robot_uid"]:
            raise ValueError("Dataset robot differs from config.")
        self.obs_horizon = settings["obs_horizon"]
        self.pred_horizon = settings["pred_horizon"]
        if entry["control_mode"] not in {"pd_ee_delta_pos", "pd_ee_delta_pose", "pd_joint_delta_pos"}:
            raise ValueError("This padding assumes delta arm commands and one final gripper coordinate.")
        self.trajectories = {"observations": [], "actions": []}
        self.slices = []
        with h5py.File(entry["demo_path"], "r") as handle:
            for traj_idx, key in enumerate(entry["trajectory_keys"]):
                group = handle[key]
                if not isinstance(group["obs"], h5py.Dataset):
                    raise ValueError("Replay to numeric obs_mode=state first.")
                obs = np.asarray(group["obs"], dtype=np.float32)
                act = np.asarray(group["actions"], dtype=np.float32)
                if (obs.ndim != 2 or act.ndim != 2 or obs.shape != (len(act) + 1, entry["obs_dim"])
                        or act.shape[1] != entry["action_dim"] or len(act) < 1):
                    raise ValueError(f"Bad observation/action shape in {key}")
                if not np.isfinite(obs).all() or not np.isfinite(act).all():
                    raise ValueError(f"Nonfinite data in {key}")
                if np.any(np.abs(act) > 1.00001):
                    raise ValueError("Actions must use normalized [-1,1] controller coordinates.")
                self.trajectories["observations"].append(torch.tensor(obs, device=device))
                self.trajectories["actions"].append(torch.tensor(act, device=device))
                L = len(act)
                pad_before = self.obs_horizon - 1
                pad_after = self.pred_horizon - self.obs_horizon
                self.slices += [(traj_idx, start, start + self.pred_horizon)
                                for start in range(-pad_before, L - self.pred_horizon + pad_after)]
        self.pad_action_arm = torch.zeros(entry["action_dim"] - 1, device=device)
        if not self.slices:
            raise ValueError("No official training windows; collect longer trajectories.")
        print(f"{entry['task_id']}: {len(entry['trajectory_keys'])} demos, {len(self.slices)} official windows", flush=True)

    def __getitem__(self, index):
        traj_idx, start, end = self.slices[index]
        L = len(self.trajectories["actions"][traj_idx])
        obs_seq = self.trajectories["observations"][traj_idx][max(0, start):start + self.obs_horizon]
        act_seq = self.trajectories["actions"][traj_idx][max(0, start):end]
        if start < 0:
            obs_seq = torch.cat([obs_seq[0].repeat(-start, 1), obs_seq], dim=0)
            act_seq = torch.cat([act_seq[0].repeat(-start, 1), act_seq], dim=0)
        if end > L:
            pad_action = torch.cat((self.pad_action_arm, act_seq[-1, -1][None]), dim=0)
            act_seq = torch.cat([act_seq, pad_action.repeat(end - L, 1)], dim=0)
        assert obs_seq.shape[0] == self.obs_horizon and act_seq.shape[0] == self.pred_horizon
        return {"observations": obs_seq, "actions": act_seq}

    def __len__(self):
        return len(self.slices)


class TaskBoundPolicy(nn.Module):
    """Expose official get_action(obs), binding only the task selection."""
    def __init__(self, model, task_index):
        super().__init__()
        self.model, self.task_index = model, task_index

    def get_action(self, obs):
        return self.model.get_action(obs, self.task_index)


class SeedFirstReset:
    """Set the first reset seed while leaving the official evaluator untouched."""
    def __init__(self, env, seed):
        self.env, self.seed = env, seed

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, **kwargs):
        if self.seed is not None and kwargs.get("seed") is None:
            kwargs["seed"] = self.seed
            self.seed = None
        return self.env.reset(**kwargs)


def make_official_env(entry, settings, args, video_dir=None):
    from diffusion_policy.make_env import make_eval_envs
    kwargs = dict(robot_uids=entry["robot_uid"], control_mode=entry["control_mode"],
                  reward_mode="sparse", obs_mode="state", max_episode_steps=entry["max_episode_steps"],
                  render_mode="rgb_array", human_render_camera_configs=dict(shader_pack="default"))
    extra = entry.get("env_kwargs", {})
    protected = set(kwargs) | {"num_envs", "sim_backend", "reconfiguration_freq"}
    if protected.intersection(extra):
        raise ValueError("env_kwargs cannot override robot/controller/evaluation settings.")
    kwargs.update(extra)
    env = make_eval_envs(entry["task_id"], args.num_eval_envs, args.sim_backend, kwargs,
                         dict(obs_horizon=settings["obs_horizon"]), video_dir=video_dir)
    try:
        if tuple(env.single_observation_space.shape) != (settings["obs_horizon"], entry["obs_dim"]):
            raise ValueError("Runtime observation space differs from dataset.")
        action = env.single_action_space
        if tuple(action.shape) != (entry["action_dim"],) or not np.all(action.low == -1) or not np.all(action.high == 1):
            raise ValueError("Runtime action space differs from normalized dataset contract.")
    except Exception:
        env.close()
        raise
    return env


def default_settings(config):
    for key, expected in (("num_train_timesteps", 100), ("num_inference_steps", 100), ("kernel_size", 5)):
        if key in config and config[key] != expected:
            raise ValueError(f"Compatibility version uses official {key}={expected}; your config requests {config[key]}.")
    settings = dict(obs_horizon=2, act_horizon=8, pred_horizon=16,
                    diffusion_step_embed_dim=64, unet_dims=[64, 128, 256], n_groups=8,
                    encoder_dim=128, task_embedding_dim=32, adapter_rank=16)
    for key in settings:
        if key in config:
            settings[key] = config[key]
    if "down_dims" in config:
        settings["unet_dims"] = list(config["down_dims"])
    for key in settings:
        if key != "unet_dims" and (type(settings[key]) is not int or settings[key] < 1):
            raise ValueError(f"{key} must be a positive integer")
    dims = settings["unet_dims"]
    if not dims or any(type(d) is not int or d < 1 or d % settings["n_groups"] for d in dims) or dims[0] % 8:
        raise ValueError("UNet widths must match GroupNorm divisibility.")
    if settings["obs_horizon"] + settings["act_horizon"] - 1 > settings["pred_horizon"]:
        raise ValueError("Prediction horizon is too short.")
    if settings["pred_horizon"] % (2 ** (len(dims) - 1)) or settings["diffusion_step_embed_dim"] < 4 or settings["diffusion_step_embed_dim"] % 2:
        raise ValueError("Invalid UNet horizon or timestep embedding dimension.")
    return settings
