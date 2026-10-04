"""Prepare a fixed ManiSkill 3.0.1 state dataset manifest; no model training.

The H5 files remain unchanged. Training chooses observation/action horizons and
builds windows later. This script does not import any co-training module.
"""
from __future__ import annotations

import argparse
import dataclasses
import enum
import hashlib
import importlib.metadata
import json
import math
import os
import re
from pathlib import Path

import h5py
import numpy as np


ACTION_DIMS = {"pd_ee_delta_pose": 7, "pd_joint_delta_pos": 8}
DEFAULT_ROBOTS = {
    "PickCube-v1": "panda", "PushCube-v1": "panda", "PullCubeTool-v1": "panda",
    "StackCube-v1": "panda_wristcam", "StackPyramid-v1": "panda_wristcam",
    "PegInsertionSide-v1": "panda_wristcam",
}
DEFAULT_EVALUATION = {
    "validation_seeds": list(range(100000, 100100)),
    "test_seeds": list(range(200000, 200200)),
    "policy_seed_base": 1000000,
}


# 1. Small helpers: JSON, paths, deterministic selection, file hashes.
def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def positive_int(value, label):
    if type(value) is not int or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def reset_seed(value, label):
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    if type(value) is not int or not 0 <= value < 2**32:
        raise ValueError(f"{label} must be an integer in [0, 2**32)")
    return value


def resolve_input(value, directory):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Input paths must be nonempty strings")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else directory / path).resolve()


def json_digest(value):
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def subset_seed(task, seed):
    # Keep the previous selection algorithm so existing experiments stay comparable.
    text = json.dumps(["cotraining-subset-v1", seed, task], separators=(",", ":"), ensure_ascii=False)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")


# 2. Actual environment audit. This is data metadata, not a model definition.
def type_name(value):
    cls = value if isinstance(value, type) else type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def serialize(value):
    """Serialize actual controller settings without silently converting to repr()."""
    import torch
    if isinstance(value, enum.Enum):
        return {"__type__": type_name(value), "value": serialize(value.value)}
    if isinstance(value, type):
        return {"__type__": "python_type", "qualified_name": type_name(value)}
    if dataclasses.is_dataclass(value):
        result = {"__type__": type_name(value)}
        result.update({field.name: serialize(getattr(value, field.name)) for field in dataclasses.fields(value)})
        if hasattr(value, "controller_cls"):
            result["controller_cls"] = serialize(value.controller_cls)
        return result
    if isinstance(value, torch.Tensor):
        return {"__type__": "torch.Tensor", "dtype": str(value.dtype),
                "shape": list(value.shape), "values": serialize(value.detach().cpu().tolist())}
    if isinstance(value, np.ndarray):
        return {"__type__": "numpy.ndarray", "dtype": str(value.dtype),
                "shape": list(value.shape), "values": serialize(value.tolist())}
    if isinstance(value, np.generic):
        return serialize(value.item())
    if isinstance(value, Path):
        return {"__type__": "pathlib.Path", "value": value.as_posix()}
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise ValueError("Controller dictionary keys must be strings")
        return {key: serialize(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [serialize(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValueError(f"Unsupported/nonfinite controller setting: {type_name(value)}")


def observation_schema(raw):
    import torch
    fields = []
    def visit(value, path):
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, f"{path}.{key}" if path else key)
        else:
            tensor = torch.as_tensor(value)
            if tensor.ndim < 1 or tensor.shape[0] != 1:
                raise ValueError(f"{path}: expected a single-environment state")
            start = fields[-1]["end"] if fields else 0
            fields.append({"path": path, "shape": list(tensor.shape[1:]), "dtype": str(tensor.dtype),
                           "start": start, "end": start + tensor.numel()})
    visit(raw, "")
    return fields


def action_signature(spec):
    return json_digest({key: spec[key] for key in
                        ("action_dim", "robot_uid", "control_mode", "control_freq", "action_contract")})


def audit_environment(entry, robot, control):
    import gymnasium as gym
    import mani_skill
    import mani_skill.envs  # register tasks
    import torch
    from mani_skill.utils import common, gym_utils
    version = importlib.metadata.version("mani_skill")
    if version.split("+")[0] != "3.0.1":
        raise ValueError(f"Use ManiSkill 3.0.1, got {version}")
    kwargs = dict(robot_uids=robot, control_mode=control, obs_mode="state", num_envs=1,
                  sim_backend="physx_cpu", render_mode=None, render_backend="none", reconfiguration_freq=1)
    if entry.get("max_episode_steps") is not None:
        kwargs["max_episode_steps"] = entry["max_episode_steps"]
    print(f"Auditing real CPU environment: {entry['env_id']}", flush=True)
    env = gym.make(entry["env_id"], **kwargs)
    try:
        obs, _ = env.reset(seed=0)
        base = env.unwrapped
        state = torch.as_tensor(obs)
        if state.ndim != 2 or state.shape[0] != 1 or not torch.isfinite(state).all():
            raise ValueError("Environment must return finite state [1, obs_dim]")
        if base.agent.uid != robot or base.control_mode != control:
            raise ValueError("Actual robot/controller differs from requested settings")
        raw = base.get_obs(unflattened=True)
        flat = common.flatten_state_dict(raw, use_torch=True, device=base.device)
        fields = observation_schema(raw)
        if flat.shape != state.shape or not torch.equal(flat, state) or not fields or fields[-1]["end"] != state.shape[1]:
            raise ValueError("Actual state flattening/schema does not match observations")
        space = getattr(base, "single_action_space", base.action_space)
        controller = base.agent.controller
        children = getattr(controller, "controllers", None)
        mapping = getattr(controller, "action_mapping", None)
        dim = ACTION_DIMS[control]
        if (tuple(space.shape) != (dim,) or not np.all(space.low == -1) or not np.all(space.high == 1)
                or not isinstance(children, dict) or set(children) != {"arm", "gripper"}
                or not isinstance(mapping, dict) or tuple(mapping.get("arm", ())) != (0, dim - 1)
                or tuple(mapping.get("gripper", ())) != (dim - 1, dim)):
            raise ValueError("Unexpected normalized arm/gripper action space or mapping")
        names = (["translation_cmd_x", "translation_cmd_y", "translation_cmd_z",
                  "rotation_cmd_x", "rotation_cmd_y", "rotation_cmd_z"] if control == "pd_ee_delta_pose"
                 else [f"joint_delta:{name}" for name in children["arm"].config.joint_names]) + ["gripper_cmd"]
        configs = {"__type__": type_name(controller), "action_mapping": serialize(mapping), "children": {
            name: {"__type__": type_name(child), "config": serialize(child.config),
                   "control_freq": float(child.control_freq)} for name, child in children.items()}}
        spec = {"task_id": entry["env_id"], "obs_dim": int(state.shape[1]), "action_dim": dim,
                "robot_uid": robot, "control_mode": control, "control_freq": float(base.control_freq),
                "action_contract": {"action_names": names, "low": space.low.tolist(),
                                    "high": space.high.tolist(), "controller_config": configs}}
        return {"format_version": 1, "task_spec": spec, "action_signature": action_signature(spec),
                "runtime": {"maniskill_version": version, "maniskill_module": mani_skill.__file__,
                            "gymnasium_version": gym.__version__, "torch_version": torch.__version__,
                            "sim_backend": "physx_cpu", "render_backend": "none", "obs_mode": "state",
                            "num_envs": 1, "seed": 0, "sim_freq": float(base.sim_freq),
                            "max_episode_steps": gym_utils.find_max_episode_steps_value(env), "env_kwargs": kwargs},
                "observation_schema": {"state_dim": spec["obs_dim"], "fields": fields,
                                       "sha256": json_digest(fields), "flatten_check_passed": True},
                "controller_config_sha256": json_digest(configs)}
    finally:
        env.close()


def load_audit(entry, robot, control):
    audit = read_json(entry["audit"]) if entry.get("audit") else audit_environment(entry, robot, control)
    spec = audit.get("task_spec", {})
    if not isinstance(spec, dict):
        raise ValueError("Audit task_spec must be a JSON object")
    contract = spec.get("action_contract", {})
    fields = audit.get("observation_schema", {})
    runtime = audit.get("runtime", {})
    if not all(isinstance(value, dict) for value in (contract, fields, runtime)):
        raise ValueError("Audit contract/schema/runtime must be JSON objects")
    names = contract.get("action_names", [])
    if not isinstance(names, list):
        raise ValueError("Audit action_names must be a list")
    if (audit.get("format_version") != 1 or (spec.get("task_id"), spec.get("robot_uid"), spec.get("control_mode"))
            != (entry["env_id"], robot, control) or spec.get("action_dim") != ACTION_DIMS[control]
            or type(spec.get("obs_dim")) is not int or spec["obs_dim"] < 1
            or not isinstance(spec.get("control_freq"), (int, float))
            or not math.isfinite(spec["control_freq"]) or spec["control_freq"] <= 0
            or len(names) != ACTION_DIMS[control] or any(not isinstance(n, str) or not n for n in names)
            or len(set(names)) != len(names) or contract.get("low") != [-1.0] * ACTION_DIMS[control]
            or contract.get("high") != [1.0] * ACTION_DIMS[control]
            or not isinstance(contract.get("controller_config"), dict) or not contract["controller_config"]
            or audit.get("action_signature") != action_signature(spec)
            or str(runtime.get("maniskill_version", "")).split("+")[0] != "3.0.1"
            or fields.get("state_dim") != spec["obs_dim"] or fields.get("flatten_check_passed") is not True
            or not isinstance(fields.get("fields"), list) or not fields["fields"]
            or fields.get("sha256") != json_digest(fields["fields"])):
        raise ValueError(f"{entry['env_id']}: inconsistent task/controller/observation audit")
    return audit


# 3. Dataset preparation: metadata, success filtering, exact trajectory subset.
def check_metadata(metadata, task, robot):
    info = metadata.get("env_info", {})
    if not isinstance(info, dict):
        raise ValueError(f"{task}: env_info must be a JSON object")
    kwargs = info.get("env_kwargs", {})
    if info.get("env_id") != task or not isinstance(kwargs, dict):
        raise ValueError(f"{task}: invalid dataset env_info")
    recorded = kwargs.get("robot_uids", kwargs.get("robot_uid"))
    if isinstance(recorded, list) and len(recorded) == 1:
        recorded = recorded[0]
    basis = "explicit_metadata" if recorded is not None else "inferred_from_official_v3.0.1_task_default"
    recorded = recorded if recorded is not None else DEFAULT_ROBOTS.get(task)
    if recorded != robot or kwargs.get("obs_mode", "state") != "state":
        raise ValueError(f"{task}: dataset robot/state mode does not match the requested settings")
    allowed = {"robot_uid", "robot_uids", "control_mode", "obs_mode", "reward_mode", "render_mode",
               "render_backend", "shader_dir", "sim_backend", "num_envs", "device", "max_episode_steps",
               "reconfiguration_freq", "human_render_camera_configs", "sensor_configs", "viewer_camera_configs",
               "parallel_in_single_scene", "sim_config", "robot_init_qpos_noise"}
    if (set(kwargs) - allowed or kwargs.get("robot_init_qpos_noise", 0.02) != 0.02
            or kwargs.get("sim_config") not in (None, {})):
        raise ValueError(f"{task}: custom task/physics settings require an explicit environment audit extension")
    episodes = metadata.get("episodes")
    if not isinstance(episodes, list) or any(not isinstance(ep, dict) for ep in episodes):
        raise ValueError(f"{task}: metadata must contain an episodes list")
    result = {}
    for episode in episodes:
        identifier = episode.get("episode_id")
        if type(identifier) is not int or identifier < 0 or f"traj_{identifier}" in result:
            raise ValueError(f"{task}: invalid/duplicate episode_id")
        result[f"traj_{identifier}"] = episode
    return result, kwargs, {"robot_uid": recorded, "basis": basis}


def success_status(group, episode, key):
    if "success" in group:
        signal, actions = group["success"], group.get("actions")
        if not isinstance(signal, h5py.Dataset) or not isinstance(actions, h5py.Dataset) or actions.ndim != 2:
            raise ValueError(f"{key}: success/actions must be numeric datasets")
        values = signal[...]
        length = len(actions)
        if (length < 1 or values.shape not in ((length,), (length, 1)) or values.dtype.kind not in "biuf"
                or not np.isfinite(values).all() or not np.isin(values, [0, 1]).all()):
            raise ValueError(f"{key}: success must contain one Boolean/0/1 per action")
        return bool(values.any()), "hdf5_success_once"
    if "success_once" in episode:
        value = episode["success_once"]
        if not isinstance(value, (bool, int)) or value not in (0, 1):
            raise ValueError(f"{key}: invalid success_once")
        return bool(value), "metadata_success_once"
    return None, "no_machine_success_signal"


def inspect_trajectories(handle, keys, spec):
    lengths = []
    for key in keys:
        group = handle[key]
        matrices = []
        for name in ("obs", "actions"):
            dataset = group.get(name)
            if not isinstance(dataset, h5py.Dataset) or dataset.ndim != 2 or dataset.dtype.kind not in "iuf":
                raise ValueError(f"{key}/{name}: expected numeric state/action matrix")
            value = np.asarray(dataset, dtype=np.float64)
            if not np.isfinite(value).all() or np.any(np.abs(value) > np.finfo(np.float32).max):
                raise ValueError(f"{key}/{name}: invalid/nonfinite float32 data")
            matrices.append(value)
        obs, actions = matrices
        if (len(actions) < 1 or obs.shape != (len(actions) + 1, spec["obs_dim"])
                or actions.shape[1] != spec["action_dim"] or np.any(np.abs(actions) > 1)):
            raise ValueError(f"{key}: inconsistent state/action shapes or normalized bounds")
        lengths.append(len(actions))
    return lengths


def prepare_task(entry, robot, control, seed, evaluation, output_dir):
    task = entry["env_id"]
    metadata = read_json(entry["metadata_path"])
    episodes, kwargs, provenance = check_metadata(metadata, task, robot)
    if kwargs.get("control_mode", control) != control:
        raise ValueError(f"{task}: recorded control_mode differs")
    confirmed = entry.get("confirm_successful_converted", False)
    candidates, quality, seeds, failed, unknown = [], {}, {}, 0, 0
    with h5py.File(entry["demo_path"], "r") as handle:
        keys = sorted((key for key in handle if re.fullmatch(r"traj_[0-9]+", key)), key=lambda key: int(key[5:]))
        for key in keys:
            if not isinstance(handle[key], h5py.Group) or key not in episodes:
                raise ValueError(f"{task}/{key}: missing trajectory group or episode metadata")
            success, basis = success_status(handle[key], episodes[key], key)
            if success is False:
                failed += 1
                continue
            if success is None:
                unknown += 1
                if not confirmed:
                    continue
                basis = "user_confirmed_successful_converted_without_machine_signal"
            episode = episodes[key]
            value = episode.get("episode_seed")
            if value is None:
                reset = episode.get("reset_kwargs", {})
                if not isinstance(reset, dict):
                    raise ValueError(f"{task}/{key}: reset_kwargs must be a JSON object")
                value = reset.get("seed")
            seeds[key] = reset_seed(value, f"{task}/{key}.reset_seed")
            candidates.append(key)
            quality[key] = basis
        if len(candidates) < entry["num_demos"]:
            raise ValueError(f"{task}: need {entry['num_demos']} successes, found {len(candidates)} "
                             f"({failed} failures, {unknown} missing success signals)")
        if len(set(seeds.values())) != len(seeds):
            raise ValueError(f"{task}: duplicate eligible training reset seeds")
        derived = subset_seed(task, seed)
        order = np.random.default_rng(derived).permutation(len(candidates))
        selected = [candidates[int(i)] for i in order[:entry["num_demos"]]]
        train_seeds = [seeds[key] for key in selected]
        heldout = set(evaluation["validation_seeds"]) | set(evaluation["test_seeds"])
        if heldout.intersection(train_seeds):
            raise ValueError(f"{task}: selected training reset seeds overlap reserved evaluation seeds")
        if any(episodes[key].get("control_mode", kwargs.get("control_mode")) != control for key in selected):
            raise ValueError(f"{task}: selected episodes have missing/inconsistent control_mode")
        audit = load_audit(entry, robot, control)
        lengths = inspect_trajectories(handle, selected, audit["task_spec"])
    schema_hash = metadata.get("observation_schema_sha256")
    if schema_hash is not None and schema_hash != audit["observation_schema"]["sha256"]:
        raise ValueError(f"{task}: recorded observation schema differs from the environment")
    cap = entry.get("max_episode_steps") or audit["runtime"]["max_episode_steps"]
    positive_int(cap, f"{task}.max_episode_steps")
    print(f"{task}: selected {len(selected)} successes; hashing source files", flush=True)
    try:
        paths = {key: Path(os.path.relpath(entry[key], output_dir)).as_posix()
                 for key in ("demo_path", "metadata_path")}
    except ValueError as exc:
        raise ValueError("Place output JSON and data on the same drive/filesystem") from exc
    return {**paths, "audit": audit, "trajectory_keys": selected, "train_reset_seeds": train_seeds,
            "subset_rng_seed": derived, "demo_sha256": file_digest(entry["demo_path"]),
            "metadata_sha256": file_digest(entry["metadata_path"]),
            "eval": {"env_id": task, "robot_uids": robot, "control_mode": control,
                     "max_episode_steps": cap, "env_kwargs": {}},
            "quality": {"available_trajectory_count": len(keys), "eligible_trajectory_count": len(candidates),
                        "excluded_explicit_failure_count": failed, "missing_machine_success_signal_count": unknown,
                        "user_confirmed_successful_converted": confirmed,
                        "selected_success_basis": {key: quality[key] for key in selected},
                        "selected_trajectory_lengths": lengths, "selected_decision_count": sum(lengths),
                        "duplicate_eligible_reset_seeds": 0, "robot_provenance": provenance,
                        "observation_schema_status": ("recorded_hash_matches_audit" if schema_hash is not None else
                                                      "numeric_shape_checked; flattened_field_order_not_encoded_in_recording"),
                        "action_contract_status": "runtime_audited; recording robot/controller and normalized bounds checked"}}


# 4. Inputs and CLI. Only dataset/environment parameters belong here.
def load_inputs(args):
    if not args.data_root or not args.tasks:
        raise ValueError("Provide both --data-root and --tasks")
    caps = args.max_episode_steps or [None] * len(args.tasks)
    if len(caps) != len(args.tasks):
        raise ValueError("--max-episode-steps must have one value per task")
    root = args.data_root.expanduser().resolve()
    filename = f"trajectory.state.{args.control_mode}.physx_cpu.h5"
    spec = {"robot_uid": args.robot_uid, "control_mode": args.control_mode, "subset_seed": args.subset_seed,
            "tasks": [{"env_id": name, "demo_path": str(root / name / "motionplanning" / filename),
                       "num_demos": args.num_demos, "max_episode_steps": cap}
                      for name, cap in zip(args.tasks, caps)]}
    base = Path.cwd()
    robot, control, seed = spec.get("robot_uid", "panda"), spec.get("control_mode", "pd_ee_delta_pose"), spec.get("subset_seed", 20260929)
    if not isinstance(robot, str) or not robot.strip() or control not in ACTION_DIMS:
        raise ValueError("Provide an explicit robot and supported delta control mode")
    if type(seed) is not int or seed < 0:
        raise ValueError("subset_seed must be a nonnegative integer")
    evaluation = {**DEFAULT_EVALUATION, **spec.get("evaluation", {})}
    if set(evaluation) != set(DEFAULT_EVALUATION):
        raise ValueError("Unknown evaluation seed fields")
    for key in ("validation_seeds", "test_seeds"):
        values = evaluation[key]
        if not isinstance(values, list) or not values or any(type(v) is not int for v in values):
            raise ValueError(f"{key} must be a nonempty list of integer reset seeds")
        values = [reset_seed(v, key) for v in values]
        if len(values) != len(set(values)):
            raise ValueError(f"{key} contains duplicate seeds")
    reset_seed(evaluation["policy_seed_base"], "policy_seed_base")
    if set(evaluation["validation_seeds"]) & set(evaluation["test_seeds"]):
        raise ValueError("Validation/test reserved seeds overlap")
    entries = spec.get("tasks")
    if not isinstance(entries, list) or not 1 <= len(entries) <= 6:
        raise ValueError("tasks must contain 1-6 distinct task entries")
    tasks, names = [], set()
    for raw in entries:
        if not isinstance(raw, dict) or set(raw) - {"env_id", "demo_path", "metadata_path", "num_demos",
                                                   "max_episode_steps", "audit", "confirm_successful_converted"}:
            raise ValueError("Each task must contain supported dataset/environment fields only")
        entry = dict(raw)
        name = entry.get("env_id")
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError("Task IDs must be nonempty and distinct")
        names.add(name)
        positive_int(entry.get("num_demos"), f"{name}.num_demos")
        if entry.get("max_episode_steps") is not None:
            positive_int(entry["max_episode_steps"], f"{name}.max_episode_steps")
        if type(entry.get("confirm_successful_converted", False)) is not bool:
            raise ValueError("confirm_successful_converted must be Boolean")
        entry["demo_path"] = resolve_input(entry.get("demo_path"), base)
        entry["metadata_path"] = resolve_input(entry["metadata_path"], base) if "metadata_path" in entry else entry["demo_path"].with_suffix(".json")
        if "audit" in entry:
            entry["audit"] = resolve_input(entry["audit"], base)
        tasks.append(entry)
    return tasks, robot, control, seed, evaluation


def build_manifest(args):
    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise ValueError(f"Output already exists: {output}; choose a new name or use --overwrite")
    tasks, robot, control, seed, evaluation = load_inputs(args)
    protected = {entry[key] for entry in tasks for key in ("demo_path", "metadata_path", "audit") if key in entry}
    if output in protected:
        raise ValueError("Output must not overwrite input data, metadata, or audit")
    prepared = {entry["env_id"]: prepare_task(entry, robot, control, seed, evaluation, output.parent) for entry in tasks}
    if len({entry["audit"]["action_signature"] for entry in prepared.values()}) != 1:
        raise ValueError("Joint tasks must share the same audited robot/controller/action contract")
    return {"format_version": 2, "maniskill_version": "3.0.1", "task_order": list(prepared),
            "subset_seed": seed, "tasks": prepared, "evaluation": evaluation}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Root containing <task>/motionplanning/ converted data")
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--robot-uid", default="panda")
    parser.add_argument("--control-mode", choices=tuple(ACTION_DIMS), default="pd_ee_delta_pose")
    parser.add_argument("--num-demos", type=int, default=200, help="Successful trajectories per task")
    parser.add_argument("--subset-seed", type=int, default=20260929)
    parser.add_argument("--max-episode-steps", type=int, nargs="+", help="One evaluation time limit per task")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = build_manifest(args)
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w" if args.overwrite else "x", encoding="utf-8") as stream:
            stream.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        print(f"Saved fixed dataset manifest: {output}", flush=True)
    except (ValueError, TypeError, OSError, ImportError, RuntimeError, KeyError) as exc:
        parser.exit(2, f"Preparation failed: {exc}\n")


if __name__ == "__main__":
    main()
