from collections import defaultdict
from collections.abc import Mapping

import numpy as np
import torch
from tqdm import tqdm
from mani_skill.utils import common


def _numpy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _check_mask(mapping, key, width):
    mask_key = "_" + key
    if mask_key in mapping:
        mask = _numpy(mapping[mask_key]).reshape(-1)
        if mask.dtype.kind != "b" or len(mask) != width or not mask.all():
            raise ValueError(
                f"Info mask {mask_key!r} marks incomplete values for "
                f"{width} completed environments."
            )


def _completed_metrics(info, num_envs):
    if "final_info" in info:
        _check_mask(info, "final_info", num_envs)
        final_info = info["final_info"]
        if isinstance(final_info, Mapping):
            _check_mask(final_info, "episode", num_envs)
            rows = [final_info.get("episode")]
        else:
            rows = []
            for item in final_info:
                if isinstance(item, Mapping):
                    _check_mask(item, "episode", 1)
                    rows.append(item.get("episode"))
    elif "episode" in info:
        _check_mask(info, "episode", num_envs)
        rows = [info["episode"]]
    else:
        raise KeyError(
            "No completed episode metrics: expected info['final_info'] "
            f"or info['episode']; received keys {list(info)}."
        )

    fields = defaultdict(list)
    for row in rows:
        if not isinstance(row, Mapping):
            raise KeyError("Completed episode info is missing an 'episode' dictionary.")
        for key, value in row.items():
            # Gymnasium adds _<key> masks alongside batched info fields.
            if key.startswith("_"):
                continue
            values = _numpy(value).reshape(-1)
            _check_mask(row, key, len(values))
            if values.dtype.kind not in "biuf" or not np.isfinite(values).all():
                raise ValueError(f"Episode metric {key!r} must contain finite numeric values.")
            fields[key].append(values)

    if not fields:
        raise KeyError("Completed episode info contains no numeric metrics.")
    metrics = {key: np.concatenate(values) for key, values in fields.items()}
    for key, values in metrics.items():
        if len(values) != num_envs:
            raise ValueError(
                f"Episode metric {key!r} has {len(values)} values for "
                f"{num_envs} completed environments; episode info is incomplete."
            )
    for key in ("success_once", "success_at_end"):
        if key not in metrics:
            raise KeyError(f"Missing required completed episode metric {key!r}.")
        if not np.isin(metrics[key], [0, 1]).all():
            raise ValueError(f"Episode metric {key!r} must contain Boolean/0/1 values.")
    return metrics


def evaluate(n: int, agent, eval_envs, device, sim_backend: str,
             progress_bar: bool = True):
    """Return metric arrays with one value per evaluated episode.

    As in the official evaluator, all environments must truncate together.
    If n is not divisible by num_envs, the last complete batch is included.
    """
    if n < 1 or eval_envs.num_envs < 1:
        raise ValueError("Evaluation needs a positive episode and environment count.")
    agent.eval()
    pbar = tqdm(total=n) if progress_bar else None
    eval_metrics = defaultdict(list)
    try:
        with torch.no_grad():
            obs, info = eval_envs.reset()
            eps_count = 0
            while eps_count < n:
                obs = common.to_tensor(obs, device)
                action_seq = agent.get_action(obs)
                if action_seq.shape[1] < 1:
                    raise ValueError("Policy returned an empty action chunk.")
                if sim_backend == "physx_cpu":
                    action_seq = action_seq.cpu().numpy()
                for i in range(action_seq.shape[1]):
                    obs, rew, terminated, truncated, info = eval_envs.step(action_seq[:, i])
                    done = _numpy(truncated).astype(bool).reshape(-1)
                    if done.any():
                        break

                if done.any():
                    if len(done) != eval_envs.num_envs or not done.all():
                        raise RuntimeError(
                            "All evaluation environments must truncate at the same time, "
                            "as required by the official DP rollout protocol."
                        )
                    metrics = _completed_metrics(info, eval_envs.num_envs)
                    for key, values in metrics.items():
                        eval_metrics[key].append(values)
                    eps_count += eval_envs.num_envs
                    if pbar is not None:
                        pbar.update(eval_envs.num_envs)

                    # final_info means the vector wrapper already reset in the
                    # terminal step. With CPU next-step/disabled autoreset,
                    # reset now so the next chunk uses initial observations,
                    # rather than terminal observations or a discarded action.
                    if (eps_count < n and sim_backend == "physx_cpu"
                            and "final_info" not in info):
                        obs, info = eval_envs.reset()
    finally:
        agent.train()
        if pbar is not None:
            pbar.close()

    return {key: np.concatenate(values) for key, values in eval_metrics.items()}
