#!/usr/bin/env python3
"""Single-file LunarLander IIF pipeline (no subprocess dispatch).

This inlines the logic from:
- train_lunarlander_with_given_states.py
- merge_training_buffers.py
- compute_TracIn_surrogate_ghost.py
- produce_minigrid_selected_dataset_shuffle.py
- train_lunarlander_with_given_batches.py
- test_lunarlander_return.py

so each stage is invoked via direct function calls.
"""

from __future__ import annotations

import argparse
import os
import pickle
import random
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np

# Keep consistency with the original scripts.
np.float_ = np.float64

import torch
import torch as th
import torch.nn.functional as F
from torch.optim import Adam, SGD
from tqdm import tqdm

import gymnasium as gym
from gymnasium import error as gym_error
from gymnasium import spaces
from gymnasium.wrappers import RecordEpisodeStatistics

import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from wandb.integration.sb3 import WandbCallback


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_cpu_list(cpus: str) -> set[int]:
    """Parse a taskset-style CPU string (e.g., '0-3,6')."""
    parsed: set[int] = set()
    for part in cpus.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-")
            parsed.update(range(int(start), int(end) + 1))
        else:
            parsed.add(int(part))
    return parsed


def apply_cpu_affinity(cpus: str) -> None:
    """Pin the current process to the requested CPUs (best-effort)."""
    if not cpus:
        return
    cpu_set = parse_cpu_list(cpus)
    if not cpu_set:
        return
    try:
        os.sched_setaffinity(0, cpu_set)
    except AttributeError:
        # Not available on the platform; continue without pinning.
        pass


class Tee:
    """Simple tee that writes to stdout and a log file simultaneously."""

    def __init__(self, file_obj):
        self.file_obj = file_obj
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file_obj.write(data)

    def flush(self):
        self.stdout.flush()
        self.file_obj.flush()

    def isatty(self):
        # Report non-interactive to satisfy callers that check terminal capabilities.
        return False


@contextmanager
def log_to_file(log_path: Path, append: bool = False):
    """Redirect stdout/stderr to a log file while still echoing to the console."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with log_path.open(mode) as f:
        tee = Tee(f)
        with redirect_stdout(tee), redirect_stderr(tee):
            yield


def build_env(env_name: str):
    """Create an environment wrapped with RecordEpisodeStatistics."""
    if env_name == "FrozenLake-v1":
        from gymnasium.envs.toy_text.frozen_lake import generate_random_map

        env = gym.make(env_name, map_name="4x4", is_slippery=False)
    elif env_name == "highway-v0":
        import highway_env

        env = gym.make(
            "highway-v0",
            render_mode=None,
            config={
                "observation": {"type": "Kinematics"},
                "action": {"type": "DiscreteMetaAction"},
                "vehicles_count": 10,
            },
        )
    elif "merge" in env_name:
        import highway_env

        env = gym.make(env_name, render_mode=None)
    elif env_name.startswith("LunarLander"):
        target_env = env_name
        try:
            env = gym.make(target_env)
        except gym_error.Error:
            target_env = "LunarLander-v3"
            env = gym.make(target_env)
    else:
        env = gym.make(env_name)
    return RecordEpisodeStatistics(env)


def make_env(env_name: str):
    """Factory for vectorized environments."""

    def _init():
        return build_env(env_name)

    return _init


# ---------------------------------------------------------------------------
# Train with rollouts (train_lunarlander_with_given_states)
# ---------------------------------------------------------------------------

class RolloutSaveStatesCallback(BaseCallback):
    """Collect rollout data and log episodic stats to wandb."""

    def __init__(self, save_path: Path, verbose: int = 0):
        super().__init__(verbose)
        self.save_path = save_path
        self.all_states = []
        self.all_actions = []
        self.all_rewards = []
        self.all_advantages = []
        self.all_episode_starts = []

    def _on_step(self) -> bool:
        return True

    def on_rollout_end(self) -> None:
        if hasattr(self.model.rollout_buffer, "observations"):
            obs = self.model.rollout_buffer.observations.copy()
            self.all_states.extend(obs)

            actions = self.model.rollout_buffer.actions.copy()
            self.all_actions.extend(actions)

            rewards = self.model.rollout_buffer.rewards.copy()
            self.all_rewards.extend(rewards)

            advantages = self.model.rollout_buffer.advantages.copy()
            self.all_advantages.extend(advantages)

            episode_starts = self.model.rollout_buffer.episode_starts.copy()
            self.all_episode_starts.extend(episode_starts)

        # Log episodic reward/length to wandb (if available).
        if hasattr(self.model, "ep_info_buffer") and len(self.model.ep_info_buffer) > 0:
            rews = [info["r"] for info in self.model.ep_info_buffer if "r" in info]
            lens = [info["l"] for info in self.model.ep_info_buffer if "l" in info]
            if rews:
                step = self.model.num_timesteps
                wandb.log(
                    {
                        "rollout/ep_rew_mean": float(np.mean(rews)),
                        "rollout/ep_len_mean": float(np.mean(lens)) if lens else None,
                        "global_step": step,
                    },
                    step=step,
                )

    def on_training_end(self) -> None:
        save_path = Path(self.save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        np.save(save_path / "all_states.npy", np.array(self.all_states))
        np.save(save_path / "all_actions.npy", np.array(self.all_actions))
        np.save(save_path / "all_rewards.npy", np.array(self.all_rewards))
        np.save(save_path / "all_advantages.npy", np.array(self.all_advantages))
        np.save(save_path / "all_episode_starts.npy", np.array(self.all_episode_starts))


def train_policy_with_given_states(
    *,
    total_timesteps: int,
    n_steps: int,
    learning_rate: float,
    seed: int,
    save_path: Path,
    optimizer_class: str = "SGD",
    initial_policy: str = "",
    rollout_begin_idx: int = 0,
    total_rollouts: int = 20,
    env_name: str = "LunarLander-v3",
    project_name: str = "lunarlander-rl",
    run_name: str = "ppo_lunarlander_run",
    wandb_run_id: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
) -> None:
    """Inline version of train_lunarlander_with_given_states.py."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    wandb.init(
        project=project_name,
        name=wandb_run_name or run_name,
        id=wandb_run_id,
        resume="allow" if wandb_run_id else None,
        reinit=True,
        config={
            "total_timesteps": total_timesteps,
            "n_steps": n_steps,
            "learning_rate": learning_rate,
            "seed": seed,
            "optimizer_class": optimizer_class,
            "initial_policy": initial_policy,
            "rollout_begin_idx": rollout_begin_idx,
            "total_rollouts": total_rollouts,
            "env_name": env_name,
        },
        sync_tensorboard=True,
        save_code=True,
    )
    wandb.define_metric("global_step")
    wandb.define_metric("rollout/*", step_metric="global_step")

    env = DummyVecEnv([make_env(env_name)])
    env = VecMonitor(env)

    policy_kwargs = dict(
        optimizer_class=th.optim.SGD if optimizer_class == "SGD" else th.optim.Adam,
    )

    global_steps = n_steps * rollout_begin_idx
    device = "cpu"

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device=device,
        learning_rate=learning_rate,
        tensorboard_log="./tb_logs/",
        seed=seed,
        policy_kwargs=policy_kwargs,
    )

    if initial_policy:
        import copy

        policy_path = Path(initial_policy)
        policy_path = policy_path.with_suffix("") if policy_path.suffix == ".zip" else policy_path
        if not policy_path.with_suffix(".zip").exists() and not policy_path.exists():
            print(f"Initial policy file not found ({initial_policy}); continuing from fresh policy.")
        else:
            model_init = PPO.load(str(policy_path), device=device)
            model.policy = copy.deepcopy(model_init.policy)
            print("Loaded model with initial policy:", initial_policy)
    else:
        model.save(save_path / "policy_grad_0.zip")

    model._update_counter = 320 * rollout_begin_idx
    model.save(save_path / f"policy_grad_{model._update_counter}.zip")
    model.num_timesteps = global_steps

    # Patch the optimizer's step to save checkpoints after each gradient update.
    original_optimizer_step = model.policy.optimizer.step

    def patched_optimizer_step(*args, **kwargs):
        result = original_optimizer_step(*args, **kwargs)
        checkpoint_filename = save_path / f"policy_grad_{model._update_counter}.zip"
        model.save(checkpoint_filename)
        return result

    model.policy.optimizer.step = patched_optimizer_step

    original_get = model.rollout_buffer.get

    def patched_get(batch_size):
        for mini_batch in original_get(batch_size):
            model._update_counter += 1
            batch_filename = save_path / f"batch_{model._update_counter}.pkl"
            with open(batch_filename, "wb") as f:
                pickle.dump(mini_batch, f)
            yield mini_batch

    model.rollout_buffer.get = patched_get

    callback = CallbackList(
        [
            WandbCallback(verbose=2),
            RolloutSaveStatesCallback(save_path=save_path, verbose=1),
        ]
    )

    print("===============================================================")
    print(f"Begin training at rollout_idx = {rollout_begin_idx} and global_steps = {global_steps}")
    print(f"Target rollouts = {total_rollouts} with total global_steps = {total_timesteps}")

    for rollout_idx in range(rollout_begin_idx, total_rollouts):
        print(f"Training on rollout {rollout_idx + 1}/{total_rollouts} with {model.n_steps} transitions.")

        global_steps += model.n_steps
        print(f"Global progress: {1.0 - (global_steps / total_timesteps):.3f}")

        model.learn(
            total_timesteps=n_steps,
            callback=callback,
            reset_num_timesteps=False,
        )

        model.save(save_path / f"policy_after_rollout_{rollout_idx + 1}.zip")

    final_model_path = save_path / "final_policy.zip"
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")

    wandb.finish()
    env.close()


# ---------------------------------------------------------------------------
# Merge training buffers (merge_training_buffers.py)
# ---------------------------------------------------------------------------

def merge_training_buffers(
    *,
    model_path: Path,
    k: int,
    ed: int,
    env_short: str = "lunarlander",
    n_updates: int = 32,
    n_epochs: int = 10,
) -> None:
    """Inline version of merge_training_buffers.py."""
    model_path = Path(model_path)
    for i in range(k, ed):
        obs_list = []
        act_list = []
        adv_list = []
        prob_list = []
        ret_list = []
        for j in range(i * n_updates * n_epochs, i * n_updates * n_epochs + n_updates):
            data = pickle.load(open(model_path / f"batch_{j + 1}.pkl", "rb"))
            obs_list.append(data.observations)
            act_list.append(data.actions)
            adv_list.append(data.advantages)
            prob_list.append(data.old_log_prob)
            ret_list.append(data.returns)

        obs = torch.cat(obs_list, dim=0).unsqueeze(1)
        act = torch.cat(act_list, dim=0).unsqueeze(1)
        adv = torch.cat(adv_list, dim=0).unsqueeze(1)
        prob = torch.cat(prob_list, dim=0).unsqueeze(1)
        ret = torch.cat(ret_list, dim=0).unsqueeze(1)

        buffer_save = {
            "observations": obs,
            "actions": act,
            "advantages": adv,
            "log_probs": prob,
            "returns": ret,
        }

        print(f"Buffer {i} shape: {obs.shape}, {act.shape}, {adv.shape}, {prob.shape}")

        with open(model_path / f"buffer_{i}.pkl", "wb") as f:
            pickle.dump(buffer_save, f)
        print(f"Saved buffer_{i}")


# ---------------------------------------------------------------------------
# TracIn computation (compute_TracIn_surrogate_ghost.py)
# ---------------------------------------------------------------------------

def get_action_prob(model: PPO, obs: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return log prob and probability of selected actions."""
    dist = model.policy.get_distribution(obs)
    logits = dist.distribution.logits

    log_probs = F.log_softmax(logits, dim=-1)
    selected_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
    selected_probs = torch.exp(selected_log_probs)
    return selected_log_probs, selected_probs


def flatten_ob_act_prob_adv(ob, act, prob, adv):
    return tuple(
        np.concatenate(
            (
                ob.cpu().numpy().flatten(),
                act.cpu().numpy().flatten(),
                prob.cpu().numpy().flatten(),
                adv.cpu().numpy().flatten(),
            )
        )
    )


def compute_sample_ip_layers_vec(train_xs, train_gs, val_xs, val_gs):
    layers = list(train_xs.keys())
    n_train = next(iter(train_xs.values())).shape[0]
    sample_ip = torch.zeros(n_train, dtype=next(iter(train_xs.values())).dtype)
    for layer in layers:
        x_t = train_xs[layer]
        g_t = train_gs[layer]
        x_v = val_xs[layer]
        g_v = val_gs[layer]
        p = g_v.transpose(0, 1) @ x_v
        q = g_t.unsqueeze(2) * x_t.unsqueeze(1)
        sample_ip += (q * p.unsqueeze(0)).sum(dim=(1, 2))

        bias_sum = g_v.sum(dim=0)
        sample_ip += (g_t * bias_sum.unsqueeze(0)).sum(dim=1)
    return sample_ip


def compute_tracin_surrogate(
    *,
    model_path: str,
    rollout_file: Path,
    ref_policy: str,
    env_short: str,
    begin_step: int,
    num_updates: int,
    total_updates: int,
    output_dir: Path,
    sanity_check: bool = False,
) -> None:
    """Inline version of compute_TracIn_surrogate_ghost.py."""
    rollout_file = Path(rollout_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if ref_policy.endswith(".zip"):
        policy_path = Path(ref_policy[:-4])
    else:
        policy_path = Path(f"./models_{env_short}/{model_path}/policy_grad_{ref_policy}.zip").with_suffix("")

    if not policy_path.with_suffix(".zip").exists() and not policy_path.exists():
        raise FileNotFoundError(f"Reference policy file not found: {policy_path}")

    model_ref = PPO.load(str(policy_path), device="cpu")

    rollouts = pickle.load(open(rollout_file, "rb"))
    if isinstance(rollouts, dict):
        sample = rollouts
        sample["observations"] = sample["observations"].squeeze(1)
    else:
        sample = {
            "observations": rollouts.observations.squeeze(1),
            "actions": rollouts.actions,
            "advantages": rollouts.advantages,
        }

    print("rollout sample size =", sample["observations"].shape[0])

    device = model_ref.device
    obs = torch.tensor(sample["observations"], dtype=torch.float32, device=device)
    act = torch.tensor(sample["actions"], dtype=torch.long, device=device)
    advantages = torch.tensor(sample["advantages"], dtype=torch.float32, device=device)

    obs_shape = model_ref.policy.observation_space.shape
    expected_ndim = len(obs_shape) + 1
    if obs_shape == () and obs.shape[1] == 1:
        obs = obs.squeeze(1)
    else:
        if obs.ndim > expected_ndim:
            obs = obs.view(-1, *obs.shape[-len(obs_shape) :])
        elif obs.ndim == len(obs_shape):
            obs = obs.unsqueeze(0)

    if act.ndim > 1:
        act = act.flatten()
    if advantages.ndim > 1:
        advantages = advantages.flatten()

    chunk_size = 64
    num_chunks = 32
    chunks = advantages.view(num_chunks, chunk_size)
    chunk_means = chunks.mean(dim=1, keepdim=True)
    chunk_stds = chunks.std(dim=1, keepdim=True) + 1e-8
    normalized_chunks = (chunks - chunk_means) / chunk_stds
    adjusted_advantages = normalized_chunks.view(-1)[: advantages.shape[0]]

    _xs, _gs = {}, {}
    for name, module in model_ref.policy.named_modules():
        if isinstance(module, torch.nn.Linear) and "value_net" not in name:
            _xs[name] = None
            _gs[name] = None

            def fwd(m, inp, out, nm=name):
                (x,) = inp
                _xs[nm] = x.detach()

            def bwd(m, grad_in, grad_out, nm=name):
                (g,) = grad_out
                _gs[nm] = g.detach()

            module.register_forward_hook(fwd)
            module.register_full_backward_hook(bwd)

    model_ref.policy.optimizer.zero_grad()
    values, new_log_prob, entropy = model_ref.policy.evaluate_actions(obs, act.long().flatten())
    ratio = torch.exp(new_log_prob - new_log_prob.detach())
    policy_loss_1 = ratio * adjusted_advantages
    policy_loss = -policy_loss_1.sum()
    entropy_loss = -entropy.mean()
    loss = policy_loss + model_ref.ent_coef * entropy_loss
    loss.backward()
    train_loss = loss.item()

    if sanity_check:
        print(model_ref.policy)

    _train_xs, _train_gs = _xs, _gs
    _xs, _gs = {}, {}

    model_ref.policy.optimizer.zero_grad()
    selected_log_probs, _ = get_action_prob(model_ref, obs, act)
    val_loss = -torch.mean(advantages * selected_log_probs)
    val_loss.backward()

    _val_xs, _val_gs = _xs, _gs

    out = compute_sample_ip_layers_vec(_train_xs, _train_gs, _val_xs, _val_gs)

    d_global = {}
    n_rows = sample["observations"].shape[0]
    for i in range(n_rows):
        tpl = flatten_ob_act_prob_adv(
            sample["observations"][i],
            sample["actions"][i],
            sample.get("log_probs", sample["advantages"])[i],
            sample["advantages"][i],
        )
        d_global.setdefault(tpl, []).append(out[i].cpu().numpy())

    pickle.dump({"ip": out, "d_global": d_global}, open(output_dir / "scores.pkl", "wb"))


# ---------------------------------------------------------------------------
# Dataset selection (produce_minigrid_selected_dataset_shuffle.py)
# ---------------------------------------------------------------------------

def flatten_ob_act_prob_adv_rew_ret(ob, act, prob, adv, rew, ret):
    return tuple(
        np.concatenate(
            (
                ob.cpu().numpy().flatten(),
                act.cpu().numpy().flatten(),
                prob.cpu().numpy().flatten(),
                adv.cpu().numpy().flatten(),
                rew.cpu().numpy().flatten(),
                ret.cpu().numpy().flatten(),
            )
        )
    )


def select_dataset_shuffle(
    *,
    folder: Path,
    model_path: str,
    begin: int,
    k: int,
    ed: int,
    final_ed: int = -1,
    n: int = 0,
    method: str = "TracIn",
    curriculum: str,
    seed: int,
    valid_samples: str,
    drop_percentile: float,
    mode: str,
    env_short: str = "lunarlander",
    random_sampling: bool = False,
) -> str:
    """Inline version of produce_minigrid_selected_dataset_shuffle.py."""
    folder = Path(folder)
    assert ed == k + 1, "k and ed should be consecutive"
    if final_ed == -1:
        final_ed = ed

    drop_percentile = float(drop_percentile)

    if method == "SGDI":
        file = folder / "sgd_influence_scores.pkl"
        results = pickle.load(open(file, "rb"))
        d_global = results["sgd_influence_scores"]
    elif method in ["TracIn", "TracIn-fixed"]:
        file = folder / "scores.pkl"
        results = pickle.load(open(file, "rb"))
        d_global = results["d_global"]
    elif method == "SGDI-advnorm":
        file = folder / "sgd_influence_advnorm_scores.pkl"
        results = pickle.load(open(file, "rb"))
        d_global = results["sgd_influence_advnorm_scores"]
    elif method == "SGDI-fixed":
        file = folder / "sgd_influence_scores.pkl"
        results = pickle.load(open(file, "rb"))
        d_global = results["sgd_influence_scores"]
    else:
        if method != "random":
            raise NotImplementedError
        d_global = {}

    if mode == "random":
        new_model_path = f"{curriculum}_shuffle-adv_{method}_seed-{seed}_rollout_{begin}-{k}-{final_ed}_drop_{k}-{ed}_random-{drop_percentile}_last-{10-n}_wrt_{valid_samples}_ckpt_batch"
    elif mode == "neg":
        new_model_path = f"{curriculum}_shuffle-adv_{method}_seed-{seed}_rollout_{begin}-{k}-{final_ed}_drop_{k}-{ed}_neg--1.0_last-{10-n}_wrt_{valid_samples}_ckpt_batch"
    else:
        if random_sampling:
            new_model_path = f"{curriculum}_shuffle-adv_{method}_seed-{seed}_rollout_{begin}-{k}-{final_ed}_drop_{k}-{ed}_bottom-{drop_percentile}-random_last-{10-n}_wrt_{valid_samples}_ckpt_batch"
        else:
            new_model_path = f"{curriculum}_shuffle-adv_{method}_seed-{seed}_rollout_{begin}-{k}-{final_ed}_drop_{k}-{ed}_{mode}-{drop_percentile}_last-{10-n}_wrt_{valid_samples}_ckpt_batch"

    model_root = Path(f"./models_{env_short}/{new_model_path}")
    model_root.mkdir(parents=True, exist_ok=True)

    act_ob_to_discard = set()
    rollout_idx = k

    print("mode", mode)

    if mode in ["bottom", "neg-bottom"]:
        scores = []
        for i in range(320 * rollout_idx, 320 * rollout_idx + 32):
            batch_i = pickle.load(open(Path(f"models_{env_short}/{model_path}") / f"batch_{i + 1}.pkl", "rb"))
            for j in range(64):
                obs, act, prob, adv = (
                    batch_i.observations[j],
                    batch_i.actions[j],
                    batch_i.old_log_prob[j],
                    batch_i.advantages[j],
                )
                ob_act = flatten_ob_act_prob_adv(obs, act, prob, adv)
                scores.append(np.mean(d_global[ob_act]))
    elif mode in ["random"]:
        ob_act_list = []
        for i in range(320 * rollout_idx, 320 * rollout_idx + 32):
            batch_i = pickle.load(open(Path(f"models_{env_short}/{model_path}") / f"batch_{i + 1}.pkl", "rb"))
            for j in range(64):
                obs, act, prob, adv = (
                    batch_i.observations[j],
                    batch_i.actions[j],
                    batch_i.old_log_prob[j],
                    batch_i.advantages[j],
                )
                ob_act = flatten_ob_act_prob_adv(obs, act, prob, adv)
                ob_act_list.append(ob_act)
    elif mode in ["neg-random"]:
        scores, ob_act_list = [], []
        for i in range(320 * rollout_idx, 320 * rollout_idx + 32):
            batch_i = pickle.load(open(Path(f"models_{env_short}/{model_path}") / f"batch_{i + 1}.pkl", "rb"))
            for j in range(64):
                obs, act, prob, adv = (
                    batch_i.observations[j],
                    batch_i.actions[j],
                    batch_i.old_log_prob[j],
                    batch_i.advantages[j],
                )
                ob_act = flatten_ob_act_prob_adv(obs, act, prob, adv)
                score = np.mean(d_global[ob_act])
                ob_act_list.append(ob_act)
                scores.append(score)

    if mode in ["neg", "bottom", "neg-bottom"]:
        if drop_percentile != -1:
            if mode == "bottom":
                threshold = np.percentile(scores, drop_percentile)
            elif mode == "neg-bottom":
                threshold = sorted(scores)[int(sum(np.array(scores) < 0) * drop_percentile * 0.01)]
        else:
            assert mode == "neg"
            threshold = 0
        print("threshold", threshold, mode, drop_percentile)

        for i in range(320 * rollout_idx, 320 * rollout_idx + 32):
            batch_i = pickle.load(open(Path(f"models_{env_short}/{model_path}") / f"batch_{i + 1}.pkl", "rb"))
            for j in range(64):
                obs, act, prob, adv = (
                    batch_i.observations[j],
                    batch_i.actions[j],
                    batch_i.old_log_prob[j],
                    batch_i.advantages[j],
                )
                ob_act = flatten_ob_act_prob_adv(obs, act, prob, adv)
                if np.mean(d_global[ob_act]) < threshold:
                    act_ob_to_discard.add(ob_act)

        if drop_percentile > 0 and random_sampling:
            random.seed(seed)
            act_ob_to_discard = set(random.sample(sorted(act_ob_to_discard), 1))

    elif mode == "random":
        random.seed(seed)
        act_ob_to_discard = set(random.sample(ob_act_list, int(len(ob_act_list) * 0.01 * drop_percentile)))
    elif mode == "neg-random":
        random.seed(seed)
        scores_arr = np.array(scores)
        threshold = 0
        neg_set = [ob_act_list[i] for i in range(len(ob_act_list)) if scores_arr[i] < threshold]
        print("random sample from the negative set")
        act_ob_to_discard = set(random.sample(neg_set, int(len(neg_set) * 0.01 * drop_percentile)))

    pickle.dump(act_ob_to_discard, open("act_ob_to_discard_PG.pkl", "wb"))

    if mode == "random":
        obs_list, act_list, prob_list, adv_list, val_list, ret_list = [], [], [], [], [], []
        for i in range(320 * rollout_idx, 320 * rollout_idx + 32):
            batch_i = pickle.load(open(Path(f"models_{env_short}/{model_path}") / f"batch_{i + 1}.pkl", "rb"))
            for j in range(64):
                obs, act, prob, adv, val, ret = (
                    batch_i.observations[j],
                    batch_i.actions[j],
                    batch_i.old_log_prob[j],
                    batch_i.advantages[j],
                    batch_i.old_values[j],
                    batch_i.returns[j],
                )
                ob_act = flatten_ob_act_prob_adv(obs, act, prob, adv)
                if np.random.rand() < 1 - drop_percentile * 0.01:
                    obs_list.append(obs)
                    act_list.append(act)
                    prob_list.append(prob)
                    adv_list.append(adv)
                    val_list.append(val)
                    ret_list.append(ret)
    else:
        obs_list, act_list, prob_list, adv_list, val_list, ret_list = [], [], [], [], [], []
        for i in range(320 * rollout_idx, 320 * rollout_idx + 32):
            batch_i = pickle.load(open(Path(f"models_{env_short}/{model_path}") / f"batch_{i + 1}.pkl", "rb"))
            for j in range(64):
                obs, act, prob, adv, val, ret = (
                    batch_i.observations[j],
                    batch_i.actions[j],
                    batch_i.old_log_prob[j],
                    batch_i.advantages[j],
                    batch_i.old_values[j],
                    batch_i.returns[j],
                )
                ob_act = flatten_ob_act_prob_adv(obs, act, prob, adv)
                if ob_act not in act_ob_to_discard:
                    obs_list.append(obs)
                    act_list.append(act)
                    prob_list.append(prob)
                    adv_list.append(adv)
                    val_list.append(val)
                    ret_list.append(ret)

    print("len remaining records", len(obs_list))

    t = 320 * rollout_idx
    for _ in range(10):
        indices = np.random.permutation(range(len(obs_list)))
        n_batch = int(len(indices) // 64)
        for j in range(n_batch):
            cur_indices = indices[j * 64 : (j + 1) * 64]
            results = {
                "observations": torch.tensor(np.array([obs_list[i] for i in cur_indices])),
                "actions": torch.tensor(np.array([act_list[i] for i in cur_indices])),
                "old_values": torch.tensor(np.array([val_list[i] for i in cur_indices])),
                "old_log_prob": torch.tensor(np.array([prob_list[i] for i in cur_indices])),
                "advantages": torch.tensor(np.array([adv_list[i] for i in cur_indices])),
                "returns": torch.tensor(np.array([ret_list[i] for i in cur_indices])),
            }
            results["advantages"] = (results["advantages"] - results["advantages"].mean()) / (
                results["advantages"].std() + 1e-8
            )

            pickle.dump(results, open(model_root / f"batch_{t + 1}.pkl", "wb"))
            t += 1

    for i in range(t, ed * 320):
        results = None
        pickle.dump(results, open(model_root / f"batch_{i + 1}.pkl", "wb"))

    if mode != "random":
        print("len(act_ob_to_discard)", len(act_ob_to_discard))
    print("model_path", model_root)
    return str(model_root)


# ---------------------------------------------------------------------------
# Offline training (train_lunarlander_with_given_batches.py)
# ---------------------------------------------------------------------------

def train_policy_with_given_batches(
    *,
    seed: int,
    initial_policy: str,
    save_path: Path,
    run_name: str,
    batches_dir: Path,
    batch_begin_idx: int,
    n_batches: int,
    total_updates: int,
    batch_size: int,
    optimizer_class: str,
    learning_rate: float,
    wandb_run_id: Optional[str],
    wandb_run_name: Optional[str],
    env_name: str = "LunarLander-v3",
    project_name: str = "lunarlander-rl",
    gamma: float = 0.99,
    ent_coef: float = 0.0,
    clip_range: float = 0.2,
    normalize_advantage: bool = False,
) -> None:
    """Inline version of train_lunarlander_with_given_batches.py."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    save_path = Path(save_path)
    batches_dir = Path(batches_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    wandb.init(
        project=project_name,
        name=wandb_run_name or run_name,
        id=wandb_run_id,
        resume="allow" if wandb_run_id else None,
        reinit=True,
        config={
            "learning_rate": learning_rate,
            "gamma": gamma,
            "ent_coef": ent_coef,
            "clip_range": clip_range,
            "batch_size": batch_size,
            "batches_dir": str(batches_dir),
            "batch_begin_idx": batch_begin_idx,
            "n_batches": n_batches,
            "optimizer_class": optimizer_class,
            "seed": seed,
            "initial_policy": initial_policy,
            "total_updates": total_updates,
            "env_name": env_name,
            "normalize_advantage": normalize_advantage,
        },
        sync_tensorboard=True,
        save_code=True,
    )
    config = wandb.config

    env = build_env(env_name)
    device = "cpu"

    policy_kwargs = {"optimizer_class": SGD if optimizer_class == "SGD" else Adam}
    clip_range_value = float(clip_range)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        device=device,
        learning_rate=config.learning_rate,
        gamma=config.gamma,
        ent_coef=config.ent_coef,
        clip_range=clip_range_value,
        batch_size=config.batch_size,
        tensorboard_log="./tb_logs/",
        seed=seed,
        normalize_advantage=normalize_advantage,
        policy_kwargs=policy_kwargs,
    )

    if initial_policy:
        import copy

        policy_path = Path(initial_policy)
        policy_path = policy_path.with_suffix("") if policy_path.suffix == ".zip" else policy_path
        if not policy_path.with_suffix(".zip").exists() and not policy_path.exists():
            print(f"Initial policy file not found ({initial_policy}); continuing from a fresh policy instead.")
        else:
            model_init = PPO.load(str(policy_path), device=device, seed=seed)
            model.policy = copy.deepcopy(model_init.policy)
            print(f"Loaded initial policy from {initial_policy}")

    print("normalize_advantage:", model.normalize_advantage)
    model.policy.train()

    model.save(save_path / f"policy_grad_{batch_begin_idx}.zip")
    model._update_counter = batch_begin_idx

    for t in tqdm(range(batch_begin_idx, n_batches)):
        batch_path = batches_dir / f"batch_{t + 1}.pkl"
        with open(batch_path, "rb") as f:
            rollout_data = pickle.load(f)

        if rollout_data is None:
            print(f"encountered None batch at t={t}, exiting")
            rollout_idx = n_batches // 320
            cp_path = save_path / f"policy_after_rollout_{rollout_idx}.zip"
            model.save(cp_path)
            model.save(save_path / f"policy_grad_{n_batches}.zip")
            break

        progress = 1 - ((int(t / 320) + 1) / (total_updates / 320))
        clip_range_value = model.clip_range(progress) if callable(model.clip_range) else model.clip_range

        if isinstance(rollout_data, dict):
            actions = rollout_data["actions"]
            observations = rollout_data["observations"]
            advantages = rollout_data["advantages"]
            returns = rollout_data["returns"]
            old_log_prob = rollout_data["old_log_prob"]
        else:
            actions = rollout_data.actions
            observations = rollout_data.observations
            advantages = rollout_data.advantages
            returns = rollout_data.returns
            old_log_prob = rollout_data.old_log_prob

        actions = actions.to(device)
        observations = observations.to(device)
        advantages = advantages.to(device)
        returns = returns.to(device)
        old_log_prob = old_log_prob.to(device)

        if isinstance(model.action_space, spaces.Discrete):
            actions = actions.long().flatten()

        batch_len = len(observations)
        values, log_prob, entropy = model.policy.evaluate_actions(observations, actions)
        values = values.flatten()

        if model.normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        ratio = th.exp(log_prob - old_log_prob)
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range_value, 1 + clip_range_value)
        policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

        value_loss = F.mse_loss(returns, values)
        entropy_loss = -th.mean(entropy) if entropy is not None else -th.mean(-log_prob)
        loss = policy_loss + model.ent_coef * entropy_loss + model.vf_coef * value_loss

        model.policy.optimizer.zero_grad()
        loss.backward()
        for param in model.policy.parameters():
            if param.grad is not None:
                param.grad.data *= batch_len / config.batch_size
        model.policy.optimizer.step()

        if (t + 1) % 320 == 0:
            rollout_idx = (t + 1) // 320
            cp_path = save_path / f"policy_after_rollout_{rollout_idx}.zip"
            model.save(cp_path)
        model.save(save_path / f"policy_grad_{t + 1}.zip")

    final_model_path = save_path / "final_policy.zip"
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")

    wandb.finish()
    env.close()


# ---------------------------------------------------------------------------
# Evaluation (test_lunarlander_return.py)
# ---------------------------------------------------------------------------

def simulate_trajectory(policy: PPO, env):
    obs, info = env.reset()
    total_reward = 0.0
    done = False
    while not done:
        action, _ = policy.predict(obs, deterministic=False)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += reward
        done = terminated or truncated
    return total_reward


def evaluate_policy_path(policy_path: str, env_name: str, num_episodes: int):
    policy_path = Path(policy_path)
    policy_path_local = policy_path.with_suffix("") if policy_path.suffix == ".zip" else policy_path
    policy = PPO.load(str(policy_path_local), device="cpu")
    print("Loaded policy from:", policy_path)

    env = build_env(env_name)
    returns = []
    for _ in tqdm(range(num_episodes), desc="Evaluating"):
        ret = simulate_trajectory(policy, env)
        returns.append(ret)
    env.close()

    avg_return = float(np.mean(returns))
    std_returns = float(np.std(returns))
    print(f"Average return over {num_episodes} episodes: {avg_return:.2f}")
    print(f"Return standard deviation: {std_returns:.2f}")
    print("Returns for each episode:", returns)
    return avg_return, std_returns, returns


# ---------------------------------------------------------------------------
# Pipeline orchestration (run_iif_lunarlander_adam.py)
# ---------------------------------------------------------------------------

def run_first_rollout(
    *,
    cpus: str,
    seed: int,
    pct: str,
    valid_tag: str,
    base_dir: Path,
    output_root: Path,
    log_path: Path,
    wandb_run_id: str,
    wandb_run_name: str,
) -> None:
    mode = "neg-bottom"
    x = str(pct)
    try:
        drop_tag = f"{float(x)}"
    except ValueError:
        drop_tag = x
    k = 0
    ed = k + 1
    k_idx = k * 320
    ed_idx = ed * 320
    suffix = "" if seed == 182 else f"_seed-{seed}"

    rollout_save = (
        base_dir
        / f"./models_lunarlander/lunarlander_adam_lr-1e-3_step-310k_multi-rol-adaptive-begin-{k}-from-{k}-to-{ed}_{mode}-{drop_tag}_valid-{valid_tag}"
    )
    output_dir = output_root / (
        f"outputs_TracIn_lunarlander_seed-{seed}_adam_lr-1e-3_step-310k_valid-{valid_tag}"
        f"_begin-{k}-updates-{k_idx}-{ed_idx}_{mode}-{drop_tag}"
    )
    batches_dir = (
        base_dir
        / f"./models_lunarlander/lunarlander_adam_lr-1e-3_step-310k_shuffle-adv_TracIn_seed-{seed}"
        f"_rollout_{k}-{k}-{ed}_drop_{k}-{ed}_{mode}-{drop_tag}_last-10_wrt_{valid_tag}_ckpt_batch"
    )
    trained_save = (
        base_dir
        / f"./models_lunarlander/lunarlander_adam_lr-1e-3_step-310k_TracIn_{k}-{k}-{ed}_{mode}-{drop_tag}_valid-{valid_tag}"
    )

    with log_to_file(log_path, append=False):
        print("=== Stage: train rollouts ===")
        train_policy_with_given_states(
            total_timesteps=307200,
            n_steps=2048,
            learning_rate=1e-3,
            seed=seed,
            save_path=rollout_save,
            optimizer_class="Adam",
            initial_policy=base_dir
            / f"./models_lunarlander/lunarlander_adam_lr-1e-3_step-310k{suffix}/policy_grad_{k_idx}.zip",
            rollout_begin_idx=k,
            total_rollouts=ed,
            env_name="LunarLander-v3",
            project_name="lunarlander-rl",
            run_name="ppo_lunarlander_run",
            wandb_run_id=wandb_run_id,
            wandb_run_name=wandb_run_name,
        )

        print("=== Stage: merge buffers ===")
        merge_training_buffers(model_path=rollout_save, k=k, ed=ed, env_short="lunarlander")

        model_name = rollout_save.name
        print("=== Stage: compute TracIn ===")
        compute_tracin_surrogate(
            model_path=model_name,
            rollout_file=rollout_save / f"buffer_{k}.pkl",
            ref_policy=str(k_idx),
            env_short="lunarlander",
            begin_step=k_idx,
            num_updates=ed_idx,
            total_updates=48000,
            output_dir=output_dir,
            sanity_check=False,
        )

        print("=== Stage: select dataset ===")
        selected_batches_dir = select_dataset_shuffle(
            folder=output_dir,
            model_path=model_name,
            begin=k,
            k=k,
            ed=ed,
            final_ed=-1,
            n=0,
            method="TracIn",
            curriculum="lunarlander_adam_lr-1e-3_step-310k",
            seed=seed,
            valid_samples=valid_tag,
            drop_percentile=x,
            mode=mode,
            env_short="lunarlander",
            random_sampling=False,
        )

        print("=== Stage: offline training ===")
        train_policy_with_given_batches(
            seed=seed,
            initial_policy=base_dir
            / f"./models_lunarlander/lunarlander_adam_lr-1e-3_step-310k{suffix}/policy_grad_{k_idx}.zip",
            save_path=trained_save,
            run_name="train_with_given_states",
            batches_dir=selected_batches_dir,
            batch_begin_idx=k_idx,
            n_batches=ed_idx,
            total_updates=48000,
            batch_size=64,
            optimizer_class="Adam",
            learning_rate=1e-3,
            wandb_run_id=wandb_run_id,
            wandb_run_name=wandb_run_name,
            env_name="LunarLander-v3",
            project_name="lunarlander-rl",
            normalize_advantage=False,
        )

        print("=== Stage: evaluation ===")
        evaluate_policy_path(
            policy_path=trained_save / f"policy_grad_{ed_idx}.zip",
            env_name="LunarLander-v3",
            num_episodes=1000,
        )


def run_followup_rollout(
    *,
    begin: int,
    k: int,
    cpus: str,
    seed: int,
    pct: str,
    valid_tag: str,
    base_dir: Path,
    output_root: Path,
    log_path: Path,
    wandb_run_id: str,
    wandb_run_name: str,
) -> None:
    mode = "neg-bottom"
    x = str(pct)
    try:
        drop_tag = f"{float(x)}"
    except ValueError:
        drop_tag = x
    pk = k - 1
    ed = k + 1
    k_idx = k * 320
    ed_idx = ed * 320

    rollout_save = (
        base_dir
        / f"./models_lunarlander/lunarlander_adam_lr-1e-3_step-310k_multi-rol-adaptive-begin-{begin}-from-{k}-to-{ed}_{mode}-{drop_tag}_valid-{valid_tag}"
    )
    output_dir = output_root / (
        f"outputs_TracIn_lunarlander_seed-{seed}_adam_lr-1e-3_step-310k_valid-{valid_tag}"
        f"_begin-{begin}-updates-{k_idx}-{ed_idx}_{mode}-{drop_tag}"
    )
    batches_dir = (
        base_dir
        / f"./models_lunarlander/lunarlander_adam_lr-1e-3_step-310k_shuffle-adv_TracIn_seed-{seed}"
        f"_rollout_{begin}-{k}-{ed}_drop_{k}-{ed}_{mode}-{drop_tag}_last-10_wrt_{valid_tag}_ckpt_batch"
    )
    trained_save = (
        base_dir
        / f"./models_lunarlander/lunarlander_adam_lr-1e-3_step-310k_TracIn_{begin}-{k}-{ed}_{mode}-{drop_tag}_valid-{valid_tag}"
    )
    prev_tracin_model = (
        base_dir
        / f"./models_lunarlander/lunarlander_adam_lr-1e-3_step-310k_TracIn_{begin}-{pk}-{k}_{mode}-{drop_tag}_valid-{valid_tag}"
    )

    with log_to_file(log_path, append=False):
        print("=== Stage: train rollouts ===")
        train_policy_with_given_states(
            total_timesteps=307200,
            n_steps=2048,
            learning_rate=1e-3,
            seed=seed,
            save_path=rollout_save,
            optimizer_class="Adam",
            initial_policy=prev_tracin_model / f"policy_grad_{k_idx}.zip",
            rollout_begin_idx=k,
            total_rollouts=ed,
            env_name="LunarLander-v3",
            project_name="lunarlander-rl",
            run_name="ppo_lunarlander_run",
            wandb_run_id=wandb_run_id,
            wandb_run_name=wandb_run_name,
        )

        print("=== Stage: merge buffers ===")
        merge_training_buffers(model_path=rollout_save, k=k, ed=ed, env_short="lunarlander")

        model_name = rollout_save.name
        print("=== Stage: compute TracIn ===")
        compute_tracin_surrogate(
            model_path=model_name,
            rollout_file=rollout_save / f"buffer_{k}.pkl",
            ref_policy=str(k_idx),
            env_short="lunarlander",
            begin_step=k_idx,
            num_updates=ed_idx,
            total_updates=48000,
            output_dir=output_dir,
            sanity_check=False,
        )

        print("=== Stage: select dataset ===")
        selected_batches_dir = select_dataset_shuffle(
            folder=output_dir,
            model_path=model_name,
            begin=begin,
            k=k,
            ed=ed,
            final_ed=-1,
            n=0,
            method="TracIn",
            curriculum="lunarlander_adam_lr-1e-3_step-310k",
            seed=seed,
            valid_samples=valid_tag,
            drop_percentile=x,
            mode=mode,
            env_short="lunarlander",
            random_sampling=False,
        )

        print("=== Stage: offline training ===")
        train_policy_with_given_batches(
            seed=seed,
            initial_policy=prev_tracin_model / f"policy_grad_{k_idx}.zip",
            save_path=trained_save,
            run_name="train_with_given_states",
            batches_dir=selected_batches_dir,
            batch_begin_idx=k_idx,
            n_batches=ed_idx,
            total_updates=48000,
            batch_size=64,
            optimizer_class="Adam",
            learning_rate=1e-3,
            wandb_run_id=wandb_run_id,
            wandb_run_name=wandb_run_name,
            env_name="LunarLander-v3",
            project_name="lunarlander-rl",
            normalize_advantage=False,
        )

        print("=== Stage: evaluation ===")
        evaluate_policy_path(
            policy_path=trained_save / f"policy_grad_{ed_idx}.zip",
            env_name="LunarLander-v3",
            num_episodes=1000,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Run the LunarLander rollout/train/selection loop (single-file, no subprocess)."
    )
    parser.add_argument("seed", type=int, default=42, help="Random seed.")
    parser.add_argument("pct", type=float, default=0.2, help="Drop percentile argument (x).")
    parser.add_argument(
        "cpus",
        default="0-7",
        help="CPU list passed to sched_setaffinity, e.g., '0-7' or '0,2,4,6'.",
    )
    parser.add_argument(
        "--rollouts",
        type=int,
        default=100,
        help="Total rollout iterations (default 100 for k=0..99).",
    )
    parser.add_argument(
        "--begin",
        type=int,
        default=0,
        help="Begin rollout index (matches bash driver default 0).",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs",
        help="Root folder to store all generated outputs/logs (default: outputs/).",
    )
    parser.add_argument(
        "--wandb-run-id",
        type=str,
        default=None,
        help="Optional wandb run id to aggregate all rollouts into one run.",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Optional wandb run name override (defaults to generated name).",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    os.chdir(base_dir)

    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = base_dir / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    results_dir = output_root / "results_lunarlander"

    valid_tag = f"PG-SI-CP-ghost-2k-lambda-1-rollout-train-seed-{args.seed}"
    wandb_run_id = args.wandb_run_id or wandb.util.generate_id()
    wandb_run_name = args.wandb_run_name or f"lunarlander-seed-{args.seed}-pct-{args.pct}"

    apply_cpu_affinity(args.cpus)

    first_log = (
        results_dir
        / f"result_lunarlander_adam_rollout_train_surrogate_PG-SI-CP-ghost_adaptive_rol-0-0-1_neg-bottom-{args.pct}_gae-lambda-1_2k_lambda_seed-{args.seed}.txt"
    )
    run_first_rollout(
        cpus=args.cpus,
        seed=args.seed,
        pct=str(args.pct),
        valid_tag=valid_tag,
        base_dir=base_dir,
        output_root=output_root,
        log_path=first_log,
        wandb_run_id=wandb_run_id,
        wandb_run_name=wandb_run_name,
    )

    for i in range(1, args.rollouts):
        j = i + 1
        log_path = (
            results_dir
            / f"result_lunarlander_adam_rollout_train_surrogate_PG-SI-CP-ghost_adaptive_rol-0-{i}-{j}_neg-bottom-{args.pct}_gae-lambda-1_2k_lambda_seed-{args.seed}.txt"
        )
        run_followup_rollout(
            begin=args.begin,
            k=i,
            cpus=args.cpus,
            seed=args.seed,
            pct=str(args.pct),
            valid_tag=valid_tag,
            base_dir=base_dir,
            output_root=output_root,
            log_path=log_path,
            wandb_run_id=wandb_run_id,
            wandb_run_name=wandb_run_name,
        )


if __name__ == "__main__":
    main()
