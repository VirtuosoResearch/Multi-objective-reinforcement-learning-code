import json
import os
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import torch as th

from morl_baselines.common.weights import random_weights
from morl_baselines.multi_policy.ensemble.outer_ensemble import OuterEnsemble


class MuJoCoRewardEvaluator:

    def __init__(self, env_fn: Callable):
        import mujoco

        self.mujoco = mujoco
        self.env = env_fn()
        self.env.reset()
        self.base = self.env.unwrapped
        if not hasattr(self.base, "model") or not hasattr(self.base, "data"):
            self.env.close()
            raise TypeError("Provide a reward evaluator with capture(env) and evaluate(states, actions) for non-MuJoCo envs.")
        self.spec = mujoco.mjtState.mjSTATE_INTEGRATION

    def capture(self, env):
        base = env.unwrapped
        state = np.empty(self.mujoco.mj_stateSize(base.model, self.spec), dtype=np.float64)
        self.mujoco.mj_getState(base.model, base.data, state, self.spec)
        return state, base.data.xpos.copy()

    def evaluate(self, states, actions) -> np.ndarray:
        rewards = []
        for (state, body_positions), action in zip(states, actions):
            self.mujoco.mj_setState(self.base.model, self.base.data, state, self.spec)
            self.mujoco.mj_forward(self.base.model, self.base.data)
            self.mujoco.mj_setState(self.base.model, self.base.data, state, self.spec)
            self.base.data.xpos[:] = body_positions
            rewards.append(np.asarray(self.base.step(action)[1], dtype=np.float64))
        return np.stack(rewards)

    def close(self):
        self.env.close()


@dataclass
class RoutedTrajectory:

    obs: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_obs: np.ndarray
    dones: np.ndarray
    states: list
    preference: np.ndarray
    origin: int

    def __len__(self):
        return len(self.obs)


class RoutedEnsemble(OuterEnsemble):

    def __init__(
        self,
        env_fn: Callable,
        agent_ctor: Callable,
        ensemble_size: int = 3,
        agent_kwargs=None,
        buffer_size: int = 100_000,
        trajectories_per_iteration: int = 8,
        horizon: int = 1000,
        score_action_samples: int = 4,
        updates_per_transition: float = 1.0,
        assignment: str = "score",
        retain_assigned_replay: bool = False,
        update_preference: str = "iteration",
        reward_evaluator=None,
        action_sampler: Optional[Callable] = None,
        seed: int = 0,
        device: str = "cpu",
    ):
        if min(ensemble_size, trajectories_per_iteration, horizon, score_action_samples) < 1:
            raise ValueError("Ensemble size, trajectory count, horizon and action sample count must be positive.")
        if buffer_size < trajectories_per_iteration * horizon:
            raise ValueError("buffer_size must fit a complete iteration, including when all trajectories have one winner.")
        if not np.isfinite(updates_per_transition) or updates_per_transition < 0:
            raise ValueError("updates_per_transition must be finite and non-negative.")
        if assignment not in ("score", "uniform"):
            raise ValueError("assignment must be 'score' or 'uniform'.")
        if update_preference not in ("iteration", "random"):
            raise ValueError("update_preference must be 'iteration' or 'random'.")
        super().__init__(
            env_fn=env_fn,
            agent_ctor=agent_ctor,
            ensemble_size=ensemble_size,
            agent_kwargs=agent_kwargs,
            buffer_size=buffer_size,
            log=False,
            seed=seed,
            device=device,
        )
        for agent in self.agents:
            if not hasattr(agent, "train_preference") or not hasattr(agent, "train_preference_concentration"):
                raise TypeError("The learning subroutine must support train_preference and train_preference_concentration.")
        self.trajectories_per_iteration = int(trajectories_per_iteration)
        self.horizon = int(horizon)
        self.score_action_samples = int(score_action_samples)
        self.updates_per_transition = float(updates_per_transition)
        self.assignment = assignment
        self.retain_assigned_replay = retain_assigned_replay
        self.update_preference = update_preference
        self._training_member_targets = None
        self.reward_evaluator = reward_evaluator if reward_evaluator is not None else MuJoCoRewardEvaluator(env_fn)
        self.action_sampler = action_sampler
        self.selection_env = env_fn()
        self.iteration = 0
        self.score_env_steps = 0
        self.evaluation_env_steps = 0
        self.evaluation_discount = 1.0
        self.baseline_policy = None
        self._evaluation_member = None
        self._evaluation_preference = None

    def get_config(self):
        return {
            "algorithm": "RoutedEnsemble",
            "ensemble_size": self.ensemble_size,
            "trajectories_per_iteration": self.trajectories_per_iteration,
            "horizon": self.horizon,
            "score_action_samples": self.score_action_samples,
            "updates_per_transition": self.updates_per_transition,
            "assignment": self.assignment,
            "retain_assigned_replay": self.retain_assigned_replay,
            "update_preference": self.update_preference,
            "seed": self.seed,
            "score": "sum_h mean_a dot(w_t, r(s_h, a)), a ~ pi_j(.|s_h,w_t)",
            "replay": "assigned trajectory history" if self.retain_assigned_replay else "current iteration partition only",
        }

    def _sample_actions(self, agent, obs, preference):
        if self.action_sampler is not None:
            return np.asarray(self.action_sampler(agent, obs, preference))
        if hasattr(agent, "policy") and hasattr(agent.policy, "sample"):
            with th.no_grad():
                obs_t = th.as_tensor(obs, dtype=th.float32, device=agent.device)
                pref_t = th.as_tensor(preference, dtype=th.float32, device=agent.device).expand(len(obs), -1)
                return agent.policy.sample(obs_t, pref_t)[0].cpu().numpy()
        raise TypeError("Provide action_sampler(agent, obs_batch, preference) for this policy type.")

    def _collect_trajectory(self, preference, max_steps):
        available = np.arange(self.ensemble_size)
        if self._training_member_targets is not None:
            available = np.asarray(
                [j for j, agent in enumerate(self.agents) if agent.global_step < self._training_member_targets[j]]
            )
        origin = int(self.np_random.choice(available))
        agent, env = self.agents[origin], self.envs[origin]
        if self._training_member_targets is not None:
            max_steps = min(max_steps, self._training_member_targets[origin] - agent.global_step)
        obs, _ = env.reset(seed=int(self.np_random.integers(2**31 - 1)))
        env.action_space.seed(int(self.np_random.integers(2**31 - 1)))
        rows, states = [], []
        for _ in range(max_steps):
            states.append(self.reward_evaluator.capture(env))
            if agent.global_step < self.effective_learning_starts[origin]:
                action = env.action_space.sample()
            else:
                action = self._sample_actions(agent, np.asarray(obs)[None], preference)[0]
            next_obs, reward, terminated, truncated, _ = env.step(action)
            rows.append(
                (
                    np.array(obs, copy=True),
                    np.array(action, copy=True),
                    np.array(reward, copy=True),
                    np.array(next_obs, copy=True),
                    float(terminated),
                )
            )
            agent.global_step += 1
            self.global_step += 1  
            obs = next_obs
            if terminated or truncated:
                break
        agent.num_episodes += 1
        arrays = [np.asarray(column, dtype=np.float32) for column in zip(*rows)]
        return RoutedTrajectory(*arrays, states, preference.copy(), origin)

    def _score_trajectories(self, trajectories, preference):
        scores = np.zeros((self.ensemble_size, len(trajectories)), dtype=np.float64)
        for j, agent in enumerate(self.agents):
            for i, trajectory in enumerate(trajectories):
                if not np.array_equal(trajectory.preference, preference):
                    raise ValueError("All trajectories in an iteration must have the same preference.")
                for _ in range(self.score_action_samples):
                    actions = self._sample_actions(agent, trajectory.obs, preference)
                    rewards = self.reward_evaluator.evaluate(trajectory.states, actions)
                    if rewards.shape != trajectory.rewards.shape or not np.all(np.isfinite(rewards)):
                        raise ValueError("Reward evaluator must return a finite (trajectory_length, reward_dim) array.")
                    scores[j, i] += np.sum(rewards @ preference) / self.score_action_samples
                    self.score_env_steps += len(trajectory)
        return scores

    def _assign_trajectories(self, scores):
        if not np.all(np.isfinite(scores)):
            raise ValueError("Trajectory scores must be finite.")
        if self.assignment == "uniform":
            return self.np_random.integers(self.ensemble_size, size=scores.shape[1])
        return np.asarray([self.np_random.choice(np.flatnonzero(column == column.max())) for column in scores.T])

    def _update_partitions(self, trajectories, assignments, preference):
        counts, updates = [], []
        for j, agent in enumerate(self.agents):
            buffer = self.member_buffers[j]
            if not self.retain_assigned_replay:
                buffer.ptr = buffer.size = 0
            assigned_count = 0
            for i, trajectory in enumerate(trajectories):
                if assignments[i] == j:
                    for row in zip(
                        trajectory.obs, trajectory.actions, trajectory.rewards, trajectory.next_obs, trajectory.dones
                    ):
                        buffer.add(*row)
                        assigned_count += 1
            counts.append(assigned_count)
            old_pref, old_concentration = agent.train_preference, agent.train_preference_concentration
            agent.train_preference = preference.copy() if self.update_preference == "iteration" else None
            agent.train_preference_concentration = 0.0
            ready = len(buffer) >= agent.batch_size and agent.global_step >= self.effective_learning_starts[j]
            n_updates = int(np.ceil(assigned_count * self.updates_per_transition)) if ready else 0
            self._set_agent_buffer(agent, buffer)
            try:
                for _ in range(n_updates):
                    agent.update()
            finally:
                agent.train_preference, agent.train_preference_concentration = old_pref, old_concentration
            updates.append(n_updates)
        return counts, updates

    def train(
        self, total_timesteps, save_dir="routed_checkpoints", eval_freq=0, evaluation_callback=None, budget_mode="total"
    ):
        if total_timesteps < 0:
            raise ValueError("total_timesteps must be non-negative.")
        if budget_mode not in ("total", "per_member"):
            raise ValueError("budget_mode must be 'total' or 'per_member'.")
        multiplier = self.ensemble_size if budget_mode == "per_member" else 1
        self._training_member_targets = (
            [agent.global_step + total_timesteps for agent in self.agents] if multiplier > 1 else None
        )
        os.makedirs(save_dir, exist_ok=True)
        next_eval = self.global_step + eval_freq if eval_freq > 0 else float("inf")
        target = self.global_step + total_timesteps * multiplier
        while self.global_step < target:
            preference = random_weights(self.reward_dim, rng=self.np_random).astype(np.float32)
            trajectories = []
            for _ in range(self.trajectories_per_iteration):
                if self.global_step == target:
                    break
                trajectories.append(self._collect_trajectory(preference, min(self.horizon, target - self.global_step)))
            scores = self._score_trajectories(trajectories, preference)
            assignments = self._assign_trajectories(scores)
            counts, updates = self._update_partitions(trajectories, assignments, preference)
            self.iteration += 1
            record = {
                "iteration": self.iteration,
                "global_step": self.global_step,
                "preference": preference.tolist(),
                "origins": [t.origin for t in trajectories],
                "trajectory_lengths": [len(t) for t in trajectories],
                "scores": scores.tolist(),
                "assignments": assignments.tolist(),
                "partition_transitions": counts,
                "replay_sizes": [len(buffer) for buffer in self.member_buffers],
                "member_env_steps": [agent.global_step for agent in self.agents],
                "update_calls": updates,
                "score_env_steps": self.score_env_steps,
                "losses": [getattr(agent, "last_losses", {}) for agent in self.agents],
            }
            with open(os.path.join(save_dir, "routing.jsonl"), "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
            self._save_members(save_dir)
            if evaluation_callback is not None and self.global_step >= next_eval:
                evaluation_callback(self)
                next_eval = (self.global_step // eval_freq + 1) * eval_freq
        self._save_members(save_dir)

    @property
    def candidate_policies(self):
        return self.agents + ([self.baseline_policy] if self.baseline_policy is not None else [])

    def _rollout_return(self, member, preference, seed):
        obs, _ = self.selection_env.reset(seed=seed)
        total = np.zeros(self.reward_dim, dtype=np.float64)
        discount = 1.0
        for _ in range(self.horizon):
            action = self.candidate_policies[member].eval(obs, preference)
            obs, reward, terminated, truncated, _ = self.selection_env.step(action)
            self.evaluation_env_steps += 1
            total += discount * reward
            discount *= self.evaluation_discount
            if terminated or truncated:
                break
        return total

    def select_policy(self, preference, episodes=2, seed=10_000):
        if episodes < 1:
            raise ValueError("episodes must be positive.")
        preference = np.asarray(preference, dtype=np.float32)
        returns = np.asarray(
            [
                np.mean([self._rollout_return(j, preference, seed + k) for k in range(episodes)], axis=0)
                for j in range(len(self.candidate_policies))
            ]
        )
        self._evaluation_member = int(np.argmax(returns @ preference))
        if self.baseline_policy is not None and (returns @ preference)[-1] >= (returns @ preference).max():
            self._evaluation_member = self.ensemble_size
        self._evaluation_preference = preference.copy()
        return self._evaluation_member, returns

    def eval(self, obs, w=None):
        if (
            self._evaluation_member is None
            or w is None
            or not np.array_equal(np.asarray(w, dtype=np.float32), self._evaluation_preference)
        ):
            raise RuntimeError("Call select_policy(w) before evaluating; the selected member stays fixed for the episode.")
        return self.candidate_policies[self._evaluation_member].eval(obs, w)

    def evaluate(self, preferences, selection_episodes=2, test_episodes=3, seed=10_000):
        if test_episodes < 1:
            raise ValueError("test_episodes must be positive.")
        records = []
        for preference in preferences:
            member, candidate_returns = self.select_policy(preference, selection_episodes, seed)
            ensemble_member = int(np.argmax(candidate_returns[: self.ensemble_size] @ preference))
            test_returns = {}
            for j in range(len(self.candidate_policies)):
                test_returns[j] = np.mean(
                    [self._rollout_return(j, preference, seed + selection_episodes + k) for k in range(test_episodes)], axis=0
                )
            record = {
                "preference": np.asarray(preference).tolist(),
                "member": member,
                "selection_returns": candidate_returns.tolist(),
                "test_return": test_returns[member].tolist(),
                "ensemble_member": ensemble_member,
                "ensemble_test_return": test_returns[ensemble_member].tolist(),
                "selected_source": "single_policy" if member == self.ensemble_size else "ensemble",
            }
            oracle_member = max(test_returns, key=lambda j: float(test_returns[j] @ preference))
            record["oracle_member"] = oracle_member
            record["oracle_test_return"] = test_returns[oracle_member].tolist()
            record["candidate_test_returns"] = [test_returns[j].tolist() for j in range(len(self.candidate_policies))]
            if self.baseline_policy is not None:
                record["single_policy_test_return"] = test_returns[self.ensemble_size].tolist()
                record["validation_gain_over_single_policy"] = float(
                    (candidate_returns[member] - candidate_returns[-1]) @ preference
                )
            records.append(record)
        return records

    def close(self):
        for env in self.envs:
            env.close()
        self.selection_env.close()
        self.reward_evaluator.close()
