"""Q-Pensieve algorithm.

Adapted from the Q-Pensieve MOSAC implementation in qpensieve/MOSAC.
"""

import copy
import os
from typing import List, Optional, Tuple, Union

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.distributions import Normal

from morl_baselines.common.buffer import ReplayBuffer
from morl_baselines.common.evaluation import log_all_multi_policy_metrics, log_episode_info, policy_evaluation_mo
from morl_baselines.common.morl_algorithm import MOAgent, MOPolicy
from morl_baselines.common.networks import layer_init, mlp, polyak_update
from morl_baselines.common.weights import equally_spaced_weights, random_weights


LOG_STD_MIN = -20
LOG_STD_MAX = 2
EPSILON = 1e-6


class QMemory:
    """Fixed-size buffer for storing critic snapshots."""

    def __init__(self, capacity: int) -> None:
        self.capacity = max(0, int(capacity))
        self.buffer: List["TwinQNetwork"] = []
        self.position = 0

    def append(self, critic: "TwinQNetwork") -> None:
        """Append a critic snapshot if capacity allows."""
        if self.capacity <= 0:
            return
        if len(self.buffer) < self.capacity:
            self.buffer.append(critic)
        else:
            self.buffer[self.position] = critic
            self.position = (self.position + 1) % self.capacity

    def sample(self) -> List["TwinQNetwork"]:
        """Return a list of stored critics."""
        return list(self.buffer)


class QPensieveActor(nn.Module):
    """Actor network conditioned on preferences."""

    def __init__(self, obs_dim: int, action_dim: int, reward_dim: int, action_space: gym.spaces.Box, net_arch=None):
        super().__init__()
        if net_arch is None:
            net_arch = [256, 256]
        self.net = mlp(obs_dim + reward_dim, -1, net_arch)
        self.mean = nn.Linear(net_arch[-1], action_dim)
        self.log_std = nn.Linear(net_arch[-1], action_dim)

        self.register_buffer(
            "action_scale", th.tensor((action_space.high - action_space.low) / 2.0, dtype=th.float32)
        )
        self.register_buffer(
            "action_bias", th.tensor((action_space.high + action_space.low) / 2.0, dtype=th.float32)
        )
        self.apply(layer_init)

    def forward(self, obs: th.Tensor, preference: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        h = self.net(th.cat((obs, preference), dim=obs.dim() - 1))
        mean = self.mean(h)
        log_std = th.clamp(self.log_std(h), min=LOG_STD_MIN, max=LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs: th.Tensor, preference: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        mean, log_std = self.forward(obs, preference)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = th.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        log_prob = normal.log_prob(x_t).sum(dim=1, keepdim=True)
        log_prob -= th.log(self.action_scale * (1 - y_t.pow(2)) + EPSILON).sum(dim=1, keepdim=True)
        log_prob = log_prob.clamp(-1e3, 1e3)

        mean_action = th.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action

    def get_action(self, obs: th.Tensor, preference: th.Tensor) -> th.Tensor:
        mean, _ = self.forward(obs, preference)
        return th.tanh(mean) * self.action_scale + self.action_bias


class QNetwork(nn.Module):
    """Q-network mapping (s, a, w) to a reward vector."""

    def __init__(self, obs_dim: int, action_dim: int, reward_dim: int, net_arch=None):
        super().__init__()
        if net_arch is None:
            net_arch = [256, 256]
        self.net = mlp(obs_dim + action_dim + reward_dim, reward_dim, net_arch)
        self.apply(layer_init)

    def forward(self, obs: th.Tensor, action: th.Tensor, preference: th.Tensor) -> th.Tensor:
        return self.net(th.cat((obs, action, preference), dim=obs.dim() - 1))


class TwinQNetwork(nn.Module):
    """Twin Q-networks for double Q-learning."""

    def __init__(self, obs_dim: int, action_dim: int, reward_dim: int, net_arch=None):
        super().__init__()
        self.q1 = QNetwork(obs_dim, action_dim, reward_dim, net_arch)
        self.q2 = QNetwork(obs_dim, action_dim, reward_dim, net_arch)

    def forward(self, obs: th.Tensor, action: th.Tensor, preference: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        return self.q1(obs, action, preference), self.q2(obs, action, preference)


class QPensieve(MOAgent, MOPolicy):
    """Q-Pensieve algorithm.

    Q-PENSIEVE: BOOSTING SAMPLE EFFICIENCY OF MULTI-OBJECTIVE RL THROUGH MEMORY SHARING OF Q-SNAPSHOT.
    """

    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_size: int = int(1e6),
        batch_size: int = 256,
        net_arch: Optional[List[int]] = None,
        prefer_set_size: int = 4,
        q_replay_size: int = 4,
        q_replay_start: int = 20000,
        q_replay_freq: int = 1000,
        alpha: float = 0.2,
        autotune: bool = True,
        learning_starts: int = 10000,
        gradient_updates: int = 1,
        policy_freq: int = 1,
        train_preference: Optional[np.ndarray] = None,
        train_preference_concentration: float = 0.0,
        project_name: str = "MORL-Baselines",
        experiment_name: str = "QPensieve",
        wandb_entity: Optional[str] = None,
        log: bool = True,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
    ) -> None:
        MOAgent.__init__(self, env, device=device, seed=seed)
        MOPolicy.__init__(self, device=device)

        if not isinstance(env.action_space, gym.spaces.Box):
            raise ValueError("QPensieve only supports continuous action spaces.")

        if seed is not None:
            th.manual_seed(seed)

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.net_arch = net_arch or [256, 256]
        self.prefer_set_size = max(1, int(prefer_set_size))
        self.q_replay_size = max(0, int(q_replay_size))
        self.q_replay_start = int(q_replay_start)
        self.q_replay_freq = int(q_replay_freq)
        self.learning_starts = int(learning_starts)
        self.gradient_updates = int(gradient_updates)
        self.policy_freq = int(policy_freq)
        self.train_preference = self._normalize_preference(train_preference)
        self.train_preference_concentration = float(train_preference_concentration)

        self.policy = QPensieveActor(self.observation_dim, self.action_dim, self.reward_dim, env.action_space, self.net_arch).to(
            self.device
        )
        self.critic = TwinQNetwork(self.observation_dim, self.action_dim, self.reward_dim, self.net_arch).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        for param in self.critic_target.parameters():
            param.requires_grad = False

        self.policy_optim = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.q1_optim = optim.Adam(self.critic.q1.parameters(), lr=learning_rate)
        self.q2_optim = optim.Adam(self.critic.q2.parameters(), lr=learning_rate)

        self.autotune = autotune
        if autotune:
            self.target_entropy = -float(self.action_dim)
            self.log_alpha = th.zeros(1, requires_grad=True, device=self.device)
            self.alpha_optim = optim.Adam([self.log_alpha], lr=learning_rate)
            self.alpha_tensor = self.log_alpha.exp()
        else:
            self.alpha_tensor = th.tensor(alpha, device=self.device)
            self.log_alpha = None
            self.alpha_optim = None

        self.replay_buffer = ReplayBuffer(
            obs_shape=self.observation_shape,
            action_dim=self.action_dim,
            rew_dim=self.reward_dim,
            max_size=self.buffer_size,
        )
        self.q_memory = QMemory(self.q_replay_size)

        self._updates = 0
        self.last_losses: dict = {}

        self.log = log
        self.experiment_name = experiment_name
        if self.log:
            self.setup_wandb(project_name, experiment_name, wandb_entity)

    def get_config(self) -> dict:
        """Return algorithm configuration."""
        return {
            "env_id": self.env.unwrapped.spec.id,
            "learning_rate": self.policy_optim.param_groups[0]["lr"],
            "gamma": self.gamma,
            "tau": self.tau,
            "buffer_size": self.buffer_size,
            "batch_size": self.batch_size,
            "net_arch": self.net_arch,
            "prefer_set_size": self.prefer_set_size,
            "q_replay_size": self.q_replay_size,
            "q_replay_start": self.q_replay_start,
            "q_replay_freq": self.q_replay_freq,
            "learning_starts": self.learning_starts,
            "gradient_updates": self.gradient_updates,
            "policy_freq": self.policy_freq,
            "train_preference": None if self.train_preference is None else self.train_preference.tolist(),
            "train_preference_concentration": self.train_preference_concentration,
            "autotune": self.autotune,
            "seed": self.seed,
        }

    def _normalize_preference(self, preference: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Normalize a preference vector to the simplex if provided."""
        if preference is None:
            return None
        preference = np.asarray(preference, dtype=np.float32).reshape(-1)
        if preference.shape[0] != self.reward_dim:
            raise ValueError(
                f"train_preference must have shape ({self.reward_dim},), got {preference.shape!r}"
            )
        if np.any(preference < 0):
            raise ValueError("train_preference must be non-negative.")
        total = float(preference.sum())
        if total <= 0:
            raise ValueError("train_preference must sum to a positive value.")
        return preference / total

    def _sample_preference(self) -> np.ndarray:
        if self.train_preference is not None:
            if self.train_preference_concentration <= 0:
                return self.train_preference.copy()
            alpha = np.maximum(self.train_preference * self.train_preference_concentration, 1e-3)
            return self.np_random.dirichlet(alpha).astype(np.float32)
        return random_weights(self.reward_dim, n=1, rng=self.np_random)

    def _sample_preference_tensor(self) -> th.Tensor:
        return th.tensor(self._sample_preference(), dtype=th.float32, device=self.device)

    def _build_preference_set(self, main_preference: th.Tensor) -> List[th.Tensor]:
        preferences = [main_preference]
        for _ in range(self.prefer_set_size - 1):
            preferences.append(self._sample_preference_tensor())
        return preferences

    def _snapshot_critic(self) -> TwinQNetwork:
        snapshot = copy.deepcopy(self.critic)
        snapshot.eval()
        for param in snapshot.parameters():
            param.requires_grad = False
        return snapshot

    def _select_action(self, obs: np.ndarray, preference: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = th.tensor(obs, dtype=th.float32, device=self.device).unsqueeze(0)
        pref_t = th.tensor(preference, dtype=th.float32, device=self.device).unsqueeze(0)
        with th.no_grad():
            if deterministic:
                action = self.policy.get_action(obs_t, pref_t)
            else:
                action, _, _ = self.policy.sample(obs_t, pref_t)
        return action.squeeze(0).cpu().numpy()

    def update(self) -> None:
        """Update the policy and critic networks."""
        if len(self.replay_buffer) < self.batch_size:
            return

        for _ in range(self.gradient_updates):
            self._updates += 1
            (states, actions, rewards, next_states, dones) = self.replay_buffer.sample(
                self.batch_size, to_tensor=True, device=self.device
            )

            preference = self._sample_preference_tensor()
            preference_batch = preference.repeat(self.batch_size, 1)

            with th.no_grad():
                next_actions, next_log_prob, _ = self.policy.sample(next_states, preference_batch)
                next_q1, next_q2 = self.critic_target(next_states, next_actions, preference_batch)
                scalar_q1 = (next_q1 * preference_batch).sum(dim=1, keepdim=True)
                scalar_q2 = (next_q2 * preference_batch).sum(dim=1, keepdim=True)
                min_q = th.where(scalar_q1 < scalar_q2, next_q1, next_q2)
                entropy = -next_log_prob
                next_q = min_q + self.alpha_tensor * entropy
                target_q = rewards + (1 - dones) * self.gamma * next_q

            curr_q1, curr_q2 = self.critic(states, actions, preference_batch)
            q1_loss = ((curr_q1 - target_q).pow(2) * preference).sum(dim=1).mean()
            q2_loss = ((curr_q2 - target_q).pow(2) * preference).sum(dim=1).mean()

            self.q1_optim.zero_grad()
            q1_loss.backward()
            self.q1_optim.step()

            self.q2_optim.zero_grad()
            q2_loss.backward()
            self.q2_optim.step()

            policy_loss = self._calc_policy_loss(states, preference, preference_batch)
            if self._updates % self.policy_freq == 0:
                self.policy_optim.zero_grad()
                policy_loss.backward()
                self.policy_optim.step()

            if self.autotune:
                _, log_prob, _ = self.policy.sample(states, preference_batch)
                alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()
                self.alpha_tensor = self.log_alpha.exp()
            else:
                alpha_loss = None

            polyak_update(self.critic.q1.parameters(), self.critic_target.q1.parameters(), self.tau)
            polyak_update(self.critic.q2.parameters(), self.critic_target.q2.parameters(), self.tau)

            if self.q_replay_size > 0 and self._updates % self.q_replay_freq == 0 and self.global_step >= self.q_replay_start:
                self.q_memory.append(self._snapshot_critic())

            # stash last losses for external loggers (e.g., outer ensemble)
            self.last_losses = {
                "losses/critic_q1": q1_loss.item(),
                "losses/critic_q2": q2_loss.item(),
                "losses/policy": policy_loss.item(),
                "alpha": self.alpha_tensor.item(),
            }
            if alpha_loss is not None:
                self.last_losses["losses/alpha"] = alpha_loss.item()

            if self.log and self.global_step % 100 == 0:
                log_items = dict(self.last_losses)
                log_items["global_step"] = self.global_step
                wandb.log(log_items)

    def _calc_policy_loss(self, states: th.Tensor, preference: th.Tensor, preference_batch: th.Tensor) -> th.Tensor:
        pref_set = self._build_preference_set(preference)
        critics = [self.critic] + self.q_memory.sample()

        losses = []
        alpha = self.alpha_tensor.detach()
        for critic in critics:
            for pref in pref_set:
                pref_batch = pref.repeat(states.shape[0], 1)
                sampled_action, log_prob, _ = self.policy.sample(states, pref_batch)
                q1, q2 = critic(states, sampled_action, preference_batch)
                q1 = (q1 * preference).sum(dim=1, keepdim=True)
                q2 = (q2 * preference).sum(dim=1, keepdim=True)
                q = th.min(q1, q2)
                entropy = -log_prob
                losses.append(-(q + alpha * entropy))

        stacked = th.stack(losses, dim=1)
        return stacked.min(dim=1).values.mean()

    @th.no_grad()
    def eval(self, obs: Union[np.ndarray, th.Tensor], w: Optional[np.ndarray], torch_action: bool = False):
        """Evaluate the policy for a given observation and preference."""
        if w is None:
            w = self._sample_preference()
        if isinstance(obs, np.ndarray):
            obs_t = th.tensor(obs, dtype=th.float32, device=self.device).unsqueeze(0)
            w_t = th.tensor(w, dtype=th.float32, device=self.device).unsqueeze(0)
        else:
            obs_t = obs.to(self.device)
            w_t = th.tensor(w, dtype=th.float32, device=self.device).unsqueeze(0)
        action = self.policy.get_action(obs_t, w_t)
        if not torch_action:
            return action.squeeze(0).cpu().numpy()
        return action

    def train(
        self,
        total_timesteps: int,
        eval_env: gym.Env,
        ref_point: np.ndarray,
        known_pareto_front: Optional[List[np.ndarray]] = None,
        num_eval_weights_for_front: int = 100,
        num_eval_episodes_for_front: int = 5,
        num_eval_weights_for_eval: int = 50,
        eval_freq: int = 10000,
        reset_num_timesteps: bool = False,
        checkpoints: bool = False,
        save_freq: int = 100000,
    ) -> None:
        """Train QPensieve."""
        if self.log:
            self.register_additional_config(
                {
                    "total_timesteps": total_timesteps,
                    "ref_point": ref_point.tolist(),
                    "known_front": known_pareto_front,
                    "num_eval_weights_for_front": num_eval_weights_for_front,
                    "num_eval_episodes_for_front": num_eval_episodes_for_front,
                    "num_eval_weights_for_eval": num_eval_weights_for_eval,
                    "eval_freq": eval_freq,
                    "reset_num_timesteps": reset_num_timesteps,
                }
            )

        eval_weights = equally_spaced_weights(self.reward_dim, n=num_eval_weights_for_front)

        self.global_step = 0 if reset_num_timesteps else self.global_step
        self.num_episodes = 0 if reset_num_timesteps else self.num_episodes

        obs, _ = self.env.reset()
        preference = self._sample_preference()

        for _ in range(1, total_timesteps + 1):
            self.global_step += 1

            if self.global_step < self.learning_starts:
                action = self.env.action_space.sample()
            else:
                action = self._select_action(obs, preference, deterministic=False)

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            self.replay_buffer.add(obs, action, reward, next_obs, float(done))

            if self.global_step >= self.learning_starts:
                self.update()

            if done:
                obs, _ = self.env.reset()
                preference = self._sample_preference()
                self.num_episodes += 1
                if self.log and "episode" in info:
                    log_episode_info(info["episode"], np.dot, preference, self.global_step)
            else:
                obs = next_obs

            if self.log and self.global_step % eval_freq == 0:
                returns_test_tasks = [
                    policy_evaluation_mo(self, eval_env, ew, rep=num_eval_episodes_for_front)[3] for ew in eval_weights
                ]
                log_all_multi_policy_metrics(
                    current_front=returns_test_tasks,
                    hv_ref_point=ref_point,
                    reward_dim=self.reward_dim,
                    global_step=self.global_step,
                    n_sample_weights=num_eval_weights_for_eval,
                    ref_front=known_pareto_front,
                )

            if checkpoints and self.global_step % save_freq == 0:
                self.save(filename=f"QPensieve step={self.global_step}", save_replay_buffer=False)

        self.close_wandb()

    def save(self, save_dir: str = "weights", filename: Optional[str] = None, save_replay_buffer: bool = True) -> None:
        """Save model parameters."""
        os.makedirs(save_dir, exist_ok=True)
        filename = filename or self.experiment_name
        save_path = os.path.join(save_dir, f"{filename}.tar")
        saved = {
            "policy_state_dict": self.policy.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "critic_target_state_dict": self.critic_target.state_dict(),
            "policy_optimizer_state_dict": self.policy_optim.state_dict(),
            "q1_optimizer_state_dict": self.q1_optim.state_dict(),
            "q2_optimizer_state_dict": self.q2_optim.state_dict(),
            "alpha": self.alpha_tensor,
            "log_alpha": self.log_alpha,
        }
        if self.alpha_optim is not None:
            saved["alpha_optimizer_state_dict"] = self.alpha_optim.state_dict()
        if save_replay_buffer:
            saved["replay_buffer"] = self.replay_buffer
        th.save(saved, save_path)

    def load(self, path: str, load_replay_buffer: bool = True) -> None:
        """Load model parameters."""
        params = th.load(path, map_location=self.device, weights_only=False)
        self.policy.load_state_dict(params["policy_state_dict"])
        self.critic.load_state_dict(params["critic_state_dict"])
        self.critic_target.load_state_dict(params["critic_target_state_dict"])
        self.policy_optim.load_state_dict(params["policy_optimizer_state_dict"])
        self.q1_optim.load_state_dict(params["q1_optimizer_state_dict"])
        self.q2_optim.load_state_dict(params["q2_optimizer_state_dict"])
        if "log_alpha" in params and params["log_alpha"] is not None and self.log_alpha is not None:
            self.log_alpha.data.copy_(params["log_alpha"].to(self.device))
            self.alpha_tensor = self.log_alpha.exp()
            if self.alpha_optim is not None and "alpha_optimizer_state_dict" in params:
                self.alpha_optim.load_state_dict(params["alpha_optimizer_state_dict"])
        elif "alpha" in params:
            self.alpha_tensor = params["alpha"].to(self.device)
        if load_replay_buffer and "replay_buffer" in params:
            self.replay_buffer = params["replay_buffer"]
