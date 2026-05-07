"""Generic outer-loop ensemble trainer usable with any MORL agent that exposes a critic and replay buffer."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import csv
import os
import time
from typing import Callable, Iterable, List, Literal, Optional, Sequence, Tuple, Union

import gymnasium as gym
import numpy as np
import torch as th

from morl_baselines.common.buffer import ReplayBuffer
from morl_baselines.common.evaluation import log_all_multi_policy_metrics, log_episode_info, policy_evaluation_mo
from morl_baselines.common.morl_algorithm import MOAgent, MOPolicy
from morl_baselines.common.weights import equally_spaced_weights, random_weights


ScalarizationFn = Callable[[np.ndarray, np.ndarray], Union[float, np.ndarray]]
CriticScorer = Callable[
    [MOPolicy, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    np.ndarray,
]


@dataclass
class EnsembleMemberConfig:
    """Lightweight carrier for member-specific settings."""

    preference: Optional[np.ndarray] = None
    scalarization: Optional[ScalarizationFn] = None
    scorer: Optional[CriticScorer] = None


def _to_numpy(tensor_or_array: Union[th.Tensor, np.ndarray]) -> np.ndarray:
    """Convert torch tensors (possibly nested) to numpy arrays."""
    if isinstance(tensor_or_array, th.Tensor):
        return tensor_or_array.detach().cpu().numpy()
    if isinstance(tensor_or_array, (list, tuple)):
        # handle list/tuple of tensors or arrays
        return np.stack([_to_numpy(x) for x in tensor_or_array], axis=0)
    return np.asarray(tensor_or_array)


def _default_scalarization(values: np.ndarray, preference: np.ndarray) -> np.ndarray:
    """Weighted-sum scalarization used if none is provided."""
    return np.einsum("bd,d->b", values, preference)


def _build_default_scorer(preference: np.ndarray, scalarization: ScalarizationFn) -> CriticScorer:
    """Return a scorer that tries common critic names and falls back to rewards."""

    def scorer(
        agent: MOPolicy,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        dones: np.ndarray,
        pref: np.ndarray,
    ) -> np.ndarray:
        """Score a batch using the agent's critic. Uses reward if no critic found."""
        device = getattr(agent, "device", th.device("cpu"))
        obs_t = th.as_tensor(obs, dtype=th.float32, device=device)
        act_t = th.as_tensor(actions, dtype=th.float32, device=device)
        pref_t = th.as_tensor(pref, dtype=th.float32, device=device)

        def _maybe_min(two_outputs):
            """Handle twin critics by taking the element-wise min."""
            first, second = two_outputs
            return th.minimum(first, second)

        with th.no_grad():
            if hasattr(agent, "critic_target"):
                out = agent.critic_target(obs_t, act_t, pref_t.expand(obs_t.shape[0], -1))
            elif hasattr(agent, "critic"):
                out = agent.critic(obs_t, act_t, pref_t.expand(obs_t.shape[0], -1))
            elif hasattr(agent, "qf1"):
                # MOSAC-style critic without preference input
                out = getattr(agent, "qf1")(obs_t, act_t)
                if hasattr(agent, "qf2"):
                    out = _maybe_min((out, getattr(agent, "qf2")(obs_t, act_t)))
            else:
                # no critic available, fall back to raw rewards
                return scalarization(rewards, pref)

            if isinstance(out, (tuple, list)) and len(out) == 2:
                out = _maybe_min(out)

            vec_values = _to_numpy(out)
        return scalarization(vec_values, pref)

    # bind the preference inside the closure to avoid accidental mutation
    return scorer


class OuterEnsemble(MOAgent, MOPolicy):
    """Outer-loop ensemble that trains K agents on a shared pool with optional filtering.

    The shared replay buffer can be dispatched to members using either critic-based top-k selection or uniform random
    sampling. During evaluation, the ensemble can route actions using either critic-based best-member selection or
    uniform random member selection.
    """

    def __init__(
        self,
        env_fn: Callable[[], gym.Env],
        agent_ctor: Callable[..., MOPolicy],
        ensemble_size: int,
        agent_kwargs: Optional[Sequence[dict]] = None,
        member_configs: Optional[Sequence[EnsembleMemberConfig]] = None,
        buffer_size: int = int(1e6),
        min_samples_before_training: int = 1000,
        score_interval: int = 1,
        update_steps: int = 1,
        top_fraction: Optional[float] = None,
        score_subset: Optional[int] = None,
        dispatch_selection: Literal["top", "uniform", "partition", "soft_partition"] = "top",
        dispatch_warmup_selection: Optional[Literal["top", "uniform", "partition", "soft_partition"]] = None,
        dispatch_warmup_steps: int = 0,
        dispatch_temperature: float = 0.75,
        dispatch_backfill_min_size: int = 0,
        normalize_member_scores: bool = True,
        member_selection: Literal["best", "uniform", "nearest"] = "best",
        member_eval_preference: Literal["requested", "member"] = "requested",
        project_name: str = "MORL-Baselines",
        experiment_name: str = "OuterEnsemble",
        wandb_entity: Optional[str] = None,
        log: bool = True,
        member_wandb_runs: bool = False,
        delegate_to_members: bool = False,
        prefill_then_delegate: bool = False,
        shared_buffer_only: bool = False,
        online_dispatch: bool = False,
        dispatch_batch_size: int = 512,
        retain_own_samples: bool = True,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
    ) -> None:
        """Create the ensemble wrapper.

        Args:
            env_fn: Callable that builds a fresh environment; one env is created per agent.
            agent_ctor: Constructor or factory returning an MOPolicy.
            ensemble_size: Number of agents.
            agent_kwargs: Optional per-agent kwargs; if shorter, last one is reused.
            member_configs: Optional per-agent EnsembleMemberConfig for preferences and custom scorers.
            buffer_size: Size of the shared replay buffer.
            min_samples_before_training: Do not score/train until this many samples are collected.
            score_interval: How often (in environment steps) to rescore and dispatch top samples.
            update_steps: Number of agent.update() calls after each dispatch.
            top_fraction: Fraction of the shared buffer each agent keeps; default=1.0.
            score_subset: If set, randomly subsample at most this many items from the shared buffer for dispatch.
            dispatch_selection: How to pick items from the shared pool for each member:
                "top" (critic-based top-k per member), "uniform", "partition" (argmax assignment across members), or
                "soft_partition" (sample assignment from temperature-scaled member scores).
            dispatch_warmup_selection: Optional dispatch rule used during the first dispatch_warmup_steps outer steps.
            dispatch_warmup_steps: Number of outer-loop steps to use dispatch_warmup_selection before dispatch_selection.
            dispatch_temperature: Softmax temperature used by "soft_partition"; lower values make routing sharper.
            dispatch_backfill_min_size: If > 0, backfill each member with extra high-scoring shared samples until this
                many items are available for updates. This reduces early starvation at the cost of some overlap.
            normalize_member_scores: Whether to z-normalize each member's dispatch scores before cross-member comparison.
            member_selection: How to pick which member acts during evaluation: "best" (critic-based),
                "nearest" (closest configured preference), or "uniform".
            member_eval_preference: Preference passed to the selected member at evaluation time. "requested" passes the
                task/eval preference; "member" passes the member's configured preference.
            log: Whether to enable wandb logging.
            member_wandb_runs: When True (and ensemble_size>1), also let each agent keep its own wandb run in addition to the central ensemble run.
            delegate_to_members: If True, skip outer-loop updates and simply call each member's own train() (budget split evenly).
            prefill_then_delegate: If True, first collect data with the outer loop (up to min_samples_before_training),
                build a per-member filtered buffer from the shared pool, then call each member's train(). Useful when
                you want to reuse shared data collection / filtering but still rely on the native training loop for updates.
            shared_buffer_only: If True, collect experience only in the shared buffer during outer-loop training and
                rebuild member buffers from shared samples during dispatch instead of mixing in private per-agent updates.
            online_dispatch: If True, route each newly collected mini-batch directly into member buffers and update
                members at native per-step cadence, instead of rescoring the full shared buffer periodically.
            dispatch_batch_size: Number of newly collected samples to accumulate before an online dispatch pass.
            retain_own_samples: If True, each member always keeps its own freshly collected samples in its replay buffer
                in addition to any routed samples. This stabilizes training when routing is noisy.
        """
        first_env = env_fn()
        MOAgent.__init__(self, env=first_env, device=device, seed=seed)
        MOPolicy.__init__(self, id=None, device=device)

        if dispatch_selection not in ("top", "uniform", "partition", "soft_partition"):
            raise ValueError(
                "dispatch_selection must be 'top', 'uniform', 'partition', or 'soft_partition', "
                f"got {dispatch_selection!r}"
            )
        if dispatch_warmup_selection is not None and dispatch_warmup_selection not in (
            "top",
            "uniform",
            "partition",
            "soft_partition",
        ):
            raise ValueError(
                "dispatch_warmup_selection must be one of 'top', 'uniform', 'partition', 'soft_partition', or None, "
                f"got {dispatch_warmup_selection!r}"
            )
        if member_selection not in ("best", "uniform", "nearest"):
            raise ValueError(f"member_selection must be 'best', 'nearest', or 'uniform', got {member_selection!r}")
        if member_eval_preference not in ("requested", "member"):
            raise ValueError(
                "member_eval_preference must be 'requested' or 'member', "
                f"got {member_eval_preference!r}"
            )

        self.envs = [first_env] + [env_fn() for _ in range(ensemble_size - 1)]
        self.ensemble_size = ensemble_size
        self.agent_kwargs = list(agent_kwargs) if agent_kwargs is not None else [{} for _ in range(ensemble_size)]
        if len(self.agent_kwargs) < ensemble_size:
            self.agent_kwargs.extend([self.agent_kwargs[-1]] * (ensemble_size - len(self.agent_kwargs)))

        # Logging strategy: single agent should own its wandb run; multi-agent logs centrally, optionally per-agent too.
        self.member_log = bool(log) if ensemble_size == 1 else bool(member_wandb_runs)
        self.log = bool(log) and ensemble_size > 1
        self.delegate_to_members = delegate_to_members
        self.prefill_then_delegate = prefill_then_delegate
        self.shared_buffer_only = shared_buffer_only
        self.online_dispatch = online_dispatch
        self.dispatch_batch_size = max(1, int(dispatch_batch_size))
        self.retain_own_samples = bool(retain_own_samples)

        ctor_params = inspect.signature(agent_ctor).parameters
        pass_id = "id" in ctor_params
        self.agents: List[MOPolicy] = []
        for i in range(ensemble_size):
            kwargs = dict(self.agent_kwargs[i])
            kwargs.setdefault("device", self.device)
            # single-agent case should keep its own wandb; multi-agent logs centrally
            kwargs["log"] = self.member_log
            if pass_id:
                kwargs.setdefault("id", i)
            self.agents.append(agent_ctor(env=self.envs[i], **kwargs))

        # use member gamma if available, otherwise default
        self.gamma = getattr(self.agents[0], "gamma", 0.99)
        self.learning_starts_min = min(getattr(a, "learning_starts", 0) for a in self.agents)
        # Ensure outer-loop dispatch does not start before members would normally start updating.
        self.min_samples_before_training = max(min_samples_before_training, self.learning_starts_min)

        self.member_configs: List[EnsembleMemberConfig] = []
        for idx in range(ensemble_size):
            cfg = EnsembleMemberConfig()
            if member_configs and idx < len(member_configs):
                cfg = member_configs[idx]
            if cfg.preference is None:
                cfg.preference = random_weights(self.reward_dim, rng=self.np_random)
            if cfg.scalarization is None:
                cfg.scalarization = _default_scalarization
            if cfg.scorer is None:
                cfg.scorer = _build_default_scorer(cfg.preference, cfg.scalarization)
            self.member_configs.append(cfg)

        self.shared_buffer = ReplayBuffer(
            obs_shape=self.observation_shape,
            action_dim=self.action_dim,
            rew_dim=self.reward_dim,
            max_size=buffer_size,
        )
        self.member_buffers: List[ReplayBuffer] = []
        for agent in self.agents:
            member_buffer = ReplayBuffer(
                obs_shape=self.observation_shape,
                action_dim=self.action_dim,
                rew_dim=self.reward_dim,
                max_size=self.shared_buffer.max_size,
            )
            self.member_buffers.append(member_buffer)
            self._set_agent_buffer(agent, member_buffer)

        self._obs = [env.reset()[0] for env in self.envs]
        # default keep all samples per agent; user can shrink with top_fraction
        self.top_fraction = top_fraction if top_fraction is not None else 1.0
        self.filter_top_only = self.top_fraction < 1.0
        self.score_interval = max(1, score_interval)
        self.update_steps = max(1, update_steps)
        self.score_subset = score_subset
        self.dispatch_selection = dispatch_selection
        self.dispatch_warmup_selection = dispatch_warmup_selection
        self.dispatch_warmup_steps = max(0, int(dispatch_warmup_steps))
        self.dispatch_temperature = max(1e-6, float(dispatch_temperature))
        self.dispatch_backfill_min_size = max(0, int(dispatch_backfill_min_size))
        self.normalize_member_scores = bool(normalize_member_scores)
        self.member_selection = member_selection
        self.member_eval_preference = member_eval_preference
        self._pending_obs: List[np.ndarray] = []
        self._pending_actions: List[np.ndarray] = []
        self._pending_rewards: List[np.ndarray] = []
        self._pending_next_obs: List[np.ndarray] = []
        self._pending_dones: List[float] = []
        self._pending_origins: List[int] = []
        self.member_own_sample_counts = np.zeros((self.ensemble_size,), dtype=np.int64)
        self.member_shared_sample_counts = np.zeros((self.ensemble_size,), dtype=np.int64)

        # Per-member learning start threshold (used for action warmup + gating updates).
        # Do not scale by ensemble_size: each member's global_step is incremented once per env step.
        self.effective_learning_starts = []
        for agent in self.agents:
            ls = max(0, getattr(agent, "learning_starts", 0))
            self.effective_learning_starts.append(ls)

        # local CSV logger paths per agent for fallback diagnostics
        self.local_loss_files: List[Optional[str]] = [None] * self.ensemble_size
        self.local_loss_dir: Optional[str] = None
        if self.log:
            ts = int(time.time())
            base_dir = os.path.join("logs", f"ensemble_losses_{ts}")
            os.makedirs(base_dir, exist_ok=True)
            self.local_loss_dir = base_dir
            for i in range(self.ensemble_size):
                path = os.path.join(base_dir, f"agent_{i}.csv")
                self.local_loss_files[i] = path
                if not os.path.exists(path):
                    with open(path, "w", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow(["global_step", "loss_name", "value"])

        if self.log:
            self.setup_wandb(
                project_name=project_name,
                experiment_name=experiment_name,
                entity=wandb_entity,
            )

    # --- MOAgent / MOPolicy required API -------------------------------------------------
    def get_config(self) -> dict:
        return {
            "ensemble_size": self.ensemble_size,
            "buffer_size": self.shared_buffer.max_size,
            "min_samples_before_training": self.min_samples_before_training,
            "score_interval": self.score_interval,
            "update_steps": self.update_steps,
            "top_fraction": self.top_fraction,
            "dispatch_selection": self.dispatch_selection,
            "dispatch_warmup_selection": self.dispatch_warmup_selection,
            "dispatch_warmup_steps": self.dispatch_warmup_steps,
            "dispatch_temperature": self.dispatch_temperature,
            "dispatch_backfill_min_size": self.dispatch_backfill_min_size,
            "normalize_member_scores": self.normalize_member_scores,
            "member_selection": self.member_selection,
            "member_eval_preference": self.member_eval_preference,
            "shared_buffer_only": self.shared_buffer_only,
            "online_dispatch": self.online_dispatch,
            "dispatch_batch_size": self.dispatch_batch_size,
            "retain_own_samples": self.retain_own_samples,
            "seed": self.seed,
            "gamma": self.gamma,
        }

    def eval(self, obs: np.ndarray, w: Optional[np.ndarray] = None):
        """Route evaluation to the member predicted to work best for the given preference."""
        preference = w if w is not None else random_weights(self.reward_dim, rng=self.np_random)
        idx = self._select_agent_idx(obs, preference)
        action_preference = self._action_preference_for_member(idx, preference)
        return self.agents[idx].eval(obs, action_preference)

    def update(self) -> None:  # pragma: no cover - training handled by train()
        """OuterEnsemble does not use a single-step update; call train()."""
        return

    # --- Core logic ----------------------------------------------------------------------
    def train(
        self,
        total_timesteps: int,
        eval_env: Optional[gym.Env] = None,
        eval_freq: int = 10000,
        checkpoints: bool = False,
        save_freq: int = 10000,
        save_dir: str = "weights",
        num_eval_episodes: int = 3,
        ref_point: Optional[np.ndarray] = None,
        known_pareto_front: Optional[List[np.ndarray]] = None,
        num_eval_weights_for_front: int = 50,
        num_eval_episodes_for_front: int = 3,
        num_eval_weights_for_eval: int = 50,
    ) -> None:
        """Main training loop."""
        # Delegate path: use each member's native train (for parity with single-agent behavior).
        if self.ensemble_size == 1 or self.delegate_to_members or self.prefill_then_delegate:
            # optional prefill: collect shared experiences first
            if self.prefill_then_delegate and total_timesteps > 0 and self.ensemble_size > 1:
                warmup_steps = max(self.min_samples_before_training, 1)
                for _ in range(warmup_steps):
                    self._collect_one_step()
                    self.global_step += 1

            # Split the budget deterministically across members (so later agents don't get 0 when total_timesteps is small).
            base_steps = total_timesteps // max(1, self.ensemble_size)
            remainder = total_timesteps % max(1, self.ensemble_size)
            per_agent_budgets = [base_steps + (1 if i < remainder else 0) for i in range(self.ensemble_size)]
            # Keep a monotonically increasing step for the ensemble wandb run even though we train members sequentially.
            ensemble_step_base = self.global_step

            if self.log:
                import wandb

                wandb.log(
                    {
                        "ensemble/delegate_mode": 1.0,
                        "ensemble/num_members": float(self.ensemble_size),
                        "ensemble/total_timesteps": float(total_timesteps),
                        "global_step": ensemble_step_base,
                    }
                )
            for idx, agent in enumerate(self.agents):
                agent_step_base = agent.global_step
                # if we prefetched, seed each agent's replay buffer from the filtered shared pool so native train() can update immediately
                if self.prefill_then_delegate and len(self.shared_buffer) > 0:
                    # Filter the shared pool per member before delegating so each expert can start from its own top slice.
                    agent_buf = self._build_member_buffer_from_shared(idx, agent)
                    self._set_agent_buffer(agent, agent_buf)
                    # make sure learning_starts does not delay updates
                    if hasattr(agent, "learning_starts"):
                        setattr(agent, "learning_starts", 0)

                # Wrap update() to stream each member's losses into the ensemble wandb run.
                # This works even when the member's own wandb logging is disabled.
                orig_update = getattr(agent, "update", None)
                orig_close_wandb = getattr(agent, "close_wandb", None)

                if self.log:
                    import wandb

                    wandb.log(
                        {
                            f"agent_{idx}/member_global_step": agent.global_step,
                            f"agent_{idx}/status": 1.0,
                            f"agent_{idx}/train_started": 1.0,
                            "global_step": ensemble_step_base,
                        },
                        commit=True,
                    )

                def _wrapped_update(  # type: ignore[no-redef]
                    _idx: int = idx,
                    _agent: MOPolicy = agent,
                    _orig_update=orig_update,
                    _ensemble_step_base: int = ensemble_step_base,
                    _agent_step_base: int = agent_step_base,
                ) -> None:
                    if _orig_update is not None:
                        _orig_update()
                    # Sync ensemble global_step to the member's progress for meaningful wandb x-axis.
                    self.global_step = _ensemble_step_base + (_agent.global_step - _agent_step_base)
                    if self.log:
                        try:
                            import wandb

                            wandb.log(
                                {
                                    f"agent_{_idx}/member_global_step": _agent.global_step,
                                    f"agent_{_idx}/updates": getattr(_agent, "_updates", np.nan),
                                    "global_step": self.global_step,
                                },
                                commit=False,
                            )
                        except Exception:
                            # Never let logging break training.
                            pass
                    if self.log and hasattr(_agent, "last_losses") and _agent.last_losses:
                        self._log_agent_losses(_idx, _agent, commit=True)
                    if self.log and eval_env is not None and eval_freq > 0 and self.global_step % eval_freq == 0:
                        self._log_eval(
                            eval_env,
                            num_eval_episodes=num_eval_episodes,
                            ref_point=ref_point,
                            known_pareto_front=known_pareto_front,
                            num_eval_weights_for_front=num_eval_weights_for_front,
                            num_eval_episodes_for_front=num_eval_episodes_for_front,
                            num_eval_weights_for_eval=num_eval_weights_for_eval,
                        )
                    if checkpoints and save_freq > 0 and self.global_step % save_freq == 0:
                        self._save_members(save_dir=save_dir)

                if callable(orig_update):
                    setattr(agent, "update", _wrapped_update)
                # Some algorithms call close_wandb() unconditionally at the end of train(). When we are running a
                # central ensemble wandb run, prevent members from finishing it.
                if self.log and not self.member_log and callable(orig_close_wandb):
                    setattr(agent, "close_wandb", lambda: None)
                train_sig = inspect.signature(getattr(agent, "train"))
                pass_args = {
                    "total_timesteps": per_agent_budgets[idx],
                    "eval_env": eval_env,
                    "eval_freq": eval_freq,
                    "num_eval_episodes": num_eval_episodes,
                    "ref_point": ref_point,
                    "known_pareto_front": known_pareto_front,
                    "num_eval_weights_for_front": num_eval_weights_for_front,
                    "num_eval_episodes_for_front": num_eval_episodes_for_front,
                    "num_eval_weights_for_eval": num_eval_weights_for_eval,
                    "reset_num_timesteps": False,
                }
                filtered = {k: v for k, v in pass_args.items() if k in train_sig.parameters}
                try:
                    agent.train(**filtered)
                finally:
                    if callable(orig_update):
                        setattr(agent, "update", orig_update)
                    if callable(orig_close_wandb):
                        setattr(agent, "close_wandb", orig_close_wandb)

                # Advance ensemble step base by the member's progress (even if update() never ran).
                ensemble_step_base = ensemble_step_base + max(0, agent.global_step - agent_step_base)
                self.global_step = ensemble_step_base

                if self.log and eval_env is not None:
                    # Log at least one ensemble evaluation point after each member finishes.
                    self._log_eval(
                        eval_env,
                        num_eval_episodes=num_eval_episodes,
                        ref_point=ref_point,
                        known_pareto_front=known_pareto_front,
                        num_eval_weights_for_front=num_eval_weights_for_front,
                        num_eval_episodes_for_front=num_eval_episodes_for_front,
                        num_eval_weights_for_eval=num_eval_weights_for_eval,
                    )
                    try:
                        import wandb

                        wandb.log(
                            {
                                f"agent_{idx}/train_finished": 1.0,
                                f"agent_{idx}/member_global_step": agent.global_step,
                                f"agent_{idx}/updates": getattr(agent, "_updates", np.nan),
                                "global_step": self.global_step,
                            }
                        )
                    except Exception:
                        pass

            if self.log:
                self.close_wandb()
            return

        for step in range(1, total_timesteps + 1):
            if step == 1 and (self.shared_buffer_only or self.online_dispatch):
                for idx, agent in enumerate(self.agents):
                    self.member_buffers[idx] = ReplayBuffer(
                        obs_shape=self.observation_shape,
                        action_dim=self.action_dim,
                        rew_dim=self.reward_dim,
                        max_size=self.shared_buffer.max_size,
                    )
                    self._set_agent_buffer(agent, self.member_buffers[idx])
            self._collect_one_step()
            self.global_step += 1

            if (
                not self.online_dispatch
                and len(self.shared_buffer) >= self.min_samples_before_training
                and step % self.score_interval == 0
            ):
                self._score_and_dispatch()

            if self.log and eval_env is not None and step % eval_freq == 0:
                self._log_eval(
                    eval_env,
                    num_eval_episodes=num_eval_episodes,
                    ref_point=ref_point,
                    known_pareto_front=known_pareto_front,
                    num_eval_weights_for_front=num_eval_weights_for_front,
                    num_eval_episodes_for_front=num_eval_episodes_for_front,
                    num_eval_weights_for_eval=num_eval_weights_for_eval,
                )
            if checkpoints and save_freq > 0 and self.global_step % save_freq == 0:
                self._save_members(save_dir=save_dir)

        if self.online_dispatch:
            self._flush_pending_dispatch(force=True)

        if checkpoints:
            self._save_members(save_dir=save_dir)

        if self.log:
            self.close_wandb()

    # --- Helpers -------------------------------------------------------------------------
    def _collect_one_step(self) -> None:
        """Collect one transition per agent and add them to the shared buffer."""
        for idx, agent in enumerate(self.agents):
            # Mirror common train() loops: global_step increments once per env step.
            agent.global_step += 1
            pref = self.member_configs[idx].preference

            # Warm-up exploration: sample random actions until the agent would normally start learning.
            if agent.global_step < self.effective_learning_starts[idx]:
                action = self.envs[idx].action_space.sample()
            elif hasattr(agent, "_select_action"):
                action = agent._select_action(self._obs[idx], pref, deterministic=False)
            else:
                action = agent.eval(self._obs[idx], pref)

            next_obs, reward, terminated, truncated, info = self.envs[idx].step(action)
            done = terminated or truncated
            if self.online_dispatch:
                if self.retain_own_samples:
                    own_buffer = self.member_buffers[idx]
                    own_buffer.add(self._obs[idx], action, reward, next_obs, float(done))
                    self.member_own_sample_counts[idx] += 1
                self._pending_obs.append(np.asarray(self._obs[idx], dtype=np.float32))
                self._pending_actions.append(np.asarray(action, dtype=np.float32))
                self._pending_rewards.append(np.asarray(reward, dtype=np.float32))
                self._pending_next_obs.append(np.asarray(next_obs, dtype=np.float32))
                self._pending_dones.append(float(done))
                self._pending_origins.append(idx)
            else:
                self.shared_buffer.add(self._obs[idx], action, reward, next_obs, float(done))

            agent_buffer = None
            if not self.shared_buffer_only and not self.online_dispatch:
                # Hybrid mode keeps each member's raw data locally in addition to the shared pool.
                agent_buffer = self.member_buffers[idx]
                agent_buffer.add(self._obs[idx], action, reward, next_obs, float(done))

            self._obs[idx] = next_obs
            if done:
                if self.log and "episode" in info:
                    log_episode_info(info["episode"], np.dot, pref, self.global_step, id=idx, verbose=False)
                self._obs[idx], _ = self.envs[idx].reset()

            agent.num_episodes = getattr(agent, "num_episodes", 0) + int(done)

            # perform a baseline per-step update similar to single-agent training
            if (
                not self.shared_buffer_only
                and not self.online_dispatch
                and agent_buffer is not None
                and agent.global_step >= self.effective_learning_starts[idx]
                and len(agent_buffer) >= getattr(agent, "batch_size", 1)
            ):
                self._maybe_update_agent(idx, agent, agent_buffer, require_learning_starts=True)

        if self.online_dispatch:
            self._flush_pending_dispatch(force=False)
            for idx, agent in enumerate(self.agents):
                agent_buffer = self.member_buffers[idx]
                self._maybe_update_agent(idx, agent, agent_buffer, require_learning_starts=True)
            if self.log and self.global_step % 100 == 0:
                self._log_online_agent_debug()

    def _dispatch_recent_batch(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        dones: np.ndarray,
        origins: Optional[np.ndarray] = None,
    ) -> None:
        """Route a newly collected mini-batch directly into per-member buffers.

        When origins are provided, each member only receives other members' samples from the shared route. Its own
        samples are retained separately by retain_own_samples, so this avoids duplicate self-samples.
        """
        active_dispatch_selection = self._active_dispatch_selection()
        if active_dispatch_selection in ("partition", "soft_partition"):
            partitions = self._partition_shared_indices(
                obs,
                actions,
                rewards,
                next_obs,
                dones,
                soft=active_dispatch_selection == "soft_partition",
                backfill_min_size=0,
            )
            for idx, selected_idx in enumerate(partitions):
                if origins is not None:
                    selected_idx = selected_idx[origins[selected_idx] != idx]
                agent_buffer = self.member_buffers[idx]
                for sample_idx in selected_idx:
                    agent_buffer.add(
                        obs[sample_idx],
                        actions[sample_idx],
                        rewards[sample_idx],
                        next_obs[sample_idx],
                        dones[sample_idx],
                    )
                    self.member_shared_sample_counts[idx] += 1
            return

        for idx, agent in enumerate(self.agents):
            candidate_idx = np.arange(len(obs), dtype=np.int64)
            if origins is not None:
                candidate_idx = np.flatnonzero(origins != idx).astype(np.int64, copy=False)
            if len(candidate_idx) == 0:
                continue
            k = max(1, int(len(candidate_idx) * self.top_fraction))
            selected_idx = self._select_shared_indices(
                idx,
                agent,
                obs[candidate_idx],
                actions[candidate_idx],
                rewards[candidate_idx],
                next_obs[candidate_idx],
                dones[candidate_idx],
                k=k,
                dispatch_selection=active_dispatch_selection,
            )
            selected_idx = candidate_idx[selected_idx]
            agent_buffer = self.member_buffers[idx]
            for sample_idx in selected_idx:
                agent_buffer.add(
                    obs[sample_idx],
                    actions[sample_idx],
                    rewards[sample_idx],
                    next_obs[sample_idx],
                    dones[sample_idx],
                )
                self.member_shared_sample_counts[idx] += 1

    def _flush_pending_dispatch(self, force: bool) -> None:
        """Dispatch accumulated online samples once enough have been collected."""
        while len(self._pending_obs) >= self.dispatch_batch_size or (force and len(self._pending_obs) > 0):
            batch_size = min(self.dispatch_batch_size, len(self._pending_obs))
            obs = np.asarray(self._pending_obs[:batch_size], dtype=np.float32)
            actions = np.asarray(self._pending_actions[:batch_size], dtype=np.float32)
            rewards = np.asarray(self._pending_rewards[:batch_size], dtype=np.float32)
            next_obs = np.asarray(self._pending_next_obs[:batch_size], dtype=np.float32)
            dones = np.asarray(self._pending_dones[:batch_size], dtype=np.float32)
            origins = np.asarray(self._pending_origins[:batch_size], dtype=np.int64)

            del self._pending_obs[:batch_size]
            del self._pending_actions[:batch_size]
            del self._pending_rewards[:batch_size]
            del self._pending_next_obs[:batch_size]
            del self._pending_dones[:batch_size]
            del self._pending_origins[:batch_size]

            self._dispatch_recent_batch(
                obs=obs,
                actions=actions,
                rewards=rewards,
                next_obs=next_obs,
                dones=dones,
                origins=origins,
            )

    def _log_online_agent_debug(self) -> None:
        """Emit lightweight per-agent stats while using online dispatch."""
        if not self.log:
            return
        import wandb

        metrics = {"global_step": self.global_step, "ensemble/pending_dispatch": len(self._pending_obs)}
        for idx, agent in enumerate(self.agents):
            agent_buffer = self.member_buffers[idx]
            metrics[f"agent_{idx}/buffer_size"] = len(agent_buffer)
            metrics[f"agent_{idx}/own_samples"] = int(self.member_own_sample_counts[idx])
            metrics[f"agent_{idx}/shared_samples"] = int(self.member_shared_sample_counts[idx])
            metrics[f"agent_{idx}/shared_to_own_ratio"] = float(
                self.member_shared_sample_counts[idx] / max(1, self.member_own_sample_counts[idx])
            )
            internal_buffer = getattr(agent, "replay_buffer", None)
            if internal_buffer is not None:
                metrics[f"agent_{idx}/internal_buffer_size"] = len(internal_buffer)
            metrics[f"agent_{idx}/member_global_step"] = agent.global_step
            metrics[f"agent_{idx}/updates"] = getattr(agent, "_updates", np.nan)
        wandb.log(metrics, commit=True)

    def _maybe_update_agent(
        self,
        idx: int,
        agent: MOPolicy,
        agent_buffer: ReplayBuffer,
        require_learning_starts: bool,
    ) -> None:
        """Bind the member buffer and perform an update when enough data is available."""
        batch_size = getattr(agent, "batch_size", 1)
        if len(agent_buffer) < batch_size:
            return
        if require_learning_starts and agent.global_step < self.effective_learning_starts[idx]:
            return
        self._set_agent_buffer(agent, agent_buffer)
        agent.update()
        if self.log and hasattr(agent, "last_losses") and agent.last_losses and agent.global_step % 50 == 0:
            self._log_agent_losses(idx, agent, commit=True)

    def _score_and_dispatch(self) -> None:
        """Score the shared buffer with each critic and train members on their top slice."""
        obs, actions, rewards, next_obs, dones = self.shared_buffer.get_all_data(max_samples=self.score_subset)
        if len(obs) == 0:
            return

        active_dispatch_selection = self._active_dispatch_selection()
        partitioned_indices = None
        if active_dispatch_selection in ("partition", "soft_partition"):
            partitioned_indices = self._partition_shared_indices(
                obs,
                actions,
                rewards,
                next_obs,
                dones,
                soft=active_dispatch_selection == "soft_partition",
            )

        for idx, agent in enumerate(self.agents):
            if partitioned_indices is not None:
                selected_idx = partitioned_indices[idx]
            else:
                k = max(1, int(len(obs) * self.top_fraction))
                selected_idx = self._select_shared_indices(
                    idx,
                    agent,
                    obs,
                    actions,
                    rewards,
                    next_obs,
                    dones,
                    k=k,
                    dispatch_selection=active_dispatch_selection,
                )

            rebuild_member_buffer = self.filter_top_only or self.shared_buffer_only
            if rebuild_member_buffer:
                # In shared-buffer mode, each member trains from a fresh slice of the shared pool.
                agent_buffer = ReplayBuffer(
                    obs_shape=self.observation_shape,
                    action_dim=self.action_dim,
                    rew_dim=self.reward_dim,
                    max_size=self.shared_buffer.max_size,
                )
                self._set_agent_buffer(agent, agent_buffer)
                self.member_buffers[idx] = agent_buffer
            else:
                # Hybrid mode appends top-shared samples to the member's private buffer.
                agent_buffer = self.member_buffers[idx]

            for i in selected_idx:
                agent_buffer.add(obs[i], actions[i], rewards[i], next_obs[i], dones[i])

            for _ in range(self.update_steps):
                self._maybe_update_agent(idx, agent, agent_buffer, require_learning_starts=True)

            # Log per-agent losses to the ensemble wandb run if available
            if self.log:
                # always log a light heartbeat so we can see training progress even if last_losses is empty
                debug_metrics = {
                    f"agent_{idx}/buffer_size": len(agent_buffer),
                    f"agent_{idx}/updates": getattr(agent, "_updates", np.nan),
                    "global_step": self.global_step,
                }
                import wandb

                wandb.log(debug_metrics, commit=False)

                if hasattr(agent, "last_losses") and agent.last_losses:
                    self._log_agent_losses(idx, agent, commit=True)

    def _build_member_buffer_from_shared(self, idx: int, agent: MOPolicy) -> ReplayBuffer:
        """Build a fresh per-member buffer from the shared pool using the member's scorer.

        Used by prefill_then_delegate so we can reuse each agent's native train() loop but still start from
        the outer-loop filtered dataset.
        """
        obs, actions, rewards, next_obs, dones = self.shared_buffer.get_all_data(max_samples=self.score_subset)
        if len(obs) == 0:
            return ReplayBuffer(
                obs_shape=self.observation_shape,
                action_dim=self.action_dim,
                rew_dim=self.reward_dim,
                max_size=self.shared_buffer.max_size,
            )
        active_dispatch_selection = self._active_dispatch_selection()
        if active_dispatch_selection in ("partition", "soft_partition"):
            selected_idx = self._partition_shared_indices(
                obs,
                actions,
                rewards,
                next_obs,
                dones,
                soft=active_dispatch_selection == "soft_partition",
            )[idx]
        else:
            k = max(1, int(len(obs) * self.top_fraction))
            selected_idx = self._select_shared_indices(
                idx,
                agent,
                obs,
                actions,
                rewards,
                next_obs,
                dones,
                k=k,
                dispatch_selection=active_dispatch_selection,
            )
        buf = ReplayBuffer(
            obs_shape=self.observation_shape,
            action_dim=self.action_dim,
            rew_dim=self.reward_dim,
            max_size=self.shared_buffer.max_size,
        )
        for i in selected_idx:
            buf.add(obs[i], actions[i], rewards[i], next_obs[i], dones[i])
        return buf

    def _select_shared_indices(
        self,
        idx: int,
        agent: MOPolicy,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        dones: np.ndarray,
        k: int,
        dispatch_selection: Optional[str] = None,
    ) -> np.ndarray:
        """Pick indices from the (already sampled) shared pool for member idx."""
        n = len(obs)
        if n == 0:
            return np.empty((0,), dtype=np.int64)
        k = int(np.clip(k, 1, n))

        active_dispatch_selection = dispatch_selection or self._active_dispatch_selection()
        if active_dispatch_selection == "uniform":
            return self.np_random.choice(n, size=k, replace=False).astype(np.int64, copy=False)

        scores = self._score_member_batch(idx, agent, obs, actions, rewards, next_obs, dones)
        scores = np.asarray(scores).reshape(-1)
        return np.argpartition(-scores, kth=k - 1)[:k].astype(np.int64, copy=False)

    def _active_dispatch_selection(self) -> str:
        """Return the dispatch rule currently in effect, accounting for warmup."""
        if self.dispatch_warmup_selection is not None and self.global_step < self.dispatch_warmup_steps:
            return self.dispatch_warmup_selection
        return self.dispatch_selection

    def _partition_shared_indices(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        dones: np.ndarray,
        soft: bool = False,
        backfill_min_size: Optional[int] = None,
    ) -> List[np.ndarray]:
        """Assign each shared sample to the member with the highest score."""
        if len(obs) == 0:
            return [np.empty((0,), dtype=np.int64) for _ in range(self.ensemble_size)]

        score_matrix = []
        for idx, agent in enumerate(self.agents):
            scores = self._score_member_batch(idx, agent, obs, actions, rewards, next_obs, dones)
            score_matrix.append(np.asarray(scores).reshape(-1))
        score_matrix = np.stack(score_matrix, axis=0).astype(np.float32, copy=False)
        compare_matrix = self._normalize_score_matrix(score_matrix) if self.normalize_member_scores else score_matrix

        if soft:
            logits = compare_matrix / self.dispatch_temperature
            logits -= np.max(logits, axis=0, keepdims=True)
            probs = np.exp(logits)
            probs /= np.clip(np.sum(probs, axis=0, keepdims=True), a_min=1e-8, a_max=None)
            draws = self.np_random.random(size=probs.shape[1])
            cumulative = np.cumsum(probs, axis=0)
            best_members = np.argmax(cumulative >= draws[None, :], axis=0)
        else:
            best_members = np.argmax(compare_matrix, axis=0)
        partitions: List[np.ndarray] = []
        for idx in range(self.ensemble_size):
            assigned = np.flatnonzero(best_members == idx)
            if assigned.size == 0:
                partitions.append(np.empty((0,), dtype=np.int64))
                continue

            if self.top_fraction < 1.0:
                member_scores = compare_matrix[idx, assigned]
                k = max(1, int(len(assigned) * self.top_fraction))
                top_local = np.argpartition(-member_scores, kth=k - 1)[:k]
                assigned = assigned[top_local]

            partitions.append(assigned.astype(np.int64, copy=False))
        effective_backfill_min_size = self.dispatch_backfill_min_size if backfill_min_size is None else backfill_min_size
        if effective_backfill_min_size > 0:
            partitions = self._backfill_partitions(partitions, compare_matrix, min_size=effective_backfill_min_size)
        return partitions

    def _score_member_batch(
        self,
        idx: int,
        agent: MOPolicy,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        dones: np.ndarray,
    ) -> np.ndarray:
        """Compute dispatch scores for one member on a shared batch."""
        cfg = self.member_configs[idx]
        return cfg.scorer(agent, obs, actions, rewards, next_obs, dones, cfg.preference)

    def _normalize_score_matrix(self, score_matrix: np.ndarray) -> np.ndarray:
        """Normalize each member's score scale before cross-member comparison."""
        centered = score_matrix - np.mean(score_matrix, axis=1, keepdims=True)
        scales = np.std(centered, axis=1, keepdims=True)
        scales = np.where(scales < 1e-6, 1.0, scales)
        normalized = centered / scales
        return np.clip(normalized, -5.0, 5.0)

    def _backfill_partitions(
        self,
        partitions: List[np.ndarray],
        score_matrix: np.ndarray,
        min_size: int,
    ) -> List[np.ndarray]:
        """Ensure each member receives enough samples for stable updates."""
        if min_size <= 0:
            return partitions

        total_samples = score_matrix.shape[1]
        target_size = min(min_size, total_samples)
        out: List[np.ndarray] = []
        for idx, assigned in enumerate(partitions):
            assigned = np.asarray(assigned, dtype=np.int64)
            if assigned.size >= target_size:
                out.append(assigned)
                continue

            scores = score_matrix[idx]
            ranked = np.argsort(-scores, kind="stable")
            needed = target_size - assigned.size
            existing = set(assigned.tolist())
            extras = []
            for sample_idx in ranked:
                if int(sample_idx) in existing:
                    continue
                extras.append(int(sample_idx))
                if len(extras) >= needed:
                    break
            if extras:
                assigned = np.concatenate((assigned, np.asarray(extras, dtype=np.int64)))
            out.append(assigned)
        return out

    def _select_agent_idx(self, obs: np.ndarray, pref: np.ndarray) -> int:
        """Pick which member acts for a given (obs, preference)."""
        if self.member_selection == "uniform":
            return int(self.np_random.integers(self.ensemble_size))
        if self.member_selection == "nearest":
            return self._select_nearest_preference_agent(pref)
        return self._select_best_agent(obs, pref)

    def _action_preference_for_member(self, idx: int, requested_pref: np.ndarray) -> np.ndarray:
        """Return the preference used as input to a selected member's conditioned policy."""
        if self.member_eval_preference == "member":
            return np.asarray(self.member_configs[idx].preference, dtype=np.float32)
        return np.asarray(requested_pref, dtype=np.float32)

    def _select_nearest_preference_agent(self, pref: np.ndarray) -> int:
        """Choose the member whose configured preference is closest to the requested one."""
        pref = np.asarray(pref, dtype=np.float32).reshape(-1)
        distances = [
            np.linalg.norm(pref - np.asarray(cfg.preference, dtype=np.float32).reshape(-1)) for cfg in self.member_configs
        ]
        return int(np.argmin(distances))

    def _select_best_agent(self, obs: np.ndarray, pref: np.ndarray) -> int:
        """Choose the agent with the highest scalarized critic value for the current state."""
        scores = []
        for idx, agent in enumerate(self.agents):
            cfg = self.member_configs[idx]
            action = agent.eval(obs, self._action_preference_for_member(idx, pref))
            obs_t = th.as_tensor(obs, dtype=th.float32, device=getattr(agent, "device", th.device("cpu"))).unsqueeze(0)
            act_t = th.as_tensor(action, dtype=th.float32, device=getattr(agent, "device", th.device("cpu"))).unsqueeze(
                0
            )
            pref_t = th.as_tensor(pref, dtype=th.float32, device=getattr(agent, "device", th.device("cpu")))
            with th.no_grad():
                if hasattr(agent, "critic"):
                    q_vals = agent.critic(obs_t, act_t, pref_t.expand(1, -1))
                elif hasattr(agent, "qf1"):
                    q_vals = getattr(agent, "qf1")(obs_t, act_t)
                    if hasattr(agent, "qf2"):
                        q_vals = th.minimum(q_vals, getattr(agent, "qf2")(obs_t, act_t))
                else:
                    scores.append(cfg.scalarization(np.atleast_2d(pref), pref)[0])
                    continue
            if isinstance(q_vals, (tuple, list)) and len(q_vals) == 2:
                q_vals = th.minimum(q_vals[0], q_vals[1])
            q_vec = _to_numpy(q_vals).reshape(1, -1)
            scores.append(cfg.scalarization(q_vec, pref)[0])
        return int(np.argmax(scores))

    def _log_eval(
        self,
        eval_env: gym.Env,
        num_eval_episodes: int,
        ref_point: Optional[np.ndarray],
        known_pareto_front: Optional[List[np.ndarray]],
        num_eval_weights_for_front: int,
        num_eval_episodes_for_front: int,
        num_eval_weights_for_eval: int,
    ) -> None:
        """Run evaluations and log HV/EUM and per-agent returns."""
        import wandb

        # Log a single random preference rollout for quick feedback
        eval_pref = random_weights(self.reward_dim, rng=self.np_random)
        _, _, _, discounted_vec_return = policy_evaluation_mo(
            self, eval_env, w=eval_pref, rep=num_eval_episodes, scalarization=np.dot
        )
        wandb.log(
            {
                "ensemble/eval_discounted_vec_return": discounted_vec_return,
                "ensemble/eval_pref": eval_pref,
                "global_step": self.global_step,
            }
        )

        # Pareto metrics for the ensemble (best-agent selection)
        eval_weights = equally_spaced_weights(self.reward_dim, n=num_eval_weights_for_front)
        returns_test_tasks = [
            policy_evaluation_mo(self, eval_env, w=ew, rep=num_eval_episodes_for_front, scalarization=np.dot)[3]
            for ew in eval_weights
        ]
        if ref_point is not None:
            log_all_multi_policy_metrics(
                current_front=returns_test_tasks,
                hv_ref_point=ref_point,
                reward_dim=self.reward_dim,
                global_step=self.global_step,
                n_sample_weights=num_eval_weights_for_eval,
                ref_front=known_pareto_front,
            )

        # Per-agent scalarized return on the same eval_pref
        per_agent = {}
        for idx, agent in enumerate(self.agents):
            _, _, _, vec_ret = policy_evaluation_mo(agent, eval_env, w=eval_pref, rep=1, scalarization=np.dot)
            per_agent[f"agent_{idx}/discounted_vec_return"] = vec_ret
            # also log last losses if present
            if hasattr(agent, "last_losses") and agent.last_losses:
                for k, v in agent.last_losses.items():
                    per_agent[f"agent_{idx}/{k}"] = v
        per_agent["global_step"] = self.global_step
        wandb.log(per_agent)

    def _get_agent_buffer(self, agent: MOPolicy, create_if_missing: bool = False) -> ReplayBuffer:
        """Helper to get or create an agent-specific replay buffer."""
        uses_custom_getter = type(agent).get_buffer is not MOPolicy.get_buffer
        uses_custom_setter = type(agent).set_buffer is not MOPolicy.set_buffer

        if uses_custom_getter:
            buf = agent.get_buffer()
        elif hasattr(agent, "replay_buffer"):
            buf = getattr(agent, "replay_buffer")
        else:
            buf = getattr(agent, "buffer", None)
        if buf is None and create_if_missing:
            buf = ReplayBuffer(
                obs_shape=self.observation_shape,
                action_dim=self.action_dim,
                rew_dim=self.reward_dim,
                max_size=self.shared_buffer.max_size,
            )
            if uses_custom_setter:
                agent.set_buffer(buf)
            elif hasattr(agent, "replay_buffer"):
                setattr(agent, "replay_buffer", buf)
            else:
                setattr(agent, "buffer", buf)
        return buf

    def _set_agent_buffer(self, agent: MOPolicy, buffer: ReplayBuffer) -> None:
        """Assign a replay buffer to an agent, mirroring _get_agent_buffer logic."""
        if type(agent).set_buffer is not MOPolicy.set_buffer:
            agent.set_buffer(buffer)
        elif hasattr(agent, "replay_buffer"):
            setattr(agent, "replay_buffer", buffer)
        else:
            setattr(agent, "buffer", buffer)

    def _log_agent_losses(self, idx: int, agent: MOPolicy, commit: bool = True) -> None:
        """Log the latest losses for one agent to wandb and the fallback CSV."""
        if not self.log or not hasattr(agent, "last_losses") or not agent.last_losses:
            return
        import wandb

        prefixed = {f"agent_{idx}/{k}": v for k, v in agent.last_losses.items()}
        prefixed["global_step"] = self.global_step
        wandb.log(prefixed, commit=commit)

        if self.local_loss_files[idx]:
            with open(self.local_loss_files[idx], "a", newline="") as f:
                writer = csv.writer(f)
                for k, v in agent.last_losses.items():
                    writer.writerow([self.global_step, k, v])

    def _save_members(self, save_dir: str) -> None:
        """Overwrite the latest ensemble member checkpoints on disk."""
        os.makedirs(save_dir, exist_ok=True)
        for idx, agent in enumerate(self.agents):
            if not hasattr(agent, "save"):
                continue
            agent.save(save_dir=save_dir, filename=f"ensemble_latest_agent_{idx}", save_replay_buffer=False)
        marker_path = os.path.join(save_dir, "ensemble_latest_step.txt")
        with open(marker_path, "w", encoding="utf-8") as f:
            f.write(str(self.global_step))

    def close_wandb(self) -> None:
        """Upload per-agent CSVs as artifacts before finishing wandb."""
        if not self.log:
            return
        import wandb

        if self.local_loss_dir is not None and os.path.isdir(self.local_loss_dir):
            art = wandb.Artifact("ensemble_agent_losses", type="metrics")
            for path in self.local_loss_files:
                if path and os.path.exists(path):
                    art.add_file(path, name=os.path.relpath(path, self.local_loss_dir))
            wandb.log_artifact(art)
        wandb.finish()
