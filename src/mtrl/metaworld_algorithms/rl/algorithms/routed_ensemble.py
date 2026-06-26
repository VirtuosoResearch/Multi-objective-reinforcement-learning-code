import dataclasses
from pathlib import Path
from typing import Protocol, Self, override

import jax
import jax.flatten_util as flatten_util
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from flax import serialization, struct
from jaxtyping import Array, Float

from metaworld_algorithms.config.envs import EnvConfig
from metaworld_algorithms.config.networks import (
    ContinuousActionPolicyConfig,
    QValueFunctionConfig,
)
from metaworld_algorithms.config.optim import OptimizerConfig
from metaworld_algorithms.config.rl import AlgorithmConfig, OffPolicyTrainingConfig
from metaworld_algorithms.rl.algorithms.mtsac import MTSAC, MTSACConfig
from metaworld_algorithms.rl.algorithms.utils import TrainState
from metaworld_algorithms.rl.buffers import MultiTaskReplayBuffer
from metaworld_algorithms.types import Action, LogDict, Observation, ReplayBufferSamples

from .base import OffPolicyAlgorithm


class ScoreFn(Protocol):
    def __call__(self, grad_vector: jax.Array, grad_ema: jax.Array) -> jax.Array: ...


@dataclasses.dataclass(frozen=True)
class GradientSimilarityScore:
    eps: float = 1e-8

    def __call__(self, grad_vector: jax.Array, grad_ema: jax.Array) -> jax.Array:
        denom = jnp.linalg.norm(grad_vector) * jnp.linalg.norm(grad_ema) + self.eps
        return jnp.dot(grad_vector, grad_ema) / denom


@dataclasses.dataclass(frozen=True)
class RoutedEnsembleConfig(AlgorithmConfig):
    num_experts: int = 3
    actor_config: ContinuousActionPolicyConfig = ContinuousActionPolicyConfig()
    critic_config: QValueFunctionConfig = QValueFunctionConfig()
    temperature_optimizer_config: OptimizerConfig = OptimizerConfig(max_grad_norm=None)
    initial_temperature: float = 1.0
    num_critics: int = 2
    tau: float = 0.005
    use_task_weights: bool = False
    max_q_value: float | None = 5000

    lambda_mu: float = 0.1
    centroid_eta: float = 0.1
    grad_ema_beta: float = 0.1
    centroid_init: str = "random"
    score_type: str = "gradient_similarity"
    score_eps: float = 1e-8
    update_mode: str = "single_policy"
    routing_cache_steps: int = 1
    fixed_task_assignments: tuple[tuple[int, ...], ...] | None = None
    init_updates: int = 0
    init_policy_path: str | None = None
    save_init_policy_path: str | None = None
    max_assignment_fraction: float = 0.5


class RoutedEnsembleMTSAC(OffPolicyAlgorithm[RoutedEnsembleConfig]):
    # Ensemble of independent MTSAC experts with task-based routing and
    # optional routing cache for multi-step reuse of assignments.
    experts: tuple[MTSAC, ...]
    centroids: jax.Array
    grad_ema: jax.Array
    task_expert_map: npt.NDArray[np.int64] = struct.field(pytree_node=False)
    routing_cache_steps: int = struct.field(pytree_node=False)
    routing_cache_remaining: int = struct.field(pytree_node=False)
    cached_expert_batches: tuple[ReplayBufferSamples | None, ...] = struct.field(
        pytree_node=False
    )
    cached_assignment_counts: npt.NDArray[np.int64] = struct.field(pytree_node=False)
    cached_sample_counts: npt.NDArray[np.int64] = struct.field(pytree_node=False)

    num_experts: int = struct.field(pytree_node=False)
    lambda_mu: float = struct.field(pytree_node=False)
    centroid_eta: float = struct.field(pytree_node=False)
    grad_ema_beta: float = struct.field(pytree_node=False)
    score_fn: ScoreFn = struct.field(pytree_node=False)
    score_type: str = struct.field(pytree_node=False)
    update_mode: str = struct.field(pytree_node=False)
    init_updates_remaining: int = struct.field(pytree_node=False)
    init_policy_path: str | None = struct.field(pytree_node=False)
    save_init_policy_path: str | None = struct.field(pytree_node=False)
    init_policy_saved: bool = struct.field(pytree_node=False)
    sampling_key: jax.Array = struct.field(pytree_node=False)
    max_assignment_fraction: float = struct.field(pytree_node=False)
    eval_task_expert_map: npt.NDArray[np.int64] | None = struct.field(
        pytree_node=False
    )

    @override
    @staticmethod
    def initialize(
        config: RoutedEnsembleConfig, env_config: EnvConfig, seed: int = 1
    ) -> "RoutedEnsembleMTSAC":
        # Build per-expert MTSAC configs and initialize expert parameters.
        base_config = MTSACConfig(
            num_tasks=config.num_tasks,
            gamma=config.gamma,
            actor_config=config.actor_config,
            critic_config=config.critic_config,
            temperature_optimizer_config=config.temperature_optimizer_config,
            initial_temperature=config.initial_temperature,
            num_critics=config.num_critics,
            tau=config.tau,
            use_task_weights=config.use_task_weights,
            max_q_value=config.max_q_value,
        )

        rng = np.random.default_rng(seed)
        expert_seeds = rng.integers(0, 2**31 - 1, size=config.num_experts)
        experts = tuple(
            MTSAC.initialize(base_config, env_config, seed=int(expert_seed))
            for expert_seed in expert_seeds
        )
        init_updates_remaining = config.init_updates
        init_policy_saved = config.save_init_policy_path is None

        run_results_candidates = [
            Path("run_results"),
            Path(__file__).resolve().parents[4]
            / "mtrl"
            / "examples"
            / "multi_task"
            / "run_results",
        ]
        init_policy_paths: list[Path] = []
        for candidate in run_results_candidates:
            if candidate.is_dir():
                init_policy_paths = [
                    candidate / f"init_policy_{idx}.msgpack" for idx in range(1, 4)
                ]
                if all(path.is_file() for path in init_policy_paths):
                    break
                init_policy_paths = []

        if init_policy_paths:
            loaded_experts: list[MTSAC] = []
            for idx, path in enumerate(init_policy_paths):
                if idx >= config.num_experts:
                    break
                loaded_experts.append(
                    serialization.from_bytes(experts[idx], path.read_bytes())
                )
            if loaded_experts:
                experts = tuple(
                    loaded_experts
                    + list(experts[len(loaded_experts) : config.num_experts])
                )
                init_updates_remaining = 0
                print(
                    "Loaded init policies:",
                    ", ".join(str(path) for path in init_policy_paths[: len(loaded_experts)]),
                )
        elif config.init_policy_path is not None:
            init_path = Path(config.init_policy_path)
            if init_path.is_file():
                restored = serialization.from_bytes(
                    experts[0], init_path.read_bytes()
                )
                experts = tuple([restored] * config.num_experts)
                init_updates_remaining = 0
                print(f"Loaded init policy: {init_path}")
            else:
                print(
                    f"WARNING: init_policy_path not found: {config.init_policy_path}"
                )
        else:
            print("WARNING: init policies not found in run_results.")

        centroid_key = jax.random.PRNGKey(seed)
        # Initialize per-expert task preference centroids for routing.
        if config.centroid_init == "uniform":
            centroids = jnp.full(
                (config.num_experts, config.num_tasks),
                1.0 / config.num_tasks,
                dtype=jnp.float32,
            )
        elif config.centroid_init == "random":
            centroids = jax.random.dirichlet(
                centroid_key,
                alpha=jnp.ones((config.num_tasks,)),
                shape=(config.num_experts,),
            ).astype(jnp.float32)
        else:
            raise ValueError(f"Unknown centroid_init: {config.centroid_init}")

        # For centroid_only, override with fixed per-expert task preferences.
        if config.score_type == "centroid_only":
            fixed_centroids = np.zeros(
                (config.num_experts, config.num_tasks), dtype=np.float32
            )
            if config.fixed_task_assignments is None:
                base = config.num_tasks // config.num_experts
                remainder = config.num_tasks % config.num_experts
                task_idx = 0
                for expert_idx in range(config.num_experts):
                    k_i = base + (1 if expert_idx < remainder else 0)
                    if k_i == 0:
                        continue
                    assigned = np.arange(task_idx, task_idx + k_i)
                    fixed_centroids[expert_idx, assigned] = 1.0 / k_i
                    task_idx += k_i
            else:
                if len(config.fixed_task_assignments) != config.num_experts:
                    raise ValueError(
                        "fixed_task_assignments must have one list per expert."
                    )
                for expert_idx, task_ids in enumerate(
                    config.fixed_task_assignments
                ):
                    if len(task_ids) == 0:
                        continue
                    task_ids_array = np.asarray(task_ids, dtype=np.int64)
                    if np.any(task_ids_array < 0) or np.any(
                        task_ids_array >= config.num_tasks
                    ):
                        raise ValueError(
                            "fixed_task_assignments task id out of range."
                        )
                    k_i = int(task_ids_array.shape[0])
                    fixed_centroids[expert_idx, task_ids_array] = 1.0 / k_i
            centroids = jnp.asarray(fixed_centroids)

        # Select routing score function; some modes reuse gradient similarity scoring.
        if config.score_type == "gradient_similarity":
            score_fn: ScoreFn = GradientSimilarityScore(eps=config.score_eps)
        elif config.score_type in {
            "reward_rank",
            "centroid_only",
            "random_fixed",
            "reinforce",
        }:
            score_fn = GradientSimilarityScore(eps=config.score_eps)
        else:
            raise ValueError(f"Unknown score_type: {config.score_type}")

        # Fixed task-to-expert assignment for the random_fixed routing mode.
        task_expert_map = rng.integers(
            0, config.num_experts, size=config.num_tasks, dtype=np.int64
        )
        cached_expert_batches: tuple[ReplayBufferSamples | None, ...] = tuple(
            [None] * config.num_experts
        )
        cached_assignment_counts = np.zeros(config.num_experts, dtype=np.int64)
        cached_sample_counts = np.zeros(config.num_experts, dtype=np.int64)

        flat_params, _ = flatten_util.ravel_pytree(experts[0].actor.params)
        # Track per-expert EMA of policy gradients for similarity routing.
        grad_ema = jnp.zeros(
            (config.num_experts, flat_params.shape[0]), dtype=flat_params.dtype
        )

        sampling_key = jax.random.PRNGKey(seed + 1)

        return RoutedEnsembleMTSAC(
            num_tasks=config.num_tasks,
            gamma=config.gamma,
            experts=experts,
            centroids=centroids,
            grad_ema=grad_ema,
            task_expert_map=task_expert_map,
            routing_cache_steps=config.routing_cache_steps,
            routing_cache_remaining=0,
            cached_expert_batches=cached_expert_batches,
            cached_assignment_counts=cached_assignment_counts,
            cached_sample_counts=cached_sample_counts,
            num_experts=config.num_experts,
            lambda_mu=config.lambda_mu,
            centroid_eta=config.centroid_eta,
            grad_ema_beta=config.grad_ema_beta,
            score_fn=score_fn,
            score_type=config.score_type,
            update_mode=config.update_mode,
            init_updates_remaining=init_updates_remaining,
            init_policy_path=config.init_policy_path,
            save_init_policy_path=config.save_init_policy_path,
            init_policy_saved=init_policy_saved,
            sampling_key=sampling_key,
            max_assignment_fraction=config.max_assignment_fraction,
            eval_task_expert_map=None,
        )

    @override
    def spawn_replay_buffer(
        self, env_config: EnvConfig, config: OffPolicyTrainingConfig, seed: int = 1
    ) -> MultiTaskReplayBuffer:
        return MultiTaskReplayBuffer(
            total_capacity=config.buffer_size,
            num_tasks=self.num_tasks,
            env_obs_space=env_config.observation_space,
            env_action_space=env_config.action_space,
            seed=seed,
        )

    @override
    def get_num_params(self) -> dict[str, int]:
        totals: dict[str, int] = {
            "actor_num_params": 0,
            "critic_num_params": 0,
        }
        for expert in self.experts:
            params = expert.get_num_params()
            totals["actor_num_params"] += params["actor_num_params"]
            totals["critic_num_params"] += params["critic_num_params"]
        return totals

    def _select_experts(
        self, task_ids: Float[npt.NDArray, "task num_tasks"]
    ) -> npt.NDArray[np.int64]:
        # Route each task to an expert using fixed mapping or centroid proximity.
        if self.score_type == "random_fixed":
            task_indices = np.asarray(np.argmax(task_ids, axis=1))
            return self.task_expert_map[task_indices]
        distances = jnp.linalg.norm(
            task_ids[:, None, :] - self.centroids[None, :, :], axis=-1
        )
        return np.asarray(jax.device_get(jnp.argmin(distances, axis=1)))

    def _compute_policy_grad_vector(
        self,
        actor: TrainState,
        observations: jax.Array,
        actions: jax.Array,
    ) -> jax.Array:
        # Flattened policy-gradient vector from negative log-likelihood loss.
        def loss_fn(params) -> jax.Array:
            dist = actor.apply_fn(params, observations)
            log_prob = dist.log_prob(actions)
            if log_prob.ndim > 1:
                log_prob = log_prob.sum(axis=-1)
            return -jnp.mean(log_prob)

        grads = jax.grad(loss_fn)(actor.params)
        flat_grads, _ = flatten_util.ravel_pytree(grads)
        return flat_grads

    def _compute_reward_score(
        self, expert: MTSAC, observations: jax.Array
    ) -> jax.Array:
        # Score expert by expected minimum Q value under its greedy action.
        dist = expert.actor.apply_fn(expert.actor.params, observations)
        actions = dist.mode()
        q_values = expert.critic.apply_fn(expert.critic.params, observations, actions)
        min_q_values = jnp.min(q_values, axis=0)
        return jnp.mean(min_q_values)

    def _compute_sample_values(
        self, expert: MTSAC, observations: jax.Array, actions: jax.Array
    ) -> jax.Array:
        q_values = expert.critic.apply_fn(expert.critic.params, observations, actions)
        return jnp.min(q_values, axis=0)

    @staticmethod
    @jax.jit
    def _compute_expert_values(
        experts: tuple[MTSAC, ...],
        observations: jax.Array,
        actions: jax.Array,
    ) -> jax.Array:
        def expert_value(expert: MTSAC, expert_actions: jax.Array) -> jax.Array:
            q_values = expert.critic.apply_fn(
                expert.critic.params, observations, expert_actions
            )
            return jnp.min(q_values, axis=0)

        if actions.ndim == 2:
            return jnp.stack(
                [expert_value(expert, actions) for expert in experts], axis=0
            )
        return jnp.stack(
            [
                expert_value(expert, actions[idx])
                for idx, expert in enumerate(experts)
            ],
            axis=0,
        )

    def _assign_samples_by_value(
        self, values: npt.NDArray[np.floating]
    ) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64]]:
        num_experts, batch_size = values.shape
        max_count = max(1, int(np.floor(self.max_assignment_fraction * batch_size)))
        preferences = np.argsort(-values, axis=0)
        if num_experts > 1:
            top_values = values[preferences[0], np.arange(batch_size)]
            second_values = values[preferences[1], np.arange(batch_size)]
            priorities = top_values - second_values
        else:
            priorities = values[preferences[0], np.arange(batch_size)]
        order = np.argsort(-priorities)

        assignments = np.full(batch_size, -1, dtype=np.int64)
        counts = np.zeros(num_experts, dtype=np.int64)

        for idx in order:
            for rank in range(num_experts):
                expert_idx = int(preferences[rank, idx])
                if counts[expert_idx] < max_count:
                    assignments[idx] = expert_idx
                    counts[expert_idx] += 1
                    break
            if assignments[idx] == -1:
                expert_idx = int(preferences[0, idx])
                assignments[idx] = expert_idx
                counts[expert_idx] += 1

        return assignments, counts

    def _compute_log_prob_scores(
        self,
        expert: MTSAC,
        observations: jax.Array,
        actions: jax.Array,
    ) -> jax.Array:
        dist = expert.actor.apply_fn(expert.actor.params, observations)
        log_prob = dist.log_prob(actions)
        if log_prob.ndim > 1:
            log_prob = log_prob.sum(axis=-1)
        return log_prob

    def _maybe_save_init_policy(self, expert: MTSAC) -> bool:
        if self.save_init_policy_path is None or self.init_policy_saved:
            return self.init_policy_saved
        save_path = Path(self.save_init_policy_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_bytes(serialization.to_bytes(expert))
        return True

    def _split_data_by_tasks(
        self,
        data: ReplayBufferSamples,
        task_ids: Float[Array, "batch num_tasks"],
    ) -> ReplayBufferSamples:
        # Group batch data by task index; assumes task ids are one-hot.
        tasks = jnp.argmax(task_ids, axis=1)
        sorted_indices = jnp.argsort(tasks)

        def group_by_task_leaf(
            leaf: Float[Array, "batch data_dim"],
        ) -> Float[Array, "task task_batch data_dim"]:
            leaf_sorted = leaf[sorted_indices]
            return leaf_sorted.reshape(self.num_tasks, -1, leaf.shape[1])

        return jax.tree.map(group_by_task_leaf, data)

    @override
    def sample_action(self, observation: Observation) -> tuple[Self, Action]:
        if self.init_updates_remaining > 0:
            experts = list(self.experts)
            experts[0], actions = experts[0].sample_action(observation)
            return self.replace(experts=tuple(experts)), actions

        sampling_key, expert_key = jax.random.split(self.sampling_key)
        expert_idx = int(
            jax.device_get(jax.random.randint(expert_key, (), 0, self.num_experts))
        )
        experts = list(self.experts)
        experts[expert_idx], actions = experts[expert_idx].sample_action(observation)
        return (
            self.replace(experts=tuple(experts), sampling_key=sampling_key),
            actions,
        )

    @override
    def eval_action(self, observations: Observation) -> Action:
        if self.init_updates_remaining > 0:
            return self.experts[0].eval_action(observations)

        if self.eval_task_expert_map is not None:
            task_ids = observations[..., -self.num_tasks :]
            if observations.ndim == 1:
                task_indices = np.array([int(np.argmax(task_ids))])
            else:
                task_indices = np.asarray(np.argmax(task_ids, axis=1))
            expert_indices = self.eval_task_expert_map[task_indices]
            actions_by_expert = np.stack(
                [expert.eval_action(observations) for expert in self.experts], axis=0
            )
            num_envs = observations.shape[0] if observations.ndim > 1 else 1
            selected_actions = actions_by_expert[expert_indices, np.arange(num_envs)]
            if observations.ndim == 1:
                return selected_actions.reshape(-1)
            return selected_actions

        actions_by_expert = np.stack(
            [expert.eval_action(observations) for expert in self.experts], axis=0
        )
        obs = jnp.asarray(observations)
        values = self._compute_expert_values(
            self.experts, obs, jnp.asarray(actions_by_expert)
        )
        if values.ndim > 2:
            values = jnp.squeeze(values, axis=-1)
        expert_indices = np.asarray(jax.device_get(jnp.argmax(values, axis=0)))
        if expert_indices.ndim > 1:
            expert_indices = np.squeeze(expert_indices, axis=-1)

        num_envs = observations.shape[0] if observations.ndim > 1 else 1
        selected_actions = actions_by_expert[expert_indices, np.arange(num_envs)]
        if observations.ndim == 1:
            return selected_actions.reshape(-1)
        return selected_actions

    @override
    def update(self, data: ReplayBufferSamples) -> tuple[Self, LogDict]:
        # Train experts on batches assigned by value-based routing.
        experts = list(self.experts)
        centroids = self.centroids
        grad_ema = self.grad_ema

        if self.init_updates_remaining > 0:
            experts[0], expert_logs = experts[0].update(data)
            init_updates_remaining = self.init_updates_remaining - 1
            logs = {
                f"pretrain/{key}": float(jax.device_get(value))
                for key, value in expert_logs.items()
            }
            logs["pretrain/steps_remaining"] = float(init_updates_remaining)
            init_policy_saved = self.init_policy_saved
            if init_updates_remaining == 0:
                base_expert = experts[0]
                experts = [base_expert] * self.num_experts
                grad_ema = jnp.zeros_like(self.grad_ema)
                init_policy_saved = self._maybe_save_init_policy(base_expert)
            return (
                self.replace(
                    experts=tuple(experts),
                    centroids=centroids,
                    grad_ema=grad_ema,
                    init_updates_remaining=init_updates_remaining,
                    init_policy_saved=init_policy_saved,
                ),
                logs,
            )

        obs = jnp.asarray(data.observations)
        actions = jnp.asarray(data.actions)
        value_matrix = self._compute_expert_values(self.experts, obs, actions)
        values = np.asarray(jax.device_get(value_matrix))
        if values.ndim > 2:
            values = np.squeeze(values, axis=-1)
        assignments, sample_counts = self._assign_samples_by_value(values)

        expert_batches: list[ReplayBufferSamples | None] = []
        for expert_idx in range(self.num_experts):
            sample_mask = assignments == expert_idx
            count = int(np.sum(sample_mask))
            if count == 0:
                expert_batches.append(None)
                continue
            expert_batches.append(
                ReplayBufferSamples(
                    observations=data.observations[sample_mask],
                    actions=data.actions[sample_mask],
                    next_observations=data.next_observations[sample_mask],
                    dones=data.dones[sample_mask],
                    rewards=data.rewards[sample_mask],
                )
            )

        logs_weighted: dict[str, float] = {}
        total_weight = 0
        for expert_idx, expert in enumerate(experts):
            expert_data = expert_batches[expert_idx]
            if expert_data is None:
                continue
            experts[expert_idx], expert_logs = expert.update(expert_data)
            weight = int(sample_counts[expert_idx])
            total_weight += weight
            for key, value in expert_logs.items():
                logs_weighted[key] = logs_weighted.get(
                    key, 0.0
                ) + float(jax.device_get(value)) * weight

        logs = (
            {key: value / total_weight for key, value in logs_weighted.items()}
            if total_weight > 0
            else {}
        )
        for expert_idx, count in enumerate(sample_counts):
            logs[f"routing/expert_{expert_idx}_count"] = float(count)
            logs[f"routing/expert_{expert_idx}_fraction"] = float(
                count / max(1, data.observations.shape[0])
            )
        logs["routing/max_assignment_fraction"] = float(
            self.max_assignment_fraction
        )

        return (
            self.replace(
                experts=tuple(experts),
                centroids=centroids,
                grad_ema=grad_ema,
                routing_cache_remaining=0,
                cached_expert_batches=tuple([None] * self.num_experts),
                cached_assignment_counts=sample_counts,
                cached_sample_counts=sample_counts,
            ),
            logs,
        )
