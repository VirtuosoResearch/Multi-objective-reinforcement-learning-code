import dataclasses
from typing import Protocol, Self, override

import jax
import jax.flatten_util as flatten_util
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt
from flax import struct
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
    num_experts: int = 4
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
        elif config.score_type in {"reward_rank", "centroid_only", "random_fixed"}:
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
        # Sample actions from all experts, then select by routed expert per task.
        task_ids = observation[..., -self.num_tasks :]
        expert_indices = self._select_experts(task_ids)

        new_experts = []
        all_actions = []
        for expert in self.experts:
            expert, actions = expert.sample_action(observation)
            new_experts.append(expert)
            all_actions.append(actions)

        actions_by_expert = np.stack(all_actions, axis=0)
        selected_actions = actions_by_expert[expert_indices, np.arange(self.num_tasks)]
        return self.replace(experts=tuple(new_experts)), selected_actions

    @override
    def eval_action(self, observations: Observation) -> Action:
        # Deterministic action selection routed by task-specific expert.
        task_ids = observations[..., -self.num_tasks :]
        expert_indices = self._select_experts(task_ids)

        actions_by_expert = np.stack(
            [expert.eval_action(observations) for expert in self.experts], axis=0
        )
        return actions_by_expert[expert_indices, np.arange(self.num_tasks)]

    @override
    def update(self, data: ReplayBufferSamples) -> tuple[Self, LogDict]:
        # Train experts either on shared data (single_policy) or routed data.
        experts = list(self.experts)
        centroids = self.centroids
        grad_ema = self.grad_ema
        logs_accum: dict[str, list[float]] = {}

        if self.update_mode == "single_policy":
            # Update every expert on the same batch; no routing.
            for expert_idx, expert in enumerate(experts):
                experts[expert_idx], expert_logs = expert.update(data)
                for key, value in expert_logs.items():
                    logs_accum.setdefault(key, []).append(
                        float(jax.device_get(value))
                    )
            logs = {key: float(np.mean(values)) for key, values in logs_accum.items()}
            centroid_values = np.asarray(jax.device_get(centroids))
            for expert_idx in range(self.num_experts):
                for task_idx in range(self.num_tasks):
                    logs[
                        f"routing/preference_expert_{expert_idx}_task_{task_idx}"
                    ] = float(centroid_values[expert_idx, task_idx])
            return (
                self.replace(
                    experts=tuple(experts),
                    centroids=centroids,
                    grad_ema=grad_ema,
                ),
                logs,
            )

        if self.update_mode != "routed":
            raise ValueError(f"Unknown update_mode: {self.update_mode}")

        # Optionally reuse cached routing assignments for multiple steps.
        use_cache = self.routing_cache_steps > 1 and self.routing_cache_remaining > 0
        assignment_counts = self.cached_assignment_counts
        sample_counts = self.cached_sample_counts
        expert_batches = list(self.cached_expert_batches)

        if not use_cache:
            # Split batch by task and assign each task to the best expert.
            task_ids = data.observations[..., -self.num_tasks :]
            split_data = self._split_data_by_tasks(data, task_ids)

            assignments = np.zeros(self.num_tasks, dtype=np.int64)

            for task_idx in range(self.num_tasks):
                task_data = ReplayBufferSamples(
                    observations=split_data.observations[task_idx],
                    actions=split_data.actions[task_idx],
                    next_observations=split_data.next_observations[task_idx],
                    dones=split_data.dones[task_idx],
                    rewards=split_data.rewards[task_idx],
                )
                obs = jnp.asarray(task_data.observations)
                actions = jnp.asarray(task_data.actions)

                grad_vectors = None
                if self.score_type == "random_fixed":
                    expert_idx = int(self.task_expert_map[task_idx])
                    scores = None
                elif self.score_type == "gradient_similarity":
                    # Prefer experts whose gradient aligns with their EMA.
                    grad_vectors = []
                    scores = []
                    for expert_idx, expert in enumerate(experts):
                        grad_vector = self._compute_policy_grad_vector(
                            expert.actor, obs, actions
                        )
                        grad_vectors.append(grad_vector)
                        scores.append(self.score_fn(grad_vector, grad_ema[expert_idx]))
                    scores = jnp.stack(scores)
                elif self.score_type == "reward_rank":
                    # Prefer experts with higher expected min-Q on task data.
                    scores = jnp.stack(
                        [self._compute_reward_score(expert, obs) for expert in experts]
                    )
                elif self.score_type == "centroid_only":
                    # Ignore scores and route solely by centroid proximity.
                    scores = jnp.zeros((self.num_experts,), dtype=centroids.dtype)
                else:
                    raise ValueError(f"Unknown score_type: {self.score_type}")

                w_t = jax.nn.one_hot(task_idx, self.num_tasks)
                if scores is not None:
                    # Combine task-score with a centroid penalty to discourage drift.
                    centroid_penalty = -self.lambda_mu * jnp.sum(
                        (centroids - w_t) ** 2, axis=1
                    )
                    total_scores = scores + centroid_penalty
                    expert_idx = int(jax.device_get(jnp.argmax(total_scores)))
                assignments[task_idx] = expert_idx

                # Update routed expert's centroid and gradient EMA.
                if self.score_type != "centroid_only":
                    centroids = centroids.at[expert_idx].set(
                        (1 - self.centroid_eta) * centroids[expert_idx]
                        + self.centroid_eta * w_t
                    )
                if grad_vectors is not None:
                    grad_ema = grad_ema.at[expert_idx].set(
                        (1 - self.grad_ema_beta) * grad_ema[expert_idx]
                        + self.grad_ema_beta * grad_vectors[expert_idx]
                    )

            # Dispatch each sample to the expert assigned to its task.
            sample_task_ids = np.argmax(np.asarray(task_ids), axis=1)
            expert_batches = []
            sample_counts = np.zeros(self.num_experts, dtype=np.int64)
            for expert_idx in range(self.num_experts):
                sample_mask = assignments[sample_task_ids] == expert_idx
                count = int(np.sum(sample_mask))
                sample_counts[expert_idx] = count
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

            assignment_counts = np.bincount(assignments, minlength=self.num_experts)
            if self.routing_cache_steps > 1:
                # Cache routing decisions to reuse for the next steps.
                routing_cache_remaining = self.routing_cache_steps - 1
                cached_expert_batches = tuple(expert_batches)
                cached_assignment_counts = assignment_counts
                cached_sample_counts = sample_counts
            else:
                routing_cache_remaining = 0
                cached_expert_batches = tuple([None] * self.num_experts)
                cached_assignment_counts = assignment_counts
                cached_sample_counts = sample_counts
        else:
            # Reuse cached routing decisions and expert-specific batches.
            routing_cache_remaining = self.routing_cache_remaining - 1
            cached_expert_batches = self.cached_expert_batches
            cached_assignment_counts = self.cached_assignment_counts
            cached_sample_counts = self.cached_sample_counts

        # Update each expert on its routed sub-batch and weight logs by samples.
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
        for expert_idx, count in enumerate(assignment_counts):
            logs[f"routing/expert_{expert_idx}_count"] = float(count)
        centroid_values = np.asarray(jax.device_get(centroids))
        for expert_idx in range(self.num_experts):
            for task_idx in range(self.num_tasks):
                logs[
                    f"routing/preference_expert_{expert_idx}_task_{task_idx}"
                ] = float(centroid_values[expert_idx, task_idx])

        return (
            self.replace(
                experts=tuple(experts),
                centroids=centroids,
                grad_ema=grad_ema,
                routing_cache_remaining=routing_cache_remaining,
                cached_expert_batches=cached_expert_batches,
                cached_assignment_counts=cached_assignment_counts,
                cached_sample_counts=cached_sample_counts,
            ),
            logs,
        )
