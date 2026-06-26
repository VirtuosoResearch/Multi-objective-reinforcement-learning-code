"""Discrete finite-horizon (H=2) multi-objective MDP utilities for offline learning.

States and actions are discrete with three possible values each (default: -1, 0, 1).
Rewards are 2-dimensional and intentionally conflicting: actions that increase
objective 1 tend to decrease objective 2 and vice versa. This makes the optimal
policy depend on the preference vector w. Policies can be stochastic with a minimum
exploration probability per action.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np

PolicyFn = Callable[[int, int, Optional[np.random.Generator]], int]


@dataclass
class Trajectory:
    """Container for a single horizon-2 trajectory."""

    s0: int
    a1: int
    s1: int
    r1: np.ndarray  # shape: (2,)
    a2: int
    s2: int
    r2: np.ndarray  # shape: (2,)


class FiniteHorizonMultiObjectiveMDP:
    """Finite-horizon discrete MDP with multi-objective rewards."""

    def __init__(
        self,
        states: Optional[Iterable[int]] = None,
        actions: Optional[Iterable[int]] = None,
        horizon: int = 2,
        reward_std: float = 0.0,
        slip_prob: float = 0.1,
        seed: Optional[int] = 0,
    ) -> None:
        if horizon != 2:
            raise ValueError("This helper targets horizon H=2.")
        self.states = np.array(list(states) if states is not None else [-1, 0, 1])
        self.actions = np.array(list(actions) if actions is not None else [-1, 0, 1])
        self.H = horizon
        self.M = 2
        self.reward_std = float(reward_std)
        self.slip_prob = float(slip_prob)
        self.rng = np.random.default_rng(seed)

    def sample_initial_state(self, rng: Optional[np.random.Generator] = None) -> int:
        rng = rng or self.rng
        return int(rng.choice(self.states))

    def _reward_mean(self, state: int, action: int, t: int) -> np.ndarray:
        """Conflicting rewards: positive actions help obj1, negative actions help obj2."""
        # Slight state effect so the best action can depend on s and preference.
        bias = 0.2 * state
        obj1 = 1.5 * (action == 1) + 1.2 * (action == 0) + 0.2 * (action == -1) + bias
        obj2 = 1.5 * (action == -1) + 1.2 * (action == 0) + 0.2 * (action == 1) - bias
        if t == 1:
            obj1 += 0.2  # small bonus at the last step for variance
            obj2 += 0.2
        return np.array([obj1, obj2], dtype=float)

    def step(
        self,
        state: int,
        action: int,
        t: int,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[int, np.ndarray]:
        """Sample next state and reward for step t (t=0 or 1)."""
        if t >= self.H:
            raise ValueError(f"t={t} must be < horizon={self.H}")
        rng = rng or self.rng

        base_next = int(np.clip(state + action, self.states.min(), self.states.max()))
        if rng.random() < self.slip_prob:
            next_state = int(rng.choice(self.states))
        else:
            next_state = base_next

        reward_mean = self._reward_mean(state, action, t)
        reward = reward_mean + rng.normal(0.0, self.reward_std, size=self.M)
        return next_state, reward

    def rollout(
        self,
        policy: PolicyFn,
        rng: Optional[np.random.Generator] = None,
    ) -> Trajectory:
        """Simulate a single horizon-2 trajectory with a policy a_{h+1} = pi(s_h)."""
        rng = rng or self.rng
        s0 = self.sample_initial_state(rng)
        a1 = int(policy(s0, 0, rng))
        s1, r1 = self.step(s0, a1, 0, rng)
        a2 = int(policy(s1, 1, rng))
        s2, r2 = self.step(s1, a2, 1, rng)
        return Trajectory(s0=s0, a1=a1, s1=s1, r1=r1, a2=a2, s2=s2, r2=r2)


def evaluate_policy_value(
    mdp: FiniteHorizonMultiObjectiveMDP,
    policy: PolicyFn,
    w: Sequence[float],
    num_rollouts: int = 1_000,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Monte Carlo estimate of V^pi(s0; w) = E[w^T (r1 + r2)]."""
    rng = rng or np.random.default_rng()
    w_vec = np.asarray(w, dtype=float)
    returns = []
    for _ in range(num_rollouts):
        traj_rng = np.random.default_rng(rng.integers(0, 1_000_000))
        traj = mdp.rollout(policy, traj_rng)
        total = float(np.dot(w_vec, traj.r1 + traj.r2))
        returns.append(total)
    return float(np.mean(returns))


def estimate_q_value(
    mdp: FiniteHorizonMultiObjectiveMDP,
    policy: PolicyFn,
    state: int,
    action: int,
    w: Sequence[float],
    num_rollouts: int = 500,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Monte Carlo estimate of Q^pi(s, a; w) for the first decision step."""
    rng = rng or np.random.default_rng()
    w_vec = np.asarray(w, dtype=float)
    returns = []
    for _ in range(num_rollouts):
        traj_rng = np.random.default_rng(rng.integers(0, 1_000_000))
        s1, r1 = mdp.step(state, action, 0, traj_rng)
        a2 = int(policy(s1, 1, traj_rng))
        s2, r2 = mdp.step(s1, a2, 1, traj_rng)
        total = float(np.dot(w_vec, r1 + r2))
        returns.append(total)
    return float(np.mean(returns))


def generate_dataset(
    mdp: FiniteHorizonMultiObjectiveMDP,
    num_trajectories: int = 2_000,
    behavior_policy: Optional[PolicyFn] = None,
    seed: Optional[int] = 0,
    behavior_min_prob: float = 0.05,
) -> List[Trajectory]:
    """Generate a synthetic offline dataset D of horizon-2 trajectories."""
    rng = np.random.default_rng(seed)

    if behavior_policy is None:
        # Mildly state-biased but exploratory policy with probability floor epsilon/|A|.
        def behavior_policy(state: int, t: int, rng: np.random.Generator) -> int:
            logits = np.array([0.4, 0.2, -0.4]) + 0.3 * state
            base = np.exp(logits - logits.max())
            base = base / base.sum()
            eps = float(behavior_min_prob)
            uniform = np.ones_like(base) / base.size
            probs = (1.0 - eps) * base + eps * uniform
            probs = probs / probs.sum()
            return int(rng.choice(mdp.actions, p=probs))

    dataset: List[Trajectory] = []
    for _ in range(num_trajectories):
        traj_rng = np.random.default_rng(rng.integers(0, 1_000_000))
        traj = mdp.rollout(behavior_policy, traj_rng)
        dataset.append(traj)
    return dataset


def _init_tables(states: np.ndarray, actions: np.ndarray):
    shape = (len(states), len(actions))
    reward_sum = np.zeros(shape + (2,), dtype=float)
    counts = np.zeros(shape, dtype=float)
    next_counts = np.zeros(shape + (len(states),), dtype=float)
    return reward_sum, counts, next_counts


def learn_policy_from_dataset(
    dataset: Sequence[Trajectory],
    w: Sequence[float],
    reg: float = 1e-3,
    min_prob: float = 0.05,
    temperature: float = 0.7,
) -> Tuple[PolicyFn, dict]:
    """Estimate model from D and compute a stochastic policy for preference vector w.

    Each state's action distribution is softened (softmax temperature) and then mixed
    with uniform so every action has probability at least `min_prob / |A|`.
    """
    if not dataset:
        raise ValueError("Dataset is empty.")
    if not 0.0 <= min_prob < 1.0:
        raise ValueError("min_prob must be in [0, 1).")
    w_vec = np.asarray(w, dtype=float)
    states = np.array([-1, 0, 1])
    actions = np.array([-1, 0, 1])
    s_to_idx = {s: i for i, s in enumerate(states)}
    a_to_idx = {a: i for i, a in enumerate(actions)}

    r1_sum, c1, p1 = _init_tables(states, actions)
    r2_sum, c2, p2 = _init_tables(states, actions)

    for traj in dataset:
        s0i, a1i, s1i = s_to_idx[traj.s0], a_to_idx[traj.a1], s_to_idx[traj.s1]
        s1i2, a2i, s2i = s_to_idx[traj.s1], a_to_idx[traj.a2], s_to_idx[traj.s2]

        r1_sum[s0i, a1i] += traj.r1
        c1[s0i, a1i] += 1
        p1[s0i, a1i, s1i] += 1

        r2_sum[s1i2, a2i] += traj.r2
        c2[s1i2, a2i] += 1
        p2[s1i2, a2i, s2i] += 1

    def mean_with_reg(sum_arr: np.ndarray, counts: np.ndarray) -> np.ndarray:
        expanded = counts[..., None]
        return (sum_arr + reg) / (expanded + reg * sum_arr.shape[-1])

    r1_mean = mean_with_reg(r1_sum, c1)
    r2_mean = mean_with_reg(r2_sum, c2)

    def normalize_transitions(counts: np.ndarray) -> np.ndarray:
        probs = counts + reg
        probs = probs / probs.sum(axis=-1, keepdims=True)
        return probs

    p1_probs = normalize_transitions(p1)

    def _softmax(scores: np.ndarray, temp: float) -> np.ndarray:
        temp = max(temp, 1e-6)
        scaled = scores / temp
        scaled -= scaled.max()
        exps = np.exp(scaled)
        return exps / exps.sum()

    def _mix_with_uniform(probs: np.ndarray, eps: float) -> np.ndarray:
        uniform = np.ones_like(probs) / probs.size
        mixed = (1.0 - eps) * probs + eps * uniform
        return mixed / mixed.sum()

    # Stage-2 action distribution and expected value per state.
    stage2_probs = np.zeros((len(states), len(actions)), dtype=float)
    stage2_value = np.zeros(len(states), dtype=float)
    for si, s in enumerate(states):
        scores = np.array([float(np.dot(w_vec, r2_mean[si, ai])) for ai in range(len(actions))])
        probs = _softmax(scores, temperature)
        probs = _mix_with_uniform(probs, min_prob)
        stage2_probs[si] = probs
        stage2_value[si] = float(np.dot(probs, scores))

    # Stage-1 action distribution uses expected stage-2 value.
    stage1_probs = np.zeros((len(states), len(actions)), dtype=float)
    for si, s in enumerate(states):
        scores = []
        for ai, a in enumerate(actions):
            immediate = float(np.dot(w_vec, r1_mean[si, ai]))
            cont = float(np.dot(p1_probs[si, ai], stage2_value))
            scores.append(immediate + cont)
        scores = np.array(scores, dtype=float)
        probs = _softmax(scores, temperature)
        probs = _mix_with_uniform(probs, min_prob)
        stage1_probs[si] = probs

    default_rng = np.random.default_rng()

    def _sample(probs: np.ndarray, rng: Optional[np.random.Generator]) -> int:
        rng = rng or default_rng
        return int(rng.choice(actions, p=probs))

    def policy(state: int, t: int, rng: Optional[np.random.Generator] = None) -> int:
        si = s_to_idx[state]
        if t == 0:
            return _sample(stage1_probs[si], rng)
        return _sample(stage2_probs[si], rng)

    info = {
        "states": states,
        "actions": actions,
        "stage1_probs": stage1_probs,
        "stage2_probs": stage2_probs,
        "temperature": temperature,
        "min_prob": min_prob,
    }
    return policy, info


__all__ = [
    "FiniteHorizonMultiObjectiveMDP",
    "Trajectory",
    "evaluate_policy_value",
    "estimate_q_value",
    "generate_dataset",
    "learn_policy_from_dataset",
]
