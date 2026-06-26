from typing import Sequence

import numpy as np

from morl_baselines.common.utils import linearly_decaying_value
from morl_baselines.common.weights import random_weights
from morl_baselines.multi_policy.pareto_q_learning.pql import PQL


class PreferenceGuidedPQL(PQL):
    """PQL member that uses a fixed preference to score action Q-sets."""

    def __init__(self, *args, preference: np.ndarray, **kwargs):
        super().__init__(*args, **kwargs)
        self.preference = np.asarray(preference, dtype=np.float64)

    def score_hypervolume(self, state: int):
        q_sets = [self.get_q_set(state, action) for action in range(self.num_actions)]
        return [max(float(np.dot(np.asarray(vec), self.preference)) for vec in q_set) for q_set in q_sets]


def train_routed_pql_ensemble(
    members: Sequence[PreferenceGuidedPQL],
    rollout_env,
    total_timesteps: int,
    seed: int,
) -> None:
    """Train PQL members with lightweight trajectory routing.

    Each iteration samples a preference, rolls out one member chosen uniformly
    at random, scores the observed states with every member's current Q-sets,
    and replays the trajectory into the highest-scoring member.
    """

    rng = np.random.default_rng(seed)
    reward_dim = members[0].num_objectives
    global_step = 0

    while global_step < total_timesteps:
        preference = random_weights(reward_dim, rng=rng)
        rollout_member = members[int(rng.integers(len(members)))]
        transitions, states = _rollout_episode(rollout_member, rollout_env, total_timesteps - global_step)
        if not transitions:
            break

        scores = [_score_trajectory(member, states, preference) for member in members]
        routed_member = members[int(np.argmax(scores))]
        _replay_transitions(routed_member, transitions)

        global_step += len(transitions)
        for member in members:
            member.global_step = global_step
            member.epsilon = linearly_decaying_value(
                member.initial_epsilon,
                member.epsilon_decay_steps,
                member.global_step,
                0,
                member.final_epsilon,
            )


def _rollout_episode(member: PreferenceGuidedPQL, env, max_steps: int):
    transitions = []
    states = []
    obs, _ = env.reset()
    state = _state_index(member, obs)
    terminated = False
    truncated = False

    while not (terminated or truncated) and len(transitions) < max_steps:
        action = member.select_action(state, member.score_hypervolume)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        next_state = _state_index(member, next_obs)
        transitions.append((state, action, np.asarray(reward, dtype=np.float64), next_state))
        states.append(state)
        state = next_state

    return transitions, states


def _score_trajectory(member: PreferenceGuidedPQL, states: Sequence[int], preference: np.ndarray) -> float:
    score = 0.0
    for state in states:
        action_scores = []
        for action in range(member.num_actions):
            q_set = member.get_q_set(state, action)
            action_scores.append(max(float(np.dot(np.asarray(vec), preference)) for vec in q_set))
        score += max(action_scores)
    return score


def _replay_transitions(member: PreferenceGuidedPQL, transitions) -> None:
    for state, action, reward, next_state in transitions:
        member.counts[state, action] += 1
        member.non_dominated[state][action] = member.calc_non_dominated(next_state)
        member.avg_reward[state, action] += (reward - member.avg_reward[state, action]) / member.counts[state, action]


def _state_index(member: PreferenceGuidedPQL, obs) -> int:
    return int(np.ravel_multi_index(obs, member.env_shape))
