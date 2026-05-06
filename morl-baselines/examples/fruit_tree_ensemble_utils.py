import argparse
from typing import Iterable, List, Optional, Sequence

import mo_gymnasium as mo_gym
import numpy as np
import wandb
from mo_gymnasium.wrappers import MORecordEpisodeStatistics

from morl_baselines.common.evaluation import log_all_multi_policy_metrics, policy_evaluation_mo
from morl_baselines.common.morl_algorithm import MOPolicy
from morl_baselines.common.pareto import filter_pareto_dominated
from morl_baselines.common.weights import equally_spaced_weights


ENV_ID = "fruit-tree-v0"


def fruit_tree_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--ensemble-size", type=int, default=3, help="Number of independently trained members.")
    parser.add_argument("--timesteps", type=int, default=int(5e6), help="Training timesteps per ensemble member.")
    parser.add_argument("--eval-episodes", type=int, default=1, help="Evaluation episodes per reported weight.")
    parser.add_argument("--metric-weights", type=int, default=50, help="Number of sampled weights for EUM/MUL metrics.")
    parser.add_argument("--seed", type=int, default=1, help="Base random seed.")
    parser.add_argument("--log", default=True, action="store_true", help="Enable ensemble-level wandb logging.")
    parser.add_argument("--no-log", dest="log", action="store_false", help="Disable ensemble-level wandb logging.")
    parser.add_argument("--member-log", default=True, action="store_true", help="Enable wandb logging inside each member algorithm.")
    parser.add_argument("--no-member-log", dest="member_log", action="store_false", help="Disable member-level wandb logging.")
    parser.add_argument("--project-name", type=str, default="MORL-Baselines", help="wandb project name.")
    parser.add_argument("--wandb-entity", type=str, default=None, help="wandb entity.")
    return parser


def make_env(record_stats: bool = False, gamma: float = 0.99):
    env = mo_gym.make(ENV_ID)
    if record_stats:
        env = MORecordEpisodeStatistics(env, gamma=gamma)
    return env


def reward_dim() -> int:
    env = make_env()
    dim = env.unwrapped.reward_space.shape[0]
    env.close()
    return dim


def ref_point(dim: Optional[int] = None) -> np.ndarray:
    if dim is None:
        dim = reward_dim()
    return np.zeros(dim)


def member_preferences(dim: int, ensemble_size: int) -> List[np.ndarray]:
    return [np.asarray(w, dtype=np.float64) for w in equally_spaced_weights(dim=dim, n=ensemble_size)]


def evaluate_members(members: Sequence[MOPolicy], preferences: Sequence[np.ndarray], gamma: float, eval_episodes: int):
    evaluations = []
    for member, preference in zip(members, preferences):
        eval_env = make_env()
        _, _, _, discounted_return = policy_evaluation_mo(
            member,
            eval_env,
            w=preference,
            scalarization=np.dot,
            rep=eval_episodes,
        )
        eval_env.close()
        member.gamma = gamma
        evaluations.append(discounted_return)
    return evaluations


class PreferenceRoutedEnsemble(MOPolicy):
    """Small tabular ensemble router for algorithms that do not expose a critic."""

    def __init__(
        self,
        members: Sequence[MOPolicy],
        preferences: Sequence[np.ndarray],
        member_values: Optional[Sequence[np.ndarray]] = None,
        gamma: float = 0.99,
    ):
        super().__init__()
        self.members = list(members)
        self.preferences = [np.asarray(w, dtype=np.float64) for w in preferences]
        self.member_values = None if member_values is None else [np.asarray(v, dtype=np.float64) for v in member_values]
        self.gamma = gamma

    def eval(self, obs: np.ndarray, w: Optional[np.ndarray] = None) -> int:
        if w is None:
            w = np.ones_like(self.preferences[0]) / len(self.preferences[0])
        member = self.members[self._select_member(np.asarray(w, dtype=np.float64))]
        return member.eval(obs, w)

    def update(self):
        return None

    def _select_member(self, w: np.ndarray) -> int:
        if self.member_values is not None:
            scores = []
            for value in self.member_values:
                value = np.asarray(value, dtype=np.float64)
                if value.ndim == 1:
                    scores.append(np.dot(value, w))
                else:
                    scores.append(np.max(value @ w))
        else:
            scores = [np.dot(preference, w) for preference in self.preferences]
        return int(np.argmax(scores))


def print_front(name: str, vectors: Iterable[np.ndarray]):
    front = filter_pareto_dominated(list(vectors))
    print(f"{name} non-dominated front ({len(front)} vectors):")
    for vec in front:
        print(vec)


def start_wandb_run(args, experiment_name: str, extra_config: Optional[dict] = None):
    if not args.log:
        return None

    config = {
        "env_id": ENV_ID,
        "ensemble_size": args.ensemble_size,
        "timesteps_per_member": args.timesteps,
        "eval_episodes": args.eval_episodes,
        "seed": args.seed,
    }
    if extra_config is not None:
        config.update(extra_config)

    run = wandb.init(
        project=args.project_name,
        entity=args.wandb_entity,
        config=config,
        name=f"{ENV_ID}__{experiment_name}__{args.seed}",
        save_code=True,
    )
    wandb.define_metric("*", step_metric="global_step")
    wandb.log({"global_step": args.timesteps})
    return run


def log_ensemble_results(
    front: Iterable[np.ndarray],
    preferences: Sequence[np.ndarray],
    evaluations: Sequence[tuple],
    step: int,
    hv_ref_point: np.ndarray,
    known_front: Optional[Sequence[np.ndarray]] = None,
    n_sample_weights: int = 50,
):
    filtered_front = filter_pareto_dominated(list(front))
    reward_dim = len(filtered_front[0])

    log_all_multi_policy_metrics(
        current_front=[np.asarray(vec) for vec in filtered_front],
        hv_ref_point=hv_ref_point,
        reward_dim=reward_dim,
        global_step=step,
        n_sample_weights=n_sample_weights,
        ref_front=None if known_front is None else [np.asarray(vec) for vec in known_front],
    )

    # Mirror the key metrics under an ensemble-specific namespace so they are
    # easy to separate from member-level runs when both are enabled.
    from morl_baselines.common.performance_indicators import cardinality, expected_utility, hypervolume, igd, maximum_utility_loss
    from pymoo.util.ref_dirs import get_reference_directions

    ensemble_metrics = {
        "ensemble/eval/hypervolume": hypervolume(hv_ref_point, filtered_front),
        "ensemble/eval/eum": expected_utility(filtered_front, weights_set=equally_spaced_weights(reward_dim, n_sample_weights)),
        "ensemble/eval/cardinality": cardinality(filtered_front),
        "global_step": step,
    }
    if known_front is not None:
        known_front_list = [np.asarray(vec) for vec in known_front]
        ensemble_metrics["ensemble/eval/igd"] = igd(known_front=known_front_list, current_estimate=filtered_front)
        ensemble_metrics["ensemble/eval/mul"] = maximum_utility_loss(
            front=filtered_front,
            reference_set=known_front_list,
            weights_set=get_reference_directions("energy", reward_dim, n_sample_weights).astype(np.float32),
        )
    wandb.log(ensemble_metrics)

    wandb.log(
        {
            "ensemble/front_size": len(filtered_front),
            "global_step": step,
        }
    )
    wandb.log(
        {
            "ensemble/front": wandb.Table(
                columns=[f"objective_{i}" for i in range(len(filtered_front[0]))],
                data=[np.asarray(vec).tolist() for vec in filtered_front],
            )
        }
    )

    rows = []
    for preference, evaluation in zip(preferences, evaluations):
        rows.append(
            np.asarray(preference).tolist()
            + [float(evaluation[0]), float(evaluation[1])]
            + np.asarray(evaluation[2]).tolist()
            + np.asarray(evaluation[3]).tolist()
        )

    columns = (
        [f"pref_{i}" for i in range(reward_dim)]
        + ["scalarized_return", "discounted_scalarized_return"]
        + [f"vec_return_{i}" for i in range(reward_dim)]
        + [f"discounted_vec_return_{i}" for i in range(reward_dim)]
    )
    wandb.log({"ensemble/evaluations": wandb.Table(columns=columns, data=rows)})


def finish_wandb_run(run) -> None:
    if run is not None:
        wandb.finish()
