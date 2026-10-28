import argparse
from typing import Iterable, Optional, Sequence

import mo_gymnasium as mo_gym
import numpy as np
import wandb

from morl_baselines.common.evaluation import log_all_multi_policy_metrics
from morl_baselines.common.pareto import filter_pareto_dominated
from morl_baselines.common.performance_indicators import cardinality, expected_utility, hypervolume, igd, maximum_utility_loss
from morl_baselines.common.weights import equally_spaced_weights
from pql_routed_ensemble_utils import PreferenceGuidedPQL, train_routed_pql_ensemble


ENV_ID = "deep-sea-treasure-concave-v0"


def parse_args():
    parser = argparse.ArgumentParser(description="Pareto Q-Learning seed ensemble on Deep-Sea Treasure.")
    parser.add_argument("--ensemble-size", type=int, default=3, help="Number of independently trained members.")
    parser.add_argument("--timesteps", type=int, default=10000, help="Training timesteps per ensemble member.")
    parser.add_argument("--log-every", type=int, default=100, help="Member evaluation/logging frequency.")
    parser.add_argument("--metric-weights", type=int, default=50, help="Number of sampled weights for EUM/MUL metrics.")
    parser.add_argument("--seed", type=int, default=1, help="Base random seed.")
    parser.add_argument("--project-name", type=str, default="MORL-Baselines", help="wandb project name.")
    parser.add_argument("--experiment-name", type=str, default="PQL-DST-Ensemble", help="wandb run name.")
    parser.add_argument("--wandb-entity", type=str, default=None, help="wandb entity.")
    parser.add_argument("--no-log", dest="log", action="store_false", help="Disable ensemble-level wandb logging.")
    parser.add_argument("--member-log", action="store_true", help="Enable wandb logging inside each PQL member.")
    parser.add_argument(
        "--training-mode",
        choices=["routed", "independent"],
        default="routed",
        help="Use routed trajectory replay or train each preference-guided member independently.",
    )
    parser.add_argument(
        "--routing",
        choices=["greedy", "random"],
        default="greedy",
        help="How to route each evaluation preference to ensemble members.",
    )
    parser.set_defaults(log=True)
    return parser.parse_args()


def make_env():
    return mo_gym.make(ENV_ID)


def print_front(name: str, vectors: Iterable[np.ndarray]):
    front = filter_pareto_dominated(list(vectors))
    print(f"{name} non-dominated front ({len(front)} vectors):")
    for vec in front:
        print(vec)


def start_wandb_run(args, extra_config: Optional[dict] = None):
    if not args.log:
        return None

    config = {
        "env_id": ENV_ID,
        "ensemble_size": args.ensemble_size,
        "timesteps_per_member": args.timesteps,
        "log_every": args.log_every,
        "metric_weights": args.metric_weights,
        "seed": args.seed,
        "member_log": args.member_log,
        "algo": "pql_ensemble",
        "routing": args.routing,
        "training_mode": args.training_mode,
    }
    if extra_config is not None:
        config.update(extra_config)

    run = wandb.init(
        project=args.project_name,
        entity=args.wandb_entity,
        config=config,
        name=f"{ENV_ID}__{args.experiment_name}__{args.seed}",
        save_code=True,
    )
    wandb.define_metric("*", step_metric="global_step")
    wandb.log({"global_step": args.timesteps})
    return run


def log_ensemble_results(
    front: Sequence[np.ndarray],
    preferences: Sequence[np.ndarray],
    evaluations: Sequence[tuple],
    step: int,
    hv_ref_point: np.ndarray,
    known_front: Optional[Sequence[np.ndarray]] = None,
    n_sample_weights: int = 50,
):
    filtered_front = filter_pareto_dominated([np.asarray(vec) for vec in front])
    reward_dim = len(filtered_front[0])

    log_all_multi_policy_metrics(
        current_front=filtered_front,
        hv_ref_point=hv_ref_point,
        reward_dim=reward_dim,
        global_step=step,
        n_sample_weights=n_sample_weights,
        ref_front=None if known_front is None else [np.asarray(vec) for vec in known_front],
    )

    ensemble_metrics = {
        "ensemble/eval/hypervolume": hypervolume(hv_ref_point, filtered_front),
        "ensemble/eval/eum": expected_utility(filtered_front, weights_set=equally_spaced_weights(reward_dim, n_sample_weights)),
        "ensemble/eval/cardinality": cardinality(filtered_front),
        "ensemble/front_size": len(filtered_front),
        "global_step": step,
    }
    if known_front is not None:
        from pymoo.util.ref_dirs import get_reference_directions

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
            "ensemble/front": wandb.Table(
                columns=[f"objective_{i}" for i in range(reward_dim)],
                data=[np.asarray(vec).tolist() for vec in filtered_front],
            ),
            "global_step": step,
        }
    )

    columns = [f"pref_{i}" for i in range(reward_dim)] + [
        "scalarized_return",
        "discounted_scalarized_return",
    ] + [f"vec_return_{i}" for i in range(reward_dim)] + [f"discounted_vec_return_{i}" for i in range(reward_dim)]
    rows = []
    for preference, evaluation in zip(preferences, evaluations):
        rows.append(
            np.asarray(preference).tolist()
            + [float(evaluation[0]), float(evaluation[1])]
            + np.asarray(evaluation[2]).tolist()
            + np.asarray(evaluation[3]).tolist()
        )
    wandb.log({"ensemble/evaluations": wandb.Table(columns=columns, data=rows)})


def main():
    args = parse_args()
    gamma = 0.99
    ref_point = np.array([0.0, -25.0])
    known_env = make_env()
    known_front = known_env.unwrapped.pareto_front(gamma=gamma)
    reward_dim = known_env.unwrapped.reward_space.shape[0]
    known_env.close()
    preferences = [np.asarray(w, dtype=np.float64) for w in equally_spaced_weights(dim=reward_dim, n=args.ensemble_size)]

    members = []
    member_envs = []
    member_eval_envs = []
    for i, preference in enumerate(preferences):
        env = make_env()
        eval_env = make_env()
        agent = PreferenceGuidedPQL(
            env,
            ref_point,
            preference=preference,
            gamma=gamma,
            initial_epsilon=1.0,
            epsilon_decay_steps=50000,
            final_epsilon=0.2,
            seed=args.seed + i,
            project_name=args.project_name,
            experiment_name=f"Pareto Q-Learning DST Ensemble {i}",
            wandb_entity=args.wandb_entity,
            log=args.member_log,
        )
        members.append(agent)
        member_envs.append(env)
        member_eval_envs.append(eval_env)

    if args.training_mode == "routed":
        rollout_env = make_env()
        train_routed_pql_ensemble(members, rollout_env, total_timesteps=args.timesteps, seed=args.seed)
        rollout_env.close()
    else:
        for agent, eval_env in zip(members, member_eval_envs):
            agent.train(
                total_timesteps=args.timesteps,
                log_every=args.log_every,
                action_eval="hypervolume",
                known_pareto_front=known_front,
                ref_point=ref_point,
                eval_env=eval_env,
            )

    fronts = []
    member_fronts = []
    for agent in members:
        member_front = [np.asarray(vec) for vec in agent.get_local_pcs(state=0)]
        member_fronts.append(member_front)
        fronts.extend(member_front)
        if args.member_log:
            agent.close_wandb()

    for env in member_envs + member_eval_envs:
        env.close()

    print_front("PQL DST ensemble", fronts)
    evaluations = []
    rng = np.random.default_rng(args.seed)
    for preference in preferences:
        candidates = fronts
        if args.routing == "random":
            candidates = member_fronts[int(rng.integers(len(member_fronts)))]
        scalarized = max(float(np.dot(np.asarray(vec), preference)) for vec in candidates)
        best = max(candidates, key=lambda vec: float(np.dot(np.asarray(vec), preference)))
        evaluations.append((scalarized, scalarized, np.asarray(best), np.asarray(best)))

    run = start_wandb_run(args, extra_config={"preference_guided_members": True})
    if run is not None:
        log_ensemble_results(
            fronts,
            preferences,
            evaluations,
            step=args.timesteps,
            hv_ref_point=ref_point,
            known_front=known_front,
            n_sample_weights=args.metric_weights,
        )
        wandb.finish()


if __name__ == "__main__":
    main()
