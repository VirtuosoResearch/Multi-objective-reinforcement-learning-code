import numpy as np

from fruit_tree_ensemble_utils import (
    finish_wandb_run,
    fruit_tree_parser,
    log_ensemble_results,
    make_env,
    member_preferences,
    print_front,
    ref_point,
    reward_dim,
    start_wandb_run,
)
from pql_routed_ensemble_utils import PreferenceGuidedPQL, train_routed_pql_ensemble


def main():
    parser = fruit_tree_parser("Pareto Q-Learning seed ensemble on Fruit-Tree.")
    parser.add_argument(
        "--routing",
        choices=["greedy", "random"],
        default="greedy",
        help="How to route each evaluation preference to ensemble members.",
    )
    parser.add_argument(
        "--training-mode",
        choices=["routed", "independent"],
        default="routed",
        help="Use routed trajectory replay or train each preference-guided member independently.",
    )
    args = parser.parse_args()

    gamma = 0.99
    dim = reward_dim()
    known_front = make_env().unwrapped.pareto_front(gamma=gamma)
    preferences = member_preferences(dim, args.ensemble_size)
    run = start_wandb_run(
        args,
        "PQL-FruitTree-Ensemble",
        extra_config={
            "algo": "pql_ensemble",
            "routing": args.routing,
            "training_mode": args.training_mode,
            "preference_guided_members": True,
        },
    )

    members = []
    member_envs = []
    member_eval_envs = []
    for i, preference in enumerate(preferences):
        env = make_env()
        eval_env = make_env()
        agent = PreferenceGuidedPQL(
            env,
            ref_point(dim),
            preference=preference,
            gamma=gamma,
            initial_epsilon=1.0,
            epsilon_decay_steps=args.timesteps,
            final_epsilon=0.2,
            seed=args.seed + i,
            experiment_name=f"Pareto Q-Learning Fruit-Tree Ensemble {i}",
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
                log_every=100,
                action_eval="hypervolume",
                known_pareto_front=known_front,
                ref_point=ref_point(dim),
                eval_env=eval_env,
            )

    fronts = []
    member_fronts = []
    for agent in members:
        member_front = [np.asarray(vec) for vec in agent.get_local_pcs(state=0)]
        member_fronts.append(member_front)
        fronts.extend(member_front)

    for env in member_envs + member_eval_envs:
        env.close()

    print_front("PQL ensemble", fronts)
    evaluations = []
    rng = np.random.default_rng(args.seed)
    for preference in preferences:
        candidates = fronts
        if args.routing == "random":
            candidates = member_fronts[int(rng.integers(len(member_fronts)))]
        scalarized = max(float(np.dot(np.asarray(vec), preference)) for vec in candidates)
        best = max(candidates, key=lambda vec: float(np.dot(np.asarray(vec), preference)))
        evaluations.append((scalarized, scalarized, np.asarray(best), np.asarray(best)))
    if run is not None:
        log_ensemble_results(
            fronts,
            preferences,
            evaluations,
            step=args.timesteps,
            hv_ref_point=ref_point(dim),
            known_front=known_front,
            n_sample_weights=args.metric_weights,
        )
        finish_wandb_run(run)


if __name__ == "__main__":
    main()
