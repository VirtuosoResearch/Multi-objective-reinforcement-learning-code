from dataclasses import dataclass
from pathlib import Path

import tyro

from metaworld_algorithms.config.networks import (
    ContinuousActionPolicyConfig,
    QValueFunctionConfig,
)
from metaworld_algorithms.config.nn import SoftModulesConfig
from metaworld_algorithms.config.optim import OptimizerConfig
from metaworld_algorithms.config.rl import OffPolicyTrainingConfig
from metaworld_algorithms.envs import MetaworldConfig
from metaworld_algorithms.rl.algorithms import RoutedEnsembleConfig
from metaworld_algorithms.run import Run


@dataclass(frozen=True)
class Args:
    seed: int = 1
    track: bool = True
    wandb_project: str | None = 'multitask-RL'
    wandb_entity: str | None = "zszhang-northeastern-university"
    data_dir: Path = Path("./run_results")
    resume: bool = False


def main() -> None:
    args = tyro.cli(Args)

    num_tasks = 10
    num_experts = 3
    expert_task_ids = [
        [1, 2, 4, 5, 7],
        [3, 6],
        [0, 8, 9],
    ]

    run = Run(
        run_name="mt10_routed_ensemble",
        seed=args.seed,
        data_dir=args.data_dir,
        env=MetaworldConfig(
            env_id="MT10",
            terminate_on_success=False,
        ),
        algorithm=RoutedEnsembleConfig(
            num_tasks=num_tasks,
            gamma=0.99,
            num_experts=num_experts,
            actor_config=ContinuousActionPolicyConfig(
                network_config=SoftModulesConfig(
                    num_tasks=num_tasks,
                    optimizer=OptimizerConfig(max_grad_norm=1.0),
                )
            ),
            critic_config=QValueFunctionConfig(
                network_config=SoftModulesConfig(
                    num_tasks=num_tasks,
                    optimizer=OptimizerConfig(max_grad_norm=1.0),
                )
            ),
            num_critics=2,
            use_task_weights=True,
            lambda_mu=0.1,
            centroid_eta=0.1,
            grad_ema_beta=0.1,
            #score_type="gradient_similarity",
            score_type="centroid_only",
            #update_mode="single_policy",
            update_mode="routed",
            fixed_task_assignments=tuple(tuple(ids) for ids in expert_task_ids),
        ),
        training_config=OffPolicyTrainingConfig(
            total_steps=int(2e7),
            buffer_size=int(1e6),
            batch_size=1280,
        ),
        checkpoint=True,
        resume=args.resume,
    )

    if args.track:
        assert args.wandb_project is not None and args.wandb_entity is not None
        run.enable_wandb(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=run,
            resume="allow",
        )

    run.start()


if __name__ == "__main__":
    main()
