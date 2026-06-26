import os
import argparse
import mo_gymnasium as mo_gym
import numpy as np

from morl_baselines.multi_policy import EnsembleMemberConfig, OuterEnsemble
from morl_baselines.multi_policy.qpensieve.qpensieve import QPensieve
from morl_baselines.common.weights import equally_spaced_weights


def make_env():
    return mo_gym.make("mo-halfcheetah-v4")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble-size", type=int, default=3, help="Number of ensemble members.")
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=int(1e6),
        help="Outer-loop training steps. In shared-buffer mode each step collects one transition per member into the shared buffer.",
    )
    parser.add_argument("--pretrain-steps", type=int, default=0, help="Steps to pretrain each member before ensembling.")
    parser.add_argument("--weights-dir", type=str, default="pretrained_agents", help="Directory to save/load pretrained agents.")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="ensemble_checkpoints_halfcheetah",
        help="Directory for ensemble checkpoints. Only the latest checkpoint is kept by overwriting fixed filenames.",
    )
    parser.add_argument("--save-freq", type=int, default=10_000, help="Save the latest ensemble checkpoint every N steps.")
    parser.add_argument("--learning-starts", type=int, default=2000, help="Member learning_starts for QPensieve.")
    parser.add_argument("--score-interval", type=int, default=20, help="Outer-loop steps between shared-buffer dispatches.")
    parser.add_argument("--update-steps", type=int, default=64, help="Gradient updates per member after each dispatch.")
    parser.add_argument(
        "--dispatch-batch-size",
        type=int,
        default=512,
        help="Number of newly collected samples accumulated before an online dispatch pass.",
    )
    parser.add_argument(
        "--dispatch-warmup-steps",
        type=int,
        default=0,
        help="Use uniform shared-buffer dispatch for the first N outer-loop steps before switching to the main rule.",
    )
    parser.add_argument(
        "--dispatch-temperature",
        type=float,
        default=0.75,
        help="Soft routing temperature used by soft-partition dispatch. Lower is sharper.",
    )
    parser.add_argument(
        "--dispatch-backfill-min-size",
        type=int,
        default=4096,
        help="Minimum number of shared samples retained per member after dispatch by backfilling with extra high-scoring items.",
    )
    parser.add_argument(
        "--top-fraction",
        type=float,
        default=1.0,
        help="Fraction of assigned shared-buffer samples kept for each member during dispatch.",
    )
    parser.add_argument(
        "--member-preference-concentration",
        type=float,
        default=10.0,
        help="Dirichlet concentration used to sample training preferences around each member's target preference. Set <= 0 to keep it fixed.",
    )
    parser.add_argument(
        "--member-selection",
        type=str,
        default="nearest",
        choices=["best", "nearest", "uniform"],
        help="How the ensemble routes actions at eval time: 'nearest' uses the closest member preference, 'best' uses critic scores, 'uniform' samples a member uniformly.",
    )
    parser.add_argument(
        "--dispatch-selection",
        type=str,
        default="soft_partition",
        choices=["top", "uniform", "partition", "soft_partition"],
        help="How the shared buffer is dispatched to members when using the outer-loop: 'soft_partition' samples assignments from member scores, 'partition' uses argmax, 'top' uses per-member top-k, 'uniform' samples uniformly.",
    )
    args = parser.parse_args()

    ensemble_size = args.ensemble_size
    total_timesteps = args.total_timesteps
    pretrain_steps = args.pretrain_steps
    weights_dir = args.weights_dir
    checkpoint_dir = args.checkpoint_dir
    save_freq = args.save_freq
    learning_starts = args.learning_starts
    score_interval = args.score_interval
    update_steps = args.update_steps
    dispatch_batch_size = args.dispatch_batch_size
    dispatch_warmup_steps = args.dispatch_warmup_steps
    dispatch_temperature = args.dispatch_temperature
    dispatch_backfill_min_size = args.dispatch_backfill_min_size
    top_fraction = args.top_fraction
    member_preference_concentration = args.member_preference_concentration
    member_selection = args.member_selection
    dispatch_selection = args.dispatch_selection

    wandb_project = f"ensembled-{ensemble_size}-qpensieve-halfcheetah-sharedbuf-pretrain-{pretrain_steps}"
    # Spread member preferences over the reward simplex for diversity
    prefs = equally_spaced_weights(dim=2, n=ensemble_size)
    member_cfgs = [EnsembleMemberConfig(preference=w) for w in prefs]

    # Shared hyperparameters for all QPensieve instances (can be overridden per member via agent_kwargs)
    base_kwargs = dict(
        learning_rate=3e-4,
        batch_size=256,
        buffer_size=int(1e6),
        prefer_set_size=4,
        q_replay_size=4,
        q_replay_start=5_000,
        q_replay_freq=1_000,
        learning_starts=learning_starts,
        project_name="MORL-Baselines",
        experiment_name=wandb_project,
        log=True,
    )
    agent_kwargs = [
        dict(
            base_kwargs,
            train_preference=np.asarray(w, dtype=np.float32),
            train_preference_concentration=member_preference_concentration,
        )
        for w in prefs
    ]

    # --- Phase 1: pre-train each member independently for a fixed budget ---
    os.makedirs(weights_dir, exist_ok=True)
    ref_point = np.array([-500.0, -5000.0])
    pretrained_paths = [None] * ensemble_size
    for i in range(ensemble_size):
        print(f"Pretraining agent {i+1}/{ensemble_size}...")
        pre_kwargs = dict(agent_kwargs[i])
        pre_kwargs["log"] = False  # record the 50k pretrain in wandb
        pre_kwargs["experiment_name"] = f"{wandb_project}-Pretrain-agent-{i}"
        env = make_env()
        eval_env = make_env()
        agent = QPensieve(env=env, **pre_kwargs)
        agent.train(
            total_timesteps=pretrain_steps,
            eval_env=eval_env,
            ref_point=ref_point,
            eval_freq=pretrain_steps + 1,  # skip interim evals
            num_eval_weights_for_front=10,
            num_eval_episodes_for_front=2,
            num_eval_weights_for_eval=10,
        )
        save_path = os.path.join(weights_dir, f"qpensieve_agent_{i}.tar")
        agent.save(save_dir=weights_dir, filename=f"qpensieve_agent_{i}", save_replay_buffer=False)
        env.close()
        eval_env.close()
        pretrained_paths[i] = save_path
    print("Pretraining complete.")

    ensemble = OuterEnsemble(
        env_fn=make_env,
        agent_ctor=QPensieve,
        ensemble_size=ensemble_size,
        agent_kwargs=agent_kwargs,
        member_configs=member_cfgs,
        buffer_size=int(1e6),
        min_samples_before_training=2000,
        score_interval=score_interval,
        update_steps=update_steps,
        top_fraction=top_fraction,
        project_name="MORL-Baselines",
        experiment_name=wandb_project,
        log=True,
        dispatch_selection=dispatch_selection,
        dispatch_warmup_selection="uniform",
        dispatch_warmup_steps=dispatch_warmup_steps,
        dispatch_temperature=dispatch_temperature,
        dispatch_backfill_min_size=dispatch_backfill_min_size,
        member_selection=member_selection,
        # Centralized logging: write each member's train losses into the ensemble run.
        member_wandb_runs=False,
        delegate_to_members=False,
        prefill_then_delegate=False,
        shared_buffer_only=False,
        online_dispatch=True,
        dispatch_batch_size=dispatch_batch_size,
        retain_own_samples=True,
    )

    # Load only network weights so outer-loop training starts from the shared buffer.
    for i in range(ensemble_size):
        if pretrained_paths[i]:
            ensemble.agents[i].load(pretrained_paths[i], load_replay_buffer=False)

    eval_env = make_env()
    print(f"Saving ensemble checkpoints to {checkpoint_dir!r} every {save_freq} steps.")
    ensemble.train(
        total_timesteps=total_timesteps,
        eval_env=eval_env,
        eval_freq=10_000,
        checkpoints=True,
        save_freq=save_freq,
        save_dir=checkpoint_dir,
        num_eval_episodes=3,
        ref_point=ref_point,
        num_eval_weights_for_front=50,
        num_eval_episodes_for_front=3,
        num_eval_weights_for_eval=50,
    )


if __name__ == "__main__":
    main()
