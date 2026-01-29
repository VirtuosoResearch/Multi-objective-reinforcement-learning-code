import argparse
import os

import mo_gymnasium as mo_gym
import numpy as np

from morl_baselines.common.weights import equally_spaced_weights
from morl_baselines.multi_policy import EnsembleMemberConfig, OuterEnsemble
from morl_baselines.multi_policy.qpensieve.qpensieve import QPensieve


def make_env():
    return mo_gym.make("mo-hopper-v4")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble-size", type=int, default=3, help="Number of ensemble members.")
    parser.add_argument("--pretrain-steps", type=int, default=0, help="Steps to pretrain each member before ensembling.")
    parser.add_argument("--weights-dir", type=str, default="pretrained_agents", help="Directory to save/load pretrained agents.")
    args = parser.parse_args()

    ensemble_size = args.ensemble_size
    pretrain_steps = args.pretrain_steps
    weights_dir = args.weights_dir

    wandb_project = f"ensembled-{ensemble_size}-qpensieve-hopper-pretrain-{pretrain_steps}"
    prefs = equally_spaced_weights(dim=3, n=ensemble_size)
    member_cfgs = [EnsembleMemberConfig(preference=w) for w in prefs]

    base_kwargs = dict(
        learning_rate=3e-4,
        batch_size=256,
        buffer_size=int(1e6),
        prefer_set_size=4,
        q_replay_size=4,
        q_replay_start=20_000,
        q_replay_freq=5_000,
        project_name="MORL-Baselines",
        experiment_name=wandb_project,
        log=True,
    )
    agent_kwargs = [dict(base_kwargs) for _ in range(ensemble_size)]

    os.makedirs(weights_dir, exist_ok=True)
    ref_point = np.array([-100.0, -100.0, -100.0])
    pretrained_paths = [None] * ensemble_size
    for i in range(ensemble_size):
        print(f"Pretraining agent {i+1}/{ensemble_size}...")
        pre_kwargs = dict(base_kwargs)
        pre_kwargs["log"] = False
        pre_kwargs["experiment_name"] = f"{wandb_project}-Pretrain-agent-{i}"
        env = make_env()
        eval_env = make_env()
        agent = QPensieve(env=env, **pre_kwargs)
        agent.train(
            total_timesteps=pretrain_steps,
            eval_env=eval_env,
            ref_point=ref_point,
            eval_freq=pretrain_steps + 1,
            num_eval_weights_for_front=10,
            num_eval_episodes_for_front=2,
            num_eval_weights_for_eval=10,
        )
        save_path = os.path.join(weights_dir, f"qpensieve_hopper_agent_{i}.tar")
        agent.save(save_dir=weights_dir, filename=f"qpensieve_hopper_agent_{i}", save_replay_buffer=True)
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
        score_interval=200,
        update_steps=4,
        top_fraction=0.5,
        project_name="MORL-Baselines",
        experiment_name=wandb_project,
        log=True,
        member_wandb_runs=False,
        delegate_to_members=True,
        prefill_then_delegate=False,
    )

    for i in range(ensemble_size):
        if pretrained_paths[i]:
            ensemble.agents[i].load(pretrained_paths[i], load_replay_buffer=True)

    eval_env = make_env()
    ensemble.train(
        total_timesteps=int(1e6),
        eval_env=eval_env,
        eval_freq=10_000,
        num_eval_episodes=3,
        ref_point=ref_point,
        num_eval_weights_for_front=50,
        num_eval_episodes_for_front=3,
        num_eval_weights_for_eval=50,
    )


if __name__ == "__main__":
    main()

