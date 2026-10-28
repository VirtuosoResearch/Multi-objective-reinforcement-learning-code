import argparse
import json
import os
import random

import mo_gymnasium as mo_gym
import numpy as np
import torch
import wandb

from morl_baselines.common.checkpoint import wait_for_single_policy
from morl_baselines.common.performance_indicators import hypervolume
from morl_baselines.common.routed_evaluation import routed_hypervolumes
from morl_baselines.common.weights import equally_spaced_weights
from morl_baselines.multi_policy import RoutedEnsemble
from morl_baselines.multi_policy.qpensieve.qpensieve import QPensieve
from morl_baselines.multi_policy.qpensieve.config import reference_qpensieve_kwargs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", choices=["mo-halfcheetah-v4", "mo-hopper-v4", "mo-ant-v4"], default="mo-halfcheetah-v4")
    parser.add_argument(
        "--total-timesteps", type=int, default=100_000, help="Training steps in the selected budget convention."
    )
    parser.add_argument("--budget-mode", choices=["per_member", "total"], default="per_member")
    parser.add_argument("--ensemble-size", type=int, default=3)
    parser.add_argument("--trajectories-per-iteration", type=int, default=6)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--score-action-samples", type=int, default=4)
    parser.add_argument("--assignment", choices=["score", "uniform"], default="score")
    parser.add_argument("--updates-per-transition", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--net-arch", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--q-replay-size", type=int, default=4)
    parser.add_argument(
        "--learning-starts", type=int, default=2000, help="Random-action warmup steps per member."
    )
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument(
        "--strict-paper-updates",
        action="store_true",
        help="Discard old assigned replay and fix updates to the iteration preference.",
    )
    parser.add_argument(
        "--single-policy-checkpoint", help="Reuse a completed independent baseline instead of training it again."
    )
    parser.add_argument(
        "--eval-freq", type=int, default=20_000, help="Evaluate at iteration boundaries; 0 skips intermediate evaluation."
    )
    parser.add_argument("--eval-weights", type=int, default=50)
    parser.add_argument("--selection-episodes", type=int, default=2)
    parser.add_argument("--test-episodes", type=int, default=3)
    parser.add_argument("--skip-final-eval", action="store_true")
    parser.add_argument(
        "--skip-single-policy-baseline", action="store_true", help="Skip the independent baseline and fallback comparison."
    )
    parser.add_argument(
        "--ref-point", type=float, nargs="+", help="HV reference point; defaults to an environment-specific point."
    )
    parser.add_argument(
        "--eval-discount", type=float, default=0.99, help="Discount for vector returns used in HV; 1 means undiscounted."
    )
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "MORL-Baselines"))
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb-run-name")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output-dir", default="routed_checkpoints")
    args = parser.parse_args()
    if args.threads < 1 or args.eval_weights < 1:
        parser.error("--threads and --eval-weights must be positive")
    if args.selection_episodes < 1 or args.test_episodes < 1 or not 0 < args.eval_discount <= 1:
        parser.error("Evaluation episode counts must be positive and 0 < eval-discount <= 1")
    if args.total_timesteps < 1 or args.eval_freq < 0 or args.batch_size < 1:
        parser.error("Training steps/batch size must be positive and eval-freq must be non-negative")

    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    if os.path.exists(os.path.join(args.output_dir, "config.json")):
        parser.error("output-dir already contains config.json; choose a new experiment directory")

    def env_fn():
        return mo_gym.make(args.env_id)

    probe = env_fn()
    reward_dim = probe.unwrapped.reward_space.shape[0]
    probe.close()
    if args.ref_point is None:
        args.ref_point = [-500.0, -5000.0] if args.env_id == "mo-halfcheetah-v4" else [-100.0] * reward_dim
    if len(args.ref_point) != reward_dim or not np.all(np.isfinite(args.ref_point)):
        parser.error("--ref-point must have one finite coordinate per reward objective")
    capacity = max(args.buffer_size, args.trajectories_per_iteration * args.horizon)
    agent_kwargs = [
        dict(
            seed=args.seed + i,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            buffer_size=capacity,
            net_arch=args.net_arch,
            q_replay_size=args.q_replay_size,
            q_replay_start=5000,
            q_replay_freq=1000,
            prefer_set_size=4,
            train_preference=None,
            train_preference_concentration=0.0,
        )
        for i in range(args.ensemble_size)
    ]
    ensemble = RoutedEnsemble(
        env_fn=env_fn,
        agent_ctor=QPensieve,
        ensemble_size=args.ensemble_size,
        agent_kwargs=agent_kwargs,
        buffer_size=capacity,
        trajectories_per_iteration=args.trajectories_per_iteration,
        horizon=args.horizon,
        score_action_samples=args.score_action_samples,
        updates_per_transition=args.updates_per_transition,
        assignment=args.assignment,
        retain_assigned_replay=not args.strict_paper_updates,
        update_preference="iteration" if args.strict_paper_updates else "random",
        seed=args.seed,
        device="cpu",
    )
    ensemble.evaluation_discount = args.eval_discount
    step_multiplier = args.ensemble_size if args.budget_mode == "per_member" else 1
    baseline_kwargs = reference_qpensieve_kwargs(args.seed, args.net_arch, args.batch_size)
    config = {
        **vars(args),
        **ensemble.get_config(),
        "member_kwargs": agent_kwargs,
        "single_policy_kwargs": baseline_kwargs,
        "ensemble_total_env_budget": args.total_timesteps * step_multiplier,
    }
    with open(os.path.join(args.output_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    last_eval_step = -1
    evaluation_steps = set()
    best_validation_hv = {"ensemble": -float("inf"), "single_policy": -float("inf")}
    wandb_run = None

    def write_report(report, phase):
        report["phase"] = phase
        report["return_discount"] = args.eval_discount
        report["ref_point"] = args.ref_point
        with open(os.path.join(args.output_dir, "evaluation.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(report) + "\n")
        scalars = {key: value for key, value in report.items() if isinstance(value, (int, float))}
        print(json.dumps({"phase": phase, **scalars}), flush=True)
        if wandb_run is not None:
            payload = {f"{phase}/{key}": value for key, value in scalars.items()}
            for j, value in enumerate(report.get("member_hypervolumes", [])):
                payload[f"{phase}/member_{j}_hypervolume"] = value
            wandb_run.log(payload)
            if phase == "comparison":
                wandb_run.summary.update(scalars)
                wandb_run.summary["selected_policy"] = report["selected_policy"]

    def evaluate(model):
        nonlocal last_eval_step
        preferences = equally_spaced_weights(reward_dim, args.eval_weights)
        records = model.evaluate(preferences, args.selection_episodes, args.test_episodes, seed=args.seed + 10_000)
        mean_utility = float(np.mean([np.dot(r["preference"], r["test_return"]) for r in records]))
        report = {
            "global_step": int(np.ceil(model.global_step / step_multiplier)),
            "ensemble_env_steps": model.global_step,
            "mean_scalarized_test_return": mean_utility,
            "score_env_steps": model.score_env_steps,
            "evaluation_env_steps": model.evaluation_env_steps,
            "policies": records,
        }
        report.update(routed_hypervolumes(records, args.ref_point, model.ensemble_size))
        for source, field in [
            ("ensemble", "ensemble_test_return"),
            ("single_policy", "single_policy_test_return"),
            ("oracle", "oracle_test_return"),
        ]:
            if all(field in r for r in records):
                report[f"{source}_mean_scalarized_test_return"] = float(
                    np.mean([np.dot(r["preference"], r[field]) for r in records])
                )
        if model.baseline_policy is not None:
            report["single_policy_training_steps"] = model.baseline_policy.global_step
            report["single_policy_selections"] = sum(r["selected_source"] == "single_policy" for r in records)
        phase = "comparison" if model.baseline_policy is not None else "ensemble"
        if phase == "ensemble":
            evaluation_steps.add(report["global_step"])
            if report["ensemble_validation_hypervolume"] > best_validation_hv["ensemble"]:
                best_validation_hv["ensemble"] = report["ensemble_validation_hypervolume"]
                model._save_members(os.path.join(args.output_dir, "best_ensemble"))
                with open(os.path.join(args.output_dir, "best_ensemble", "selection.json"), "w", encoding="utf-8") as f:
                    json.dump({"global_step": model.global_step, "validation_hypervolume": best_validation_hv["ensemble"]}, f)
        write_report(report, phase)
        last_eval_step = model.global_step

    try:
        requested_mode = args.wandb_mode
        run_name = (
            args.wandb_run_name
            or f"{args.env_id}-routed-{args.total_timesteps}-seed-{args.seed}-{os.environ.get('SLURM_JOB_ID', 'local')}"
        )
        init_kwargs = dict(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            config=config,
            dir=os.path.abspath(args.output_dir),
            mode=requested_mode,
            settings=wandb.Settings(init_timeout=45),
        )
        try:
            wandb_run = wandb.init(**init_kwargs)
        except wandb.errors.Error:
            if requested_mode != "online":
                raise
            print("W&B online initialization failed; HV will be recorded in an offline run for later sync.", flush=True)
            init_kwargs["mode"] = "offline"
            wandb_run = wandb.init(**init_kwargs)
        actual_mode = wandb_run.settings.mode
        for phase in ("ensemble", "single_policy", "comparison"):
            wandb_run.define_metric(f"{phase}/*", step_metric=f"{phase}/global_step")
        tracking = {
            "id": wandb_run.id,
            "mode": actual_mode,
            "url": wandb_run.url if actual_mode == "online" else None,
            "project": args.wandb_project,
            "entity": wandb_run.entity,
        }
        with open(os.path.join(args.output_dir, "wandb.json"), "w", encoding="utf-8") as f:
            json.dump(tracking, f, indent=2)
        print(json.dumps({"wandb": tracking}), flush=True)
        ensemble.train(
            args.total_timesteps,
            save_dir=args.output_dir,
            eval_freq=args.eval_freq * step_multiplier,
            evaluation_callback=evaluate,
            budget_mode=args.budget_mode,
        )
        if not args.skip_final_eval and last_eval_step != ensemble.global_step:
            evaluate(ensemble)
        if not args.skip_single_policy_baseline:
            print(
                f"Using an independent single-policy baseline trained for {args.total_timesteps} steps; "
                f"ensemble total environment budget is {args.total_timesteps * step_multiplier}.",
                flush=True,
            )
            baseline_env = env_fn()
            baseline_env.reset(seed=args.seed)
            baseline_env.action_space.seed(args.seed)
            baseline = QPensieve(env=baseline_env, device="cpu", log=False, **baseline_kwargs)
            ensemble.baseline_policy = baseline
            np.random.seed(args.seed)
            original_update = baseline.update
            original_close_wandb = baseline.close_wandb

            def update_and_evaluate_baseline():
                original_update()
                if baseline.global_step not in evaluation_steps:
                    return
                preferences = equally_spaced_weights(reward_dim, args.eval_weights)
                validation_returns, test_returns = [], []
                for pref in preferences:
                    validation_returns.append(
                        np.mean(
                            [
                                ensemble._rollout_return(ensemble.ensemble_size, pref, args.seed + 10_000 + k)
                                for k in range(args.selection_episodes)
                            ],
                            axis=0,
                        )
                    )
                    test_returns.append(
                        np.mean(
                            [
                                ensemble._rollout_return(
                                    ensemble.ensemble_size, pref, args.seed + 10_000 + args.selection_episodes + k
                                )
                                for k in range(args.test_episodes)
                            ],
                            axis=0,
                        )
                    )
                validation_hv = float(hypervolume(np.asarray(args.ref_point), validation_returns))
                hv = float(hypervolume(np.asarray(args.ref_point), test_returns))
                if validation_hv > best_validation_hv["single_policy"]:
                    best_validation_hv["single_policy"] = validation_hv
                    baseline.save(
                        save_dir=os.path.join(args.output_dir, "best_single_policy"),
                        filename="single_policy",
                        save_replay_buffer=False,
                    )
                write_report(
                    {
                        "global_step": baseline.global_step,
                        "hypervolume": hv,
                        "single_policy_hypervolume": hv,
                        "validation_hypervolume": validation_hv,
                        "validation_returns": np.asarray(validation_returns).tolist(),
                        "test_returns": np.asarray(test_returns).tolist(),
                    },
                    "single_policy",
                )

            baseline.update = update_and_evaluate_baseline
            baseline.close_wandb = lambda: None
            try:
                if args.single_policy_checkpoint:
                    source_dir = os.path.dirname(os.path.abspath(args.single_policy_checkpoint))
                    baseline_steps = wait_for_single_policy(args.single_policy_checkpoint, args.total_timesteps)
                    with open(os.path.join(source_dir, "config.json"), encoding="utf-8") as f:
                        source_config = json.load(f)
                    if source_config["env_id"] != args.env_id or source_config["agent_kwargs"] != baseline_kwargs:
                        raise ValueError("Independent baseline environment and hyperparameters must match this comparison.")
                    baseline.load(args.single_policy_checkpoint, load_replay_buffer=False)
                    baseline.global_step = baseline_steps
                    print(f"Loaded independently trained {baseline_steps}-step baseline from {source_dir}", flush=True)
                else:
                    baseline.train(
                        args.total_timesteps,
                        eval_env=None,
                        ref_point=np.asarray(args.ref_point),
                        num_eval_weights_for_front=args.eval_weights,
                    )
            finally:
                baseline.update = original_update
                baseline.close_wandb = original_close_wandb
            baseline.save(save_dir=args.output_dir, filename="single_policy", save_replay_buffer=False)
            last_eval_step = -1
        if not args.skip_final_eval and last_eval_step != ensemble.global_step:
            evaluate(ensemble)
    finally:
        if wandb_run is not None:
            wandb_run.finish()
        if ensemble.baseline_policy is not None:
            ensemble.baseline_policy.env.close()
        ensemble.close()


if __name__ == "__main__":
    main()
