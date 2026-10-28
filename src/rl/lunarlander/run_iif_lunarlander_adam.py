#!/usr/bin/env python3
"""Pure-Python orchestration for running the full IIF LunarLander pipeline.

This folds the two bash drivers into a single Python entrypoint so each rollout
is executed directly from here (while still delegating to the underlying
training/selection Python scripts)."""

import argparse
import subprocess
from pathlib import Path
import uuid
from typing import Dict


def run_taskset(cmd: list[str], cpus: str, log_path: Path, append: bool, cwd: Path) -> None:
    """Run a command pinned to the requested CPUs and tee output to a log file."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    full_cmd = ["taskset", "-c", cpus, *cmd]
    mode = "a" if append else "w"
    with log_path.open(mode) as f:
        f.write("$ " + " ".join(full_cmd) + "\n")
        f.flush()
        subprocess.run(full_cmd, stdout=f, stderr=subprocess.STDOUT, check=True, cwd=cwd)


def run_first_rollout(*, cpus: str, seed: int, pct: str, valid_tag: str, base_dir: Path,
                      output_root: Path, scripts: Dict[str, Path], log_path: Path,
                      wandb_run_id: str, wandb_run_name: str) -> None:
    """Execute the first rollout (k=0) end-to-end."""
    mode = "neg-bottom"
    x = str(pct)
    try:
        drop_tag = f"{float(x)}"
    except ValueError:
        drop_tag = x
    k = 0
    ed = k + 1
    k_idx = k * 320
    ed_idx = ed * 320
    suffix = "" if seed == 182 else f"_seed-{seed}"

    rollout_save = f"./models_lunarlander/lunarlander_adam_lr-1e-3_step-310k_multi-rol-adaptive-begin-{k}-from-{k}-to-{ed}_{mode}-{drop_tag}_valid-{valid_tag}"
    output_dir = output_root / (
        f"outputs_TracIn_lunarlander_seed-{seed}_adam_lr-1e-3_step-310k_valid-{valid_tag}"
        f"_begin-{k}-updates-{k_idx}-{ed_idx}_{mode}-{drop_tag}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    batches_dir = (
        f"./models_lunarlander/lunarlander_adam_lr-1e-3_step-310k_shuffle-adv_TracIn_seed-{seed}"
        f"_rollout_{k}-{k}-{ed}_drop_{k}-{ed}_{mode}-{drop_tag}_last-10_wrt_{valid_tag}_ckpt_batch"
    )
    trained_save = (
        f"./models_lunarlander/lunarlander_adam_lr-1e-3_step-310k_TracIn_{k}-{k}-{ed}_{mode}-{drop_tag}_valid-{valid_tag}"
    )

    # Train and collect rollout.
    run_taskset(
        [
            "python",
            str(scripts["train_states"]),
            "--total_timesteps",
            "307200",
            "--optimizer_class",
            "Adam",
            "--learning_rate",
            "1e-3",
            "--seed",
            str(seed),
            "--initial_policy",
            f"./models_lunarlander/lunarlander_adam_lr-1e-3_step-310k{suffix}/policy_grad_{k_idx}.zip",
            "--rollout_begin_idx",
            str(k),
            "--total_rollouts",
            str(ed),
            "--save_path",
            rollout_save,
            "--wandb_run_id",
            wandb_run_id,
            "--wandb_run_name",
            wandb_run_name,
        ],
        cpus=cpus,
        log_path=log_path,
        append=True,
        cwd=base_dir,
    )

    # Move training rollout into validation buffer.
    run_taskset(
        [
            "python",
            str(scripts["merge_buffers"]),
            "--model_path",
            rollout_save,
            "--env_short",
            "lunarlander",
            "--k",
            str(k),
            "--ed",
            str(ed),
        ],
        cpus=cpus,
        log_path=log_path,
        append=True,
        cwd=base_dir,
    )

    model_name = Path(rollout_save).name

    # Compute TracIn on collected rollout.
    run_taskset(
        [
            "python",
            str(scripts["tracin"]),
            "--env_short",
            "lunarlander",
            "--rollout_file",
            f"{rollout_save}/buffer_{k}.pkl",
            "--model_path",
            model_name,
            "--ref_policy",
            str(k_idx),
            "--begin_step",
            str(k_idx),
            "--num_updates",
            str(ed_idx),
            "--total_updates",
            "48000",
            "--output_dir",
            str(output_dir),
        ],
        cpus=cpus,
        log_path=log_path,
        append=True,
        cwd=base_dir,
    )

    # Select dataset based on TracIn scores.
    run_taskset(
        [
            "python",
            str(scripts["select_dataset"]),
            "--env_short",
            "lunarlander",
            "--folder",
            str(output_dir),
            "--model_path",
            model_name,
            "--begin",
            str(k),
            "--k",
            str(k),
            "--ed",
            str(ed),
            "--method",
            "TracIn",
            "--curriculum",
            "lunarlander_adam_lr-1e-3_step-310k",
            "--seed",
            str(seed),
            "--valid_samples",
            valid_tag,
            "--drop_percentile",
            x,
            "--mode",
            mode,
        ],
        cpus=cpus,
        log_path=log_path,
        append=True,
        cwd=base_dir,
    )

    # Train on selected dataset.
    run_taskset(
        [
            "python",
            str(scripts["train_batches"]),
            "--seed",
            str(seed),
            "--initial_policy",
            f"./models_lunarlander/lunarlander_adam_lr-1e-3_step-310k{suffix}/policy_grad_{k_idx}.zip",
            "--save_path",
            trained_save,
            "--run_name",
            "train_with_given_states",
            "--batches_dir",
            batches_dir,
            "--batch_begin_idx",
            str(k_idx),
            "--n_batches",
            str(ed_idx),
            "--total_updates",
            "48000",
            "--batch_size",
            "64",
            "--optimizer_class",
            "Adam",
            "--learning_rate",
            "1e-3",
            "--wandb_run_id",
            wandb_run_id,
            "--wandb_run_name",
            wandb_run_name,
        ],
        cpus=cpus,
        log_path=log_path,
        append=True,
        cwd=base_dir,
    )

    # Test final policy.
    run_taskset(
        [
            "python",
            str(scripts["test_return"]),
            "--policy_path",
            f"{trained_save}/policy_grad_{ed_idx}.zip",
            "--num_episodes",
            "1000",
        ],
        cpus=cpus,
        log_path=log_path,
        append=True,
        cwd=base_dir,
    )


def run_followup_rollout(*, begin: int, k: int, cpus: str, seed: int, pct: str,
                         valid_tag: str, base_dir: Path, output_root: Path, scripts: Dict[str, Path],
                         log_path: Path, wandb_run_id: str, wandb_run_name: str) -> None:
    """Execute follow-up rollout (k>=1) end-to-end."""
    mode = "neg-bottom"
    x = str(pct)
    try:
        drop_tag = f"{float(x)}"
    except ValueError:
        drop_tag = x
    pk = k - 1
    ed = k + 1
    k_idx = k * 320
    ed_idx = ed * 320

    rollout_save = (
        f"./models_lunarlander/lunarlander_adam_lr-1e-3_step-310k_multi-rol-adaptive-begin-{begin}"
        f"-from-{k}-to-{ed}_{mode}-{drop_tag}_valid-{valid_tag}"
    )
    output_dir = output_root / (
        f"outputs_TracIn_lunarlander_seed-{seed}_adam_lr-1e-3_step-310k_valid-{valid_tag}"
        f"_begin-{begin}-updates-{k_idx}-{ed_idx}_{mode}-{drop_tag}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    batches_dir = (
        f"./models_lunarlander/lunarlander_adam_lr-1e-3_step-310k_shuffle-adv_TracIn_seed-{seed}"
        f"_rollout_{begin}-{k}-{ed}_drop_{k}-{ed}_{mode}-{drop_tag}_last-10_wrt_{valid_tag}_ckpt_batch"
    )
    trained_save = (
        f"./models_lunarlander/lunarlander_adam_lr-1e-3_step-310k_TracIn_{begin}-{k}-{ed}_{mode}-{drop_tag}_valid-{valid_tag}"
    )
    prev_tracin_model = (
        f"./models_lunarlander/lunarlander_adam_lr-1e-3_step-310k_TracIn_{begin}-{pk}-{k}_{mode}-{drop_tag}_valid-{valid_tag}"
    )

    # Train and collect rollout.
    run_taskset(
        [
            "python",
            str(scripts["train_states"]),
            "--total_timesteps",
            "307200",
            "--optimizer_class",
            "Adam",
            "--learning_rate",
            "1e-3",
            "--seed",
            str(seed),
            "--initial_policy",
            f"{prev_tracin_model}/policy_grad_{k_idx}.zip",
            "--rollout_begin_idx",
            str(k),
            "--total_rollouts",
            str(ed),
            "--save_path",
            rollout_save,
            "--wandb_run_id",
            wandb_run_id,
            "--wandb_run_name",
            wandb_run_name,
        ],
        cpus=cpus,
        log_path=log_path,
        append=False,
        cwd=base_dir,
    )

    # Move training rollout into validation buffer.
    run_taskset(
        [
            "python",
            str(scripts["merge_buffers"]),
            "--model_path",
            rollout_save,
            "--env_short",
            "lunarlander",
            "--k",
            str(k),
            "--ed",
            str(ed),
        ],
        cpus=cpus,
        log_path=log_path,
        append=True,
        cwd=base_dir,
    )

    model_name = Path(rollout_save).name

    # Compute TracIn on collected rollout.
    run_taskset(
        [
            "python",
            str(scripts["tracin"]),
            "--env_short",
            "lunarlander",
            "--rollout_file",
            f"{rollout_save}/buffer_{k}.pkl",
            "--model_path",
            model_name,
            "--ref_policy",
            str(k_idx),
            "--begin_step",
            str(k_idx),
            "--num_updates",
            str(ed_idx),
            "--total_updates",
            "48000",
            "--output_dir",
            str(output_dir),
        ],
        cpus=cpus,
        log_path=log_path,
        append=True,
        cwd=base_dir,
    )

    # Select dataset based on TracIn scores.
    run_taskset(
        [
            "python",
            str(scripts["select_dataset"]),
            "--env_short",
            "lunarlander",
            "--folder",
            str(output_dir),
            "--model_path",
            model_name,
            "--begin",
            str(begin),
            "--k",
            str(k),
            "--ed",
            str(ed),
            "--method",
            "TracIn",
            "--curriculum",
            "lunarlander_adam_lr-1e-3_step-310k",
            "--seed",
            str(seed),
            "--valid_samples",
            valid_tag,
            "--drop_percentile",
            x,
            "--mode",
            mode,
        ],
        cpus=cpus,
        log_path=log_path,
        append=True,
        cwd=base_dir,
    )

    # Train on selected dataset.
    run_taskset(
        [
            "python",
            str(scripts["train_batches"]),
            "--seed",
            str(seed),
            "--initial_policy",
            f"{prev_tracin_model}/policy_grad_{k_idx}.zip",
            "--save_path",
            trained_save,
            "--run_name",
            "train_with_given_states",
            "--batches_dir",
            batches_dir,
            "--batch_begin_idx",
            str(k_idx),
            "--n_batches",
            str(ed_idx),
            "--total_updates",
            "48000",
            "--batch_size",
            "64",
            "--optimizer_class",
            "Adam",
            "--learning_rate",
            "1e-3",
            "--wandb_run_id",
            wandb_run_id,
            "--wandb_run_name",
            wandb_run_name,
        ],
        cpus=cpus,
        log_path=log_path,
        append=True,
        cwd=base_dir,
    )

    # Test final policy.
    run_taskset(
        [
            "python",
            str(scripts["test_return"]),
            "--policy_path",
            f"{trained_save}/policy_grad_{ed_idx}.zip",
            "--num_episodes",
            "1000",
        ],
        cpus=cpus,
        log_path=log_path,
        append=True,
        cwd=base_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the LunarLander rollout/train/selection loop (Python-only)."
    )
    parser.add_argument("seed", type=int, help="Random seed.")
    parser.add_argument("pct", help="Drop percentile argument (x).")
    parser.add_argument(
        "cpus",
        help="CPU list passed to taskset, e.g., '0-7' or '0,2,4,6'.",
    )
    parser.add_argument(
        "--rollouts",
        type=int,
        default=100,
        help="Total rollout iterations (default 100 for k=0..99).",
    )
    parser.add_argument(
        "--begin",
        type=int,
        default=0,
        help="Begin rollout index (matches bash driver default 0).",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="outputs",
        help="Root folder to store all generated outputs/logs (default: outputs/).",
    )
    parser.add_argument(
        "--wandb-run-id",
        type=str,
        default=None,
        help="Optional wandb run id to aggregate all rollouts into one run.",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Optional wandb run name override (defaults to generated name).",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = base_dir / output_root
    output_root.mkdir(parents=True, exist_ok=True)

    results_dir = output_root / "results_lunarlander"
    scripts: Dict[str, Path] = {
        "train_states": base_dir / "train_lunarlander_with_given_states.py",
        "merge_buffers": base_dir / "merge_training_buffers.py",
        "tracin": base_dir / "compute_TracIn_surrogate_ghost.py",
        "select_dataset": base_dir / "produce_minigrid_selected_dataset_shuffle.py",
        "train_batches": base_dir / "train_lunarlander_with_given_batches.py",
        "test_return": base_dir / "test_lunarlander_return.py",
    }

    valid_tag = f"PG-SI-CP-ghost-2k-lambda-1-rollout-train-seed-{args.seed}"
    wandb_run_id = args.wandb_run_id or uuid.uuid4().hex
    wandb_run_name = args.wandb_run_name or f"lunarlander-seed-{args.seed}-pct-{args.pct}"

    # First rollout (k=0).
    first_log = (
        results_dir
        / f"result_lunarlander_adam_rollout_train_surrogate_PG-SI-CP-ghost_adaptive_rol-0-0-1_neg-bottom-{args.pct}_gae-lambda-1_2k_lambda_seed-{args.seed}.txt"
    )
    run_first_rollout(
        cpus=args.cpus,
        seed=args.seed,
        pct=str(args.pct),
        valid_tag=valid_tag,
        base_dir=base_dir,
        output_root=output_root,
        scripts=scripts,
        log_path=first_log,
        wandb_run_id=wandb_run_id,
        wandb_run_name=wandb_run_name,
    )

    # Follow-up rollouts (k = 1 .. rollouts-1).
    for i in range(1, args.rollouts):
        j = i + 1
        log_path = (
            results_dir
            / f"result_lunarlander_adam_rollout_train_surrogate_PG-SI-CP-ghost_adaptive_rol-0-{i}-{j}_neg-bottom-{args.pct}_gae-lambda-1_2k_lambda_seed-{args.seed}.txt"
        )
        run_followup_rollout(
            begin=args.begin,
            k=i,
            cpus=args.cpus,
            seed=args.seed,
            pct=str(args.pct),
            valid_tag=valid_tag,
            base_dir=base_dir,
            output_root=output_root,
            scripts=scripts,
            log_path=log_path,
            wandb_run_id=wandb_run_id,
            wandb_run_name=wandb_run_name,
        )


if __name__ == "__main__":
    main()
