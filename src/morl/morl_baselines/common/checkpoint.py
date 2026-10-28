"""Wait for an independently trained checkpoint only when comparison needs it."""

import json
import os
from pathlib import Path
import subprocess
import time


def wait_for_single_policy(checkpoint, expected_steps, timeout=43_200, poll_interval=30):
    """Wait for the step marker written after checkpoint saving; stop on failed Slurm jobs."""
    checkpoint = Path(checkpoint)
    source_dir = checkpoint.resolve().parent
    marker = source_dir / "single_policy_step.txt"
    start = time.monotonic()
    job_id = source_dir.name.rsplit("job-", 1)[-1]
    announced = False
    while True:
        try:
            with open(source_dir / "config.json", encoding="utf-8") as f:
                config = json.load(f)
            if config["total_timesteps"] != expected_steps:
                raise ValueError("The baseline's configured training budget differs from the requested comparison.")
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        try:
            steps = int(marker.read_text())
        except (FileNotFoundError, ValueError):
            steps = -1
        if steps == expected_steps and checkpoint.is_file():
            return steps
        if steps > expected_steps:
            raise ValueError("The baseline checkpoint has exceeded the requested training budget.")
        if os.environ.get("SLURM_JOB_ID") and job_id.isdigit():
            result = subprocess.run(
                ["sacct", "-j", job_id, "--noheader", "--parsable2", "--format=State"],
                capture_output=True,
                text=True,
                timeout=15,
            )
            states = result.stdout.strip().splitlines()
            if states and states[0].split("|", 1)[0].startswith(
                ("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "BOOT_FAIL", "DEADLINE", "COMPLETED")
            ):
                raise RuntimeError(f"Baseline job {job_id} ended ({states[0]}) without the required checkpoint.")
        if time.monotonic() - start >= timeout:
            raise TimeoutError(f"Timed out waiting for the {expected_steps}-step baseline checkpoint: {checkpoint}")
        if not announced:
            print(
                f"Ensemble training has finished; waiting for the independent {expected_steps}-step baseline checkpoint.",
                flush=True,
            )
            announced = True
        time.sleep(poll_interval)
