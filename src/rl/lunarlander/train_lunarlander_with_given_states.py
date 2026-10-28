import argparse
import os
import numpy as np
np.float_ = np.float64
import torch
from torch.optim import SGD
import wandb
import pickle  # For saving mini-batches
from stable_baselines3.common.callbacks import CallbackList
from wandb.integration.sb3 import WandbCallback  # Requires wandb integration for SB3
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
import torch as th
import torch.nn as nn
import gymnasium as gym
from gymnasium import error as gym_error
from gymnasium.wrappers import RecordEpisodeStatistics

from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

# =======================
# (Removed CustomCNN and RestrictedActionSpaceWrapper as they are specific to MiniGrid)
# =======================

# =======================
# Environment creation
# =======================
def make_env(env_name):
    def _init():
        if env_name == 'FrozenLake-v1':
            from gymnasium.envs.toy_text.frozen_lake import generate_random_map
            env = gym.make(env_name, map_name="4x4", is_slippery=False)
        elif env_name == "highway-v0":
            import highway_env
            env = gym.make("highway-v0", render_mode=None, config={
                "observation": {
                    "type": "Kinematics"
                },
                "action": {
                    "type": "DiscreteMetaAction"
                },
                "vehicles_count": 10,
            })
        elif "merge" in env_name:
            import highway_env
            env = gym.make(env_name, render_mode=None)
        elif env_name.startswith("LunarLander"):
            target_env = env_name
            try:
                env = gym.make(target_env)
            except gym_error.Error:
                target_env = "LunarLander-v3"
                env = gym.make(target_env)
        else:
            env = gym.make(env_name)
        env = RecordEpisodeStatistics(env)  # for episode stats
        return env
    return _init

# =======================
# Callback to save states (observations) from rollouts
# =======================
class SaveStatesCallback(BaseCallback):
    def __init__(self, save_path, verbose=0):
        super(SaveStatesCallback, self).__init__(verbose)
        self.save_path = save_path
        self.all_states = []
        self.all_actions = []
        self.all_rewards = []
        self.all_advantages = []
        self.all_episode_starts = []
    
    def _on_step(self) -> bool:
        return True
    
    def on_rollout_end(self) -> None:
        if hasattr(self.model.rollout_buffer, "observations"):
            obs = self.model.rollout_buffer.observations.copy()
            self.all_states.extend(obs)
            
            actions = self.model.rollout_buffer.actions.copy()
            self.all_actions.extend(actions)
            
            rewards = self.model.rollout_buffer.rewards.copy()
            self.all_rewards.extend(rewards)
            
            advantages = self.model.rollout_buffer.advantages.copy()
            self.all_advantages.extend(advantages)
            
            episode_starts = self.model.rollout_buffer.episode_starts.copy()
            self.all_episode_starts.extend(episode_starts)

        # Log episodic reward/length to wandb (if available).
        if hasattr(self.model, "ep_info_buffer") and len(self.model.ep_info_buffer) > 0:
            rews = [info["r"] for info in self.model.ep_info_buffer if "r" in info]
            lens = [info["l"] for info in self.model.ep_info_buffer if "l" in info]
            if rews:
                step = self.model.num_timesteps
                wandb.log(
                    {
                        "rollout/ep_rew_mean": float(np.mean(rews)),
                        "rollout/ep_len_mean": float(np.mean(lens)) if lens else None,
                        "global_step": step,
                    },
                    step=step,
                )
    
    def on_training_end(self) -> None:
        all_states_array = np.array(self.all_states)
        save_file = os.path.join(self.save_path, "all_states.npy")
        np.save(save_file, all_states_array)
        print(f"Saved all states to {save_file}")
        
        all_actions_array = np.array(self.all_actions)
        save_file = os.path.join(self.save_path, "all_actions.npy")
        np.save(save_file, all_actions_array)
        print(f"Saved all actions to {save_file}")
        
        all_rewards_array = np.array(self.all_rewards)
        save_file = os.path.join(self.save_path, "all_rewards.npy")
        np.save(save_file, all_rewards_array)
        print(f"Saved all rewards to {save_file}")
        
        all_advantages_array = np.array(self.all_advantages)
        save_file = os.path.join(self.save_path, "all_advantages.npy")
        np.save(save_file, all_advantages_array)
        print(f"Saved all advantages to {save_file}")
        
        all_episode_starts_array = np.array(self.all_episode_starts)
        save_file = os.path.join(self.save_path, "all_episode_starts.npy")
        np.save(save_file, all_episode_starts_array)
        print(f"Saved all episode starts to {save_file}")

# =======================
# Argument parsing
# =======================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PPO on LunarLander-v3 with data attribution logging."
    )
    # Training hyperparameters
    parser.add_argument("--total_timesteps", type=int, default=40960,
                        help="Total timesteps for training.")
    parser.add_argument("--n_steps", type=int, default=2048,
                        help="Number of steps per rollout.")
    parser.add_argument("--learning_rate", type=float, default=0.01,
                        help="Learning rate for the optimizer.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed.")
    
    # WandB settings.
    parser.add_argument("--project_name", type=str, default="lunarlander-rl",
                        help="WandB project name.")
    parser.add_argument("--run_name", type=str, default="ppo_lunarlander_run",
                        help="WandB run name.")
    parser.add_argument("--wandb_run_id", type=str, default=None,
                        help="Optional wandb run id to resume/aggregate logs.")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Optional wandb run name override.")
    parser.add_argument("--save_path", type=str, default="saved_models",
                        help="Directory to save models and mini-batches.")
    parser.add_argument('--optimizer_class', type=str, default="SGD")
    
    parser.add_argument('--initial_policy', type=str, default='')
    parser.add_argument('--rollout_begin_idx', type=int, default=0)

    parser.add_argument("--total_rollouts", type=int, default=20,
                        help="Number of fixed rollouts provided.")
    # Change default environment name to LunarLander-v3
    parser.add_argument('--env_name', type=str, default="LunarLander-v3")
    return parser.parse_args()

# =======================
# Main training function
# =======================
def main():
    # Parse hyperparameters from command line.
    args = parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)  # Create the save directory if it doesn't exist.

    # Initialize wandb for experiment tracking.
    wandb.init(
        project=args.project_name,
        name=args.wandb_run_name or args.run_name,
        id=args.wandb_run_id,
        resume="allow" if args.wandb_run_id else None,
        reinit=True,
        config=vars(args),
        sync_tensorboard=True,
        save_code=True,
    )
    wandb.define_metric("global_step")
    wandb.define_metric("rollout/*", step_metric="global_step")
    config = wandb.config

    # Create environment wrapped in DummyVecEnv and VecMonitor
    env = DummyVecEnv([make_env(args.env_name)])
    env = VecMonitor(env)
    
    print('env:', args.env_name)
    print('observation space:', env.observation_space)
    # print('number of actions:', env.action_space.n)
    
    # Define policy keyword arguments
    # For LunarLander-v3, we use MlpPolicy (thus no need for a custom CNN extractor)
    policy_kwargs = dict(
        optimizer_class=th.optim.SGD if args.optimizer_class == "SGD" else th.optim.Adam,
    )

    # Compute total_global_steps as the sum of lengths of all fixed rollouts.
    total_timesteps = args.total_timesteps
    global_steps = args.n_steps * args.rollout_begin_idx  # Initialize global steps counter
    
    # Use GPU if available.
    device = "cpu"
    print(f"Using device: {device}")

    model = PPO(
        "MlpPolicy",  # Changed from CnnPolicy to MlpPolicy
        env,
        verbose=1,
        device=device,
        learning_rate=config.learning_rate,
        tensorboard_log="./tb_logs/",
        seed=args.seed,
        policy_kwargs=policy_kwargs
    )
    
    print(model.policy.optimizer)  # Print the optimizer to verify it's set correctly
    
    if args.initial_policy:
        import copy
        # Remove .zip extension if present, as PPO.load() automatically appends it
        policy_path = args.initial_policy
        if policy_path.endswith('.zip'):
            policy_path = policy_path[:-4]
        
        # Check if file exists (with or without .zip extension)
        if not os.path.exists(policy_path + '.zip') and not os.path.exists(policy_path):
            print(
                f"Initial policy file not found ({args.initial_policy}); "
                "continuing from a fresh policy instead."
            )
        else:
            model_init = PPO.load(
                policy_path, 
                device=device, 
            )
            model.policy = copy.deepcopy(model_init.policy)
            print('Loaded model with initial policy:', args.initial_policy)
            
    else:
        model.save(os.path.join(args.save_path, "policy_grad_0.zip"))

    # Initialize a counter for gradient updates; this will be used to index both batches and checkpoints.
    model._update_counter = 320 * args.rollout_begin_idx
    model.save(os.path.join(args.save_path, f"policy_grad_{model._update_counter}.zip"))

    model.num_timesteps = global_steps

    # Patch the optimizer's step method to save the checkpoint (pi_t) after each gradient update.
    original_optimizer_step = model.policy.optimizer.step
    def patched_optimizer_step(*args_, **kwargs):
        result = original_optimizer_step(*args_, **kwargs)
        checkpoint_filename = os.path.join(save_path, f"policy_grad_{model._update_counter}.zip")
        model.save(checkpoint_filename)
        return result
    model.policy.optimizer.step = patched_optimizer_step

    # Create your state-saving callback.
    save_states_callback = SaveStatesCallback(save_path=save_path, verbose=1)
    callback = CallbackList([WandbCallback(verbose=2), save_states_callback])
    
    # Now loop over each fixed rollout.
    total_rollouts = args.total_rollouts
    
    print('===============================================================')
    print(f'Begin training at rollout_idx = {args.rollout_begin_idx} and global_steps = {global_steps}')
    print(f'Target rollouts = {total_rollouts} with total global_steps = {total_timesteps}')
    
    # Patch the rollout buffer's get() method to save each mini-batch.
    original_get = model.rollout_buffer.get
    def patched_get(batch_size):
        for mini_batch in original_get(batch_size):
            model._update_counter += 1
            batch_filename = os.path.join(save_path, f"batch_{model._update_counter}.pkl")
            with open(batch_filename, "wb") as f:
                pickle.dump(mini_batch, f)
            yield mini_batch
    model.rollout_buffer.get = patched_get
    
    for rollout_idx in range(args.rollout_begin_idx, total_rollouts):
        print(f"Training on rollout {rollout_idx+1}/{total_rollouts} with {model.n_steps} transitions.")
        
        # Update global steps.
        global_steps += model.n_steps
        print(f"Global progress: {1.0 - (global_steps / total_timesteps):.3f}")

        # Run exactly one rollout worth of interaction.
        model.learn(
            total_timesteps=args.n_steps,
            callback=callback,
            reset_num_timesteps=False,
        )        
        
        # Optionally, save a checkpoint after each rollout.
        model.save(os.path.join(save_path, f"policy_after_rollout_{rollout_idx+1}.zip"))

    final_model_path = os.path.join(save_path, "final_policy.zip")
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    wandb.finish()

if __name__ == "__main__":
    main()
