

"""
Main Training Script for MBPO-Breakout
======================================

This script provides the main training loop with:
- Wandb integration for experiment tracking
- Bayesian optimization sweep support
- Early stopping when target reward is reached
- Modular agent selection (MBPO, DDQN, PPO)
- Checkpoint saving and resumption

Usage:
    # Standard training
    python main.py
    
    # With config override
    python main.py --config path/to/config.yaml
    
    # Wandb sweep
    python main.py --sweep

Author: CMPS458 RL Project
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import wandb

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from agents import create_agent, MBPOAgent
from env_factory import make_vec_env, verify_env_setup
from utils import (
    EarlyStopping,
    MetricsLogger,
    load_config,
    save_checkpoint,
    set_seed,
    find_latest_video,
)
def record_video_episode(config, agent, global_step, episode_idx: int, video_dir="videos"):
    """Run a single deterministic episode in a dedicated video-recording env.

    After the episode ends, find the most recent video produced and rename it
    to include the global episode index to avoid overwriting previous videos.
    """
    from env_factory import make_env
    from utils import find_latest_video
    import shutil
    import numpy as np

    # Use a fresh local directory for this eval to avoid RecordVideo overwrite warnings
    local_video_dir = os.path.join(video_dir, f"eval-{episode_idx:06d}")
    # Ensure the local video directory is fresh to avoid RecordVideo overwrite warnings
    if os.path.exists(local_video_dir):
        try:
            import shutil
            shutil.rmtree(local_video_dir)
        except Exception:
            # If we cannot remove, fallback to appending a timestamp
            local_video_dir = os.path.join(video_dir, f"eval-{episode_idx:06d}-{int(time.time())}")
    env_fn = make_env(
        env_id=config.get("env", {}).get("name", "BreakoutNoFrameskip-v4"),
        seed=config.get("seed", 42),
        idx=0,
        capture_video=True,
        video_dir=local_video_dir,
        video_trigger_freq=1,
    )

    env = env_fn()
    obs, info = env.reset()
    done = False
    total_reward = 0.0
    while not done:
        action, _ = agent.get_action(obs, deterministic=True)
        # Handle action being an array
        if isinstance(action, np.ndarray):
            action = int(action.item() if action.size == 1 else action[0])
        obs, reward, terminated, truncated, info = env.step(action)
        # reward can be scalar or array-like from wrappers — handle both
        try:
            total_reward += float(reward)
        except Exception:
            # If reward is an array, sum
            total_reward += float(np.asarray(reward).sum())
        done = bool(terminated or truncated)

    env.close()

    # Find the most recent video in the local folder and move it to the main video dir
    latest = find_latest_video(local_video_dir)
    if latest is not None:
        os.makedirs(local_video_dir, exist_ok=True)
        dst = os.path.join(local_video_dir, f"eval-episode-{episode_idx:06d}.mp4")
        try:
            shutil.move(str(latest), dst)
            print(f"[Video] Saved episode video: {dst} (reward: {total_reward})")
        except Exception:
            print(f"[Video] Recording finished but failed to move video: {latest}")
    else:
        print(f"[Video] Episode finished but no video file found in {local_video_dir}")

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="MBPO Breakout Training")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run as wandb sweep agent",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging",
    )
    return parser.parse_args()


def train(config: Dict, args: argparse.Namespace) -> float:
    """
    Main training loop.
    
    Args:
        config: Configuration dictionary
        args: Command line arguments
    
    Returns:
        Final mean reward achieved
    """
    # Setup
    seed = args.seed if args.seed is not None else config.get("seed", 42)
    set_seed(seed)
    
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Environment config
    env_cfg = config.get("env", {})
    training_cfg = config.get("training", {})
    early_stop_cfg = config.get("early_stopping", {})
    
    # Create environment
    print("\nCreating environment...")
    env = make_vec_env(
        env_id=env_cfg.get("name", "BreakoutNoFrameskip-v4"),
        num_envs=env_cfg.get("num_envs", 4),
        seed=seed,
        capture_video=env_cfg.get("record_video", True),
        video_dir=env_cfg.get("video_dir", "videos"),
        video_trigger_freq=env_cfg.get("video_trigger_freq", 50),
        noop_max=env_cfg.get("noop_max", 30),
        frame_skip=env_cfg.get("frame_skip", 4),
        frame_stack=env_cfg.get("frame_stack", 4),
    )
    
    # Get observation/action shapes
    obs_shape = env.single_observation_space.shape
    num_actions = env.single_action_space.n
    num_envs = env_cfg.get("num_envs", 4)
    
    print(f"Observation shape: {obs_shape}")
    print(f"Number of actions: {num_actions}")
    print(f"Number of parallel envs: {num_envs}")
    
    # Create agent
    algorithm = config.get("algorithm", "mbpo")
    print(f"\nCreating {algorithm.upper()} agent...")
    agent = create_agent(algorithm, obs_shape, num_actions, config, device)
    
    # Initialize wandb
    use_wandb = config.get("wandb", {}).get("enabled", True) and not args.no_wandb
    if use_wandb:
        wandb_cfg = config.get("wandb", {})
        run_name = f"{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        wandb.init(
            project=wandb_cfg.get("project", "MBPO-Breakout"),
            entity=wandb_cfg.get("entity"),
            name=run_name,
            config=config,
            tags=wandb_cfg.get("tags", [algorithm, "breakout"]),
            sync_tensorboard=False,
        )
    
    # Training parameters (robust parsing for numeric config values)
    def _parse_int(value, default: int):
        """Parse integers from config values. Accepts ints, numeric strings,
        and strings with underscores (e.g. '10_000_000')."""
        if value is None:
            return default
        if isinstance(value, int):
            return value
        try:
            # Handle strings like '10_000_000' or '10000000'
            return int(str(value).replace("_", ""))
        except Exception:
            return default

    total_timesteps = _parse_int(training_cfg.get("total_timesteps", 10_000_000), 10_000_000)
    learning_starts = _parse_int(training_cfg.get("learning_starts", 50_000), 50_000)
    train_freq = training_cfg.get("train_freq", 4)
    log_freq = config.get("logging", {}).get("log_freq", 1000)
    save_freq = config.get("logging", {}).get("save_freq", 100_000)
    checkpoint_dir = config.get("logging", {}).get("checkpoint_dir", "checkpoints")
    model_rollout_freq = training_cfg.get("model_rollout_freq", 250)
    
    # Early stopping
    early_stop_enabled = early_stop_cfg.get("enabled", True)
    early_stopper = EarlyStopping(
        target_reward=early_stop_cfg.get("target_reward", 400.0),
        window_size=early_stop_cfg.get("window_size", 100),
        patience=early_stop_cfg.get("patience", 50),
    )
    
    # Metrics logger
    metrics_logger = MetricsLogger()
    
    # Initialize episode tracking
    episode_rewards = np.zeros(num_envs)
    episode_lengths = np.zeros(num_envs)
    
    # Reset environment
    obs, info = env.reset()
    
    # Training loop
    print(f"\nStarting training for {total_timesteps:,} timesteps...")
    print("=" * 60)
    
    global_step = 0
    episode_count = 0
    best_mean_reward = float("-inf")
    
    last_video_episode = 0
    # Action distribution debug counters
    action_counts = np.zeros(num_actions, dtype=np.int64)
    action_debug_steps = config.get("debug", {}).get("action_debug_steps", 10_000)
    action_log_freq = config.get("debug", {}).get("action_log_freq", 1_000)
    while global_step < total_timesteps:
        # Get actions for all environments in a single batch
        actions, _ = agent.get_action(obs, deterministic=False)
        
        # Update action distribution counters (for debugging early bias)
        if global_step < action_debug_steps:
            # bincount over current parallel actions
            counts = np.bincount(actions, minlength=num_actions)
            action_counts[:len(counts)] += counts
            if (global_step % action_log_freq) == 0 and use_wandb:
                # Log the distribution to wandb and print
                dist = {f"action/{i}": int(action_counts[i]) for i in range(num_actions)}
                print(f"[ActionDebug] Step {global_step}: {dist}")
                wandb.log({**dist, "debug/step": int(global_step)})
        
        # Step environment
        next_obs, rewards, terminateds, truncateds, infos = env.step(actions)
        dones = np.logical_or(terminateds, truncateds)
        
        # Store transitions
        for i in range(num_envs):
            agent.store_transition(obs[i], actions[i], rewards[i], next_obs[i], dones[i])
            episode_rewards[i] += rewards[i]
            episode_lengths[i] += 1
            
            # Episode finished
            if dones[i]:
                episode_count += 1
                
                # Log episode stats
                metrics_logger.update({
                    "episode/reward": episode_rewards[i],
                    "episode/length": episode_lengths[i],
                })
                
                # Check early stopping
                should_stop = early_stop_enabled and early_stopper.update(episode_rewards[i])
                
                if use_wandb:
                    # Log per-episode metrics to Wandb without specifying `step`.
                    # Let wandb assign a logical ordering to avoid non-monotonic
                    # step warnings when other code logs with `global_step`.
                    wandb.log({
                        "episode/reward": float(episode_rewards[i]),
                        "episode/length": int(episode_lengths[i]),
                        "episode/count": int(episode_count),
                        "episode/mean_reward_100": early_stopper.get_mean_reward(),
                    })
                
                if should_stop:
                    print(f"\n{'=' * 60}")
                    print(f"Early stopping triggered: {early_stopper.stop_reason}")
                    print(f"Final mean reward: {early_stopper.get_mean_reward():.2f}")
                    print(f"Total episodes: {episode_count}")
                    print(f"Total timesteps: {global_step:,}")
                    
                    # Save final checkpoint
                    save_path = f"{checkpoint_dir}/final_{algorithm}.pt"
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    agent.save(save_path)
                    print(f"Final model saved to {save_path}")
                    
                    # Log final video if available
                    if use_wandb:
                        video_path = find_latest_video(env_cfg.get("video_dir", "videos"))
                        if video_path:
                            wandb.log({"final_video": wandb.Video(video_path)})
                    
                    env.close()
                    if use_wandb:
                        wandb.finish()
                    return early_stopper.get_mean_reward()
                
                # Reset episode stats
                episode_rewards[i] = 0
                episode_lengths[i] = 0
        
        # Update observation
        obs = next_obs
        global_step += num_envs

        # Record a video every 100 global episodes
        if episode_count // 100 > last_video_episode // 100:
            record_video_episode(config, agent, global_step, episode_count)
            last_video_episode = episode_count
        
        # Training updates
        if global_step >= learning_starts and global_step % train_freq == 0:
            # Update agent
            update_metrics = agent.update()
            metrics_logger.update(update_metrics)
            
            # Debug: log training metrics every 2500 steps to monitor entropy/gradients
            if global_step % 2500 == 0:
                dynamics_loss = update_metrics.get('dynamics/total_loss', 0)
                print(f"[Training] Step {global_step}: "
                      f"policy_loss={update_metrics.get('policy/loss', 0):.4f}, "
                      f"grad_norm={update_metrics.get('policy/grad_norm', 0):.4f}, "
                      f"entropy={update_metrics.get('policy/entropy', 0):.4f}, "
                      f"dynamics_loss={dynamics_loss:.4f}")
        
        # Model rollouts (for MBPO)
        if (algorithm == "mbpo" and 
            global_step >= learning_starts and 
            global_step % model_rollout_freq == 0):
            if hasattr(agent, 'generate_model_rollouts'):
                n_generated = agent.generate_model_rollouts()
                metrics_logger.update({"model/rollouts_generated": n_generated})
        
        # Commit staged rollouts to buffer
        if (algorithm == "mbpo" and 
            global_step >= learning_starts and 
            hasattr(agent.model_buffer, 'commit_to_buffer')):
            agent.model_buffer.commit_to_buffer(num_commits=2)
        
        # Logging
        if global_step % log_freq == 0:
            log_metrics = metrics_logger.get_means()

            # Avoid logging episode-level metrics with `global_step`.
            # Wandb requires each metric to be logged with monotonically
            # increasing `step` values. Metrics produced per-episode use
            # `episode_count` as their step; if we also log those same
            # metric names here with `global_step` (which is larger) we
            # can later attempt to write the same metric with a smaller
            # step and get warnings. Filter out episode/* keys here so
            # episode metrics are only logged at episode time.
            log_metrics = {k: v for k, v in log_metrics.items() if not k.startswith("episode/")}

            log_metrics["train/global_step"] = global_step
            log_metrics["train/episodes"] = episode_count

            if use_wandb:
                wandb.log(log_metrics, step=global_step)
            
            # Print progress
            mean_reward = early_stopper.get_mean_reward()
            print(
                f"Step: {global_step:>10,} | "
                f"Episodes: {episode_count:>6} | "
                f"Mean Reward (100): {mean_reward:>7.2f} | "
                f"Best: {early_stopper.best_mean_reward:>7.2f}"
            )
        
        # Save checkpoint
        if global_step % save_freq == 0:
            mean_reward = early_stopper.get_mean_reward()
            
            # Save latest
            save_path = f"{checkpoint_dir}/latest_{algorithm}.pt"
            os.makedirs(checkpoint_dir, exist_ok=True)
            agent.save(save_path)
            
            # Save best
            if mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                best_path = f"{checkpoint_dir}/best_{algorithm}.pt"
                agent.save(best_path)
                print(f"New best model saved! Mean reward: {mean_reward:.2f}")
            
            # Log video
            if use_wandb:
                video_path = find_latest_video(env_cfg.get("video_dir", "videos"))
                if video_path:
                    wandb.log({"train/video": wandb.Video(video_path)}, step=global_step)
    
    # Training completed
    print(f"\n{'=' * 60}")
    print("Training completed!")
    print(f"Final mean reward: {early_stopper.get_mean_reward():.2f}")
    print(f"Best mean reward: {best_mean_reward:.2f}")
    
    env.close()
    if use_wandb:
        wandb.finish()
    
    return early_stopper.get_mean_reward()


def sweep_train() -> None:
    """Training function for wandb sweep."""
    # Initialize wandb run
    wandb.init()
    
    # Load base config
    config = load_config("config/config.yaml")
    
    # Update config with sweep parameters
    sweep_config = dict(wandb.config)
    
    # Map sweep parameters to config
    if "policy_lr" in sweep_config:
        config["training"]["policy_lr"] = sweep_config["policy_lr"]
    if "dynamics_lr" in sweep_config:
        config["training"]["dynamics_lr"] = sweep_config["dynamics_lr"]
    if "gamma" in sweep_config:
        config["training"]["gamma"] = sweep_config["gamma"]
    if "batch_size" in sweep_config:
        config["training"]["batch_size"] = sweep_config["batch_size"]
    if "model_rollout_length" in sweep_config:
        config["training"]["model_rollout_length"] = sweep_config["model_rollout_length"]
    if "real_ratio" in sweep_config:
        config["training"]["real_ratio"] = sweep_config["real_ratio"]
    if "ensemble_size" in sweep_config:
        config["model"]["ensemble_size"] = sweep_config["ensemble_size"]
    if "tau" in sweep_config:
        config["training"]["tau"] = sweep_config["tau"]
    if "fc_dim" in sweep_config:
        config["model"]["fc_dim"] = sweep_config["fc_dim"]
    
    # Create dummy args
    args = argparse.Namespace(
        seed=None,
        device=None,
        no_wandb=True,  # Wandb already initialized
    )
    
    # Run training
    final_reward = train(config, args)
    
    # Log final metric for sweep
    wandb.log({"eval/mean_reward": final_reward})


def main() -> None:
    """Main entry point."""
    args = parse_args()
    
    if args.sweep:
        # Running as wandb sweep agent
        sweep_train()
    else:
        # Standard training
        config = load_config(args.config)
        
        # Verify environment setup first
        print("Verifying environment setup...")
        verify_env_setup()
        
        train(config, args)


if __name__ == "__main__":
    main()
