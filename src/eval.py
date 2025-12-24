"""
Evaluation Script for MBPO-SpaceInvaders
===================================

This script provides evaluation utilities for trained agents:
- Run deterministic episodes
- Compute average scores with confidence intervals
- Record and log evaluation videos to Wandb
- Generate evaluation reports

Usage:
    # Evaluate a trained model
    python eval.py --checkpoint checkpoints/best_mbpo.pt
    
    # Evaluate with custom number of episodes
    python eval.py --checkpoint checkpoints/best_mbpo.pt --episodes 20
    
    # Evaluate and log to wandb
    python eval.py --checkpoint checkpoints/best_mbpo.pt --wandb

Author: CMPS458 RL Project
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import wandb

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from agents import create_agent
from env_factory import make_eval_env
from utils import load_config, get_video_files, set_seed


def evaluate_agent(
    agent,
    env,
    n_episodes: int = 10,
    deterministic: bool = True,
    verbose: bool = True,
) -> Tuple[List[float], List[int], List[str]]:
    """
    Evaluate agent over multiple episodes.
    
    Args:
        agent: Trained agent with get_action method
        env: Evaluation environment
        n_episodes: Number of episodes to run
        deterministic: Whether to use deterministic actions
        verbose: Whether to print episode results
    
    Returns:
        rewards: List of episode rewards
        lengths: List of episode lengths
        video_files: List of recorded video file paths
    """
    episode_rewards = []
    episode_lengths = []
    
    if verbose:
        print(f"\nRunning {n_episodes} evaluation episodes...")
        print("-" * 50)
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0
        episode_length = 0
        
        while not done:
            # Get action from agent
            action, _ = agent.get_action(obs, deterministic=deterministic)
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            total_reward += reward
            episode_length += 1
        
        episode_rewards.append(total_reward)
        episode_lengths.append(episode_length)
        
        if verbose:
            print(f"Episode {episode + 1:>3}/{n_episodes} | "
                  f"Reward: {total_reward:>7.1f} | "
                  f"Length: {episode_length:>5}")
    
    # Get recorded videos
    video_dir = Path(env.video_folder) if hasattr(env, 'video_folder') else Path("videos/eval")
    video_files = get_video_files(str(video_dir), n_latest=n_episodes)
    
    return episode_rewards, episode_lengths, video_files


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """
    Compute statistics for a list of values.
    
    Args:
        values: List of numeric values
    
    Returns:
        Dictionary with mean, std, min, max, median, ci_95
    """
    values = np.array(values)
    n = len(values)
    
    mean = float(np.mean(values))
    std = float(np.std(values))
    
    # 95% confidence interval
    ci_95 = 1.96 * std / np.sqrt(n) if n > 0 else 0.0
    
    return {
        "mean": mean,
        "std": std,
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "median": float(np.median(values)),
        "ci_95_lower": mean - ci_95,
        "ci_95_upper": mean + ci_95,
    }


def generate_report(
    episode_rewards: List[float],
    episode_lengths: List[int],
    config: Dict,
    checkpoint_path: str,
) -> str:
    """
    Generate evaluation report as formatted string.
    
    Args:
        episode_rewards: List of episode rewards
        episode_lengths: List of episode lengths
        config: Configuration dictionary
        checkpoint_path: Path to evaluated checkpoint
    
    Returns:
        Formatted report string
    """
    reward_stats = compute_statistics(episode_rewards)
    length_stats = compute_statistics(episode_lengths)
    
    report = f"""
================================================================================
                        MBPO-SpaceInvaders Evaluation Report
================================================================================

Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Checkpoint: {checkpoint_path}
Algorithm: {config.get('algorithm', 'mbpo').upper()}

--------------------------------------------------------------------------------
                              Episode Rewards
--------------------------------------------------------------------------------
  Mean Reward:        {reward_stats['mean']:>10.2f}
  Std Deviation:      {reward_stats['std']:>10.2f}
  Min Reward:         {reward_stats['min']:>10.2f}
  Max Reward:         {reward_stats['max']:>10.2f}
  Median Reward:      {reward_stats['median']:>10.2f}
  95% CI:             [{reward_stats['ci_95_lower']:.2f}, {reward_stats['ci_95_upper']:.2f}]

--------------------------------------------------------------------------------
                              Episode Lengths
--------------------------------------------------------------------------------
  Mean Length:        {length_stats['mean']:>10.1f}
  Std Deviation:      {length_stats['std']:>10.1f}
  Min Length:         {length_stats['min']:>10.0f}
  Max Length:         {length_stats['max']:>10.0f}

--------------------------------------------------------------------------------
                              Individual Episodes
--------------------------------------------------------------------------------
"""
    
    for i, (reward, length) in enumerate(zip(episode_rewards, episode_lengths)):
        report += f"  Episode {i + 1:>3}: Reward = {reward:>7.1f}, Length = {length:>5}\n"
    
    report += f"""
================================================================================
                              Configuration
================================================================================
  Environment:        {config.get('env', {}).get('name', 'SpaceInvadersNoFrameskip-v4')}
  Frame Stack:        {config.get('env', {}).get('frame_stack', 4)}
  Ensemble Size:      {config.get('model', {}).get('ensemble_size', 5)}
  FC Dimension:       {config.get('model', {}).get('fc_dim', 512)}
  
================================================================================
"""
    
    return report


def log_to_wandb(
    episode_rewards: List[float],
    episode_lengths: List[int],
    video_files: List[str],
    config: Dict,
    checkpoint_path: str,
) -> None:
    """
    Log evaluation results to Wandb.
    
    Args:
        episode_rewards: List of episode rewards
        episode_lengths: List of episode lengths
        video_files: List of video file paths
        config: Configuration dictionary
        checkpoint_path: Path to evaluated checkpoint
    """
    reward_stats = compute_statistics(episode_rewards)
    length_stats = compute_statistics(episode_lengths)
    
    # Create wandb run
    wandb_cfg = config.get("wandb", {})
    wandb.init(
        project=wandb_cfg.get("project", "MBPO-SpaceInvaders"),
        entity=wandb_cfg.get("entity"),
        name=f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config=config,
        tags=["evaluation", config.get("algorithm", "mbpo")],
        job_type="evaluation",
    )
    
    # Log summary metrics
    wandb.log({
        "eval/mean_reward": reward_stats["mean"],
        "eval/std_reward": reward_stats["std"],
        "eval/min_reward": reward_stats["min"],
        "eval/max_reward": reward_stats["max"],
        "eval/median_reward": reward_stats["median"],
        "eval/ci_95_lower": reward_stats["ci_95_lower"],
        "eval/ci_95_upper": reward_stats["ci_95_upper"],
        "eval/mean_length": length_stats["mean"],
        "eval/n_episodes": len(episode_rewards),
    })
    
    # Log episode table
    episode_data = [
        [i + 1, reward, length]
        for i, (reward, length) in enumerate(zip(episode_rewards, episode_lengths))
    ]
    table = wandb.Table(
        columns=["Episode", "Reward", "Length"],
        data=episode_data,
    )
    wandb.log({"eval/episodes": table})
    
    # Log videos
    print(f"\nLogging {len(video_files)} videos to wandb...")
    for i, video_path in enumerate(video_files[:5]):  # Log up to 5 videos
        if os.path.exists(video_path):
            try:
                wandb.log({f"eval/video_{i + 1}": wandb.Video(video_path)})
                print(f"  Logged: {video_path}")
            except Exception as e:
                print(f"  Failed to log {video_path}: {e}")
    
    # Create histogram of rewards
    wandb.log({
        "eval/reward_histogram": wandb.Histogram(episode_rewards),
    })
    
    # Log checkpoint artifact
    artifact = wandb.Artifact(
        f"eval-checkpoint-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        type="model",
    )
    artifact.add_file(checkpoint_path)
    wandb.log_artifact(artifact)
    
    wandb.finish()
    print("Results logged to Wandb successfully!")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="MBPO SpaceInvaders Evaluation")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log results to wandb",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Disable video recording",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for evaluation report",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Use stochastic (non-deterministic) actions",
    )
    return parser.parse_args()


def main() -> None:
    """Main evaluation function."""
    args = parse_args()
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create environment
    print("\nCreating evaluation environment...")
    env_cfg = config.get("env", {})
    env = make_eval_env(
        env_id=env_cfg.get("name", "SpaceInvadersNoFrameskip-v4"),
        seed=args.seed,
        video_dir="videos/eval",
        record_every=1 if not args.no_video else 999999,
        noop_max=env_cfg.get("noop_max", 30),
        frame_skip=env_cfg.get("frame_skip", 4),
        frame_stack=env_cfg.get("frame_stack", 4),
    )
    
    # Get observation/action shapes
    obs_shape = env.observation_space.shape
    num_actions = env.action_space.n
    
    print(f"Observation shape: {obs_shape}")
    print(f"Number of actions: {num_actions}")
    
    # Create agent
    algorithm = config.get("algorithm", "mbpo")
    print(f"\nCreating {algorithm.upper()} agent...")
    agent = create_agent(algorithm, obs_shape, num_actions, config, device)
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    agent.load(args.checkpoint)
    
    # Run evaluation
    episode_rewards, episode_lengths, video_files = evaluate_agent(
        agent=agent,
        env=env,
        n_episodes=args.episodes,
        deterministic=not args.stochastic,
        verbose=True,
    )
    
    # Compute and print statistics
    reward_stats = compute_statistics(episode_rewards)
    
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Mean Reward:   {reward_stats['mean']:.2f} ± {reward_stats['std']:.2f}")
    print(f"95% CI:        [{reward_stats['ci_95_lower']:.2f}, {reward_stats['ci_95_upper']:.2f}]")
    print(f"Min/Max:       [{reward_stats['min']:.0f}, {reward_stats['max']:.0f}]")
    print("=" * 60)
    
    # Generate report
    report = generate_report(
        episode_rewards=episode_rewards,
        episode_lengths=episode_lengths,
        config=config,
        checkpoint_path=args.checkpoint,
    )
    
    # Save report if output specified
    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"\nReport saved to: {args.output}")
    
    # Log to wandb if enabled
    if args.wandb:
        print("\nLogging to Wandb...")
        log_to_wandb(
            episode_rewards=episode_rewards,
            episode_lengths=episode_lengths,
            video_files=video_files,
            config=config,
            checkpoint_path=args.checkpoint,
        )
    
    # Cleanup
    env.close()
    
    print("\nEvaluation complete!")
    
    # Return results for programmatic use
    return {
        "mean_reward": reward_stats["mean"],
        "std_reward": reward_stats["std"],
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
    }


if __name__ == "__main__":
    main()
