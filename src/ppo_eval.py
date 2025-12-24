"""
PPO Model Evaluation Script
============================

Evaluate a trained PPO model and record videos.

Usage:
    python evaluate_ppo.py --checkpoint checkpoints/ppo_spaceinvaders/best_model.pt --episodes 10
"""

import argparse
import numpy as np
import torch
import yaml
from pathlib import Path

from env_factory import make_eval_env
from models import ActorCriticPPO



def load_model(checkpoint_path: str, device: str = 'cuda') -> ActorCriticPPO:
    """Load a trained PPO model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device,weights_only=False)
    
    # If it's a full checkpoint, extract model state
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model_state = checkpoint['model_state_dict']
        config = checkpoint.get('config', {})
        num_actions = 6  # Space Invaders default
    else:
        # Direct state dict
        model_state = checkpoint
        num_actions = 6
    
    # Create model
    model = ActorCriticPPO(num_actions=num_actions, in_channels=4, fc_dim=512)
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    return model


def evaluate_model(
    model: ActorCriticPPO,
    env_id: str = "SpaceInvadersNoFrameskip-v4",
    num_episodes: int = 10,
    seed: int = 42,
    deterministic: bool = True,
    record_video: bool = True,
    video_dir: str = "videos/eval",
    device: str = 'cuda'
):
    """
    Evaluate a trained model.
    
    Args:
        model: Trained ActorCritic model
        env_id: Environment ID
        num_episodes: Number of episodes to evaluate
        seed: Random seed
        deterministic: Use deterministic actions (argmax)
        record_video: Whether to record videos
        video_dir: Directory to save videos
        device: Device for inference
    """
    print(f"\n{'='*60}")
    print(f"Evaluating on {env_id}")
    print(f"{'='*60}\n")
    
    # Create evaluation environment
    env = make_eval_env(
        env_id=env_id,
        seed=seed,
        video_dir=video_dir,
        record_every=1 if record_video else 999999,
    )
    
    episode_rewards = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs, _ = env.reset()
        obs = torch.Tensor(obs).unsqueeze(0).to(device)
        
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            with torch.no_grad():
                action, _, _, value = model.get_action_and_value(obs, deterministic=deterministic)
            
            obs, reward, terminated, truncated, info = env.step(action.cpu().item())
            obs = torch.Tensor(obs).unsqueeze(0).to(device)
            done = terminated or truncated
            
            episode_reward += reward
            episode_length += 1
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        print(f"Episode {episode + 1:2d}/{num_episodes}: "
              f"Reward = {episode_reward:6.1f}, Length = {episode_length:4d}")
    
    env.close()
    
    # Compute statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    max_reward = np.max(episode_rewards)
    min_reward = np.min(episode_rewards)
    mean_length = np.mean(episode_lengths)
    
    print(f"\n{'='*60}")
    print("Evaluation Results")
    print(f"{'='*60}")
    print(f"Mean Reward:   {mean_reward:7.2f} ± {std_reward:.2f}")
    print(f"Max Reward:    {max_reward:7.2f}")
    print(f"Min Reward:    {min_reward:7.2f}")
    print(f"Mean Length:   {mean_length:7.2f}")
    print(f"{'='*60}\n")
    
    if record_video:
        print(f"Videos saved to: {video_dir}")
    
    return {
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'max_reward': max_reward,
        'min_reward': min_reward,
        'mean_length': mean_length,
        'all_rewards': episode_rewards,
        'all_lengths': episode_lengths,
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate PPO model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--env-id', type=str, default='SpaceInvadersNoFrameskip-v4',
                        help='Environment ID')
    parser.add_argument('--episodes', type=int, default=100,
                        help='Number of episodes to evaluate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--no-video', action='store_true',
                        help='Disable video recording')
    parser.add_argument('--video-dir', type=str, default='videos/eval',
                        help='Directory to save videos')
    parser.add_argument('--stochastic', action='store_true',
                        help='Use stochastic actions instead of deterministic')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use for inference')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    else:
        device = args.device
    
    # Load model
    model = load_model(args.checkpoint, device=device)
    
    # Evaluate
    results = evaluate_model(
        model=model,
        env_id=args.env_id,
        num_episodes=args.episodes,
        seed=args.seed,
        deterministic=not args.stochastic,
        record_video=not args.no_video,
        video_dir=args.video_dir,
        device=device,
    )
    
    # Save results to file
    results_file = Path(args.video_dir) / "evaluation_results.txt"
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_file, 'w') as f:
        f.write(f"Evaluation Results\n")
        f.write(f"{'='*60}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Environment: {args.env_id}\n")
        f.write(f"Episodes: {args.episodes}\n")
        f.write(f"{'='*60}\n\n")
        f.write(f"Mean Reward:   {results['mean_reward']:.2f} ± {results['std_reward']:.2f}\n")
        f.write(f"Max Reward:    {results['max_reward']:.2f}\n")
        f.write(f"Min Reward:    {results['min_reward']:.2f}\n")
        f.write(f"Mean Length:   {results['mean_length']:.2f}\n\n")
        f.write(f"All Episode Rewards:\n")
        for i, r in enumerate(results['all_rewards'], 1):
            f.write(f"  Episode {i:2d}: {r:.2f}\n")
    
    print(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()