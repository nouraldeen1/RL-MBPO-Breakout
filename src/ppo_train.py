"""
PPO Trainer for SpaceInvadersNoFrameskip-v4
============================================

Complete PPO implementation with:
- Hyperparameters loaded from YAML config
- Vectorized environment support
- GAE advantage estimation
- Model checkpointing and evaluation
- W&B logging integration
- Video recording

Usage:
    python train_ppo.py --config config/ppo_config.yaml

Author: RL Project
"""

import argparse
import os
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.distributions import Categorical

# Import your existing modules
from env_factory import make_vec_env, make_eval_env
from models import ActorCriticPPO
from utils import set_seed, MetricsLogger

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Logging will be disabled.")


class PPOTrainer:
    """
    Proximal Policy Optimization trainer for Atari environments.
    
    Implements the PPO algorithm with Generalized Advantage Estimation (GAE)
    and supports vectorized environments for efficient data collection.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize PPO trainer with configuration.
        
        Args:
            config: Configuration dictionary loaded from YAML
        """
        self.config = config
        
        # Set random seed for reproducibility
        set_seed(config['environment']['seed'])
        
        # Setup device
        if config['device']['cuda'] and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{config['device']['device_id']}")
        else:
            self.device = torch.device('cpu')
        print(f"Using device: {self.device}")
        
        # Extract hyperparameters
        self.env_config = config['environment']
        self.ppo_config = config['ppo']
        self.train_config = config['training']
        self.log_config = config['logging']
        
        # Compute derived parameters
        self.num_envs = self.env_config['num_envs']
        self.num_steps = self.ppo_config['num_steps']
        self.batch_size = self.num_envs * self.num_steps
        self.minibatch_size = self.batch_size // self.ppo_config['num_minibatches']
        
        # Create environments
        self._setup_environments()
        
        # Create model and optimizer
        self._setup_model()
        
        # Setup logging
        self._setup_logging()
        
        # Training state
        self.global_step = 0
        self.iteration = 0
        self.best_eval_reward = -float('inf')
        
        # Storage for rollouts
        self._setup_storage()
        
        print(f"\nPPO Trainer initialized:")
        print(f"  Environment: {self.env_config['env_id']}")
        print(f"  Num Envs: {self.num_envs}")
        print(f"  Num Steps: {self.num_steps}")
        print(f"  Batch Size: {self.batch_size}")
        print(f"  Minibatch Size: {self.minibatch_size}")
        print(f"  Total Timesteps: {self.train_config['total_timesteps']}")
        print(f"  Num Iterations: {self.ppo_config['num_iterations']}")
    
    def _setup_environments(self):
        """Create training and evaluation environments."""
        # Training environment (vectorized)
        self.envs = make_vec_env(
            env_id=self.env_config['env_id'],
            num_envs=self.num_envs,
            seed=self.env_config['seed'],
            capture_video=self.train_config.get('record_video', False),
            video_dir=self.train_config.get('video_dir', 'videos'),
            video_trigger_freq=self.train_config.get('video_frequency', 100),
            noop_max=self.env_config.get('noop_max', 30),
            frame_skip=self.env_config.get('frame_skip', 4),
            frame_stack=self.env_config.get('frame_stack', 4),
        )
        
        # Get action space size
        single_action_space = self.envs.single_action_space
        self.num_actions = single_action_space.n
        
        # Get observation shape
        obs_shape = self.envs.single_observation_space.shape
        
        print(f"Action space: {self.num_actions} actions")
        print(f"Observation shape: {obs_shape}")
        
        # Validate observation shape
        if len(obs_shape) != 3:
            raise ValueError(f"Expected 3D observations (C, H, W), got shape {obs_shape}")
        if obs_shape[1:] != (84, 84):
            print(f"Warning: Expected 84x84 frames, got {obs_shape[1:]}")
    
    def _setup_model(self):
        """Create actor-critic model and optimizer."""
        # Get actual number of input channels from environment
        obs_shape = self.envs.single_observation_space.shape
        in_channels = obs_shape[0]  # First dimension is channels for (C, H, W)
        
        # Create model
        self.agent = ActorCriticPPO(
            num_actions=self.num_actions,
            in_channels=in_channels,
            fc_dim=self.config['model']['fc_dim']
        ).to(self.device)
        
        # Create optimizer
        self.optimizer = optim.Adam(
            self.agent.parameters(),
            lr=self.ppo_config['learning_rate'],
            eps=1e-5
        )
        
        num_params = sum(p.numel() for p in self.agent.parameters())
        print(f"Model created with {num_params:,} parameters")
        print(f"  Input channels: {in_channels}")
        print(f"  Action space: {self.num_actions}")
        print(f"  FC dimension: {self.config['model']['fc_dim']}")
    
    def _setup_storage(self):
        """Setup storage tensors for rollouts."""
        # Get observation shape from environment
        # For frame-stacked Atari: (num_frames, height, width)
        obs_shape = self.envs.single_observation_space.shape
        
        self.obs = torch.zeros((self.num_steps, self.num_envs) + obs_shape).to(self.device)
        self.actions = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.logprobs = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.rewards = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.dones = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.values = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
    
    def _setup_logging(self):
        """Setup logging (W&B if available)."""
        self.use_wandb = self.log_config.get('use_wandb', False) and WANDB_AVAILABLE
        
        if self.use_wandb:
            wandb.init(
                project=self.log_config.get('wandb_project', 'ppo-spaceinvaders'),
                entity=self.log_config.get('wandb_entity'),
                name=self.log_config.get('run_name', 'ppo_run'),
                config=self.config,
                sync_tensorboard=False,
            )
        
        self.metrics_logger = MetricsLogger()
    
    def compute_gae(self, next_obs: torch.Tensor, next_done: torch.Tensor) -> torch.Tensor:
        """
        Compute Generalized Advantage Estimation (GAE).
        
        Args:
            next_obs: Next observation after rollout
            next_done: Done flag for next observation
            
        Returns:
            Advantage estimates for each timestep
        """
        with torch.no_grad():
            # Get value for next observation using full forward pass
            _, _, _, next_value = self.agent.get_action_and_value(next_obs)
            next_value = next_value.reshape(1, -1)
            
            advantages = torch.zeros_like(self.rewards).to(self.device)
            lastgaelam = 0
            
            for t in reversed(range(self.num_steps)):
                if t == self.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.dones[t + 1]
                    nextvalues = self.values[t + 1]
                
                delta = self.rewards[t] + self.ppo_config['gamma'] * nextvalues * nextnonterminal - self.values[t]
                advantages[t] = lastgaelam = delta + self.ppo_config['gamma'] * self.ppo_config['gae_lambda'] * nextnonterminal * lastgaelam
            
            returns = advantages + self.values
        
        return advantages, returns
    
    def collect_rollout(self, next_obs: torch.Tensor, next_done: torch.Tensor):
        """
        Collect a rollout of experience.
        
        Args:
            next_obs: Starting observation
            next_done: Starting done flag
            
        Returns:
            Tuple of (next_obs, next_done, episode_info)
        """
        episode_rewards = []
        episode_lengths = []
        
        for step in range(self.num_steps):
            self.global_step += self.num_envs
            self.obs[step] = next_obs
            self.dones[step] = next_done
            
            # Get action and value from agent
            with torch.no_grad():
                action, logprob, _, value = self.agent.get_action_and_value(next_obs)
                self.values[step] = value.flatten()
            
            self.actions[step] = action
            self.logprobs[step] = logprob
            
            # Execute action in environment
            next_obs_np, reward, terminations, truncations, infos = self.envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            
            self.rewards[step] = torch.tensor(reward).to(self.device).view(-1)
            next_obs = torch.Tensor(next_obs_np).to(self.device)
            next_done = torch.Tensor(next_done).to(self.device)
            
            # Validate shapes
            if step == 0 and self.iteration == 1:
                print(f"\nDebug - Rollout shapes (first iteration only):")
                print(f"  next_obs shape: {next_obs.shape}")
                print(f"  action shape: {action.shape}")
                print(f"  reward shape: {self.rewards[step].shape}")
                print(f"  value shape: {value.shape}")
            
            # Track episode statistics
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        episode_rewards.append(info["episode"]["r"])
                        episode_lengths.append(info["episode"]["l"])
        
        return next_obs, next_done, {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths
        }
    
    def update_policy(self, advantages: torch.Tensor, returns: torch.Tensor) -> Dict:
        """
        Update policy using PPO algorithm.
        
        Args:
            advantages: Computed advantages
            returns: Computed returns
            
        Returns:
            Dictionary of training metrics
        """
        # Flatten batch
        b_obs = self.obs.reshape((-1,) + self.obs.shape[2:])
        b_logprobs = self.logprobs.reshape(-1)
        b_actions = self.actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = self.values.reshape(-1)
        
        # Normalize advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
        
        # Training metrics
        clipfracs = []
        pg_losses = []
        value_losses = []
        entropy_losses = []
        approx_kls = []
        
        # Optimizing the policy and value network
        b_inds = np.arange(self.batch_size)
        
        for epoch in range(self.ppo_config['update_epochs']):
            np.random.shuffle(b_inds)
            
            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_inds = b_inds[start:end]
                
                _, newlogprob, entropy, newvalue = self.agent.get_action_and_value(
                    b_obs[mb_inds], b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                
                with torch.no_grad():
                    # Calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs.append(((ratio - 1.0).abs() > self.ppo_config['clip_coef']).float().mean().item())
                
                mb_advantages = b_advantages[mb_inds]
                
                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1 - self.ppo_config['clip_coef'], 1 + self.ppo_config['clip_coef']
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                
                # Value loss
                newvalue = newvalue.view(-1)
                if self.ppo_config['clip_vloss']:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -self.ppo_config['clip_coef'],
                        self.ppo_config['clip_coef'],
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()
                
                # Entropy loss
                entropy_loss = entropy.mean()
                
                # Total loss
                loss = pg_loss - self.ppo_config['ent_coef'] * entropy_loss + v_loss * self.ppo_config['vf_coef']
                
                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.ppo_config['max_grad_norm'])
                self.optimizer.step()
                
                # Track metrics
                pg_losses.append(pg_loss.item())
                value_losses.append(v_loss.item())
                entropy_losses.append(entropy_loss.item())
                approx_kls.append(approx_kl.item())
            
            # Early stopping based on KL divergence
            if self.ppo_config.get('target_kl') is not None:
                if approx_kl > self.ppo_config['target_kl']:
                    break
        
        # Explained variance
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        
        return {
            'loss/policy_loss': np.mean(pg_losses),
            'loss/value_loss': np.mean(value_losses),
            'loss/entropy': np.mean(entropy_losses),
            'loss/approx_kl': np.mean(approx_kls),
            'loss/clipfrac': np.mean(clipfracs),
            'loss/explained_variance': explained_var,
        }
    
    def evaluate(self, num_episodes: int = 5) -> Dict:
        """
        Evaluate the current policy.
        
        Args:
            num_episodes: Number of episodes to evaluate
            
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\nEvaluating for {num_episodes} episodes...")
        
        # Create evaluation environment
        eval_env = make_eval_env(
            env_id=self.env_config['env_id'],
            seed=self.env_config['seed'] + 1000,
            video_dir=f"{self.train_config['video_dir']}/eval",
            record_every=1,  # Record all evaluation episodes
        )
        
        episode_rewards = []
        episode_lengths = []
        
        for ep in range(num_episodes):
            obs, _ = eval_env.reset()
            obs = torch.Tensor(obs).unsqueeze(0).to(self.device)
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                with torch.no_grad():
                    action, _, _, _ = self.agent.get_action_and_value(obs, deterministic=True)
                
                obs, reward, terminated, truncated, _ = eval_env.step(action.cpu().item())
                obs = torch.Tensor(obs).unsqueeze(0).to(self.device)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            print(f"  Episode {ep + 1}/{num_episodes}: Reward = {episode_reward:.2f}, Length = {episode_length}")
        
        eval_env.close()
        
        return {
            'eval/mean_reward': np.mean(episode_rewards),
            'eval/std_reward': np.std(episode_rewards),
            'eval/mean_length': np.mean(episode_lengths),
            'eval/max_reward': np.max(episode_rewards),
            'eval/min_reward': np.min(episode_rewards),
        }
    
    def save_checkpoint(self, path: Optional[str] = None):
        """Save model checkpoint."""
        if path is None:
            checkpoint_dir = Path(self.train_config['checkpoint_dir'])
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            path = checkpoint_dir / f"checkpoint_iter_{self.iteration}.pt"
        
        torch.save({
            'iteration': self.iteration,
            'global_step': self.global_step,
            'model_state_dict': self.agent.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'best_eval_reward': self.best_eval_reward,
        }, path)
        
        print(f"Checkpoint saved to {path}")
    
    def train(self):
        """Main training loop."""
        print(f"\n{'='*60}")
        print("Starting PPO Training")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        # Initialize environment
        next_obs, _ = self.envs.reset(seed=self.env_config['seed'])
        next_obs = torch.Tensor(next_obs).to(self.device)
        next_done = torch.zeros(self.num_envs).to(self.device)
        
        # Training loop
        for iteration in range(1, self.ppo_config['num_iterations'] + 1):
            self.iteration = iteration
            
            # Anneal learning rate
            if self.ppo_config.get('anneal_lr', False):
                frac = 1.0 - (iteration - 1.0) / self.ppo_config['num_iterations']
                lrnow = frac * self.ppo_config['learning_rate']
                self.optimizer.param_groups[0]['lr'] = lrnow
            
            # Collect rollout
            next_obs, next_done, episode_info = self.collect_rollout(next_obs, next_done)
            
            # Compute advantages
            advantages, returns = self.compute_gae(next_obs, next_done)
            
            # Update policy
            update_metrics = self.update_policy(advantages, returns)
            
            # Compute SPS (steps per second)
            sps = int(self.global_step / (time.time() - start_time))
            
            # Logging
            if iteration % self.log_config.get('log_frequency', 1) == 0:
                metrics = {
                    'train/iteration': iteration,
                    'train/global_step': self.global_step,
                    'train/sps': sps,
                    'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                    **update_metrics,
                }
                
                # Add episode metrics if available
                if episode_info['episode_rewards']:
                    metrics.update({
                        'train/episode_reward': np.mean(episode_info['episode_rewards']),
                        'train/episode_length': np.mean(episode_info['episode_lengths']),
                    })
                    
                    print(f"Iter {iteration:4d} | Step {self.global_step:8d} | "
                          f"Reward: {metrics['train/episode_reward']:6.2f} | "
                          f"SPS: {sps:5d}")
                
                # Log to wandb
                if self.use_wandb:
                    wandb.log(metrics, step=self.global_step)
            
            # Evaluation
            if iteration % self.train_config.get('eval_frequency', 50) == 0:
                eval_metrics = self.evaluate(self.train_config.get('eval_episodes', 5))
                
                if self.use_wandb:
                    wandb.log(eval_metrics, step=self.global_step)
                
                print(f"\n{'='*60}")
                print(f"Evaluation at iteration {iteration}")
                print(f"  Mean Reward: {eval_metrics['eval/mean_reward']:.2f} ± {eval_metrics['eval/std_reward']:.2f}")
                print(f"  Max Reward: {eval_metrics['eval/max_reward']:.2f}")
                print(f"{'='*60}\n")
                
                # Save best model
                if eval_metrics['eval/mean_reward'] > self.best_eval_reward:
                    self.best_eval_reward = eval_metrics['eval/mean_reward']
                    checkpoint_dir = Path(self.train_config['checkpoint_dir'])
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    best_path = checkpoint_dir / "best_model.pt"
                    torch.save(self.agent.state_dict(), best_path)
                    print(f"New best model saved! Reward: {self.best_eval_reward:.2f}")
            
            # Save checkpoint
            if iteration % self.train_config.get('save_frequency', 100) == 0:
                self.save_checkpoint()
        
        # Final save
        print(f"\n{'='*60}")
        print("Training Complete!")
        print(f"{'='*60}\n")
        
        self.save_checkpoint()
        
        # Final evaluation
        final_eval = self.evaluate(num_episodes=10)
        print(f"\nFinal Evaluation (10 episodes):")
        print(f"  Mean Reward: {final_eval['eval/mean_reward']:.2f} ± {final_eval['eval/std_reward']:.2f}")
        
        # Cleanup
        self.envs.close()
        
        if self.use_wandb:
            wandb.finish()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train PPO on SpaceInvaders')
    parser.add_argument('--config', type=str, default='config/ppo_config.yaml',
                        help='Path to config file')
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading config from {args.config}...")
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create trainer and train
    trainer = PPOTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()