"""
Dreamer Training Script for SpaceInvaders
==========================================

Train Dreamer agent on SpaceInvadersNoFrameskip-v4.
Optimized for Kaggle environment.

Usage:
    python train_dreamer.py --steps 1000000
"""

import argparse
import os
import time
from pathlib import Path
from collections import deque
import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym
import ale_py  
from gymnasium.wrappers import AtariPreprocessing, RecordVideo
import cv2

try:
    import wandb
    WANDB_AVAILABLE = True
except:
    WANDB_AVAILABLE = False


# =============================================================================
# Replay Buffer
# =============================================================================

class ReplayBuffer:
    """Replay buffer for storing episodes."""
    
    def __init__(self, capacity=1000, seq_len=50):
        self.capacity = capacity
        self.seq_len = seq_len
        self.episodes = deque(maxlen=capacity)
    
    def add_episode(self, episode):
        """
        Add episode to buffer.
        
        episode: dict with keys 'obs', 'actions', 'rewards', 'dones'
        """
        self.episodes.append(episode)
    
    def sample(self, batch_size):
        """Sample batch of sequences."""
        episodes = np.random.choice(self.episodes, batch_size, replace=True)
        
        batch = {
            'obs': [],
            'actions': [],
            'rewards': [],
            'dones': [],
        }
        
        for episode in episodes:
            ep_len = len(episode['obs'])
            
            if ep_len > self.seq_len:
                # Sample random start
                start = np.random.randint(0, ep_len - self.seq_len)
                end = start + self.seq_len
            else:
                # Use entire episode + padding
                start = 0
                end = ep_len
            
            # Extract sequence
            obs_seq = episode['obs'][start:end]
            action_seq = episode['actions'][start:end]
            reward_seq = episode['rewards'][start:end]
            done_seq = episode['dones'][start:end]
            
            # Pad if necessary
            if len(obs_seq) < self.seq_len:
                pad_len = self.seq_len - len(obs_seq)
                obs_seq = np.concatenate([obs_seq] + [obs_seq[-1:]] * pad_len)
                action_seq = np.concatenate([action_seq] + [action_seq[-1:]] * pad_len)
                reward_seq = np.concatenate([reward_seq, np.zeros(pad_len)])
                done_seq = np.concatenate([done_seq, np.ones(pad_len)])
            
            batch['obs'].append(obs_seq)
            batch['actions'].append(action_seq)
            batch['rewards'].append(reward_seq)
            batch['dones'].append(done_seq)
        
        # Convert to tensors
        batch = {
            'obs': torch.FloatTensor(np.stack(batch['obs'])),
            'actions': torch.FloatTensor(np.stack(batch['actions'])),
            'rewards': torch.FloatTensor(np.stack(batch['rewards'])),
            'dones': torch.FloatTensor(np.stack(batch['dones'])),
        }
        
        return batch
    
    def __len__(self):
        return len(self.episodes)


# =============================================================================
# Environment Wrapper
# =============================================================================

class DreamerAtariWrapper(gym.Wrapper):
    """Atari wrapper for Dreamer."""
    
    def __init__(self, env, size=64, grayscale=False):
        super().__init__(env)
        self.size = size
        self.grayscale = grayscale
        
        # Update observation space
        if grayscale:
            obs_shape = (1, size, size)
        else:
            obs_shape = (3, size, size)
        
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=obs_shape, dtype=np.float32
        )
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._process_obs(obs), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._process_obs(obs), reward, terminated, truncated, info
    
    def _process_obs(self, obs):
        """Resize and normalize observation."""
        # obs is already from AtariPreprocessing (84x84 grayscale) or (210x160x3)
        # We need to resize to 64x64 and convert to CHW format
        
        if obs.ndim == 2:
            # Grayscale
            obs = cv2.resize(obs, (self.size, self.size), interpolation=cv2.INTER_AREA)
            if self.grayscale:
                obs = obs[np.newaxis, ...]  # Add channel dimension
            else:
                # Convert to RGB
                obs = np.stack([obs] * 3, axis=0)
        else:
            # Color (HWC)
            obs = cv2.resize(obs, (self.size, self.size), interpolation=cv2.INTER_AREA)
            if self.grayscale:
                obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)[np.newaxis, ...]
            else:
                obs = obs.transpose(2, 0, 1)  # HWC -> CHW
        
        # Normalize to [0, 1]
        obs = obs.astype(np.float32) / 255.0
        
        return obs


def make_dreamer_env(env_id='SpaceInvadersNoFrameskip-v4', size=64, seed=0):
    """Create Dreamer-compatible Atari environment."""
    env = gym.make(env_id)
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        grayscale_obs=False,  # Keep color for Dreamer
        grayscale_newaxis=False,
        scale_obs=False,  # We'll scale in DreamerAtariWrapper
    )
    env = DreamerAtariWrapper(env, size=size, grayscale=False)
    env.action_space.seed(seed)
    return env


# =============================================================================
# Dreamer Trainer
# =============================================================================

class DreamerTrainer:
    """Dreamer training loop."""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Using device: {self.device}")
        
        # Create environment
        self.env = make_dreamer_env(
            config['env_id'],
            size=config['obs_size'],
            seed=config['seed']
        )
        
        self.action_dim = self.env.action_space.n
        self.obs_shape = self.env.observation_space.shape
        
        print(f"Environment: {config['env_id']}")
        print(f"Observation shape: {self.obs_shape}")
        print(f"Action dim: {self.action_dim}")
        
        # Create agent
        from DreamerArch import DreamerAgent
        self.agent = DreamerAgent(
            obs_shape=self.obs_shape,
            action_dim=self.action_dim,
            embed_dim=config['embed_dim'],
            stoch_dim=config['stoch_dim'],
            deter_dim=config['deter_dim'],
            hidden_dim=config['hidden_dim'],
            num_classes=config['num_classes'],
        ).to(self.device)
        
        print(f"Model parameters: {sum(p.numel() for p in self.agent.parameters()):,}")
        
        # Optimizers
        self.world_model_optimizer = torch.optim.Adam(
            list(self.agent.encoder.parameters()) +
            list(self.agent.decoder.parameters()) +
            list(self.agent.rssm.parameters()) +
            list(self.agent.reward_model.parameters()) +
            list(self.agent.discount_model.parameters()),
            lr=config['world_model_lr']
        )
        
        self.actor_optimizer = torch.optim.Adam(
            self.agent.actor.parameters(),
            lr=config['actor_lr']
        )
        
        self.critic_optimizer = torch.optim.Adam(
            self.agent.critic.parameters(),
            lr=config['critic_lr']
        )
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(
            capacity=config['buffer_capacity'],
            seq_len=config['seq_len']
        )
        
        # Training state
        self.step = 0
        self.episode = 0
        self.best_reward = -float('inf')
        
        # Setup logging
        if config['use_wandb'] and WANDB_AVAILABLE:
            wandb.init(
                project=config['wandb_project'],
                name=config['run_name'],
                config=config
            )
    
    def collect_episode(self):
        """Collect one episode."""
        obs, _ = self.env.reset()
        done = False
        
        episode = {
            'obs': [],
            'actions': [],
            'rewards': [],
            'dones': [],
        }
        
        state = self.agent.rssm.initial_state(1, self.device)
        
        while not done:
            # Encode observation
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                embed = self.agent.encoder(obs_tensor)
                
                # Get feature
                feature = self.agent.rssm.get_feature(state)
                
                # Get action
                action, action_onehot, _ = self.agent.actor.get_action(
                    feature, deterministic=False
                )
                
                # Update state (for next step)
                state, _, _ = self.agent.rssm.observe(
                    embed, action_onehot, state
                )
            
            # Step environment
            next_obs, reward, terminated, truncated, info = self.env.step(action.item())
            done = terminated or truncated
            
            # Store transition
            episode['obs'].append(obs)
            episode['actions'].append(F.one_hot(action, self.action_dim).squeeze(0).cpu().numpy())
            episode['rewards'].append(reward)
            episode['dones'].append(float(done))
            
            obs = next_obs
            self.step += 1
        
        # Convert lists to arrays
        episode['obs'] = np.array(episode['obs'])
        episode['actions'] = np.array(episode['actions'])
        episode['rewards'] = np.array(episode['rewards'])
        episode['dones'] = np.array(episode['dones'])
        
        episode_reward = sum(episode['rewards'])
        episode_length = len(episode['rewards'])
        
        return episode, episode_reward, episode_length
    
    def train_step(self):
        """One training step."""
        if len(self.replay_buffer) < self.config['batch_size']:
            return {}
        
        # Sample batch
        batch = self.replay_buffer.sample(self.config['batch_size'])
        batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Train world model
        self.world_model_optimizer.zero_grad()
        world_model_losses = self.agent.world_model_loss(
            batch['obs'], batch['actions'], batch['rewards'], batch['dones']
        )
        world_model_losses['world_model_loss'].backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.agent.encoder.parameters()) +
            list(self.agent.decoder.parameters()) +
            list(self.agent.rssm.parameters()) +
            list(self.agent.reward_model.parameters()) +
            list(self.agent.discount_model.parameters()),
            self.config['grad_clip']
        )
        self.world_model_optimizer.step()
        
        # Train actor-critic
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        
        actor_critic_losses = self.agent.actor_critic_loss(
            batch['obs'], batch['actions']
        )
        
        (actor_critic_losses['actor_loss'] + actor_critic_losses['critic_loss']).backward()
        
        torch.nn.utils.clip_grad_norm_(
            self.agent.actor.parameters(), self.config['grad_clip']
        )
        torch.nn.utils.clip_grad_norm_(
            self.agent.critic.parameters(), self.config['grad_clip']
        )
        
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        
        # Combine losses
        losses = {**world_model_losses, **actor_critic_losses}
        
        return losses
    
    def train(self):
        """Main training loop."""
        print("\nStarting Dreamer training...\n")
        
        start_time = time.time()
        
        # Collect initial episodes
        print("Collecting initial episodes...")
        while len(self.replay_buffer) < self.config['prefill']:
            episode, reward, length = self.collect_episode()
            self.replay_buffer.add_episode(episode)
            print(f"  Episode {self.episode + 1}: Reward={reward:.1f}, Length={length}")
            self.episode += 1
        
        print(f"\nCollected {len(self.replay_buffer)} episodes")
        print("Starting training...\n")
        
        # Training loop
        while self.step < self.config['total_steps']:
            # Collect episode
            episode, reward, length = self.collect_episode()
            self.replay_buffer.add_episode(episode)
            
            # Track best reward
            if reward > self.best_reward:
                self.best_reward = reward
                self.save_checkpoint('best_model.pt')
            
            # Training steps
            for _ in range(self.config['train_steps_per_episode']):
                losses = self.train_step()
            
            self.episode += 1
            
            # Logging
            if self.episode % self.config['log_freq'] == 0:
                elapsed = time.time() - start_time
                sps = self.step / elapsed
                
                print(f"Episode {self.episode:4d} | Step {self.step:7d} | "
                      f"Reward: {reward:6.1f} | Length: {length:4d} | "
                      f"Best: {self.best_reward:6.1f} | SPS: {sps:5.0f}")
                
                if losses and self.config['use_wandb'] and WANDB_AVAILABLE:
                    wandb.log({
                        'episode': self.episode,
                        'step': self.step,
                        'reward': reward,
                        'length': length,
                        'best_reward': self.best_reward,
                        **losses
                    })
            
            # Save checkpoint
            if self.episode % self.config['save_freq'] == 0:
                self.save_checkpoint(f'checkpoint_ep_{self.episode}.pt')
        
        print("\nTraining complete!")
        self.save_checkpoint('final_model.pt')
        
        if self.config['use_wandb'] and WANDB_AVAILABLE:
            wandb.finish()
    
    def save_checkpoint(self, filename):
        """Save checkpoint."""
        os.makedirs(self.config['checkpoint_dir'], exist_ok=True)
        path = os.path.join(self.config['checkpoint_dir'], filename)
        
        torch.save({
            'agent': self.agent.state_dict(),
            'world_model_optimizer': self.world_model_optimizer.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'step': self.step,
            'episode': self.episode,
            'best_reward': self.best_reward,
            'config': self.config,
        }, path)
        
        print(f"Checkpoint saved: {path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-id', type=str, default="SpaceInvadersNoFrameskip-v4")
    parser.add_argument('--steps', type=int, default=1000000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--obs-size', type=int, default=64)
    parser.add_argument('--no-wandb', action='store_true')
    args = parser.parse_args()
    
    config = {
        # Environment
        'env_id': args.env_id,
        'seed': args.seed,
        'obs_size': args.obs_size,
        
        # Model
        'embed_dim': 1024,
        'stoch_dim': 32,
        'deter_dim': 200,
        'hidden_dim': 200,
        'num_classes': 32,
        
        # Training
        'total_steps': args.steps,
        'prefill': 5,  # Initial episodes to collect
        'batch_size': 16,
        'seq_len': 50,
        'train_steps_per_episode': 100,
        
        # Optimization
        'world_model_lr': 1e-4,
        'actor_lr': 3e-5,
        'critic_lr': 3e-5,
        'grad_clip': 100.0,
        
        # Buffer
        'buffer_capacity': 1000,
        
        # Logging
        'log_freq': 10,
        'save_freq': 100,
        'checkpoint_dir': 'checkpoints/dreamer_spaceinvaders',
        'use_wandb': not args.no_wandb,
        'wandb_project': 'dreamer-spaceinvaders',
        'run_name': f'dreamer_{args.env_id}_{args.seed}',
    }
    
    trainer = DreamerTrainer(config)
    trainer.train()


if __name__ == '__main__':
    main()