"""
Utility Classes for MBPO-Breakout
=================================

This module provides utility classes and functions:
- ReplayBuffer: Experience storage with efficient sampling
- ModelReplayBuffer: Buffer for model-generated transitions
- EarlyStopping: Monitor and stop training at target performance
- Various helper functions

Author: CMPS458 RL Project
"""

from __future__ import annotations

import os
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple, Union

import numpy as np
import torch
import yaml


# =============================================================================
# Data Structures
# =============================================================================

class Transition(NamedTuple):
    """Single transition tuple."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


@dataclass
class Batch:
    """Batch of transitions for training."""
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_states: torch.Tensor
    dones: torch.Tensor
    
    def to(self, device: str) -> "Batch":
        """Move batch to specified device."""
        return Batch(
            states=self.states.to(device),
            actions=self.actions.to(device),
            rewards=self.rewards.to(device),
            next_states=self.next_states.to(device),
            dones=self.dones.to(device),
        )


# =============================================================================
# Replay Buffer
# =============================================================================

class ReplayBuffer:
    """
    Experience replay buffer for off-policy learning.
    
    Stores transitions and provides efficient random sampling.
    Uses numpy arrays for memory efficiency.
    
    Args:
        capacity: Maximum number of transitions to store
        obs_shape: Shape of observations (e.g., (4, 84, 84))
        device: Device for tensor conversion
    """
    
    def __init__(
        self,
        capacity: int,
        obs_shape: Tuple[int, ...] = (4, 84, 84),
        device: str = "cuda",
    ) -> None:
        self.capacity = capacity
        self.device = device
        self.obs_shape = obs_shape
        # By default store observations as uint8 to save memory (frames 0-255).
        # We will convert to float32 in [0,1] when sampling.
        self.store_uint8 = True

        # Pre-allocate memory
        if self.store_uint8:
            self.states = np.zeros((capacity,) + obs_shape, dtype=np.uint8)
            self.next_states = np.zeros((capacity,) + obs_shape, dtype=np.uint8)
        else:
            self.states = np.zeros((capacity,) + obs_shape, dtype=np.float32)
            self.next_states = np.zeros((capacity,) + obs_shape, dtype=np.float32)

        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        
        self.ptr = 0  # Current write position
        self.size = 0  # Current buffer size
    
    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add a single transition to the buffer."""
        # Convert and store states efficiently
        if self.store_uint8:
            # If input is float in [0,1], convert to 0-255 uint8
            if state.dtype == np.float32 or state.dtype == np.float64:
                s = np.clip((state * 255.0).round(), 0, 255).astype(np.uint8)
            else:
                s = state.astype(np.uint8)
            self.states[self.ptr] = s
            if next_state.dtype == np.float32 or next_state.dtype == np.float64:
                ns = np.clip((next_state * 255.0).round(), 0, 255).astype(np.uint8)
            else:
                ns = next_state.astype(np.uint8)
            self.next_states[self.ptr] = ns
        else:
            self.states[self.ptr] = state
            self.next_states[self.ptr] = next_state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = float(done)
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def add_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        """Add a batch of transitions to the buffer."""
        batch_size = states.shape[0]
        if batch_size == 0:
            return

        # Handle wrap-around without recursion
        end_ptr = self.ptr + batch_size
        
        if end_ptr <= self.capacity:
            # No wrap-around needed
            indices = slice(self.ptr, end_ptr)
            self._store_at_indices(indices, states, actions, rewards, next_states, dones)
            self.ptr = end_ptr % self.capacity
        else:
            # Wrap-around: split into two parts
            part1_size = self.capacity - self.ptr
            part2_size = batch_size - part1_size
            
            # Part 1: fill to end
            self._store_at_indices(
                slice(self.ptr, self.capacity),
                states[:part1_size], actions[:part1_size], rewards[:part1_size],
                next_states[:part1_size], dones[:part1_size]
            )
            # Part 2: fill from beginning
            self._store_at_indices(
                slice(0, part2_size),
                states[part1_size:], actions[part1_size:], rewards[part1_size:],
                next_states[part1_size:], dones[part1_size:]
            )
            self.ptr = part2_size
        
        self.size = min(self.size + batch_size, self.capacity)

    def _store_at_indices(
        self,
        indices: slice,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        """Store data at specified indices. Avoids creating unnecessary copies."""
        if self.store_uint8:
            # Convert in-place where possible
            if states.dtype in (np.float32, np.float64):
                np.multiply(states, 255.0, out=states if states.flags.writeable else None)
                np.clip(states, 0, 255, out=states if states.flags.writeable else None)
                self.states[indices] = states.astype(np.uint8, copy=False)
            else:
                self.states[indices] = states
            
            if next_states.dtype in (np.float32, np.float64):
                np.multiply(next_states, 255.0, out=next_states if next_states.flags.writeable else None)
                np.clip(next_states, 0, 255, out=next_states if next_states.flags.writeable else None)
                self.next_states[indices] = next_states.astype(np.uint8, copy=False)
            else:
                self.next_states[indices] = next_states
        else:
            self.states[indices] = states
            self.next_states[indices] = next_states
        
        self.actions[indices] = actions
        self.rewards[indices] = rewards
        self.dones[indices] = dones
    
    def sample(self, batch_size: int) -> Batch:
        """Sample a random batch of transitions."""
        indices = np.random.randint(0, self.size, size=batch_size)
        # Convert stored uint8 frames back to float32 [0,1] if necessary
        if self.store_uint8:
            states = self.states[indices].astype(np.float32) / 255.0
            next_states = self.next_states[indices].astype(np.float32) / 255.0
        else:
            states = self.states[indices]
            next_states = self.next_states[indices]

        return Batch(
            states=torch.from_numpy(states).to(self.device),
            actions=torch.from_numpy(self.actions[indices]).to(self.device),
            rewards=torch.from_numpy(self.rewards[indices]).to(self.device),
            next_states=torch.from_numpy(next_states).to(self.device),
            dones=torch.from_numpy(self.dones[indices]).to(self.device),
        )
    
    def __len__(self) -> int:
        return self.size
    
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples."""
        return self.size >= batch_size


class ModelReplayBuffer:
    """
    Separate buffer for model-generated (synthetic) transitions.
    
    MBPO uses a mix of real and model-generated data. This buffer
    stores the synthetic transitions generated by the dynamics ensemble.
    
    Args:
        capacity: Maximum number of synthetic transitions
        obs_shape: Observation shape
        device: Device for tensor conversion
    """
    
    def __init__(
        self,
        capacity: int,
        obs_shape: Tuple[int, ...] = (4, 84, 84),
        device: str = "cuda",
    ) -> None:
        self.buffer = ReplayBuffer(capacity, obs_shape, device)
    
    def add_rollouts(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        """Add model rollouts directly to the CPU buffer."""
        # Use non_blocking=False to ensure data is fully transferred
        # before the tensors go out of scope
        self.buffer.add_batch(
            states.detach().cpu().numpy(),
            actions.detach().cpu().numpy(),
            rewards.detach().cpu().numpy(),
            next_states.detach().cpu().numpy(),
            dones.detach().cpu().numpy(),
        )
    
    def commit_to_buffer(self, num_commits: int = 1) -> int:
        """No-op for compatibility. Data is committed immediately now."""
        return 0
    
    def sample(self, batch_size: int) -> Batch:
        """Sample from model buffer."""
        return self.buffer.sample(batch_size)
    
    def clear(self) -> None:
        """Clear the model buffer."""
        self.buffer.ptr = 0
        self.buffer.size = 0
    
    def __len__(self) -> int:
        return len(self.buffer)


class MixedReplayBuffer:
    """
    Buffer that samples from both real and model-generated data.
    
    Implements the MBPO sampling strategy where a ratio of real vs.
    synthetic data is maintained during training.
    
    Args:
        real_buffer: Buffer with real environment transitions
        model_buffer: Buffer with model-generated transitions
        real_ratio: Proportion of real data in each batch
    """
    
    def __init__(
        self,
        real_buffer: ReplayBuffer,
        model_buffer: ModelReplayBuffer,
        real_ratio: float = 0.05,
    ) -> None:
        self.real_buffer = real_buffer
        self.model_buffer = model_buffer
        self.real_ratio = real_ratio
    
    def sample(self, batch_size: int) -> Batch:
        """Sample mixed batch from real and model buffers."""
        real_batch_size = int(batch_size * self.real_ratio)
        model_batch_size = batch_size - real_batch_size
        
        # Always sample from real buffer
        real_batch = self.real_buffer.sample(real_batch_size)
        
        # Sample from model buffer if it has enough data
        if len(self.model_buffer) >= model_batch_size:
            model_batch = self.model_buffer.sample(model_batch_size)
            
            # Concatenate batches
            return Batch(
                states=torch.cat([real_batch.states, model_batch.states]),
                actions=torch.cat([real_batch.actions, model_batch.actions]),
                rewards=torch.cat([real_batch.rewards, model_batch.rewards]),
                next_states=torch.cat([real_batch.next_states, model_batch.next_states]),
                dones=torch.cat([real_batch.dones, model_batch.dones]),
            )
        else:
            # Not enough model data, use only real
            return self.real_buffer.sample(batch_size)


# =============================================================================
# Early Stopping
# =============================================================================

class EarlyStopping:
    """
    Early stopping monitor for training.
    
    Tracks episode rewards and stops training when the mean reward
    over a window reaches the target threshold.
    
    Args:
        target_reward: Stop when mean reward reaches this value
        window_size: Number of episodes for computing mean
        patience: Episodes without improvement before stopping
    """
    
    def __init__(
        self,
        target_reward: float = 400.0,
        window_size: int = 100,
        patience: int = 50,
    ) -> None:
        self.target_reward = target_reward
        self.window_size = window_size
        self.patience = patience
        
        self.episode_rewards: deque = deque(maxlen=window_size)
        self.best_mean_reward = float("-inf")
        self.episodes_without_improvement = 0
        self.total_episodes = 0
        self.should_stop = False
        self.stop_reason: Optional[str] = None
    
    def update(self, episode_reward: float) -> bool:
        """
        Update with new episode reward.
        
        Args:
            episode_reward: Total reward from completed episode
        
        Returns:
            True if training should stop
        """
        self.episode_rewards.append(episode_reward)
        self.total_episodes += 1
        
        # Not enough episodes for mean
        if len(self.episode_rewards) < self.window_size:
            return False
        
        mean_reward = np.mean(self.episode_rewards)
        
        # Check if target reached
        if mean_reward >= self.target_reward:
            self.should_stop = True
            self.stop_reason = f"Target reward {self.target_reward} reached! Mean: {mean_reward:.2f}"
            return True
        
        # Check for improvement
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            self.episodes_without_improvement = 0
        else:
            self.episodes_without_improvement += 1
        
        # Check patience
        if self.episodes_without_improvement >= self.patience:
            self.should_stop = True
            self.stop_reason = f"No improvement for {self.patience} episodes"
            return True
        
        return False
    
    def get_mean_reward(self) -> float:
        """Get current mean reward over window."""
        if not self.episode_rewards:
            return 0.0
        return float(np.mean(self.episode_rewards))
    
    def get_stats(self) -> Dict[str, float]:
        """Get current statistics."""
        return {
            "mean_reward": self.get_mean_reward(),
            "best_mean_reward": self.best_mean_reward,
            "total_episodes": self.total_episodes,
            "episodes_without_improvement": self.episodes_without_improvement,
        }


# =============================================================================
# Configuration Utilities
# =============================================================================

def load_config(config_path: str = "config/config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict, save_path: str) -> None:
    """Save configuration to YAML file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def update_config_from_sweep(config: Dict, sweep_config: Dict) -> Dict:
    """Update config with sweep parameters."""
    # Flatten nested config for easy updating
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
    
    return config


# =============================================================================
# Checkpoint Utilities
# =============================================================================

def save_checkpoint(
    path: str,
    models: Dict[str, torch.nn.Module],
    optimizers: Dict[str, torch.optim.Optimizer],
    step: int,
    episode: int,
    best_reward: float,
    config: Dict,
) -> None:
    """Save training checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    checkpoint = {
        "step": step,
        "episode": episode,
        "best_reward": best_reward,
        "config": config,
        "models": {name: model.state_dict() for name, model in models.items()},
        "optimizers": {name: opt.state_dict() for name, opt in optimizers.items()},
    }
    
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(
    path: str,
    models: Dict[str, torch.nn.Module],
    optimizers: Optional[Dict[str, torch.optim.Optimizer]] = None,
    device: str = "cuda",
) -> Dict:
    """Load training checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    
    # Load model weights
    for name, model in models.items():
        if name in checkpoint["models"]:
            model.load_state_dict(checkpoint["models"][name])
    
    # Load optimizer states
    if optimizers is not None:
        for name, opt in optimizers.items():
            if name in checkpoint["optimizers"]:
                opt.load_state_dict(checkpoint["optimizers"][name])
    
    print(f"Checkpoint loaded from {path}")
    return checkpoint


# =============================================================================
# Seed Utilities
# =============================================================================

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # For deterministic behavior (may slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# Logging Utilities
# =============================================================================

class AverageMeter:
    """Track running average of a metric."""
    
    def __init__(self, window_size: int = 100) -> None:
        self.window_size = window_size
        self.values: deque = deque(maxlen=window_size)
    
    def update(self, value: float) -> None:
        """Append a numeric value to the meter, converting tensors to floats.

        This ensures CUDA tensors are moved to CPU and converted to Python floats
        before being stored (prevents numpy from trying to convert CUDA tensors).
        """
        # Convert torch tensors (including CUDA) to Python float
        if isinstance(value, torch.Tensor):
            try:
                v = float(value.detach().cpu().item())
            except Exception:
                # Fallback: convert to CPU then to float
                v = float(value.detach().cpu().numpy())
        else:
            # Numpy scalars or Python numbers -> float
            try:
                v = float(value)
            except Exception:
                # As a last resort, store 0.0
                v = 0.0

        self.values.append(v)
    
    def mean(self) -> float:
        if not self.values:
            return 0.0
        return float(np.mean(self.values))
    
    def std(self) -> float:
        if len(self.values) < 2:
            return 0.0
        return float(np.std(self.values))
    
    def reset(self) -> None:
        self.values.clear()


class MetricsLogger:
    """Log and aggregate training metrics."""
    
    def __init__(self, window_size: int = 100) -> None:
        self.meters: Dict[str, AverageMeter] = {}
        self.window_size = window_size
    
    def update(self, metrics: Dict[str, float]) -> None:
        """Update metrics from dictionary."""
        for key, value in metrics.items():
            if key not in self.meters:
                self.meters[key] = AverageMeter(self.window_size)
            self.meters[key].update(value)
    
    def get_means(self) -> Dict[str, float]:
        """Get mean values for all metrics."""
        return {key: meter.mean() for key, meter in self.meters.items()}
    
    def reset(self) -> None:
        """Reset all meters."""
        for meter in self.meters.values():
            meter.reset()


# =============================================================================
# Video Utilities
# =============================================================================

def find_latest_video(video_dir: str = "videos") -> Optional[str]:
    """Find the most recently created video file."""
    video_path = Path(video_dir)
    if not video_path.exists():
        return None
    
    videos = list(video_path.glob("*.mp4"))
    if not videos:
        return None
    
    # Sort by modification time
    latest = max(videos, key=lambda p: p.stat().st_mtime)
    return str(latest)


def get_video_files(video_dir: str = "videos", n_latest: int = 5) -> List[str]:
    """Get the N most recent video files."""
    video_path = Path(video_dir)
    if not video_path.exists():
        return []
    
    videos = list(video_path.glob("*.mp4"))
    videos.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return [str(v) for v in videos[:n_latest]]


if __name__ == "__main__":
    # Test utilities
    print("Testing ReplayBuffer...")
    buffer = ReplayBuffer(1000, obs_shape=(4, 84, 84), device="cpu")
    
    # Add some transitions
    for i in range(100):
        state = np.random.rand(4, 84, 84).astype(np.float32)
        action = np.random.randint(0, 4)
        reward = np.random.rand()
        next_state = np.random.rand(4, 84, 84).astype(np.float32)
        done = np.random.rand() > 0.9
        buffer.add(state, action, reward, next_state, done)
    
    print(f"Buffer size: {len(buffer)}")
    
    batch = buffer.sample(32)
    print(f"Batch shapes: states={batch.states.shape}, actions={batch.actions.shape}")
    
    print("\nTesting EarlyStopping...")
    early_stop = EarlyStopping(target_reward=100, window_size=10)
    
    for i in range(20):
        reward = 50 + i * 3  # Increasing rewards
        should_stop = early_stop.update(reward)
        if should_stop:
            print(f"Stopped at episode {i}: {early_stop.stop_reason}")
            break
    
    print(f"Stats: {early_stop.get_stats()}")
    
    print("\nAll utility tests passed!")
