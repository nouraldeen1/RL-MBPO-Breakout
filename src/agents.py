"""
MBPO Agent Implementation
=========================

Model-Based Policy Optimization agent for Breakout.

MBPO Algorithm Overview:
1. Collect real data from environment
2. Train dynamics ensemble on real data
3. Generate synthetic rollouts using learned models
4. Train policy on mixture of real and synthetic data

Key Benefits:
- Sample efficient (fewer real environment interactions)
- Uncertainty-aware (ensemble provides epistemic uncertainty)
- Works with existing model-free algorithms as subroutine

Author: CMPS458 RL Project
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from models import ActorCritic, DynamicsEnsemble, create_models
from utils import Batch, MixedReplayBuffer, ModelReplayBuffer, ReplayBuffer


class MBPOAgent:
    """
    MBPO Agent with dynamics ensemble and actor-critic policy.
    
    Implements the Model-Based Policy Optimization algorithm which:
    1. Learns an ensemble of dynamics models
    2. Generates synthetic trajectories for data augmentation
    3. Trains a policy using PPO-style updates on mixed data
    
    Args:
        obs_shape: Observation shape (4, 84, 84)
        num_actions: Number of discrete actions
        config: Configuration dictionary
        device: Device for computation
    """
    
    def __init__(
        self,
        obs_shape: Tuple[int, ...] = (4, 84, 84),
        num_actions: int = 4,
        config: Dict = None,
        device: str = "cuda",
    ) -> None:
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        self.device = device
        
        # Load config
        config = config or {}
        training_cfg = config.get("training", {})
        model_cfg = config.get("model", {})
        
        # Training parameters
        self.gamma = training_cfg.get("gamma", 0.99)
        self.tau = training_cfg.get("tau", 0.005)
        def _parse_float(value, default: float):
            if value is None:
                return default
            if isinstance(value, float) or isinstance(value, int):
                return float(value)
            try:
                return float(str(value))
            except Exception:
                return default

        self.policy_lr = _parse_float(training_cfg.get("policy_lr", 3e-4), 3e-4)
        self.dynamics_lr = _parse_float(training_cfg.get("dynamics_lr", 1e-4), 1e-4)
        self.batch_size = training_cfg.get("batch_size", 256)
        self.real_ratio = training_cfg.get("real_ratio", 0.05)
        self.model_rollout_length = training_cfg.get("model_rollout_length", 1)
        self.model_rollout_batch_size = training_cfg.get("model_rollout_batch_size", 100000)
        
        # PPO parameters
        self.clip_range = config.get("ppo", {}).get("clip_range", 0.2)
        self.ent_coef = config.get("ppo", {}).get("ent_coef", 0.01)
        self.vf_coef = config.get("ppo", {}).get("vf_coef", 0.5)
        self.max_grad_norm = config.get("ppo", {}).get("max_grad_norm", 0.5)
        
        # Create models
        self.actor_critic = ActorCritic(
            num_actions=num_actions,
            in_channels=obs_shape[0],
            fc_dim=model_cfg.get("fc_dim", 512),
        ).to(device)
        
        self.dynamics = DynamicsEnsemble(
            ensemble_size=model_cfg.get("ensemble_size", 5),
            num_actions=num_actions,
            in_channels=obs_shape[0],
            hidden_dim=model_cfg.get("dynamics_hidden_dim", 256),
            fc_dim=model_cfg.get("fc_dim", 512),
        ).to(device)
        
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(
            self.actor_critic.parameters(),
            lr=self.policy_lr,
        )
        self.dynamics_optimizer = torch.optim.Adam(
            self.dynamics.parameters(),
            lr=self.dynamics_lr,
        )
        
        # Replay buffers
        buffer_size = training_cfg.get("buffer_size", 10000)
        self.real_buffer = ReplayBuffer(buffer_size, obs_shape, device)
        self.model_buffer = ModelReplayBuffer(buffer_size, obs_shape, device)
        self.mixed_buffer = MixedReplayBuffer(
            self.real_buffer, self.model_buffer, self.real_ratio
        )
        
        # Training state
        self.total_steps = 0
        self.dynamics_trained = False
    
    def get_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[int, Dict]:
        """
        Select action given state.
        
        Args:
            state: Observation array
            deterministic: Whether to use deterministic action
        
        Returns:
            action: Selected action
            info: Dictionary with log_prob and value
        """
        with torch.no_grad():
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
            action, log_prob, entropy, value = self.actor_critic.get_action_and_value(
                state_tensor, deterministic=deterministic
            )
        
        return action.cpu().item(), {
            "log_prob": log_prob.cpu().item(),
            "value": value.cpu().item(),
            "entropy": entropy.cpu().item(),
        }
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store transition in real buffer."""
        self.real_buffer.add(state, action, reward, next_state, done)
    
    def train_dynamics(self, n_updates: int = 1) -> Dict[str, float]:
        """
        Train dynamics ensemble on real data.
        
        Args:
            n_updates: Number of gradient updates
        
        Returns:
            Dictionary of training metrics
        """
        if not self.real_buffer.is_ready(self.batch_size):
            return {}
        
        total_metrics = {}
        
        for _ in range(n_updates):
            batch = self.real_buffer.sample(self.batch_size)
            
            loss, metrics = self.dynamics.compute_loss(
                batch.states,
                batch.actions,
                batch.next_states,
                batch.rewards,
            )
            
            self.dynamics_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.dynamics.parameters(), self.max_grad_norm)
            self.dynamics_optimizer.step()
            
            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0) + v / n_updates
        
        self.dynamics_trained = True
        return total_metrics
    
    def generate_model_rollouts(self) -> int:
        """
        Generate synthetic rollouts using dynamics ensemble.
        
        Samples initial states from real buffer, then uses dynamics
        models to generate trajectories. This is the core of MBPO's
        data augmentation strategy.
        
        Returns:
            Number of transitions generated
        """
        if not self.dynamics_trained or not self.real_buffer.is_ready(self.batch_size):
            return 0
        
        # Sample initial states from real buffer
        batch_size = min(self.model_rollout_batch_size, len(self.real_buffer))
        batch = self.real_buffer.sample(batch_size)
        
        states = batch.states
        total_transitions = 0
        
        with torch.no_grad():
            for _ in range(self.model_rollout_length):
                # Get actions from policy
                actions, _, _, _ = self.actor_critic.get_action_and_value(states)
                
                # Predict next states using dynamics ensemble
                # Randomly select model for each prediction (exploration)
                next_states, rewards, uncertainty = self.dynamics.predict_next_state(
                    states, actions, use_mean=False
                )
                
                # Simple termination prediction (use real data statistics)
                # For Breakout, episodes rarely terminate mid-rollout
                dones = torch.zeros(batch_size, device=self.device)
                
                # Store in model buffer
                self.model_buffer.add_rollouts(
                    states, actions, rewards.squeeze(-1), next_states, dones
                )
                
                total_transitions += batch_size
                states = next_states
        
        return total_transitions
    
    def update_policy(self, n_updates: int = 1) -> Dict[str, float]:
        """
        Update policy using PPO on mixed real/model data.
        
        Args:
            n_updates: Number of gradient updates
        
        Returns:
            Dictionary of training metrics
        """
        if not self.real_buffer.is_ready(self.batch_size):
            return {}
        
        total_metrics = {}
        
        for _ in range(n_updates):
            # Sample mixed batch
            batch = self.mixed_buffer.sample(self.batch_size)
            
            # Get current policy outputs
            _, log_probs, entropy, values = self.actor_critic.get_action_and_value(
                batch.states, action=batch.actions
            )
            
            # Compute returns and advantages (simple 1-step)
            with torch.no_grad():
                _, _, _, next_values = self.actor_critic.get_action_and_value(
                    batch.next_states
                )
                targets = batch.rewards + self.gamma * next_values * (1 - batch.dones)
            
            advantages = targets - values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Policy loss (simple policy gradient, not full PPO for efficiency)
            policy_loss = -(log_probs * advantages.detach()).mean()
            
            # Value loss
            value_loss = F.mse_loss(values, targets)
            
            # Entropy bonus
            entropy_loss = -entropy.mean()
            
            # Total loss
            loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss
            
            self.policy_optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
            self.policy_optimizer.step()
            
            # Collect metrics
            metrics = {
                "policy/loss": policy_loss.item(),
                "policy/value_loss": value_loss.item(),
                "policy/entropy": -entropy_loss.item(),
                "policy/total_loss": loss.item(),
            }
            
            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0) + v / n_updates
        
        return total_metrics
    
    def update(self) -> Dict[str, float]:
        """
        Single update step combining dynamics and policy training.
        
        Returns:
            Combined metrics dictionary
        """
        metrics = {}
        
        # Train dynamics on real data
        dynamics_metrics = self.train_dynamics(n_updates=1)
        metrics.update(dynamics_metrics)
        
        # Generate model rollouts
        n_generated = self.generate_model_rollouts()
        metrics["model/generated_transitions"] = n_generated
        
        # Update policy on mixed data
        policy_metrics = self.update_policy(n_updates=1)
        metrics.update(policy_metrics)
        
        # Buffer stats
        metrics["buffer/real_size"] = len(self.real_buffer)
        metrics["buffer/model_size"] = len(self.model_buffer)
        
        return metrics
    
    def save(self, path: str) -> None:
        """Save agent state."""
        torch.save({
            "actor_critic": self.actor_critic.state_dict(),
            "dynamics": self.dynamics.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "dynamics_optimizer": self.dynamics_optimizer.state_dict(),
            "total_steps": self.total_steps,
        }, path)
    
    def load(self, path: str) -> None:
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint["actor_critic"])
        self.dynamics.load_state_dict(checkpoint["dynamics"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
        self.dynamics_optimizer.load_state_dict(checkpoint["dynamics_optimizer"])
        self.total_steps = checkpoint.get("total_steps", 0)
        self.dynamics_trained = True


class DDQNAgent:
    """
    Double DQN Agent for comparison benchmarks.
    
    Implements Double DQN with dueling architecture as a baseline
    for comparing against MBPO.
    """
    
    def __init__(
        self,
        obs_shape: Tuple[int, ...] = (4, 84, 84),
        num_actions: int = 4,
        config: Dict = None,
        device: str = "cuda",
    ) -> None:
        from models import QNetwork
        
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        self.device = device
        
        config = config or {}
        training_cfg = config.get("training", {})
        ddqn_cfg = config.get("ddqn", {})
        model_cfg = config.get("model", {})
        
        # Parameters
        self.gamma = training_cfg.get("gamma", 0.99)
        self.lr = training_cfg.get("policy_lr", 3e-4)
        self.batch_size = training_cfg.get("batch_size", 256)
        self.target_update_freq = training_cfg.get("target_update_freq", 10000)
        
        # Epsilon parameters
        self.epsilon = ddqn_cfg.get("epsilon_start", 1.0)
        self.epsilon_end = ddqn_cfg.get("epsilon_end", 0.01)
        self.epsilon_decay = ddqn_cfg.get("epsilon_decay", 0.995)
        
        # Networks
        self.q_network = QNetwork(
            num_actions=num_actions,
            in_channels=obs_shape[0],
            fc_dim=model_cfg.get("fc_dim", 512),
            dueling=True,
        ).to(device)
        
        self.target_network = QNetwork(
            num_actions=num_actions,
            in_channels=obs_shape[0],
            fc_dim=model_cfg.get("fc_dim", 512),
            dueling=True,
        ).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.lr)
        
        # Buffer
        buffer_size = training_cfg.get("buffer_size", 1000000)
        self.buffer = ReplayBuffer(buffer_size, obs_shape, device)
        
        self.total_steps = 0
    
    def get_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[int, Dict]:
        """Select action using epsilon-greedy."""
        if not deterministic and np.random.random() < self.epsilon:
            action = np.random.randint(0, self.num_actions)
        else:
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                action = q_values.argmax(dim=-1).cpu().item()
        
        return action, {"epsilon": self.epsilon}
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Store transition in buffer."""
        self.buffer.add(state, action, reward, next_state, done)
    
    def update(self) -> Dict[str, float]:
        """Update Q-network."""
        if not self.buffer.is_ready(self.batch_size):
            return {}
        
        batch = self.buffer.sample(self.batch_size)
        
        # Current Q values
        q_values = self.q_network(batch.states)
        q_values = q_values.gather(1, batch.actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN: use online network to select action, target to evaluate
        with torch.no_grad():
            next_actions = self.q_network(batch.next_states).argmax(dim=-1)
            next_q_values = self.target_network(batch.next_states)
            next_q_values = next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            targets = batch.rewards + self.gamma * next_q_values * (1 - batch.dones)
        
        loss = F.huber_loss(q_values, targets)
        
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()
        
        # Update target network
        self.total_steps += 1
        if self.total_steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        
        return {
            "ddqn/loss": loss.item(),
            "ddqn/q_mean": q_values.mean().item(),
            "ddqn/epsilon": self.epsilon,
        }
    
    def save(self, path: str) -> None:
        """Save agent state."""
        torch.save({
            "q_network": self.q_network.state_dict(),
            "target_network": self.target_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "total_steps": self.total_steps,
        }, path)
    
    def load(self, path: str) -> None:
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.epsilon = checkpoint.get("epsilon", self.epsilon_end)
        self.total_steps = checkpoint.get("total_steps", 0)


def create_agent(algorithm: str, obs_shape: Tuple, num_actions: int, config: Dict, device: str):
    """Factory function to create agents."""
    if algorithm == "mbpo":
        return MBPOAgent(obs_shape, num_actions, config, device)
    elif algorithm == "ddqn":
        return DDQNAgent(obs_shape, num_actions, config, device)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
