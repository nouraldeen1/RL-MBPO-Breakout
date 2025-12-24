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

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import imageio

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
        self.gae_lambda = config.get("ppo", {}).get("gae_lambda", 0.95)
        # N-step returns for better credit assignment with sparse rewards
        self.n_step_returns = training_cfg.get("n_step_returns", 5)
        # Safety threshold: skip applying extremely large updates
        self.grad_norm_skip_threshold = (
            training_cfg.get("grad_norm_skip_threshold", 10.0)
        )
        
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
        # For periodic dream monitoring (disabled by default)
        self.update_count = 0
        self.dream_monitor_freq = config.get("debug", {}).get("dream_monitor_freq", 0)
        self.video_dir = config.get("env", {}).get("video_dir", "videos")
    
    def get_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Select actions for a batch of states.
        
        Args:
            state: Observation array (batch_size, C, H, W) or single (C, H, W)
            deterministic: Whether to use deterministic actions
        
        Returns:
            actions: Selected actions array (batch_size,) or single int
            info: Dictionary with batched log_probs and values
        """
        with torch.no_grad():
            # Handle single state case for compatibility
            is_single = len(state.shape) == 3
            if is_single:
                state = np.expand_dims(state, 0)

            state_tensor = torch.from_numpy(state).float().to(self.device)
            
            logits, _ = self.actor_critic.forward(state_tensor)
            actions, log_probs, entropies, values = self.actor_critic.get_action_and_value(
                state_tensor, deterministic=deterministic
            )
        
        actions_np = actions.cpu().numpy()
        
        # If input was single, return single action and scalar info
        if is_single:
            return actions_np[0], {
                "log_prob": log_probs[0].cpu().item(),
                "value": values[0].cpu().item(),
                "entropy": entropies[0].cpu().item(),
                "logits_std": logits[0].std().cpu().item(),
            }

        # Otherwise, return batch
        return actions_np, {
            "log_prob": log_probs.cpu().numpy(),
            "value": values.cpu().numpy(),
            "entropy": entropies.cpu().numpy(),
            "logits_std": logits.std(dim=-1).mean().cpu().item(),
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
            
            self.dynamics_optimizer.zero_grad(set_to_none=True)  # More memory efficient
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.dynamics.parameters(), self.max_grad_norm)
            self.dynamics_optimizer.step()
            
            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0) + v / n_updates
            
            # Explicit cleanup
            del batch, loss
        
        self.dynamics_trained = True
        return total_metrics
    
    def generate_model_rollouts(self) -> int:
        """
        Generate synthetic rollouts using dynamics ensemble.
        Store cumulative discounted rewards for each synthetic transition.
        Returns:
            Number of transitions generated
        """
        if not self.dynamics_trained or not self.real_buffer.is_ready(self.batch_size * 5):
            return 0

        total_transitions = 0
        model_idx = 0  # Always use first model for consistency
        chunk_size = 32  # Very small to minimize memory
        num_chunks = max(1, self.model_rollout_batch_size // chunk_size)
        gamma = self.gamma
        rollout_length = self.model_rollout_length

        for _ in range(num_chunks):
            batch = self.real_buffer.sample(chunk_size)
            start_states = batch.states.clone()  # (chunk_size, ...)
            del batch

            # For each trajectory, store the sequence
            states = start_states.clone()
            all_states = [states]
            all_actions = []
            all_rewards = []
            all_next_states = []
            all_dones = []

            with torch.no_grad():
                for t in range(rollout_length):
                    actions, _, _, _ = self.actor_critic.get_action_and_value(states)
                    next_states, rewards, _ = self.dynamics.predict_next_state(
                        states, actions, model_idx=model_idx, use_mean=True
                    )
                    dones = torch.zeros(chunk_size, device=self.device)
                    all_states.append(next_states)
                    all_actions.append(actions)
                    all_rewards.append(rewards.squeeze(-1))
                    all_next_states.append(next_states)
                    all_dones.append(dones)
                    states = next_states.detach()

            # Now compute cumulative discounted rewards for each starting point
            # For each t in [0, rollout_length), the reward is sum_{k=0}^{rollout_length-t-1} gamma^k * r_{t+k}
            all_states = all_states[:-1]  # exclude last next_state for alignment
            for t in range(rollout_length):
                s_np = all_states[t].cpu().numpy()
                a_np = all_actions[t].cpu().numpy()
                ns_np = all_next_states[t].cpu().numpy()
                d_np = all_dones[t].cpu().numpy()
                # Compute cumulative discounted reward for each trajectory in batch
                cum_rewards = torch.zeros(chunk_size, device=self.device)
                for k in range(rollout_length - t):
                    cum_rewards += (gamma ** k) * all_rewards[t + k]
                r_np = cum_rewards.cpu().numpy()
                self.model_buffer.buffer.add_batch(s_np, a_np, r_np, ns_np, d_np)
                total_transitions += chunk_size
                del s_np, a_np, r_np, ns_np, d_np

            del all_states, all_actions, all_rewards, all_next_states, all_dones, start_states
            torch.cuda.empty_cache()

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
            # Also compute logits diagnostics to detect blowups that zero entropy
            logits, _ = self.actor_critic.forward(batch.states)
            _, log_probs, entropy, values = self.actor_critic.get_action_and_value(
                batch.states, action=batch.actions
            )
            
            # Compute n-step return targets (approximate) to speed up credit assignment
            # G_t = R_{t+1} + gamma^n * V(s_{t+n})  (approximation when only single-step samples available)
            with torch.no_grad():
                _, _, _, next_values = self.actor_critic.get_action_and_value(
                    batch.next_states
                )
                gamma_n = float(self.gamma) ** float(self.n_step_returns)
                targets = batch.rewards + gamma_n * next_values * (1 - batch.dones)
                advantages = targets - values
            
            # Normalize advantages for stability (CRITICAL)
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            
            # Policy loss (simple policy gradient)
            policy_loss = -(log_probs * advantages.detach()).mean()
            
            # Value loss
            value_loss = F.mse_loss(values, targets)
            
            # Entropy bonus - use NEGATIVE entropy loss to MAXIMIZE entropy
            # Higher entropy = more exploration
            mean_entropy = entropy.mean()
            entropy_loss = -mean_entropy  # Negative because we want to maximize entropy
            
            # Total loss: minimize policy_loss + value_loss, maximize entropy
            # So we ADD ent_coef * entropy_loss (which is negative of entropy)
            # Actually we want: loss = policy_loss + vf*value_loss - ent_coef*entropy
            # Which is: loss = policy_loss + vf*value_loss + ent_coef*(-entropy)
            loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss
            
            self.policy_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # Clip gradients and capture gradient norm for logging/diagnostics
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.actor_critic.parameters(), self.max_grad_norm
            )

            # Safety: if the pre-clip grad norm is extremely large, skip the update
            # to avoid catastrophic parameter changes that blow up logits/entropy.
            updates_skipped = 0
            if grad_norm > self.grad_norm_skip_threshold:
                # Skip stepping this update
                updates_skipped = 1
                # Zero grads to avoid accidental application
                self.policy_optimizer.zero_grad(set_to_none=True)
            else:
                self.policy_optimizer.step()
            
            metrics = {
                "policy/loss": policy_loss.item(),
                "policy/value_loss": value_loss.item(),
                "policy/entropy": mean_entropy.item(),
                "policy/total_loss": loss.item(),
                "policy/grad_norm": float(grad_norm) if isinstance(grad_norm, (float, int)) else grad_norm,
                "policy/mean_advantage": advantages.mean().item(),
                "policy/mean_value": values.mean().item(),
                # Log logits statistics for diagnostics (detect numerical blowup)
                "policy/logits_max_abs": float(logits.abs().max().detach().cpu().item()),
                "policy/logits_std": float(logits.detach().cpu().std().item()),
                "policy/updates_skipped": updates_skipped,
            }
            
            for k, v in metrics.items():
                total_metrics[k] = total_metrics.get(k, 0) + v / n_updates
            
            # Cleanup
            del batch, loss
        
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
        
        # Periodic dream monitoring: visualize imagination from a real state
        self.update_count += 1
        if self.dream_monitor_freq and (self.update_count % self.dream_monitor_freq == 0):
            try:
                if self.real_buffer.is_ready(1):
                    sample = self.real_buffer.sample(1)
                    state = sample.states[0].cpu().numpy()
                    os.makedirs(self.video_dir, exist_ok=True)
                    filename = os.path.join(self.video_dir, f"dream-{self.total_steps:09d}.gif")
                    # Use model_rollout_length as imagination horizon
                    self.visualize_imagination(state, filename=filename, horizon=int(self.model_rollout_length))
            except Exception as e:
                print(f"Dream monitor failed: {e}")

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

    def visualize_imagination(self, state, filename: Optional[str] = None, horizon: int = 60, model_idx: int = 0, fps: int = 10) -> None:
        """
        Save a GIF showing the dynamics model's imagined rollouts starting
        from a real state. The `state` is expected to be a numpy array
        with shape (C, H, W) and values in [0, 1].
        """
        try:
            os.makedirs(self.video_dir, exist_ok=True)

            # Prepare input: ensure float32 in [0,1]
            if isinstance(state, torch.Tensor):
                state_np = state.detach().cpu().numpy()
            else:
                state_np = np.array(state)

            if state_np.max() > 1.1:
                state_input = (state_np.astype(np.float32) / 255.0)
            else:
                state_input = state_np.astype(np.float32)

            current_state = torch.from_numpy(state_input).float().to(self.device).unsqueeze(0)

            frames = []
            for i in range(horizon):
                # Actor chooses action deterministically in imagination
                actions, _info = self.get_action(current_state.cpu().numpy(), deterministic=True)
                if isinstance(actions, np.ndarray):
                    act = int(actions[0])
                else:
                    act = int(actions)

                action_tensor = torch.tensor([act], dtype=torch.long, device=self.device)

                with torch.no_grad():
                    next_state_pred, rewards_pred, _ = self.dynamics.predict_next_state(
                        current_state, action_tensor, model_idx=model_idx, use_mean=True
                    )

                # Take the last frame from the stacked channels and convert to RGB uint8
                frame_gray = (next_state_pred[0, -1].detach().cpu().numpy() * 255.0).round().astype(np.uint8)
                frame_rgb = np.stack([frame_gray, frame_gray, frame_gray], axis=-1)
                frames.append(frame_rgb)

                # Update current state for next imagination step
                current_state = next_state_pred

            if filename is None:
                filename = os.path.join(self.video_dir, f"dream-{self.total_steps:09d}.gif")

            imageio.mimsave(filename, frames, fps=fps)
            print(f"Imagination saved to {filename}")
        except Exception as e:
            print(f"Failed to save imagination GIF: {e}")


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
