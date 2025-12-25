"""
Dreamer (RSSM) Agent Implementation
===================================
A latent world model agent that learns by dreaming.
This implementation provides a Dreamer-style agent with a world model,
sequence buffer, and latent imagination policy updates.
"""

from __future__ import annotations
from typing import Dict, Optional, Tuple, List

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import imageio

from models import WorldModel
from utils import SequenceReplayBuffer, kl_divergence, ReplayBuffer

class DreamerAgent:
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
        self.config = config or {}
        
        # Hyperparameters
        training_cfg = self.config.get("training", {})
        model_cfg = self.config.get("model", {})
        ppo_cfg = self.config.get("ppo", {})

        self.latent_dim = model_cfg.get("latent_dim", 1024)
        self.deter_dim = model_cfg.get("deter_dim", 1024) # Hidden state of GRU
        self.stoch_dim = model_cfg.get("stoch_dim", 32)   # Latent z
        self.kl_coeff = model_cfg.get("kl_coeff", 0.1)
        self.imagination_horizon = training_cfg.get("imagination_horizon", 5)
        self.gamma = training_cfg.get("gamma", 0.99)
        self.ent_coef = ppo_cfg.get("ent_coef", 0.001)
        self.vf_coef = ppo_cfg.get("vf_coef", 1.0)
        self.max_grad_norm = ppo_cfg.get("max_grad_norm", 100.0)

        # 1. World Model (Encoder, RSSM, Decoder, Reward, Continue)
        # WorldModel in this repo expects (in_channels, latent_dim, action_dim)
        self.world_model = WorldModel(
            in_channels=obs_shape[0],
            latent_dim=self.latent_dim,
            action_dim=self.num_actions,
        ).to(self.device)
        
        # 2. Latent Policy / Value heads (operate on world-model latent `h`)
        latent_feat_dim = self.latent_dim
        self.policy_head = nn.Linear(latent_feat_dim, self.num_actions).to(self.device)
        self.value_head = nn.Linear(latent_feat_dim, 1).to(self.device)

        # Initialize heads similarly to ActorCritic init
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

        # Optimizers
        self.world_optimizer = torch.optim.Adam(self.world_model.parameters(), lr=2e-4)
        # Policy optimizer should update both policy and value heads
        self.policy_optimizer = torch.optim.Adam(
            list(self.policy_head.parameters()) + list(self.value_head.parameters()), lr=4e-5
        )

        # Replay buffers
        # Sequence buffer for Dreamer chunked training
        self.init_sequence_buffer()
        self.batch_size = training_cfg.get("batch_size", 32)

        self.total_steps = 0
        self.video_dir = self.config.get("env", {}).get("video_dir", "videos")
        
        # For sequence accumulation
        self.num_envs = self.config.get("env", {}).get("num_envs", 1)
        self.current_sequences = [[] for _ in range(self.num_envs)]

    def init_sequence_buffer(self):
        model_cfg = self.config.get("model", {})
        chunk_length = model_cfg.get("dreamer_chunk_length", 100)
        buffer_size = self.config.get("training", {}).get("buffer_size", 1000)
        self.sequence_buffer = SequenceReplayBuffer(buffer_size, chunk_length, self.obs_shape, self.device)

    def init_world_model(self):
        # World model already created in constructor; keep for compatibility
        return

    # --- CORE DREAMER LOGIC ---

    def train_world_model(self, batch=None, batch_size=32) -> Dict[str, float]:
        """Trains the RSSM using Reconstruction Loss + Reward Loss + KL.

        Accepts either:
        - a `Batch` sampled from `self.real_buffer` (single-step tensors), or
        - None, in which case a sequence chunk is sampled from `self.sequence_buffer`.
        """
        # Two modes: real single-step batch (from ReplayBuffer.Batch) or sequence chunk
        if batch is None:
            # Sequence chunk training
            batch = self.sequence_buffer.sample(self.batch_size)
            self.world_optimizer.zero_grad()
            total_pixel_loss = 0.0
            total_reward_loss = 0.0
            total_kl_loss = 0.0
            num_sequences = len(batch)
            for sequence in batch:
                seq_obs = sequence["states"]
                seq_actions = F.one_hot(sequence["actions"], self.num_actions).float()
                seq_rewards = sequence["rewards"]
                seq_dones = sequence["dones"]
                pixel_loss, reward_loss, kl_loss = self.world_model.compute_loss(seq_obs, seq_actions, seq_rewards, seq_dones)
                total_pixel_loss += pixel_loss.item()
                total_reward_loss += reward_loss.item()
                total_kl_loss += kl_loss.item()
                (pixel_loss + reward_loss + self.kl_coeff * kl_loss).backward()
            self.world_optimizer.step()
            return {
                "wm/pixel_loss": total_pixel_loss / num_sequences,
                "wm/reward_loss": total_reward_loss / num_sequences,
                "wm/kl_loss": total_kl_loss / num_sequences,
                "wm/total_loss": (total_pixel_loss + total_reward_loss + self.kl_coeff * total_kl_loss) / num_sequences,
            }
        else:
            # Real-sample single-step batch (Batch dataclass from ReplayBuffer.sample)
            # obs: (B, C, H, W), actions: (B,), rewards: (B,)
            obs = batch.states.to(self.device)
            actions = batch.actions.to(self.device)
            rewards = batch.rewards.to(self.device)
            dones = batch.dones.to(self.device)

            prev_hidden = torch.zeros(obs.size(0), self.deter_dim, device=self.device)
            # One-hot actions
            act_onehot = F.one_hot(actions.long(), num_classes=self.num_actions).float()

            # Forward through world model (single-step API from models.WorldModel)
            h, recon, reward_pred, done_pred = self.world_model(obs, act_onehot, prev_hidden)

            pixel_loss = F.mse_loss(recon, obs)
            reward_loss = F.mse_loss(reward_pred.squeeze(-1), rewards)

            # Simple KL regularizer (placeholder prior vs posterior)
            prior_mean = torch.zeros_like(h)
            prior_logvar = torch.zeros_like(h)
            post_mean = h
            post_logvar = torch.zeros_like(h)
            kl_loss = kl_divergence(prior_mean, prior_logvar, post_mean, post_logvar)

            loss = pixel_loss + reward_loss + self.kl_coeff * kl_loss

            self.world_optimizer.zero_grad()
            loss.backward()
            self.world_optimizer.step()

            return {
                "wm/pixel_loss": pixel_loss.item(),
                "wm/reward_loss": reward_loss.item(),
                "wm/kl_loss": kl_loss.item(),
                "wm/total_loss": loss.item(),
            }

    def update_latent_policy(self, batch_size=16) -> Dict[str, float]:
        """Actor-Critic update inside the imagined latent space."""
        # 1. Get starting states from real sequences (SequenceReplayBuffer already normalizes)
        batch = self.sequence_buffer.sample(batch_size)
        obs = torch.stack([seq["states"][0] for seq in batch])

        with torch.no_grad():
            B = obs.shape[0]
            prev_action = torch.zeros(B, self.num_actions, device=self.device)
            prev_hidden = torch.zeros(B, self.latent_dim, device=self.device)
            h, _, _, _ = self.world_model(obs, prev_action, prev_hidden)

        # 2. Imagine trajectories starting from h
        imag_h = [h]
        imag_rewards = []
        imag_log_probs = []
        imag_actions = []

        for t in range(self.imagination_horizon):
            feat = h
            action_logits = self.policy_head(feat)
            action_dist = Categorical(logits=action_logits)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            action_onehot = F.one_hot(action, num_classes=self.num_actions).float().to(self.device)

            # Step RSSM forward
            h_next = self.world_model.rssm(h, action_onehot, prev_hidden)
            reward = self.world_model.reward_model(h_next)

            imag_h.append(h_next)
            imag_rewards.append(reward.squeeze(-1))
            imag_log_probs.append(log_prob)
            imag_actions.append(action)

            prev_hidden = h_next
            h = h_next

        # Stack: feats (T+1, B, latent_dim), rewards (T, B), log_probs (T, B)
        feats = torch.stack(imag_h)
        rewards_t = torch.stack(imag_rewards)
        log_probs_t = torch.stack(imag_log_probs)

        # Value predictions for all feats
        values = self.value_head(feats.view(-1, feats.shape[-1])).view(feats.shape[0], -1)

        # Compute lambda returns
        returns = self.compute_lambda_returns(rewards_t, values)

        # Policy loss: -log_prob * (lambda_return - value)
        advantages = returns - values[:-1]
        policy_loss = -(log_probs_t * advantages.detach()).mean()

        # Value loss
        value_loss = F.mse_loss(values[:-1], returns.detach())

        # Entropy bonus
        entropy = Categorical(logits=self.policy_head(feats[:-1].view(-1, feats.shape[-1]))).entropy().mean()
        entropy_loss = -self.ent_coef * entropy

        total_loss = policy_loss + self.vf_coef * value_loss + entropy_loss

        self.policy_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_head.parameters(), self.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(self.value_head.parameters(), self.max_grad_norm)
        self.policy_optimizer.step()

        return {
            "latent_policy_loss": policy_loss.item(),
            "latent_value_loss": value_loss.item(),
            "latent_entropy_loss": entropy_loss.item(),
            "latent_total_loss": total_loss.item(),
        }

    def compute_lambda_returns(self, rewards: torch.Tensor, values: torch.Tensor, lambda_: float = 0.95) -> torch.Tensor:
        """Compute lambda returns for imagined trajectories."""
        T, B = rewards.shape
        returns = torch.zeros_like(values[:-1])  # (T, B)
        
        last_value = values[-1]  # Bootstrap from last value
        for t in reversed(range(T)):
            if t == T - 1:
                next_return = last_value
            else:
                next_return = returns[t + 1]
            returns[t] = rewards[t] + self.gamma * ((1 - lambda_) * values[t + 1] + lambda_ * next_return)
        
        return returns

    def get_action(self, state: np.ndarray, hidden_state: Optional[torch.Tensor] = None, deterministic: bool = False):
        """Action selection with persistent GRU state."""
        # Support batched or single observations
        is_single = len(state.shape) == 3
        if is_single:
            state = np.expand_dims(state, 0)

        state_tensor = torch.from_numpy(state).float().to(self.device) / 255.0 - 0.5

        batch_size = state_tensor.shape[0]

        # Prepare hidden/action tensors
        if hidden_state is None:
            prev_hidden = torch.zeros(batch_size, self.latent_dim, device=self.device)
        else:
            prev_hidden = hidden_state.to(self.device)

        prev_action = torch.zeros(batch_size, self.num_actions, device=self.device)

        with torch.no_grad():
            # Compute world-model latent from current observation
            h, _, _, _ = self.world_model(state_tensor, prev_action, prev_hidden)

            logits = self.policy_head(h)
            if deterministic:
                actions = torch.argmax(logits, dim=-1)
            else:
                dist = Categorical(logits=logits)
                actions = dist.sample()

        actions_np = actions.cpu().numpy()
        info = {
            "logits": logits.cpu().numpy(),
        }

        # Return signature: if caller provided hidden_state, include next hidden
        next_hidden = h
        if is_single:
            if hidden_state is not None:
                return actions_np[0], info, next_hidden[0]
            return actions_np[0], info

        if hidden_state is not None:
            return actions_np, info, next_hidden
        return actions_np, info

    def compute_returns(self, rewards, values):
        """Basic discounted reward calculation for imagination."""
        # rewards: (T, B), values: (T+1, B, 1) or similar
        T = rewards.shape[0]
        returns = torch.zeros_like(rewards)
        last_val = values[-1].view(-1)
        for t in reversed(range(self.imagination_horizon)):
            returns[t] = rewards[t] + self.gamma * last_val
            last_val = returns[t]
        return returns

    def store_transition(self, s, a, r, ns, d, i_env):
        # Accumulate into current sequence for this env
        self.current_sequences[i_env].append((s, a, r, ns, d))
        # If sequence reaches chunk length, store it
        if len(self.current_sequences[i_env]) >= self.sequence_buffer.chunk_length:
            chunk = self.current_sequences[i_env][:self.sequence_buffer.chunk_length]
            states = [s for s, _, _, _, _ in chunk]
            actions = [a for _, a, _, _, _ in chunk]
            rewards = [r for _, _, r, _, _ in chunk]
            next_states = [ns for _, _, _, ns, _ in chunk]
            dones = [d for _, _, _, _, d in chunk]
            self.sequence_buffer.add_chunk(states, actions, rewards, next_states, dones)
            self.current_sequences[i_env] = self.current_sequences[i_env][self.sequence_buffer.chunk_length:]
        # If episode ended, clear the sequence (don't store partial chunks)
        if d:
            self.current_sequences[i_env] = []

    def imagine_latent_rollout(self, start_obs: np.ndarray, rollout_length: int = 15):
        """Perform a short latent rollout starting from `start_obs`.

        Returns a list of latent `h` numpy arrays (detached) length `rollout_length`.
        """
        obs = torch.from_numpy(start_obs).float().to(self.device) / 255.0 - 0.5
        batch_size = obs.shape[0]
        prev_action = torch.zeros(batch_size, self.num_actions, device=self.device)
        prev_hidden = torch.zeros(batch_size, self.latent_dim, device=self.device)

        latents = []
        with torch.no_grad():
            # Initialize latent from observation
            h, _, _, _ = self.world_model(obs, prev_action, prev_hidden)
            for t in range(rollout_length):
                latents.append(h.detach().cpu().numpy())
                logits = self.policy_head(h)
                action = torch.argmax(logits, dim=-1)
                action_onehot = F.one_hot(action, num_classes=self.num_actions).float().to(self.device)
                # Step RSSM using previous latent and chosen action
                h = self.world_model.rssm(h, action_onehot, prev_hidden)
                prev_hidden = h

        return latents

    def save(self, path: str) -> None:
        """Save agent state."""
        torch.save({
            "world_model": self.world_model.state_dict(),
            "policy_head": self.policy_head.state_dict(),
            "value_head": self.value_head.state_dict(),
            "world_optimizer": self.world_optimizer.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "total_steps": self.total_steps,
        }, path)

    def load(self, path: str) -> None:
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.world_model.load_state_dict(checkpoint["world_model"])
        self.policy_head.load_state_dict(checkpoint["policy_head"])
        self.value_head.load_state_dict(checkpoint["value_head"])
        self.world_optimizer.load_state_dict(checkpoint["world_optimizer"])
        self.policy_optimizer.load_state_dict(checkpoint["policy_optimizer"])
        self.total_steps = checkpoint.get("total_steps", 0)
