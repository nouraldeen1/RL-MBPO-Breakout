"""
Agent Implementations: MBPO, DDQN, and World Models
===================================================
"""
from __future__ import annotations  # MUST BE FIRST LINE

from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# Import all models. Ensure models.py has these classes.
from models import (
    ActorCritic, 
    DynamicsEnsemble, 
    QNetwork, 
    VAE, 
    MDNRNN, 
    WorldModelsController
)
from utils import (
    Batch, 
    MixedReplayBuffer, 
    ModelReplayBuffer, 
    ReplayBuffer
)

# =============================================================================
# World Models Agent (The one you are currently using)
# =============================================================================

class WorldModelsAgent:
    """
    World Models Agent: V (VAE) + M (MDN-RNN) + C (Controller).
    Trains dynamics in latent space and a linear controller using policy gradients.
    """
    def __init__(self, obs_shape, num_actions, config, device):
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        self.device = device
        
        # Config
        train_cfg = config.get("training", {})
        model_cfg = config.get("model", {})
        
        self.batch_size = train_cfg.get("batch_size", 64)
        self.seq_len = train_cfg.get("seq_len", 32)
        self.gamma = train_cfg.get("gamma", 0.99)
        self.z_dim = model_cfg.get("z_dim", 32)
        self.hidden_dim = model_cfg.get("hidden_dim", 256)
        
        # Models
        self.vae = VAE(in_channels=obs_shape[0], z_dim=self.z_dim).to(device)
        self.mdnrnn = MDNRNN(
            z_dim=self.z_dim, 
            action_dim=num_actions, 
            hidden_dim=self.hidden_dim, 
            n_gaussians=model_cfg.get("n_gaussians", 5)
        ).to(device)
        self.controller = WorldModelsController(
            z_dim=self.z_dim, 
            hidden_dim=self.hidden_dim, 
            action_dim=num_actions
        ).to(device)
        
        # Optimizers
        self.vae_opt = torch.optim.Adam(self.vae.parameters(), lr=train_cfg.get("vae_lr", 1e-4))
        self.mdnrnn_opt = torch.optim.Adam(self.mdnrnn.parameters(), lr=train_cfg.get("mdnrnn_lr", 1e-3))
        self.ctrl_opt = torch.optim.Adam(self.controller.parameters(), lr=train_cfg.get("controller_lr", 1e-4))
        
        # Buffer
        self.buffer = ReplayBuffer(train_cfg.get("buffer_size", 100000), obs_shape, device)
        
        # Runtime State
        self.vae_trained = False
        self.mdnrnn_trained = False
        self.rnn_hidden = None # Current episode hidden state
        
    def get_action(self, obs, deterministic=False):
        """Act using Controller on (z, h)."""
        with torch.no_grad():
            # Handle input formatting
            if isinstance(obs, np.ndarray):
                obs = torch.from_numpy(obs).float().to(self.device)
            if obs.dim() == 3: 
                obs = obs.unsqueeze(0)
            
            # 1. VAE Encode
            mu, _ = self.vae.encode(obs)
            z = mu # Use mean for stability
            
            # 2. RNN State Management
            B = z.size(0)
            if self.rnn_hidden is None or self.rnn_hidden[0].size(1) != B:
                h = torch.zeros(1, B, self.hidden_dim, device=self.device)
                c = torch.zeros(1, B, self.hidden_dim, device=self.device)
                self.rnn_hidden = (h, c)
            
            # 3. Controller Act (Use previous h)
            h_prev = self.rnn_hidden[0].squeeze(0)
            action, log_prob = self.controller.get_action(z, h_prev, deterministic)
            
            # 4. RNN Step (Predict next h for next step)
            _, _, _, _, _, self.rnn_hidden = self.mdnrnn(z, action, self.rnn_hidden)
            
            return action.cpu().numpy(), log_prob.cpu().numpy()

    def store_transition(self, s, a, r, ns, d):
        self.buffer.add(s, a, r, ns, d)
        if d: 
            self.rnn_hidden = None

    def update(self):
        # Wait for enough data to form sequences
        if not self.buffer.is_ready(self.batch_size * 5): 
            return {}
        
        metrics = {}
        metrics.update(self._train_vae())
        metrics.update(self._train_mdnrnn())
        metrics.update(self._train_controller())
        return metrics

    def _train_vae(self):
        batch = self.buffer.sample(self.batch_size)
        
        # FIX: Normalize 0-255 -> 0-1
        states_norm = batch.states / 255.0
        
        loss, m = self.vae.compute_loss(states_norm)
        
        self.vae_opt.zero_grad()
        loss.backward()
        self.vae_opt.step()
        self.vae_trained = True
        return m

    def _train_mdnrnn(self):
        if not self.vae_trained: return {}
        # Sample Sequences (already normalized in utils.py sample_sequence)
        batch = self.buffer.sample_sequence(self.batch_size, self.seq_len)
        states = batch["states"]     # (B, L, C, H, W)
        actions = batch["actions"]   # (B, L)
        rewards = batch["rewards"].unsqueeze(-1)
        dones = batch["dones"].unsqueeze(-1)
        next_states = batch["next_states"]
        
        B, L = states.shape[:2]
        
        # Pre-compute Latents (Stop Gradients to VAE to save VRAM/Stability)
        with torch.no_grad():
            s_flat = states.view(B*L, *states.shape[2:])
            ns_flat = next_states.view(B*L, *states.shape[2:])
            z, _ = self.vae.encode(s_flat)
            z_next, _ = self.vae.encode(ns_flat)
            z = z.view(B, L, -1)
            z_next = z_next.view(B, L, -1)

        # Forward RNN
        pi, mu, sigma, pred_r, pred_d, _ = self.mdnrnn(z, actions)
        
        # Losses
        # 1. MDN Loss (Negative Log Likelihood)
        z_target = z_next.view(B*L, 1, -1) # (N, 1, Z)
        mu_flat = mu.view(B*L, 5, -1)      # (N, G, Z)
        sigma_flat = sigma.view(B*L, 5, -1)
        pi_flat = pi.view(B*L, 5)
        
        # log_prob per component
        log_prob_g = -0.5 * (torch.sum(((z_target - mu_flat)**2) / sigma_flat**2, dim=2) + 
                             torch.sum(torch.log(2 * np.pi * sigma_flat**2), dim=2))
        
        # LogSumExp to combine mixture
        log_prob = torch.logsumexp(torch.log(pi_flat + 1e-8) + log_prob_g, dim=1)
        loss_mdn = -log_prob.mean()
        
        loss_r = F.mse_loss(pred_r.view(B*L, 1), rewards.view(B*L, 1))
        loss_d = F.binary_cross_entropy(pred_d.view(B*L, 1), dones.view(B*L, 1))
        
        total_loss = loss_mdn + loss_r + loss_d
        
        self.mdnrnn_opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.mdnrnn.parameters(), 5.0)
        self.mdnrnn_opt.step()
        
        self.mdnrnn_trained = True
        return {"mdnrnn/loss": total_loss.item(), "mdnrnn/nll": loss_mdn.item()}

    def _train_controller(self):
        if not self.mdnrnn_trained: return {}
        
        # Start dreaming from real states
        batch = self.buffer.sample(self.batch_size)
        
        # Normalize!
        states_norm = batch.states / 255.0
        
        with torch.no_grad():
            z, _ = self.vae.encode(states_norm)
            
        h = torch.zeros(1, self.batch_size, self.hidden_dim, device=self.device)
        c = torch.zeros(1, self.batch_size, self.hidden_dim, device=self.device)
        hidden = (h, c)
        
        log_probs = []
        rewards = []
        
        # Dream Rollout
        for _ in range(15):
            h_in = hidden[0].squeeze(0)
            
            # --- SAFETY CHECK ---
            if torch.isnan(z).any() or torch.isnan(h_in).any():
                return {} # Abort update if NaNs detected
            # --------------------

            action, log_prob = self.controller.get_action(z, h_in)
            
            with torch.no_grad():
                z, r, d, hidden = self.mdnrnn.sample_next_latent(z, action, hidden)
            
            log_probs.append(log_prob)
            rewards.append(r.squeeze(-1))
            
        # REINFORCE Update
        R = torch.zeros(self.batch_size, device=self.device)
        policy_loss = 0
        returns = []
        
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.stack(returns)
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        for lp, ret in zip(log_probs, returns):
            policy_loss += -(lp * ret).mean()
            
        self.ctrl_opt.zero_grad()
        policy_loss.backward()
        
        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(self.controller.parameters(), 1.0)
        
        self.ctrl_opt.step()
        
        return {"controller/loss": policy_loss.item()}

    def save(self, path):
        torch.save({
            "vae": self.vae.state_dict(),
            "mdnrnn": self.mdnrnn.state_dict(),
            "controller": self.controller.state_dict(),
            "vae_opt": self.vae_opt.state_dict(),
            "mdnrnn_opt": self.mdnrnn_opt.state_dict(),
            "ctrl_opt": self.ctrl_opt.state_dict()
        }, path)
        
    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.vae.load_state_dict(ckpt["vae"])
        self.mdnrnn.load_state_dict(ckpt["mdnrnn"])
        self.controller.load_state_dict(ckpt["controller"])
        self.vae_trained = True
        self.mdnrnn_trained = True


# =============================================================================
# MBPO Agent (Preserved from your original code)
# =============================================================================

class MBPOAgent:
    def __init__(self, obs_shape, num_actions, config, device="cuda"):
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        self.device = device
        
        train_cfg = config.get("training", {})
        model_cfg = config.get("model", {})
        
        self.gamma = train_cfg.get("gamma", 0.99)
        self.batch_size = train_cfg.get("batch_size", 256)
        self.model_rollout_length = train_cfg.get("model_rollout_length", 1)
        
        # Models
        self.actor_critic = ActorCritic(num_actions, obs_shape[0], model_cfg.get("fc_dim", 512)).to(device)
        self.dynamics = DynamicsEnsemble(
            ensemble_size=model_cfg.get("ensemble_size", 5),
            num_actions=num_actions,
            in_channels=obs_shape[0]
        ).to(device)
        
        self.policy_optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=train_cfg.get("policy_lr", 3e-4))
        self.dynamics_optimizer = torch.optim.Adam(self.dynamics.parameters(), lr=train_cfg.get("dynamics_lr", 1e-4))
        
        # Buffers
        self.real_buffer = ReplayBuffer(train_cfg.get("buffer_size", 100000), obs_shape, device)
        self.model_buffer = ModelReplayBuffer(train_cfg.get("buffer_size", 100000), obs_shape, device)
        self.mixed_buffer = MixedReplayBuffer(self.real_buffer, self.model_buffer, train_cfg.get("real_ratio", 0.1))
        
        self.dynamics_trained = False

    def get_action(self, state, deterministic=False):
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = torch.from_numpy(state).float().to(self.device)
            if state.dim() == 3: state = state.unsqueeze(0)
            
            action, log_prob, _, _ = self.actor_critic.get_action_and_value(state, deterministic=deterministic)
            return action.cpu().numpy(), {"log_prob": log_prob.cpu().numpy()}

    def store_transition(self, s, a, r, ns, d):
        self.real_buffer.add(s, a, r, ns, d)

    def update(self):
        if not self.real_buffer.is_ready(self.batch_size): return {}
        
        # 1. Train Dynamics
        dyn_loss = 0
        batch = self.real_buffer.sample(self.batch_size)
        loss, _ = self.dynamics.compute_loss(batch.states, batch.actions, batch.next_states, batch.rewards)
        self.dynamics_optimizer.zero_grad()
        loss.backward()
        self.dynamics_optimizer.step()
        self.dynamics_trained = True
        
        # 2. Rollout
        if self.dynamics_trained:
            self._generate_rollouts()
            
        # 3. Train Policy
        batch = self.mixed_buffer.sample(self.batch_size)
        _, log_prob, entropy, value = self.actor_critic.get_action_and_value(batch.states, batch.actions)
        
        # PPO/A2C style loss (Simplified)
        with torch.no_grad():
            _, _, _, next_val = self.actor_critic.get_action_and_value(batch.next_states)
            target = batch.rewards + self.gamma * next_val * (1 - batch.dones)
            adv = target - value
        
        loss_p = -(log_prob * adv).mean()
        loss_v = F.mse_loss(value, target)
        loss = loss_p + 0.5 * loss_v - 0.01 * entropy.mean()
        
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()
        
        return {"policy/loss": loss.item()}

    def _generate_rollouts(self):
        batch = self.real_buffer.sample(256) # Rollout batch size
        state = batch.states
        for _ in range(self.model_rollout_length):
            action, _, _, _ = self.actor_critic.get_action_and_value(state)
            next_state, reward, _ = self.dynamics.predict_next_state(state, action)
            # Add to model buffer (simplified add_batch)
            self.model_buffer.buffer.add_batch(
                state.cpu().numpy(), 
                action.cpu().numpy(), 
                reward.squeeze(-1).cpu().numpy(), 
                next_state.cpu().numpy(), 
                torch.zeros_like(reward).squeeze(-1).cpu().numpy() # Assume not done in model rollout
            )
            state = next_state

    def save(self, path):
        torch.save(self.actor_critic.state_dict(), path)
    def load(self, path):
        self.actor_critic.load_state_dict(torch.load(path))


# =============================================================================
# DDQN Agent
# =============================================================================

class DDQNAgent:
    def __init__(self, obs_shape, num_actions, config, device="cuda"):
        self.q_net = QNetwork(num_actions, obs_shape[0]).to(device)
        self.target_net = QNetwork(num_actions, obs_shape[0]).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.opt = torch.optim.Adam(self.q_net.parameters(), lr=1e-4)
        self.buffer = ReplayBuffer(100000, obs_shape, device)
        self.device = device
        self.num_actions = num_actions
        self.epsilon = 1.0

    def get_action(self, state, deterministic=False):
        if not deterministic and np.random.random() < self.epsilon:
            return np.random.randint(0, self.num_actions), {}
        with torch.no_grad():
            if isinstance(state, np.ndarray):
                state = torch.from_numpy(state).float().to(self.device)
            if state.dim() == 3: state = state.unsqueeze(0)
            return self.q_net(state).argmax(dim=1).cpu().numpy(), {}

    def store_transition(self, s, a, r, ns, d):
        self.buffer.add(s, a, r, ns, d)

    def update(self):
        if not self.buffer.is_ready(64): return {}
        batch = self.buffer.sample(64)
        
        q = self.q_net(batch.states).gather(1, batch.actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_a = self.q_net(batch.next_states).argmax(1)
            next_q = self.target_net(batch.next_states).gather(1, next_a.unsqueeze(1)).squeeze(1)
            target = batch.rewards + 0.99 * next_q * (1 - batch.dones)
            
        loss = F.mse_loss(q, target)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        
        self.epsilon = max(0.01, self.epsilon * 0.995)
        return {"loss": loss.item()}
    
    def save(self, path): torch.save(self.q_net.state_dict(), path)
    def load(self, path): self.q_net.load_state_dict(torch.load(path))


# =============================================================================
# Factory
# =============================================================================

def create_agent(algorithm: str, obs_shape: Tuple, num_actions: int, config: Dict, device: str):
    if algorithm == "mbpo":
        return MBPOAgent(obs_shape, num_actions, config, device)
    elif algorithm == "ddqn":
        return DDQNAgent(obs_shape, num_actions, config, device)
    elif algorithm == "world_models":
        return WorldModelsAgent(obs_shape, num_actions, config, device)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")