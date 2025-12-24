"""
Dreamer Model Components
========================

Implementation of Dreamer algorithm (Hafner et al., 2020) for Atari games.

Components:
1. RSSM (Recurrent State-Space Model) - World model
2. Encoder - Observations to embeddings
3. Decoder - Reconstructs observations
4. Reward predictor
5. Actor - Policy network
6. Critic - Value network

References:
    Dream to Control: https://arxiv.org/abs/1912.01603
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent, OneHotCategorical
import numpy as np
import torch

class OneHotDist(OneHotCategorical):
    """
    Wrapper for OneHotCategorical that supports rsample via Straight-Through Estimator.
    This allows gradients to flow through the discrete sampling.
    """
    def __init__(self, logits=None, probs=None):
        super().__init__(logits=logits, probs=probs)

    def rsample(self, sample_shape=torch.Size()):
        # Use the Gumbel-Softmax trick for differentiation
        # hard=True gives discrete one-hot samples (Straight-Through)
        sample = F.gumbel_softmax(self.logits, tau=0.1, hard=True, dim=-1)
        return sample

# =============================================================================
# Encoder and Decoder
# =============================================================================

class ConvEncoder(nn.Module):
    """Encode observations to embeddings."""
    
    def __init__(self, in_channels=3, embed_dim=1024, depth=32):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 1*depth, 4, stride=2), nn.ReLU(),
            nn.Conv2d(1*depth, 2*depth, 4, stride=2), nn.ReLU(),
            nn.Conv2d(2*depth, 4*depth, 4, stride=2), nn.ReLU(),
            nn.Conv2d(4*depth, 8*depth, 4, stride=2), nn.ReLU(),
        )
        
        # Calculate output size (for 64x64 input)
        self.fc = nn.Linear(8*depth * 2 * 2, embed_dim)
    
    def forward(self, obs):
        """
        Args:
            obs: (batch, channels, height, width)
        Returns:
            embedding: (batch, embed_dim)
        """
        x = self.conv(obs)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)


class ConvDecoder(nn.Module):
    def __init__(self, input_dim, shape=(3, 64, 64)):
        super().__init__()
        self.c, self.h, self.w = shape
        self.depth = 32
        
        # 1. Project input to a 4x4 spatial grid
        # We use depth*32 channels to have enough capacity
        self.linear = nn.Linear(input_dim, 32 * self.depth * 4 * 4)
        
        self.net = nn.Sequential(
            # Input: (B, 32*depth, 4, 4)
            
            # Layer 1: 4x4 -> 8x8
            nn.ConvTranspose2d(32 * self.depth, 4 * self.depth, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            
            # Layer 2: 8x8 -> 16x16
            nn.ConvTranspose2d(4 * self.depth, 2 * self.depth, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            
            # Layer 3: 16x16 -> 32x32
            nn.ConvTranspose2d(2 * self.depth, 1 * self.depth, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            
            # Layer 4: 32x32 -> 64x64
            # Output channels = 3 (RGB)
            nn.ConvTranspose2d(1 * self.depth, self.c, kernel_size=3, stride=2, padding=1, output_padding=1),
            
            # Sigmoid to force pixels into [0, 1] range (matching the target images)
            nn.Sigmoid() 
        )

    def forward(self, x):
        # x shape: (Batch, input_dim)
        x = self.linear(x)
        x = x.reshape(x.shape[0], 32 * self.depth, 4, 4)
        x = self.net(x)
        return x


# =============================================================================
# RSSM (Recurrent State-Space Model)
# =============================================================================

class RSSM(nn.Module):
    """
    Recurrent State-Space Model (World Model).
    
    State consists of:
    - Deterministic state (h): Recurrent hidden state
    - Stochastic state (z): Latent state
    """
    
    def __init__(
        self,
        embed_dim=1024,
        action_dim=6,
        stoch_dim=32,
        deter_dim=200,
        hidden_dim=200,
        num_classes=32,
    ):
        super().__init__()
        self.stoch_dim = stoch_dim
        self.deter_dim = deter_dim
        self.num_classes = num_classes
        self.category_dim = stoch_dim
        
        # Recurrent model: h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
        self.rnn = nn.GRUCell(stoch_dim * num_classes + action_dim, deter_dim)
        
        # Transition predictor: p(z_t | h_t)
        self.fc_prior = nn.Linear(deter_dim, hidden_dim)
        self.fc_prior_logits = nn.Linear(hidden_dim, stoch_dim * num_classes)
        
        # Posterior: q(z_t | h_t, e_t)
        self.fc_posterior = nn.Linear(deter_dim + embed_dim, hidden_dim)
        self.fc_posterior_logits = nn.Linear(hidden_dim, stoch_dim * num_classes)
    
    def initial_state(self, batch_size, device):
        """Get initial state."""
        return {
            'deter': torch.zeros(batch_size, self.deter_dim).to(device),
            'stoch': torch.zeros(batch_size, self.stoch_dim, self.num_classes).to(device),
        }
    
    def observe(self, embed, action, state):
        """
        One step of observation (posterior).
        
        Args:
            embed: Observation embedding (batch, embed_dim)
            action: Action (batch, action_dim)
            state: Previous state dict
        
        Returns:
            next_state: Next state dict
            prior: Prior distribution
            posterior: Posterior distribution
        """
        # Flatten stochastic state for RNN input
        stoch_flat = state['stoch'].reshape(state['stoch'].size(0), -1)
        
        # Recurrent step
        deter = self.rnn(torch.cat([stoch_flat, action], dim=-1), state['deter'])
        
        # Prior p(z_t | h_t)
        prior_logits = self.fc_prior_logits(F.relu(self.fc_prior(deter)))
        prior_logits = prior_logits.reshape(deter.size(0), self.stoch_dim, self.num_classes)
        prior = Independent(OneHotCategorical(logits=prior_logits), 1)
        
        # Posterior q(z_t | h_t, e_t)
        x = torch.cat([deter, embed], dim=-1)
        posterior_logits = self.fc_posterior_logits(F.relu(self.fc_posterior(x)))
        posterior_logits = posterior_logits.reshape(deter.size(0), self.stoch_dim, self.num_classes)
        posterior = Independent(OneHotDist(logits=posterior_logits), 1)
        
        # Sample stochastic state
        stoch = posterior.rsample()
        
        next_state = {'deter': deter, 'stoch': stoch}
        
        return next_state, prior, posterior
    
    def imagine(self, action, state):
        """
        One step of imagination (prior only).
        
        Args:
            action: Action (batch, action_dim)
            state: Previous state dict
        
        Returns:
            next_state: Next state dict
            prior: Prior distribution
        """
        # Flatten stochastic state
        stoch_flat = state['stoch'].reshape(state['stoch'].size(0), -1)
        
        # Recurrent step
        deter = self.rnn(torch.cat([stoch_flat, action], dim=-1), state['deter'])
        
        # Prior p(z_t | h_t)
        prior_logits = self.fc_prior_logits(F.relu(self.fc_prior(deter)))
        prior_logits = prior_logits.reshape(deter.size(0), self.stoch_dim, self.num_classes)
        prior = Independent(OneHotDist(logits=prior_logits), 1)
        
        # Sample stochastic state
        stoch = prior.rsample()
        
        next_state = {'deter': deter, 'stoch': stoch}
        
        return next_state, prior
    
    def get_feature(self, state):
        """Get feature vector from state."""
        stoch_flat = state['stoch'].reshape(state['stoch'].size(0), -1)
        return torch.cat([state['deter'], stoch_flat], dim=-1)


# =============================================================================
# Reward and Discount Predictors
# =============================================================================

class DenseDecoder(nn.Module):
    """Dense decoder for reward/value prediction."""
    
    def __init__(self, input_dim, output_dim=1, hidden_dim=400, num_layers=2):
        super().__init__()
        
        layers = []
        curr_dim = input_dim
        
        for _ in range(num_layers):
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.ReLU())
            curr_dim = hidden_dim
        
        layers.append(nn.Linear(curr_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, feature):
        return self.net(feature)


# =============================================================================
# Actor and Critic
# =============================================================================

class Actor(nn.Module):
    """Policy network."""
    
    def __init__(self, feature_dim, action_dim, hidden_dim=400, num_layers=4):
        super().__init__()
        
        layers = []
        curr_dim = feature_dim
        
        for _ in range(num_layers):
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.ReLU())
            curr_dim = hidden_dim
        
        layers.append(nn.Linear(curr_dim, action_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, feature):
        """Return action logits."""
        return self.net(feature)
    
    def get_action(self, feature, deterministic=False):
        """Sample action from policy."""
        logits = self.forward(feature)
        dist = OneHotCategorical(logits=logits)
        
        if deterministic:
            action = dist.probs.argmax(dim=-1)
            action_onehot = F.one_hot(action, num_classes=logits.size(-1)).float()
        else:
            action_onehot = dist.sample()
            action = action_onehot.argmax(dim=-1)
        
        return action, action_onehot, dist


class Critic(nn.Module):
    """Value network."""
    
    def __init__(self, feature_dim, hidden_dim=400, num_layers=4):
        super().__init__()
        
        layers = []
        curr_dim = feature_dim
        
        for _ in range(num_layers):
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.ReLU())
            curr_dim = hidden_dim
        
        layers.append(nn.Linear(curr_dim, 1))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, feature):
        """Return value estimate."""
        return self.net(feature).squeeze(-1)


# =============================================================================
# Complete Dreamer Agent
# =============================================================================

class DreamerAgent(nn.Module):
    """Complete Dreamer agent."""
    
    def __init__(
        self,
        obs_shape=(3, 64, 64),
        action_dim=6,
        embed_dim=1024,
        stoch_dim=32,
        deter_dim=200,
        hidden_dim=200,
        num_classes=32,
    ):
        super().__init__()
        
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        
        # World model components
        self.encoder = ConvEncoder(obs_shape[0], embed_dim)
        self.rssm = RSSM(embed_dim, action_dim, stoch_dim, deter_dim, hidden_dim, num_classes)
        
        # Feature dimension
        feature_dim = deter_dim + (stoch_dim * num_classes)
        
        self.decoder = ConvDecoder(feature_dim, obs_shape)
        # Predictors
        self.reward_model = DenseDecoder(feature_dim, 1)
        self.discount_model = DenseDecoder(feature_dim, 1)
        
        # Actor-Critic
        self.actor = Actor(feature_dim, action_dim,hidden_dim)
        self.critic = Critic(feature_dim,hidden_dim)
    
    def world_model_loss(self, obs, actions, rewards, dones):
        """
        Compute world model loss (reconstruction + KL + reward).
        
        Args:
            obs: (batch, time, channels, height, width)
            actions: (batch, time, action_dim)
            rewards: (batch, time)
            dones: (batch, time)
        
        Returns:
            loss_dict: Dictionary of losses
        """
        batch_size, time_steps = obs.size(0), obs.size(1)
        device = obs.device
        
        # Encode observations
        obs_flat = obs.reshape(-1, *obs.shape[2:])
        embeds = self.encoder(obs_flat)
        embeds = embeds.reshape(batch_size, time_steps, -1)
        
        # Initialize state
        state = self.rssm.initial_state(batch_size, device)
        
        # Storage
        priors = []
        posteriors = []
        features = []
        
        # Observe sequence
        for t in range(time_steps):
            state, prior, posterior = self.rssm.observe(
                embeds[:, t], actions[:, t], state
            )
            feature = self.rssm.get_feature(state)
            
            priors.append(prior)
            posteriors.append(posterior)
            features.append(feature)
        
        # Stack features
        features = torch.stack(features, dim=1)  # (batch, time, feature_dim)
        
        # Reconstruction loss
        features_flat = features.reshape(-1, features.size(-1))
        recon = self.decoder(features_flat)
        recon_loss = F.mse_loss(recon, obs_flat)
        
        # KL loss
        kl_loss = 0
        for prior, posterior in zip(priors, posteriors):
            kl = torch.distributions.kl_divergence(posterior, prior)
            kl_loss += kl.mean()
        kl_loss = kl_loss / time_steps
        
        # Reward prediction loss
        pred_rewards = self.reward_model(features_flat).reshape(batch_size, time_steps)
        reward_loss = F.mse_loss(pred_rewards, rewards)
        
        # Discount prediction loss
        pred_discounts = self.discount_model(features_flat).reshape(batch_size, time_steps)
        discount_loss = F.binary_cross_entropy_with_logits(
            pred_discounts, (1 - dones.float())
        )
        
        # Total world model loss
        total_loss = recon_loss + kl_loss + reward_loss + discount_loss
        
        return {
            'world_model_loss': total_loss,
            'recon_loss': recon_loss.item(),
            'kl_loss': kl_loss.item(),
            'reward_loss': reward_loss.item(),
            'discount_loss': discount_loss.item(),
        }
    
    def actor_critic_loss(self, obs, actions):
        """
        Compute actor-critic loss via imagination.
        
        Args:
            obs: (batch, time, channels, height, width)
            actions: (batch, time, action_dim)
        
        Returns:
            loss_dict: Dictionary of losses
        """
        batch_size, time_steps = obs.size(0), obs.size(1)
        device = obs.device
        horizon = 15  # Imagination horizon
        
        # Encode and get initial states
        with torch.no_grad():
            obs_flat = obs.reshape(-1, *obs.shape[2:])
            embeds = self.encoder(obs_flat)
            embeds = embeds.reshape(batch_size, time_steps, -1)
            
            # Get states from observations
            state = self.rssm.initial_state(batch_size, device)
            for t in range(time_steps):
                state, _, _ = self.rssm.observe(embeds[:, t], actions[:, t], state)
        
        # Imagine trajectories
        imagined_states = [state]
        imagined_actions = []
        imagined_features = []
        
        for h in range(horizon):
            feature = self.rssm.get_feature(state)
            imagined_features.append(feature)
            
            # Sample action from policy
            _, action, _ = self.actor.get_action(feature)
            imagined_actions.append(action)
            
            # Imagine next state
            state, _ = self.rssm.imagine(action, state)
            imagined_states.append(state)
        
        # Get final feature
        imagined_features.append(self.rssm.get_feature(imagined_states[-1]))
        
        # Stack
        features = torch.stack(imagined_features, dim=1)  # (batch, horizon+1, feature_dim)
        
        # Predict rewards and values
        features_flat = features.reshape(-1, features.size(-1))
        pred_rewards = self.reward_model(features_flat).reshape(batch_size, horizon + 1)
        values = self.critic(features_flat).reshape(batch_size, horizon + 1)
        
        # Compute lambda returns (TD-lambda)
        gamma = 0.99
        lambda_ret = self._compute_lambda_returns(
            pred_rewards[:, :-1], values[:, :-1], values[:, -1], gamma, lambda_=0.95
        )
        
        # Actor loss (maximize returns)
        actor_loss = -(lambda_ret.detach() * values[:, :-1]).mean()
        
        # Critic loss
        critic_loss = F.mse_loss(values[:, :-1], lambda_ret.detach())
        
        return {
            'actor_loss': actor_loss,
            'critic_loss': critic_loss,
        }
    
    def _compute_lambda_returns(self, rewards, values, bootstrap, gamma, lambda_):
        """Compute TD-lambda returns."""
        returns = []
        next_value = bootstrap
        
        for t in reversed(range(rewards.size(1))):
            delta = rewards[:, t] + gamma * next_value - values[:, t]
            next_value = values[:, t] + delta * lambda_
            returns.insert(0, next_value)
        
        return torch.stack(returns, dim=1)