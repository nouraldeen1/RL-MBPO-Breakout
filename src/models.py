
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from utils import kl_divergence

# =============================================================================
# Dreamer World Model Components
# =============================================================================

class Encoder(nn.Module):
    """
    Encoder: CNN to latent vector z
    """
    def __init__(self, in_channels=4, latent_dim=1024):
        super().__init__()
        self.cnn = NatureCNN(in_channels=in_channels, fc_dim=latent_dim)
    def forward(self, x):
        return self.cnn(x)

class RSSM(nn.Module):
    """
    Recurrent State-Space Model (RSSM): GRU for latent dynamics
    """
    def __init__(self, latent_dim=1024, action_dim=4, in_channels=4):
        super().__init__()
        self.encoder = Encoder(in_channels=in_channels, latent_dim=latent_dim)
        self.rnn = nn.GRUCell(latent_dim + action_dim, latent_dim)
        self.img_predictor = nn.Linear(latent_dim, latent_dim)
        self.prior_mean = nn.Linear(latent_dim, latent_dim)
        self.prior_logvar = nn.Linear(latent_dim, latent_dim)
        self.post_mean = nn.Linear(latent_dim, latent_dim)
        self.post_logvar = nn.Linear(latent_dim, latent_dim)
    def forward(self, prev_latent, prev_action, prev_hidden):
        # Concatenate latent and action
        x = torch.cat([prev_latent, prev_action], dim=-1)
        h = self.rnn(x, prev_hidden)
        return h
    def observe(self, h, action, obs):
        z = self.encoder(obs)
        h_next = self.rnn(torch.cat([z, action], dim=-1), h)
        post_mean = self.post_mean(h_next)
        post_logvar = self.post_logvar(h_next)
        return z, (post_mean, post_logvar), h_next
    def imagine(self, h, action):
        z_prior = self.img_predictor(h)
        h_next = self.rnn(torch.cat([z_prior, action], dim=-1), h)
        prior_mean = self.prior_mean(h_next)
        prior_logvar = self.prior_logvar(h_next)
        return z_prior, (prior_mean, prior_logvar)
    def forward(self, prev_latent, prev_action, prev_hidden):
        # Concatenate latent and action
        x = torch.cat([prev_latent, prev_action], dim=-1)
        h = self.rnn(x, prev_hidden)
        return h

class Decoder(nn.Module):
    """
    Decoder: Deconv to reconstruct pixels from latent
    """
    def __init__(self, latent_dim=1024, out_channels=4):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 3136)
        self.deconv1 = nn.ConvTranspose2d(64, 64, 3, stride=1)
        self.deconv2 = nn.ConvTranspose2d(64, 32, 4, stride=2)
        self.deconv3 = nn.ConvTranspose2d(32, out_channels, 8, stride=4)
    def forward(self, z):
        # z: (batch, latent_dim)
        x = F.relu(self.fc(z))
        x = x.view(-1, 64, 7, 7)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))
        return x

class WorldModel(nn.Module):
    """
    Dreamer-style World Model: Encoder, RSSM, Decoder, Reward Model
    """
    def __init__(self, in_channels=4, latent_dim=1024, action_dim=4):
        super().__init__()
        self.rssm = RSSM(latent_dim=latent_dim, action_dim=action_dim, in_channels=in_channels)
        self.decoder = Decoder(latent_dim=latent_dim, out_channels=in_channels)
        self.reward_model = nn.Linear(latent_dim, 1)
        self.done_model = nn.Linear(latent_dim, 1)
    def forward(self, obs, prev_action, prev_hidden):
        z = self.rssm.encoder(obs)
        h = self.rssm(z, prev_action, prev_hidden)
        recon = self.decoder(h)
        reward = self.reward_model(h)
        done = torch.sigmoid(self.done_model(h))
        return h, recon, reward, done

    def compute_loss(self, seq_obs, seq_actions, seq_rewards, seq_dones):
        """
        Compute world model loss for a sequence.
        seq_obs: (T, C, H, W)
        seq_actions: (T, num_actions)
        seq_rewards: (T,)
        seq_dones: (T,)
        """
        T = seq_obs.shape[0]
        h = torch.zeros(1, self.rssm.rnn.hidden_size, device=seq_obs.device)  # B=1

        recon_loss = 0.0
        reward_loss = 0.0
        kl_loss = 0.0

        for t in range(T - 1):
            z_post, post_stats, h_next = self.rssm.observe(h, seq_actions[t:t+1], seq_obs[t+1:t+2])
            z_prior, prior_stats = self.rssm.imagine(h, seq_actions[t:t+1])

            recon = self.decoder(h_next)
            recon_loss += F.mse_loss(recon, seq_obs[t+1:t+2])

            pred_rew = self.reward_model(h_next)
            reward_loss += F.mse_loss(pred_rew.squeeze(-1), seq_rewards[t:t+1])

            kl_loss += kl_divergence(prior_stats[0], prior_stats[1], post_stats[0], post_stats[1])
            h = h_next

        return recon_loss, reward_loss, kl_loss

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


# =============================================================================
# Nature CNN Encoder (Shared Architecture)
# =============================================================================

class NatureCNN(nn.Module):
    """
    The standard Nature DQN convolutional encoder.
    
    Architecture:
        - Conv2d(in_channels, 32, 8x8, stride=4) + ReLU
        - Conv2d(32, 64, 4x4, stride=2) + ReLU  
        - Conv2d(64, 64, 3x3, stride=1) + ReLU
        - Flatten
        - Linear(3136, fc_dim) + ReLU
    
    This architecture has proven effective for Atari games and provides
    a good balance between expressiveness and computational efficiency.
    
    Args:
        in_channels: Number of input channels (4 for frame-stacked observations)
        fc_dim: Output dimension of the fully connected layer
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        fc_dim: int = 512,
        cnn_channels: Tuple[int, ...] = (32, 64, 64),
        cnn_kernels: Tuple[int, ...] = (8, 4, 3),
        cnn_strides: Tuple[int, ...] = (4, 2, 1),
    ) -> None:
        super().__init__()
        
        self.fc_dim = fc_dim
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, cnn_channels[0], cnn_kernels[0], stride=cnn_strides[0])
        self.conv2 = nn.Conv2d(cnn_channels[0], cnn_channels[1], cnn_kernels[1], stride=cnn_strides[1])
        self.conv3 = nn.Conv2d(cnn_channels[1], cnn_channels[2], cnn_kernels[2], stride=cnn_strides[2])
        
        # Calculate flattened size after convolutions
        # For 84x84 input: 84 -> 20 -> 9 -> 7, so 7*7*64 = 3136
        self.flatten_dim = self._calculate_flatten_dim(in_channels)
        
        # Fully connected layer
        self.fc = nn.Linear(self.flatten_dim, fc_dim)
        
        # Initialize weights using orthogonal initialization
        self._init_weights()
    
    def _calculate_flatten_dim(self, in_channels: int) -> int:
        """Calculate the flattened dimension after conv layers."""
        dummy = torch.zeros(1, in_channels, 84, 84)
        with torch.no_grad():
            x = F.relu(self.conv1(dummy))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
        return int(x.numel())
    
    def _init_weights(self) -> None:
        """Initialize weights using orthogonal initialization."""
        for module in [self.conv1, self.conv2, self.conv3]:
            nn.init.orthogonal_(module.weight, gain=math.sqrt(2))
            nn.init.zeros_(module.bias)
        
        nn.init.orthogonal_(self.fc.weight, gain=math.sqrt(2))
        nn.init.zeros_(self.fc.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CNN encoder.
        
        Args:
            x: Input tensor of shape (batch, channels, 84, 84)
        
        Returns:
            Feature tensor of shape (batch, fc_dim)
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc(x))
        return x


# =============================================================================
# Policy Network (Actor)
# =============================================================================

class PolicyNetwork(nn.Module):
    """
    Policy network for discrete action spaces (Breakout has 4 actions).
    
    Uses NatureCNN as encoder followed by a policy head that outputs
    action logits. Supports both stochastic (training) and deterministic
    (evaluation) action selection.
    
    Args:
        num_actions: Number of discrete actions
        in_channels: Number of input channels
        fc_dim: Hidden layer dimension
    """
    
    def __init__(
        self,
        num_actions: int = 4,
        in_channels: int = 4,
        fc_dim: int = 512,
    ) -> None:
        super().__init__()
        
        self.encoder = NatureCNN(in_channels=in_channels, fc_dim=fc_dim)
        self.policy_head = nn.Linear(fc_dim, num_actions)
        
        # Initialize policy head with small weights
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Compute action logits.
        
        Args:
            obs: Observation tensor of shape (batch, 4, 84, 84)
        
        Returns:
            Action logits of shape (batch, num_actions)
        """
        features = self.encoder(obs)
        logits = self.policy_head(features)
        # LOGIT CLIPPING: Prevent any single action logit from saturating
        logits = torch.clamp(logits, min=-10.0, max=10.0)
        return logits
    
    def get_action(
        self,
        obs: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action from policy.
        
        Args:
            obs: Observation tensor
            deterministic: If True, return argmax action
        
        Returns:
            action: Selected action
            log_prob: Log probability of selected action
            entropy: Policy entropy
        """
        logits = self.forward(obs)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        
        if deterministic:
            action = torch.argmax(logits, dim=-1)
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy
    
    def evaluate_actions(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate log probabilities and entropy for given actions.
        
        Args:
            obs: Observation tensor
            actions: Action tensor
        
        Returns:
            log_prob: Log probabilities of actions
            entropy: Policy entropy
        """
        logits = self.forward(obs)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        
        return log_prob, entropy


# =============================================================================
# Value Network (Critic)
# =============================================================================

class ValueNetwork(nn.Module):
    """
    Value network for estimating state values.
    
    Uses NatureCNN as encoder followed by a value head that outputs
    a single scalar value estimate.
    
    Args:
        in_channels: Number of input channels
        fc_dim: Hidden layer dimension
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        fc_dim: int = 512,
    ) -> None:
        super().__init__()
        
        self.encoder = NatureCNN(in_channels=in_channels, fc_dim=fc_dim)
        self.value_head = nn.Linear(fc_dim, 1)
        
        # Initialize value head
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Compute state value.
        
        Args:
            obs: Observation tensor of shape (batch, 4, 84, 84)
        
        Returns:
            Value estimate of shape (batch, 1)
        """
        features = self.encoder(obs)
        value = self.value_head(features)
        return value


# =============================================================================
# Q-Network (for DDQN)
# =============================================================================

class QNetwork(nn.Module):
    """
    Q-Network for DDQN algorithm.
    
    Outputs Q-values for all actions given a state observation.
    Uses dueling architecture option for better value decomposition.
    
    Args:
        num_actions: Number of discrete actions
        in_channels: Number of input channels
        fc_dim: Hidden layer dimension
        dueling: Whether to use dueling architecture
    """
    
    def __init__(
        self,
        num_actions: int = 4,
        in_channels: int = 4,
        fc_dim: int = 512,
        dueling: bool = True,
    ) -> None:
        super().__init__()
        
        self.num_actions = num_actions
        self.dueling = dueling
        
        self.encoder = NatureCNN(in_channels=in_channels, fc_dim=fc_dim)
        
        if dueling:
            # Dueling DQN: separate value and advantage streams
            self.value_stream = nn.Linear(fc_dim, 1)
            self.advantage_stream = nn.Linear(fc_dim, num_actions)
        else:
            self.q_head = nn.Linear(fc_dim, num_actions)
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize output layer weights."""
        if self.dueling:
            nn.init.orthogonal_(self.value_stream.weight, gain=1.0)
            nn.init.zeros_(self.value_stream.bias)
            nn.init.orthogonal_(self.advantage_stream.weight, gain=1.0)
            nn.init.zeros_(self.advantage_stream.bias)
        else:
            nn.init.orthogonal_(self.q_head.weight, gain=1.0)
            nn.init.zeros_(self.q_head.bias)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-values for all actions.
        
        Args:
            obs: Observation tensor of shape (batch, 4, 84, 84)
        
        Returns:
            Q-values of shape (batch, num_actions)
        """
        features = self.encoder(obs)
        
        if self.dueling:
            value = self.value_stream(features)
            advantage = self.advantage_stream(features)
            # Combine using the dueling formula
            q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        else:
            q_values = self.q_head(features)
        
        return q_values


# =============================================================================
# Dynamics Model (Single Model)
# =============================================================================

class DynamicsModel(nn.Module):
    """
    Single dynamics model for MBPO.
    
    Predicts the DELTA state (change in pixels) and reward given current
    state and action. Predicting delta is easier than full next state:
    - The model only needs to learn where the ball moved
    - Most pixels stay the same between frames
    - Reduces prediction variance and improves accuracy
    
    Architecture:
        - NatureCNN encoder for state
        - Action embedding
        - MLP decoder for delta_state and reward
    
    Args:
        num_actions: Number of discrete actions
        in_channels: Number of input channels (4 for frame stack)
        hidden_dim: Hidden layer dimension for MLP
        fc_dim: CNN encoder output dimension
        predict_reward: Whether to predict reward along with state
    """
    
    def __init__(
        self,
        num_actions: int = 4,
        in_channels: int = 4,
        hidden_dim: int = 256,
        fc_dim: int = 512,
        predict_reward: bool = True,
    ) -> None:
        super().__init__()
        
        self.num_actions = num_actions
        self.in_channels = in_channels
        self.predict_reward = predict_reward
        
        # State encoder
        self.encoder = NatureCNN(in_channels=in_channels, fc_dim=fc_dim)
        
        # Action embedding
        self.action_embed = nn.Embedding(num_actions, 64)
        
        # Transition model (predicts delta in latent space)
        self.transition_mlp = nn.Sequential(
            nn.Linear(fc_dim + 64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        
        # Delta state decoder (deconvolution to reconstruct delta)
        # Reverse the NatureCNN encoder path exactly: 7 -> 9 -> 20 -> 84
        # ConvTranspose formula: out = (in-1)*stride - 2*padding + kernel + output_padding
        self.delta_decoder = nn.Sequential(
            nn.Linear(hidden_dim, 32 * 7 * 7), # Reduced channels
            nn.ReLU(),
            nn.Unflatten(1, (32, 7, 7)),
            # Layer 1: 7 -> 9. (7-1)*1 - 2*0 + 3 + 0 = 9.
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=1, padding=0), # Reduced channels
            nn.ReLU(),
            # Layer 2: 9 -> 20. (9-1)*2 - 2*0 + 4 + 0 = 20.
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=0), # Reduced channels
            nn.ReLU(),
            # Layer 3: 20 -> 84. (20-1)*4 - 2*0 + 8 + 0 = 84.
            nn.ConvTranspose2d(16, in_channels, kernel_size=8, stride=4, padding=0), # Reduced channels
        )
        
        # Reward predictor
        if predict_reward:
            self.reward_head = nn.Sequential(
                nn.Linear(hidden_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
            )
        
        self._init_weights()
    
    def _init_weights(self) -> None:
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.orthogonal_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Predict delta state and reward.
        
        Args:
            state: Current state tensor of shape (batch, 4, 84, 84)
            action: Action tensor of shape (batch,) with integer actions
        
        Returns:
            delta_state: Predicted pixel-wise change (batch, 4, 84, 84)
            reward: Predicted reward (batch, 1) or None if not predicting
        """
        # Encode state
        state_features = self.encoder(state)
        
        # Embed action
        action_features = self.action_embed(action)
        
        # Concatenate and process
        combined = torch.cat([state_features, action_features], dim=-1)
        transition_features = self.transition_mlp(combined)
        
        # Decode delta state
        delta_state = self.delta_decoder(transition_features)
        
        # Predict reward
        reward = None
        if self.predict_reward:
            reward = self.reward_head(transition_features)
        
        return delta_state, reward
    
    def predict_next_state(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Predict next state by adding delta to current state.
        
        Args:
            state: Current state tensor
            action: Action tensor
        
        Returns:
            next_state: Predicted next state
            reward: Predicted reward
        """
        delta_state, reward = self.forward(state, action)
        next_state = state + delta_state
        # Clamp to valid range [0, 1]
        next_state = torch.clamp(next_state, 0.0, 1.0)
        return next_state, reward


# =============================================================================
# Dynamics Ensemble (MBPO Core)
# =============================================================================

class DynamicsEnsemble(nn.Module):
    """
    Ensemble of dynamics models for MBPO.
    
    The ensemble provides several key benefits:
    1. **Uncertainty Estimation**: Disagreement between models indicates epistemic uncertainty
    2. **Reduced Model Bias**: Averaging predictions reduces individual model errors
    3. **Better Exploration**: High uncertainty regions encourage exploration
    
    MBPO uses the ensemble to generate synthetic rollouts for policy training.
    The model uncertainty helps prevent the policy from exploiting model errors.
    
    Args:
        ensemble_size: Number of models in the ensemble (default: 5)
        num_actions: Number of discrete actions
        in_channels: Number of input channels
        hidden_dim: Hidden dimension for MLP layers
        fc_dim: CNN encoder output dimension
        predict_reward: Whether to predict rewards
    """
    
    def __init__(
        self,
        ensemble_size: int = 5,
        num_actions: int = 4,
        in_channels: int = 4,
        hidden_dim: int = 256,
        fc_dim: int = 512,
        predict_reward: bool = True,
    ) -> None:
        super().__init__()
        
        self.ensemble_size = ensemble_size
        self.predict_reward = predict_reward
        
        # Create ensemble as ModuleList for proper parameter registration
        self.models = nn.ModuleList([
            DynamicsModel(
                num_actions=num_actions,
                in_channels=in_channels,
                hidden_dim=hidden_dim,
                fc_dim=fc_dim,
                predict_reward=predict_reward,
            )
            for _ in range(ensemble_size)
        ])
    
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        model_idx: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Get predictions from ensemble.
        
        Args:
            state: Current state tensor (batch, 4, 84, 84)
            action: Action tensor (batch,)
            model_idx: If provided, use only this model. Otherwise, use all.
        
        Returns:
            mean_delta: Mean delta state prediction
            mean_reward: Mean reward prediction (or None)
            std_delta: Standard deviation of delta predictions (uncertainty)
        """
        if model_idx is not None:
            # Use single model
            delta, reward = self.models[model_idx](state, action)
            return delta, reward, None
        
        # Get predictions from all models
        deltas = []
        rewards = []
        
        for model in self.models:
            delta, reward = model(state, action)
            deltas.append(delta)
            if reward is not None:
                rewards.append(reward)
        
        # Stack predictions
        deltas = torch.stack(deltas, dim=0)  # (ensemble, batch, C, H, W)
        
        # Compute mean and std
        mean_delta = deltas.mean(dim=0)
        std_delta = deltas.std(dim=0)
        
        mean_reward = None
        if rewards:
            rewards = torch.stack(rewards, dim=0)
            mean_reward = rewards.mean(dim=0)
        
        return mean_delta, mean_reward, std_delta
    
    def predict_next_state(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        model_idx: Optional[int] = None,
        use_mean: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Predict next state using ensemble.
        
        Args:
            state: Current state tensor
            action: Action tensor
            model_idx: Specific model to use (if None, use ensemble)
            use_mean: Whether to use mean prediction or sample from ensemble
        
        Returns:
            next_state: Predicted next state
            reward: Predicted reward
            uncertainty: Prediction uncertainty (std of delta)
        """
        if use_mean or model_idx is not None:
            mean_delta, mean_reward, std_delta = self.forward(state, action, model_idx)
            next_state = state + mean_delta
            uncertainty = std_delta
        else:
            # For low-VRAM GPUs, just use a single random model instead of
            # the full ensemble. This avoids creating massive intermediate tensors.
            random_model_idx = torch.randint(0, self.ensemble_size, (1,)).item()
            delta, reward = self.models[random_model_idx](state, action)
            next_state = state + delta
            mean_reward = reward
            uncertainty = None
        
        # Clamp to valid range
        next_state = torch.clamp(next_state, 0.0, 1.0)
        
        return next_state, mean_reward, uncertainty
    
    def get_uncertainty(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute prediction uncertainty (epistemic uncertainty).
        
        Higher uncertainty indicates less confident predictions,
        typically in regions with less training data.
        
        Args:
            state: State tensor
            action: Action tensor
        
        Returns:
            uncertainty: Mean pixel-wise uncertainty across ensemble
        """
        _, _, std_delta = self.forward(state, action)
        # Aggregate uncertainty across spatial dimensions
        uncertainty = std_delta.mean(dim=(1, 2, 3))  # (batch,)
        return uncertainty
    
    def compute_loss(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        next_state: torch.Tensor,
        reward: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute training loss for all ensemble members.
        
        Uses MSE loss for delta state and reward predictions.
        Each model is trained independently on the same data.
        
        Args:
            state: Current state batch
            action: Action batch
            next_state: Target next state batch
            reward: Target reward batch
        
        Returns:
            total_loss: Sum of losses across ensemble
            metrics: Dictionary of loss metrics
        """
        # Target delta state
        target_delta = next_state - state
        
        total_loss = 0.0
        delta_losses = []
        reward_losses = []
        
        for model in self.models:
            pred_delta, pred_reward = model(state, action)
            
            # Delta state loss (MSE)
            delta_loss = F.mse_loss(pred_delta, target_delta)
            delta_losses.append(delta_loss.item())
            total_loss = total_loss + delta_loss
            
            # Reward loss
            if pred_reward is not None:
                reward_loss = F.mse_loss(pred_reward.squeeze(-1), reward)
                reward_losses.append(reward_loss.item())
                total_loss = total_loss + reward_loss
        
        metrics = {
            "dynamics/delta_loss": sum(delta_losses) / len(delta_losses),
            "dynamics/reward_loss": sum(reward_losses) / len(reward_losses) if reward_losses else 0.0,
            "dynamics/total_loss": total_loss.item(),
        }
        
        return total_loss, metrics


# =============================================================================
# Actor-Critic Network (for PPO/A2C)
# =============================================================================

class ActorCritic(nn.Module):
    """
    Combined actor-critic network for PPO/A2C algorithms.
    
    Shares the CNN encoder between policy and value networks for
    more efficient learning. Outputs both action distribution and
    state value estimate.
    
    Args:
        num_actions: Number of discrete actions
        in_channels: Number of input channels
        fc_dim: Hidden layer dimension
    """
    
    def __init__(
        self,
        num_actions: int = 4,
        in_channels: int = 4,
        fc_dim: int = 512,
    ) -> None:
        super().__init__()
        
        self.encoder = NatureCNN(in_channels=in_channels, fc_dim=fc_dim)
        self.policy_head = nn.Linear(fc_dim, num_actions)
        self.value_head = nn.Linear(fc_dim, 1)
        
        # Initialize heads
        nn.init.orthogonal_(self.policy_head.weight, gain=0.01)
        nn.init.zeros_(self.policy_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)
    
    def forward(
        self,
        obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute policy logits and value estimate.
        
        Args:
            obs: Observation tensor
        
        Returns:
            logits: Action logits
            value: State value estimate
        """
        features = self.encoder(obs)
        logits = self.policy_head(features)
        # LOGIT CLIPPING to keep policy uncertain enough to allow learning
        logits = torch.clamp(logits, min=-10.0, max=10.0)
        value = self.value_head(features)
        return logits, value
    
    def get_action_and_value(
        self,
        obs: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get action, log probability, entropy, and value.
        
        Args:
            obs: Observation tensor
            action: If provided, evaluate this action. Otherwise, sample.
            deterministic: If True and action is None, return argmax action
        
        Returns:
            action: Selected or provided action
            log_prob: Log probability of action
            entropy: Policy entropy
            value: State value estimate
        """
        logits, value = self.forward(obs)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        
        if action is None:
            if deterministic:
                action = torch.argmax(logits, dim=-1)
            else:
                action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value.squeeze(-1)


# =============================================================================
# Model Factory
# =============================================================================

def create_models(
    num_actions: int = 4,
    in_channels: int = 4,
    fc_dim: int = 512,
    ensemble_size: int = 5,
    hidden_dim: int = 256,
    algorithm: str = "mbpo",
    device: str = "cuda",
) -> Dict[str, nn.Module]:
    """
    Factory function to create all required models.
    
    Args:
        num_actions: Number of discrete actions
        in_channels: Number of input channels
        fc_dim: FC layer dimension
        ensemble_size: Number of dynamics models
        hidden_dim: Hidden dimension for dynamics MLP
        algorithm: Algorithm type ("mbpo", "ppo", "sac", "ddqn")
        device: Device to place models on
    
    Returns:
        Dictionary of model name -> model instance
    """
    models = {}
    
    # Dynamics ensemble (for MBPO)
    if algorithm == "mbpo":
        models["dynamics"] = DynamicsEnsemble(
            ensemble_size=ensemble_size,
            num_actions=num_actions,
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            fc_dim=fc_dim,
        )
    
    # Policy/Value networks based on algorithm
    if algorithm in ["mbpo", "ppo"]:
        models["actor_critic"] = ActorCritic(
            num_actions=num_actions,
            in_channels=in_channels,
            fc_dim=fc_dim,
        )
    elif algorithm == "sac":
        models["policy"] = PolicyNetwork(
            num_actions=num_actions,
            in_channels=in_channels,
            fc_dim=fc_dim,
        )
        models["critic"] = QNetwork(
            num_actions=num_actions,
            in_channels=in_channels,
            fc_dim=fc_dim,
        )
        models["critic_target"] = QNetwork(
            num_actions=num_actions,
            in_channels=in_channels,
            fc_dim=fc_dim,
        )
    elif algorithm == "ddqn":
        models["q_network"] = QNetwork(
            num_actions=num_actions,
            in_channels=in_channels,
            fc_dim=fc_dim,
            dueling=True,
        )
        models["target_network"] = QNetwork(
            num_actions=num_actions,
            in_channels=in_channels,
            fc_dim=fc_dim,
            dueling=True,
        )
    
    # Move all models to device
    for name, model in models.items():
        models[name] = model.to(device)
    
    return models


if __name__ == "__main__":
    # Test model shapes
    print("=" * 60)
    print("Model Shape Verification")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = 4
    
    # Create dummy input
    dummy_obs = torch.randn(batch_size, 4, 84, 84).to(device)
    dummy_action = torch.randint(0, 4, (batch_size,)).to(device)
    
    # Test NatureCNN
    print("\n1. NatureCNN Encoder:")
    encoder = NatureCNN().to(device)
    features = encoder(dummy_obs)
    print(f"   Input:  {dummy_obs.shape}")
    print(f"   Output: {features.shape}")
    print(f"   Expected: (4, 512)")
    
    # Test PolicyNetwork
    print("\n2. PolicyNetwork:")
    policy = PolicyNetwork().to(device)
    action, log_prob, entropy = policy.get_action(dummy_obs)
    print(f"   Action shape: {action.shape}")
    print(f"   Log prob shape: {log_prob.shape}")
    
    # Test DynamicsEnsemble
    print("\n3. DynamicsEnsemble:")
    dynamics = DynamicsEnsemble(ensemble_size=5).to(device)
    next_state, reward, uncertainty = dynamics.predict_next_state(dummy_obs, dummy_action)
    print(f"   Input state:  {dummy_obs.shape}")
    print(f"   Input action: {dummy_action.shape}")
    print(f"   Next state:   {next_state.shape}")
    print(f"   Reward:       {reward.shape if reward is not None else None}")
    print(f"   Uncertainty:  {uncertainty.shape if uncertainty is not None else None}")
    
    # Test ActorCritic
    print("\n4. ActorCritic:")
    ac = ActorCritic().to(device)
    action, log_prob, entropy, value = ac.get_action_and_value(dummy_obs)
    print(f"   Action: {action.shape}")
    print(f"   Value:  {value.shape}")
    
    # Count parameters
    print("\n5. Parameter Counts:")
    for name, model in [("Dynamics Ensemble", dynamics), ("Actor-Critic", ac)]:
        params = sum(p.numel() for p in model.parameters())
        print(f"   {name}: {params:,} parameters")
    
    print("\n" + "=" * 60)
    print("All models verified!")
    print("=" * 60)
