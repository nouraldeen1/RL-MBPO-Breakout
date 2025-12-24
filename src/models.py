"""
Neural Network Models
=====================
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# =============================================================================
# World Models Components
# =============================================================================

class VAE(nn.Module):
    def __init__(self, in_channels=4, z_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(256 * 5 * 5, z_dim)
        self.fc_logvar = nn.Linear(256 * 5 * 5, z_dim)
        self.fc_decode = nn.Linear(z_dim, 256 * 5 * 5)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, output_padding=1), nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose2d(32, in_channels, 4, stride=2, padding=1), nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        h = self.fc_decode(z).view(-1, 256, 5, 5)
        return self.decoder(h), mu, logvar

    def compute_loss(self, x):
        recon, mu, logvar = self.forward(x)
        recon_loss = F.mse_loss(recon, x, reduction='sum') / x.size(0)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        return recon_loss + kl_loss, {"vae/recon": recon_loss.item(), "vae/kl": kl_loss.item()}

class MDNRNN(nn.Module):
    def __init__(self, z_dim=32, action_dim=4, hidden_dim=256, n_gaussians=5):
        super().__init__()
        self.z_dim = z_dim
        self.n_gaussians = n_gaussians
        self.action_embed = nn.Embedding(action_dim, z_dim)
        self.lstm = nn.LSTM(z_dim + z_dim, hidden_dim, batch_first=True)
        self.pi = nn.Linear(hidden_dim, n_gaussians)
        self.mu = nn.Linear(hidden_dim, n_gaussians * z_dim)
        self.sigma = nn.Linear(hidden_dim, n_gaussians * z_dim)
        self.reward = nn.Linear(hidden_dim, 1)
        self.done = nn.Linear(hidden_dim, 1)

    def forward(self, z, action, hidden=None):
        a_emb = self.action_embed(action)
        is_seq = z.dim() == 3
        if not is_seq:
            z = z.unsqueeze(1)
            a_emb = a_emb.unsqueeze(1)
        
        out, hidden = self.lstm(torch.cat([z, a_emb], dim=-1), hidden)
        if not is_seq: out = out.squeeze(1)
        
        # --- STABILITY FIX START ---
        pi = F.softmax(self.pi(out), dim=-1)
        mu = self.mu(out).view(*out.shape[:-1], self.n_gaussians, self.z_dim)
        
        # Clamp log_sigma to prevent exp() from exploding or vanishing
        log_sigma = self.sigma(out).view(*out.shape[:-1], self.n_gaussians, self.z_dim)
        log_sigma = torch.clamp(log_sigma, min=-5, max=2) # bounds sigma between ~0.006 and ~7.4
        sigma = torch.exp(log_sigma)
        # --- STABILITY FIX END ---
        
        return pi, mu, sigma, self.reward(out), torch.sigmoid(self.done(out)), hidden

    def sample_next_latent(self, z, action, hidden=None):
        pi, mu, sigma, r, d, hidden = self.forward(z, action, hidden)
        idx = Categorical(pi).sample().unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.z_dim)
        mu_sel = torch.gather(mu, 1, idx).squeeze(1)
        sigma_sel = torch.gather(sigma, 1, idx).squeeze(1)
        z_next = mu_sel + sigma_sel * torch.randn_like(mu_sel)
        return z_next, r, d, hidden

class WorldModelsController(nn.Module):
    def __init__(self, z_dim=32, hidden_dim=256, action_dim=4):
        super().__init__()
        self.net = nn.Linear(z_dim + hidden_dim, action_dim)
    def get_action(self, z, h, deterministic=False):
        logits = self.net(torch.cat([z, h], dim=-1))
        dist = Categorical(probs=F.softmax(logits, dim=-1))
        action = dist.mode() if deterministic else dist.sample()
        return action, dist.log_prob(action)

# =============================================================================
# Standard/MBPO Components (Kept for compatibility)
# =============================================================================

class NatureCNN(nn.Module):
    def __init__(self, in_channels=4, fc_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, fc_dim), nn.ReLU()
        )
    def forward(self, x): return self.net(x)

class ActorCritic(nn.Module):
    def __init__(self, num_actions, in_channels=4, fc_dim=512):
        super().__init__()
        self.encoder = NatureCNN(in_channels, fc_dim)
        self.actor = nn.Linear(fc_dim, num_actions)
        self.critic = nn.Linear(fc_dim, 1)
    
    def get_action_and_value(self, x, action=None, deterministic=False):
        feats = self.encoder(x)
        logits = self.actor(feats)
        probs = Categorical(logits=logits)
        if action is None: action = probs.mode() if deterministic else probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(feats).squeeze(-1)
    
    def forward(self, x): return self.actor(self.encoder(x)), self.critic(self.encoder(x))

class QNetwork(nn.Module):
    def __init__(self, num_actions, in_channels=4, fc_dim=512, dueling=True):
        super().__init__()
        self.encoder = NatureCNN(in_channels, fc_dim)
        self.fc_val = nn.Linear(fc_dim, 1)
        self.fc_adv = nn.Linear(fc_dim, num_actions)
    def forward(self, x):
        f = self.encoder(x)
        val, adv = self.fc_val(f), self.fc_adv(f)
        return val + adv - adv.mean(1, keepdim=True)

class DynamicsModel(nn.Module):
    """Single dynamics model for Ensemble."""
    def __init__(self, in_channels, num_actions, hidden_dim=256):
        super().__init__()
        self.enc = NatureCNN(in_channels)
        self.act_emb = nn.Embedding(num_actions, 64)
        self.trunk = nn.Sequential(nn.Linear(512+64, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.dec = nn.Sequential(
            nn.Linear(hidden_dim, 32*7*7), nn.ReLU(), nn.Unflatten(1, (32,7,7)),
            nn.ConvTranspose2d(32, 32, 3, 1), nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2), nn.ReLU(),
            nn.ConvTranspose2d(16, in_channels, 8, 4)
        )
        self.reward = nn.Linear(hidden_dim, 1)
    def forward(self, s, a):
        h = self.trunk(torch.cat([self.enc(s), self.act_emb(a)], -1))
        return self.dec(h), self.reward(h)

class DynamicsEnsemble(nn.Module):
    def __init__(self, ensemble_size, num_actions, in_channels):
        super().__init__()
        self.models = nn.ModuleList([DynamicsModel(in_channels, num_actions) for _ in range(ensemble_size)])
    def predict_next_state(self, s, a):
        # Simply use first model for rollouts in this simplified version
        delta, r = self.models[0](s, a)
        return torch.clamp(s + delta, 0, 1), r, None
    def compute_loss(self, s, a, ns, r):
        loss = 0
        target_delta = ns - s
        for m in self.models:
            pd, pr = m(s, a)
            loss += F.mse_loss(pd, target_delta) + F.mse_loss(pr.squeeze(-1), r)
        return loss, {}
    
class ActorCriticPPO(nn.Module):
    """Actor-Critic model for PPO."""
    
    def __init__(self, num_actions, in_channels=4, fc_dim=512):
        super().__init__()
        self.encoder = NatureCNN(in_channels, fc_dim)
        self.actor = nn.Linear(fc_dim, num_actions)
        self.critic = nn.Linear(fc_dim, 1)
    
    def get_action_and_value(self, x, action=None, deterministic=False):
        """
        Get action and value from observation.
        
        Args:
            x: Observation tensor (batch, channels, height, width)
            action: Optional action tensor to evaluate
            deterministic: If True, use argmax; if False, sample from distribution
        
        Returns:
            action, log_prob, entropy, value
        """
        feats = self.encoder(x)
        logits = self.actor(feats)
        probs = Categorical(logits=logits)
        
        if action is None:
            if deterministic:
                # For deterministic, use argmax (not probs.mode())
                action = logits.argmax(dim=-1)
            else:
                # For stochastic, sample from distribution
                action = probs.sample()
        
        return action, probs.log_prob(action), probs.entropy(), self.critic(feats).squeeze(-1)
    
    def forward(self, x):
        """Forward pass returning logits and values."""
        feats = self.encoder(x)
        return self.actor(feats), self.critic(feats)