# MBPO-SpaceInvaders

> **Model-Based Policy Optimization for SpaceInvadersNoFrameskip-v4**  
> CMPS458 - Reinforcement Learning Project

A modular, production-ready MBRL framework that implements MBPO with support for PPO, SAC, and DDQN benchmarks.

## 📁 Project Structure

```
RL-MBPO-SpaceInvaders/
├── config/
│   ├── config.yaml           # Main configuration file
│   └── sweep_config.yaml     # Wandb Bayesian optimization sweep
├── src/
│   ├── __init__.py          # Package exports
│   ├── env_factory.py       # Environment with Nature DQN wrappers
│   ├── models.py            # Neural networks (NatureCNN, DynamicsEnsemble)
│   ├── agents.py            # MBPO and DDQN agent implementations
│   ├── utils.py             # Replay buffers, early stopping, logging
│   ├── main.py              # Training script with Wandb integration
│   └── eval.py              # Evaluation with video logging
├── checkpoints/             # Saved model weights
├── videos/                  # Recorded episode videos
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
cd RL-MBPO-SpaceInvaders

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Accept Atari ROMs license
pip install gymnasium[accept-rom-license]
```

### 2. Verify Environment Setup

```bash
cd src
python env_factory.py
```

Expected output:

```
============================================================
Environment Verification
============================================================

Environment: SpaceInvadersNoFrameskip-v4
Observation Shape: (4, 84, 84)
  Expected: (4, 84, 84) - 4 stacked grayscale frames
  Status: ✓ CORRECT

Action Space: Discrete(4)
  Number of actions: 4
  Expected: Discrete(4)
  Status: ✓ CORRECT

Action Meanings:
  0: NOOP
  1: FIRE
  2: RIGHT
  3: LEFT
```

### 3. Start Training

```bash
# Standard training
python src/main.py

# With custom config
python src/main.py --config config/config.yaml

# Disable Wandb (local training)
python src/main.py --no-wandb
```

### 4. Run Wandb Hyperparameter Sweep

```bash
# Create sweep
wandb sweep config/sweep_config.yaml

# Start agents (run on multiple machines/terminals)
wandb agent <your-sweep-id>
```

### 5. Evaluate Trained Model

```bash
# Evaluate best checkpoint
python src/eval.py --checkpoint checkpoints/best_mbpo.pt --episodes 10

# With Wandb logging
python src/eval.py --checkpoint checkpoints/best_mbpo.pt --wandb
```

## 🧠 Why This Environment Setup?

### Frame Stacking (4 frames)

Without `FrameStack(4)`, your MBPO Dynamics Model cannot see where the ball is moving. It would just see a static image. Stacking frames allows the model to learn the physics of the bounce.

### EpisodicLifeEnv

In Space Invaders and similar Atari games, if the agent loses a life but the "episode" continues, the reward signal gets confusing. This wrapper tells the agent exactly when it failed, leading to higher quality gradients and faster convergence.

### Delta State Prediction

The dynamics model predicts the **change** in pixels (`delta_state = next_state - state`) rather than the whole next image. This is easier for a CNN to learn - it only needs to learn "where the ball moved" instead of redrawing the entire screen.

### Nature DQN Preprocessing Stack

```
NoopResetEnv(30)      → Initial state variety
MaxAndSkipEnv(4)      → Reduce flickering, increase speed
EpisodicLifeEnv       → Clear reward signal on death
FireResetEnv          → Auto-start ball movement
WarpFrame(84, 84)     → Grayscale 84x84 (standard Atari size)
ScaledFloatFrame      → Normalize to [0, 1]
FrameStack(4)         → Velocity sensing for dynamics model
```

## 🏗️ Architecture

### NatureCNN Encoder

```
Conv2d(4, 32, 8x8, stride=4) + ReLU
Conv2d(32, 64, 4x4, stride=2) + ReLU
Conv2d(64, 64, 3x3, stride=1) + ReLU
Flatten
Linear(3136, 512) + ReLU
```

### Dynamics Ensemble

- **5 independent CNNs** (configurable)
- Each predicts `delta_state` and `reward`
- Ensemble provides:
  - Uncertainty estimation (disagreement)
  - Reduced model bias (averaging)
  - Better exploration (epistemic uncertainty)

### MBPO Algorithm

1. Collect real data from environment
2. Train dynamics ensemble on real data
3. Generate synthetic rollouts using learned models
4. Train policy on mixture of real (5%) and synthetic (95%) data

## ⚙️ Configuration

Key parameters in `config/config.yaml`:

```yaml
# Environment
env:
  num_envs: 4 # Parallel environments
  frame_stack: 4 # Frames for velocity sensing

# MBPO
training:
  model_rollout_length: 1 # Horizon for model rollouts
  real_ratio: 0.05 # 5% real, 95% synthetic data

# Model
model:
  ensemble_size: 5 # Number of dynamics models
  fc_dim: 512 # Hidden layer dimension

# Early Stopping
early_stopping:
  target_reward: 400 # Stop training at this reward
  window_size: 100 # Episodes for mean calculation
```

## 📊 Monitoring with Wandb

The training script logs:

- Episode rewards and lengths
- Policy/dynamics losses
- Buffer sizes (real vs model)
- Training videos (every 50 episodes)
- Best model checkpoints

Access your runs at: [wandb.ai](https://wandb.ai)

## 🧪 Verification Checklist

After generating the code:

- [ ] **Verify Observation Shape**: Run `env_factory.py`, should output `(4, 84, 84)`
- [ ] **Check Actions**: Space Invaders should have `Discrete(6)` - NOOP, FIRE, RIGHT, LEFT, RIGHTFIRE, LEFTFIRE
- [ ] **Wandb Test**: Run for 5 minutes to ensure charts and videos appear
- [ ] **Model Shapes**: Run `models.py` to verify all tensor shapes

## 📈 Expected Performance

| Metric                | Target     | Notes                    |
| --------------------- | ---------- | ------------------------ |
| Mean Reward (100 eps) | 400+       | Early stopping threshold |
| Training Time         | ~10M steps | Varies by hardware       |
| Sample Efficiency     | ~5x vs DQN | MBPO benefit             |

## 🛠️ Troubleshooting

### CUDA Out of Memory

- Reduce `batch_size` in config
- Reduce `num_envs`
- Reduce `model_rollout_batch_size`

### Import Errors

```bash
cd RL-MBPO-SpaceInvaders
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### Atari ROM Issues

```bash
pip install gymnasium[accept-rom-license]
# or manually download ROMs
```

## 📚 References

- [MBPO Paper (Janner et al., 2019)](https://arxiv.org/abs/1906.08253)
- [Nature DQN (Mnih et al., 2015)](https://www.nature.com/articles/nature14236)
- [Gymnasium Documentation](https://gymnasium.farama.org/)

## 📄 License

MIT License - See LICENSE file for details.

---

**CMPS458 - Reinforcement Learning Project**
