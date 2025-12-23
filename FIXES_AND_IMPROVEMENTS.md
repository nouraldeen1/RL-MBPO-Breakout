# MBPO-Breakout Fixes and Improvements

## Summary of Changes

This document outlines all the fixes applied to make the MBPO implementation work correctly with the Breakout environment.

## Key Issues Fixed

### 1. Configuration Parameters

**Problem**: The original configuration had several critical issues:
- `real_ratio: 0.8` - This meant 80% real data, only 20% model data (opposite of MBPO)
- `policy_lr: 3e-5` - Too conservative for Atari environments
- `learning_starts: 20_000` - Training started too late
- `ensemble_size: 3` - Too small for reliable uncertainty estimation
- Missing `seed` parameter

**Solution**: Updated [config/config.yaml](config/config.yaml):
```yaml
seed: 42 # Added for reproducibility
training:
  total_timesteps: 2_000_000 # Increased from 1M
  buffer_size: 200_000 # Increased from 100K
  batch_size: 256 # Increased from 64 for stability
  learning_starts: 10_000 # Reduced from 20K
  policy_lr: 3e-4 # Increased from 3e-5
  dynamics_lr: 1e-3 # Increased from 1e-4
  model_rollout_freq: 250 # Reduced from 1000
  model_rollout_batch_size: 400 # Increased from 256
  real_ratio: 0.05 # Fixed: 5% real, 95% model (MBPO paper)
  
model:
  fc_dim: 512 # Increased from 256
  ensemble_size: 5 # Increased from 3
  dynamics_hidden_dim: 256 # Increased from 128
```

### 2. MBPO Algorithm Understanding

**The MBPO Paper Strategy:**
- Use **5% real data** and **95% model-generated data** for policy training
- This is counter-intuitive but correct - the model data provides massive sample efficiency
- The ensemble provides uncertainty to prevent policy exploitation of model errors

**Why this works:**
1. Real environment interactions are expensive
2. Model can generate unlimited synthetic transitions
3. Ensemble disagreement catches model errors
4. Policy learns faster with more (synthetic) data

### 3. Training Schedule

**Problem**: 
- Model rollouts happened too infrequently (every 1000 steps)
- Training started too late (20K steps)
- Batch sizes were too small for stable learning

**Solution**:
- Generate model rollouts every 250 steps
- Start training at 10K steps (enough to train initial dynamics model)
- Use batch size 256 for stable gradients
- More frequent checkpoints (every 50K steps)

### 4. Model Architecture

**Improvements**:
- Increased CNN feature dimension to 512 (from 256)
- Increased dynamics hidden dimension to 256 (from 128)
- Ensemble size 5 (MBPO paper uses 5-7)
- Better initialization and gradient clipping

### 5. Debug Configuration

Created [config/config_debug.yaml](config/config_debug.yaml) for quick testing:
- Only 50K timesteps for rapid iteration
- 2 parallel environments
- Smaller networks for speed
- Wandb disabled by default
- More frequent logging and videos

## How MBPO Works (Explained)

### The Core Algorithm

```
1. Initialize replay buffer D_env (real data)
2. Initialize dynamics ensemble M = {M_1, ..., M_5}

For each iteration:
    3. Collect real transitions from environment → D_env
    4. Train dynamics ensemble M on D_env
    5. Generate synthetic rollouts using M → D_model
    6. Train policy on mixture: 5% from D_env + 95% from D_model
    7. Repeat
```

### Why Delta State Prediction?

The dynamics models predict **delta_state** (change in pixels) instead of full next state:

**Advantages:**
- Most pixels don't change between frames
- Model only learns "where the ball moved"
- Reduced variance in predictions
- Easier for CNN to learn

**Implementation:**
```python
delta_state = model(state, action)
next_state = state + delta_state
next_state = clamp(next_state, 0, 1)
```

### Ensemble Uncertainty

The ensemble provides epistemic uncertainty through disagreement:

```python
# Get predictions from all 5 models
predictions = [model_i(state, action) for model_i in ensemble]

# Mean prediction
mean_pred = mean(predictions)

# Uncertainty (std across models)
uncertainty = std(predictions)
```

**Usage:**
- High uncertainty → Model unsure → Use more real data
- Low uncertainty → Model confident → Safe to use model data

## Training Guide

### Quick Test (5-10 minutes)

```bash
# Use debug config for rapid testing
python src/main.py --config config/config_debug.yaml --no-wandb
```

This runs for 50K steps with:
- Smaller networks
- Fewer parallel envs
- More frequent logging
- No Wandb overhead

### Full Training (8-12 hours on GPU)

```bash
# Ensure you have wandb setup
pip install wandb
wandb login

# Run full training
python src/main.py --config config/config.yaml
```

This runs for 2M steps with:
- Full network capacity
- 4 parallel environments
- Wandb logging and videos
- Automatic checkpointing

### Hyperparameter Sweep

```bash
# Create sweep
wandb sweep config/sweep_config.yaml

# Run agents (can run multiple in parallel)
wandb agent <sweep-id>
```

## Expected Results

### Timeline

| Steps | Expected Behavior |
|-------|------------------|
| 0-10K | Random exploration, reward ≈ 0-5 |
| 10K-50K | Agent starts hitting bricks, reward ≈ 5-20 |
| 50K-200K | Consistent brick breaking, reward ≈ 20-50 |
| 200K-500K | Better ball control, reward ≈ 50-150 |
| 500K-1M | Strategic play, reward ≈ 150-300 |
| 1M-2M | Expert behavior, reward ≈ 300-400+ |

### Key Metrics to Watch

1. **Episode Reward**: Should steadily increase
2. **Policy Entropy**: Should decrease but stay > 0 (exploration)
3. **Dynamics Loss**: Should decrease and plateau
4. **Buffer Sizes**: real_buffer ≈ 200K, model_buffer grows continuously
5. **Grad Norm**: Should stay < 1.0 (if higher, training unstable)

### Troubleshooting

**Problem: Reward stuck at 0**
- Check action distribution (should be diverse, not all NOOP)
- Verify entropy > 0.1 (agent is exploring)
- Check if training has started (step > learning_starts)

**Problem: Policy loss exploding**
- Reduce policy_lr to 1e-4
- Check grad_norm (should be < 1.0)
- Increase grad_norm_skip_threshold

**Problem: Dynamics loss not decreasing**
- Check real_buffer has enough data
- Increase dynamics_lr to 3e-3
- Verify observations are normalized [0, 1]

**Problem: Out of memory**
- Reduce batch_size to 128
- Reduce ensemble_size to 3
- Reduce model_rollout_batch_size to 200
- Use config_debug.yaml

## Verification Steps

1. **Verify Setup**:
```bash
python verify_setup.py
```

2. **Check Environment**:
```bash
cd src
python env_factory.py
```

3. **Test Models**:
```bash
python -c "from src.models import *; from src.agents import *; print('Import successful!')"
```

4. **Run Short Test**:
```bash
python src/main.py --config config/config_debug.yaml --no-wandb
```

## Code Structure

### Key Components

1. **[src/env_factory.py](src/env_factory.py)**: Environment with Nature DQN wrappers
   - FrameStack(4): Velocity sensing
   - EpisodicLifeEnv: Clear failure signals
   - WarpFrame, ScaledFloatFrame: Preprocessing

2. **[src/models.py](src/models.py)**: Neural networks
   - `NatureCNN`: Convolutional encoder
   - `DynamicsEnsemble`: 5 dynamics models with uncertainty
   - `ActorCritic`: Policy and value networks

3. **[src/agents.py](src/agents.py)**: MBPO implementation
   - `MBPOAgent`: Full MBPO algorithm
   - `DDQNAgent`: Baseline for comparison

4. **[src/utils.py](src/utils.py)**: Utilities
   - `ReplayBuffer`: Real environment data
   - `ModelReplayBuffer`: Synthetic data
   - `MixedReplayBuffer`: 5% real + 95% model sampling

5. **[src/main.py](src/main.py)**: Training loop
   - Environment interaction
   - Agent updates
   - Logging and checkpointing
   - Wandb integration

6. **[src/eval.py](src/eval.py)**: Evaluation
   - Load trained models
   - Run deterministic episodes
   - Record videos
   - Compute statistics

## Model-Free Baselines (Bonus)

For the assignment bonus (comparing with model-free methods), the code already supports:

### DDQN (Included)
```bash
# Edit config.yaml: algorithm: "ddqn"
python src/main.py
```

### PPO (Modify agent to use PPO only)
```bash
# Edit config.yaml: algorithm: "ppo"
# Modify agents.py to create PPOAgent using ActorCritic directly
python src/main.py
```

### SAC (Would require implementation)
- SAC is typically for continuous actions
- Breakout has discrete actions
- Consider using Discrete SAC or stick with DDQN/PPO

## Publishing to Hugging Face

After training completes:

```python
# Install huggingface_hub
pip install huggingface_hub

# Login
huggingface-cli login

# Upload model (in Python script)
from huggingface_hub import HfApi
api = HfApi()

api.upload_file(
    path_or_fileobj="checkpoints/best_mbpo.pt",
    path_in_repo="model.pt",
    repo_id="your-username/mbpo-breakout",
    repo_type="model",
)

# Upload video
api.upload_file(
    path_or_fileobj="videos/best_episode.mp4",
    path_in_repo="gameplay.mp4",
    repo_id="your-username/mbpo-breakout",
    repo_type="model",
)
```

## Paper Writing Tips

### Abstract (1 paragraph)
- Background: Model-based RL for sample efficiency
- Goal: Implement MBPO on Atari Breakout
- Method: Dynamics ensemble with delta prediction
- Results: Achieved X reward after Y episodes
- Conclusion: MBPO is Z% more sample efficient than DDQN

### Introduction (4 sentences)
1. RL requires many environment interactions
2. Model-based methods learn environment dynamics for efficiency
3. MBPO combines learned models with model-free optimization
4. We implement MBPO on Breakout and compare with baselines

### Methods
- Environment setup (wrappers, preprocessing)
- Network architectures (NatureCNN, ensemble)
- MBPO algorithm (pseudo-code)
- Hyperparameters (table)

### Experiments
- Training curves (reward vs episodes)
- Sample efficiency (reward vs environment interactions)
- Ablation studies (ensemble size, real_ratio)
- Comparison with DDQN/PPO

### Results
- Learning curve plots
- Final performance statistics (mean ± std)
- Video frames showing learned behavior
- Wandb report links

### Discussion
- MBPO advantages (sample efficiency)
- MBPO challenges (model bias, memory)
- Future work (longer rollouts, better models)

## References

- MBPO Paper: https://arxiv.org/abs/1906.08253
- Nature DQN: https://www.nature.com/articles/nature14236
- Gymnasium: https://gymnasium.farama.org/
- ALE: https://ale.farama.org/

## Contact

For questions or issues, refer to the original assignment or instructor.
