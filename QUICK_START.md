# Quick Start Guide - MBPO Breakout

## Prerequisites

1. **Install Dependencies**:
```bash
pip install torch numpy gymnasium[atari] gymnasium[accept-rom-license] ale-py opencv-python pyyaml wandb tqdm matplotlib
```

2. **Verify Installation**:
```bash
python verify_setup.py
```

Expected output: All checks should pass ✓

## Quick Test (5 minutes)

Run a short training session to verify everything works:

```bash
python src/main.py --config config/config_debug.yaml --no-wandb
```

You should see:
- Environment creating successfully
- Agent initialized
- Training starting
- Episode rewards being logged
- Action distribution prints

**Expected behavior**: Agent should collect data, train dynamics model after 2K steps, and start policy updates.

## Full Training

### Option 1: Local Training (No Wandb)

```bash
python src/main.py --no-wandb
```

Checkpoints will be saved to `checkpoints/` directory.

### Option 2: With Wandb Logging

```bash
# Setup wandb (first time only)
pip install wandb
wandb login  # Enter your API key

# Run training
python src/main.py
```

Visit wandb.ai to view your training progress in real-time!

### Option 3: Custom Configuration

```bash
python src/main.py --config path/to/your/config.yaml
```

## Monitoring Training

### Console Output

You'll see prints like:
```
Step:      10,240 | Episodes:     25 | Mean Reward (100):    8.50 | Best:   12.30
[ActionDebug] Step 10000: {'action/0': 5234, 'action/1': 1892, ...}
[Training] Step 10240: policy_loss=0.8234, grad_norm=0.4123, entropy=1.2345
```

### Key Metrics

- **Mean Reward**: Average over last 100 episodes (target: 400+)
- **Action Distribution**: Should be diverse (not all one action)
- **Entropy**: Should be > 0 (agent is exploring)
- **Grad Norm**: Should be < 1.0 (stable training)

### Videos

Videos are saved to `videos/` directory:
- Automatically recorded every 100 episodes
- Watch them to see agent's behavior

## Checkpoints

Models are saved at:
- `checkpoints/latest_mbpo.pt` - Every 50K steps
- `checkpoints/best_mbpo.pt` - When best mean reward achieved

## Evaluation

After training, evaluate your model:

```bash
python src/eval.py --checkpoint checkpoints/best_mbpo.pt --episodes 10
```

This runs 10 deterministic episodes and computes statistics.

## Common Issues

### Issue: "Import numpy/torch could not be resolved"
**Solution**: These are just VS Code warnings. Run `pip install numpy torch` to fix.

### Issue: Agent always chooses NOOP (action 0)
**Solution**: 
- Check entropy > 0.1
- Verify training has started (step > learning_starts)
- May need to reduce `ent_coef` in config

### Issue: Out of memory
**Solution**:
- Use `config_debug.yaml` (smaller networks)
- Reduce `batch_size` to 128 or 64
- Reduce `ensemble_size` to 3
- Reduce `model_rollout_batch_size`

### Issue: Dynamics loss not decreasing
**Solution**:
- Check real buffer has enough data (> batch_size)
- Increase `dynamics_lr` to 3e-3
- Verify observations are scaled [0, 1]

### Issue: Training very slow
**Solution**:
- Use GPU: `python src/main.py --device cuda`
- Reduce `num_envs` to 2
- Use `config_debug.yaml`

## Next Steps

1. ✅ Run `verify_setup.py`
2. ✅ Quick test with debug config
3. ✅ Full training with main config
4. ✅ Monitor on Wandb
5. ✅ Evaluate trained model
6. ✅ Record videos for paper
7. ✅ Upload to Hugging Face

## For the Assignment

### Required:
1. ✅ MBPO implementation (Done!)
2. ✅ Train on Breakout (Run main.py)
3. ✅ Record videos (Automatic)
4. ✅ Hyperparameter tuning (Use sweep_config.yaml)
5. ✅ Publish to Hugging Face (See FIXES_AND_IMPROVEMENTS.md)

### Bonus (2 points):
Compare with model-free baselines:

**DDQN** (Already implemented):
```bash
# Edit config.yaml: algorithm: "ddqn"
python src/main.py
```

**PPO** (Use MBPO without dynamics):
```bash
# Edit config.yaml: 
# algorithm: "ppo"
# Comment out model_rollout_freq in training loop
python src/main.py
```

## Paper Structure

1. **Title**: Model-Based Policy Optimization for Atari Breakout
2. **Abstract**: 1 paragraph summary
3. **Introduction**: 4 sentences (problem, approach, results, impact)
4. **Related Work**: MBRL, MBPO, Atari benchmarks
5. **Methods**: Architecture, algorithm, hyperparameters
6. **Experiments**: Training setup, baselines, metrics
7. **Results**: Plots, tables, analysis
8. **Discussion**: Insights, limitations, future work
9. **Conclusion**: 2-3 sentences summary
10. **References**: All papers cited

Include:
- Learning curves (reward vs episodes)
- Hyperparameter table
- Comparison table (MBPO vs DDQN vs PPO)
- Video frames showing gameplay
- Wandb report links

## Wandb Report

Create a report at wandb.ai with:
1. Learning curves
2. Embedded videos
3. Hyperparameter comparisons
4. Model architecture diagrams
5. Key metrics summary

Share the report link in your paper!

## Resources

- **MBPO Paper**: https://arxiv.org/abs/1906.08253
- **Code Fixes**: See FIXES_AND_IMPROVEMENTS.md
- **Full README**: See README.md
- **Gymnasium Docs**: https://gymnasium.farama.org/

## Help

If you encounter issues:
1. Check FIXES_AND_IMPROVEMENTS.md for troubleshooting
2. Verify config parameters match recommended values
3. Test with config_debug.yaml first
4. Check action distribution and entropy logs

Good luck with your project! 🚀
