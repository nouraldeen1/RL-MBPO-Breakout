# Common Issues and Solutions

## Installation Issues

### Issue: "No module named 'ale_py'"
**Solution**:
```bash
pip install ale-py
pip install autorom[accept-rom-license]
```

### Issue: "No module named 'torch'"
**Solution**:
```bash
pip install torch
# Or for GPU support:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Issue: "gymnasium[atari] not installing"
**Solution**:
```bash
pip install gymnasium
pip install "gymnasium[atari]"
pip install autorom[accept-rom-license]
```

## Memory Issues

### Issue: "Unable to allocate X GiB for array"
**Problem**: Buffer size too large for available RAM

**Solution 1** - Reduce buffer size in config:
```yaml
training:
  buffer_size: 25_000  # Reduce from 50_000
```

**Solution 2** - Use debug config:
```bash
python src/main.py --config config/config_debug.yaml --no-wandb
```

**Solution 3** - Reduce ensemble size:
```yaml
model:
  ensemble_size: 3  # Instead of 5
```

### Issue: "CUDA out of memory"
**Problem**: GPU memory exhausted

**Solution 1** - Use CPU:
```bash
python src/main.py --device cpu
```

**Solution 2** - Reduce batch size:
```yaml
training:
  batch_size: 128  # Or even 64
  model_rollout_batch_size: 200
```

**Solution 3** - Reduce model size:
```yaml
model:
  fc_dim: 256  # Instead of 512
  dynamics_hidden_dim: 128  # Instead of 256
```

## Training Issues

### Issue: Reward stuck at 0
**Symptoms**: Agent not learning, reward stays at 0 for thousands of episodes

**Diagnosis**:
```bash
# Check action distribution
# Should see prints like: [ActionDebug] Step 5000: {'action/0': 1234, 'action/1': 2345, ...}
```

**Possible Causes & Solutions**:

1. **Agent only chooses NOOP (action 0)**
   - Check entropy: should be > 0.1
   - Increase `ent_coef` in config:
     ```yaml
     ppo:
       ent_coef: 0.1  # Increase from 0.01
     ```

2. **Training hasn't started yet**
   - Check current step vs `learning_starts`
   - Wait for step > 10,000

3. **Policy gradient too small**
   - Increase policy learning rate:
     ```yaml
     training:
       policy_lr: 1e-3  # Increase from 3e-4
     ```

### Issue: Loss is NaN or Inf
**Problem**: Numerical instability in training

**Solution 1** - Check gradient norm:
```yaml
training:
  grad_norm_skip_threshold: 5.0  # Reduce from 10.0
ppo:
  max_grad_norm: 0.3  # Reduce from 0.5
```

**Solution 2** - Reduce learning rates:
```yaml
training:
  policy_lr: 1e-4  # Reduce from 3e-4
  dynamics_lr: 3e-4  # Reduce from 1e-3
```

**Solution 3** - Check for exploding logits:
- Monitor `policy/logits_max_abs` in Wandb
- Should be < 20, if higher, reduce learning rate

### Issue: Dynamics loss not decreasing
**Problem**: Dynamics model not learning

**Diagnosis**:
- Check `dynamics/delta_loss` in Wandb
- Should decrease from ~0.1 to ~0.01

**Solutions**:

1. **Not enough data yet**
   ```python
   # Wait until real_buffer has enough samples
   # Need at least batch_size * 10 samples
   ```

2. **Learning rate too low**
   ```yaml
   training:
     dynamics_lr: 3e-3  # Increase from 1e-3
   ```

3. **Observations not normalized**
   - Check that observations are in [0, 1]
   - Verify WarpFrame and ScaledFloatFrame are applied

### Issue: Policy loss increasing
**Problem**: Policy getting worse over time

**Solutions**:

1. **Too much model data (model bias)**
   ```yaml
   training:
     real_ratio: 0.1  # Increase from 0.05
   ```

2. **Learning rate too high**
   ```yaml
   training:
     policy_lr: 1e-4  # Reduce from 3e-4
   ```

3. **Need more updates per step**
   ```yaml
   training:
     gradient_steps: 2  # Increase from 1
   ```

### Issue: Training very slow
**Problem**: Taking too long per step

**Solutions**:

1. **Use GPU**
   ```bash
   python src/main.py --device cuda
   ```

2. **Reduce parallel environments**
   ```yaml
   env:
     num_envs: 2  # Reduce from 4
   ```

3. **Reduce model rollouts**
   ```yaml
   training:
     model_rollout_freq: 500  # Increase from 250
     model_rollout_batch_size: 200  # Reduce from 400
   ```

4. **Use debug config**
   ```bash
   python src/main.py --config config/config_debug.yaml
   ```

## Wandb Issues

### Issue: "wandb: ERROR Error uploading"
**Solution**:
```bash
# Check internet connection
# Or disable wandb:
python src/main.py --no-wandb
```

### Issue: "wandb: ERROR API key not configured"
**Solution**:
```bash
wandb login
# Enter your API key from wandb.ai/authorize
```

### Issue: Videos not appearing in Wandb
**Problem**: Videos recorded but not logged

**Solution**:
- Videos are logged at save_freq (50K steps)
- Check `videos/` directory manually
- Ensure wandb is enabled in config

## Environment Issues

### Issue: "No module named 'atari_py'"
**Note**: `atari_py` is deprecated, use `ale_py`

**Solution**:
```bash
pip uninstall atari_py
pip install ale-py
pip install autorom[accept-rom-license]
```

### Issue: "ROM file not found"
**Solution**:
```bash
pip install autorom[accept-rom-license]
# Or manually:
AutoROM --accept-license
```

### Issue: Environment won't reset
**Error**: `AttributeError: 'tuple' object has no attribute 'shape'`

**Problem**: Gymnasium API changed

**Solution**: Already fixed in code, but if you encounter:
```python
# Old API
obs = env.reset()

# New API (already implemented)
obs, info = env.reset()
```

## Code Issues

### Issue: "ImportError: cannot import name 'FrameStack'"
**Solution**: Multiple fallback imports already implemented in code. If persists:
```bash
pip install --upgrade gymnasium
```

### Issue: Agent not storing transitions
**Diagnosis**:
```python
# Check buffer size
print(f"Real buffer size: {len(agent.real_buffer)}")
print(f"Model buffer size: {len(agent.model_buffer)}")
```

**Solution**: Verify `store_transition` is called in training loop

### Issue: "Model rollouts returning 0 transitions"
**Cause**: Dynamics not trained yet

**Solution**: Wait until step > learning_starts (10K)

## Performance Issues

### Issue: Reward plateauing early
**Symptoms**: Reward stuck at 20-30, not improving

**Solutions**:

1. **Increase exploration**
   ```yaml
   ppo:
     ent_coef: 0.05  # Increase from 0.01
   ```

2. **Train longer**
   ```yaml
   training:
     total_timesteps: 3_000_000  # Increase from 2M
   ```

3. **Better model rollouts**
   ```yaml
   training:
     model_rollout_length: 2  # Increase from 1
   ```

### Issue: High variance in episode rewards
**Symptoms**: Reward jumps between 0 and 100 randomly

**Solutions**:

1. **Larger batch size**
   ```yaml
   training:
     batch_size: 512  # Increase from 256
   ```

2. **More stable updates**
   ```yaml
   ppo:
     clip_range: 0.1  # Reduce from 0.2
   ```

## Debugging Tips

### Enable Verbose Logging
```yaml
debug:
  action_debug_steps: 100000  # Log for longer
  action_log_freq: 1000  # Log more frequently
```

### Monitor Key Metrics
Watch these in Wandb:
- `episode/reward` - Should increase
- `policy/entropy` - Should be > 0
- `policy/grad_norm` - Should be < 1.0
- `dynamics/delta_loss` - Should decrease
- `buffer/real_size` - Should grow to buffer_size
- `buffer/model_size` - Should grow continuously

### Print Debug Info
Add to training loop:
```python
if global_step % 1000 == 0:
    print(f"Real buffer: {len(agent.real_buffer)}")
    print(f"Model buffer: {len(agent.model_buffer)}")
    print(f"Dynamics trained: {agent.dynamics_trained}")
```

### Test Components Individually
```python
# Test environment
python src/env_factory.py

# Test models
python -c "from src.models import *; print('Models OK')"

# Test agent
python test_setup.py
```

## Quick Fixes Checklist

When something goes wrong, try these in order:

1. **Check config values**
   - real_ratio: 0.05 ✓
   - policy_lr: 3e-4 ✓
   - batch_size: 256 ✓
   - ensemble_size: 5 ✓

2. **Verify environment**
   ```bash
   python src/env_factory.py
   ```

3. **Test setup**
   ```bash
   python test_setup.py
   ```

4. **Use debug config**
   ```bash
   python src/main.py --config config/config_debug.yaml --no-wandb
   ```

5. **Check logs**
   - Action distribution diverse?
   - Entropy > 0?
   - Losses decreasing?

6. **Reduce resource usage**
   - Smaller batch size
   - Fewer parallel envs
   - Smaller networks

## Still Having Issues?

### Check These Resources:
1. [FIXES_AND_IMPROVEMENTS.md](FIXES_AND_IMPROVEMENTS.md) - Detailed explanations
2. [QUICK_START.md](QUICK_START.md) - Usage guide
3. [SUMMARY.md](SUMMARY.md) - Overview and configuration
4. [ASSIGNMENT_CHECKLIST.md](ASSIGNMENT_CHECKLIST.md) - Progress tracking

### Verify Your Setup:
```bash
# 1. Test components
python test_setup.py

# 2. Check config
python -c "from src.utils import load_config; print(load_config('config/config.yaml'))"

# 3. Quick run
python src/main.py --config config/config_debug.yaml --no-wandb
```

### Common Misunderstandings:

**"real_ratio should be high (like 0.8)"**
- ❌ Wrong: This defeats the purpose of MBPO
- ✓ Correct: Use 0.05 (5% real, 95% model)

**"Dynamics loss isn't decreasing - something's wrong"**
- ❌ Wrong: Dynamics should converge to ~0.01
- ✓ Correct: If stuck at 0.1+, increase dynamics_lr

**"Agent stuck on NOOP forever"**
- ❌ Wrong: This is entropy collapse
- ✓ Correct: Increase ent_coef to 0.1 or higher

**"Training taking forever"**
- ❌ Wrong: Waiting for perfection
- ✓ Correct: 2M steps ≈ 8-12 hours is normal

## Emergency Reset

If everything is broken, start fresh:

```bash
# 1. Clean install
rm -rf venv/
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 2. Install dependencies
pip install torch numpy gymnasium ale-py opencv-python pyyaml wandb tqdm matplotlib
pip install autorom[accept-rom-license]

# 3. Test
python test_setup.py

# 4. Run
python src/main.py --config config/config_debug.yaml --no-wandb
```

---

**If you've tried everything and still stuck:**
1. Check all config values match [SUMMARY.md](SUMMARY.md)
2. Run `test_setup.py` - all tests should pass
3. Try debug config first
4. Monitor action distribution and entropy
5. Be patient - good results take 500K+ steps
