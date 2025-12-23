"""
Simple test script to verify core functionality
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

print("=" * 70)
print("Testing MBPO-Breakout Components")
print("=" * 70)

# Test 1: Import modules
print("\n[1/5] Testing imports...")
try:
    import torch
    import numpy as np
    import gymnasium as gym
    from env_factory import make_env
    from models import NatureCNN, DynamicsEnsemble, ActorCritic
    from agents import MBPOAgent
    from utils import ReplayBuffer, load_config
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Check device
print("\n[2/5] Checking device...")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✓ Using device: {device}")

# Test 3: Load config
print("\n[3/5] Loading configuration...")
try:
    config = load_config("config/config.yaml")
    print(f"✓ Config loaded: {config.get('algorithm', 'unknown').upper()}")
    print(f"  - Ensemble size: {config['model']['ensemble_size']}")
    print(f"  - Real ratio: {config['training']['real_ratio']}")
    print(f"  - Batch size: {config['training']['batch_size']}")
except Exception as e:
    print(f"✗ Config failed: {e}")
    sys.exit(1)

# Test 4: Create environment
print("\n[4/5] Creating environment...")
try:
    env_fn = make_env(seed=42, capture_video=False)
    env = env_fn()
    obs, info = env.reset()
    print(f"✓ Environment created")
    print(f"  - Observation shape: {obs.shape}")
    print(f"  - Actions: {env.action_space.n}")
    env.close()
except Exception as e:
    print(f"✗ Environment failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Create agent
print("\n[5/5] Creating MBPO agent...")
try:
    agent = MBPOAgent(
        obs_shape=(4, 84, 84),
        num_actions=4,
        config=config,
        device=device,
    )
    print(f"✓ Agent created successfully")
    
    # Test action selection
    test_obs = np.random.rand(4, 84, 84).astype(np.float32)
    action, info = agent.get_action(test_obs, deterministic=False)
    print(f"  - Test action: {action}")
    print(f"  - Action valid: {0 <= action < 4}")
except Exception as e:
    print(f"✗ Agent creation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("ALL TESTS PASSED! ✓")
print("=" * 70)
print("\nNext steps:")
print("1. Quick test: python src/main.py --config config/config_debug.yaml --no-wandb")
print("2. Full training: python src/main.py")
print("3. See QUICK_START.md for more details")
