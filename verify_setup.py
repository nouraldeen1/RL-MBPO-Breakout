"""
Quick Verification Script
=========================

Run this script to verify the entire MBPO-Breakout setup.
It performs the Phase 2 verification steps mentioned in the requirements.

Usage:
    python verify_setup.py

Author: CMPS458 RL Project
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def main():
    print("=" * 70)
    print("MBPO-Breakout Setup Verification")
    print("=" * 70)
    
    # Step 1: Verify environment shapes
    print("\n[1/4] Verifying Environment Setup...")
    print("-" * 50)
    
    try:
        from env_factory import verify_env_setup, make_vec_env
        specs = verify_env_setup()
        
        # Check shapes
        assert specs["obs_shape"] == (4, 84, 84), f"Wrong obs shape: {specs['obs_shape']}"
        assert specs["num_actions"] == 4, f"Wrong num actions: {specs['num_actions']}"
        print("✓ Environment verification PASSED")
    except Exception as e:
        print(f"✗ Environment verification FAILED: {e}")
        return False
    
    # Step 2: Verify model shapes
    print("\n[2/4] Verifying Model Shapes...")
    print("-" * 50)
    
    try:
        import torch
        from models import NatureCNN, DynamicsEnsemble, ActorCritic
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        batch_size = 4
        
        # Test input
        dummy_obs = torch.randn(batch_size, 4, 84, 84).to(device)
        dummy_action = torch.randint(0, 4, (batch_size,)).to(device)
        
        # NatureCNN
        encoder = NatureCNN().to(device)
        features = encoder(dummy_obs)
        assert features.shape == (batch_size, 512), f"CNN output wrong: {features.shape}"
        print(f"  ✓ NatureCNN: (4, 4, 84, 84) → {features.shape}")
        
        # DynamicsEnsemble
        dynamics = DynamicsEnsemble(ensemble_size=5).to(device)
        next_state, reward, uncertainty = dynamics.predict_next_state(dummy_obs, dummy_action)
        assert next_state.shape == (batch_size, 4, 84, 84), f"Dynamics output wrong: {next_state.shape}"
        print(f"  ✓ DynamicsEnsemble: predicts next_state {next_state.shape}")
        
        # ActorCritic
        ac = ActorCritic().to(device)
        action, log_prob, entropy, value = ac.get_action_and_value(dummy_obs)
        assert action.shape == (batch_size,), f"Action shape wrong: {action.shape}"
        print(f"  ✓ ActorCritic: outputs action {action.shape}, value {value.shape}")
        
        print("✓ Model verification PASSED")
    except Exception as e:
        print(f"✗ Model verification FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 3: Verify agent creation
    print("\n[3/4] Verifying Agent Creation...")
    print("-" * 50)
    
    try:
        from agents import create_agent
        from utils import load_config
        
        config = load_config("config/config.yaml")
        
        # Create MBPO agent
        agent = create_agent(
            algorithm="mbpo",
            obs_shape=(4, 84, 84),
            num_actions=4,
            config=config,
            device=device,
        )
        
        # Test action selection
        import numpy as np
        test_state = np.random.rand(4, 84, 84).astype(np.float32)
        action, info = agent.get_action(test_state, deterministic=False)
        
        assert 0 <= action < 4, f"Invalid action: {action}"
        print(f"  ✓ MBPO Agent created successfully")
        print(f"  ✓ Action selection works: action={action}")
        
        print("✓ Agent verification PASSED")
    except Exception as e:
        print(f"✗ Agent verification FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 4: Verify config loading
    print("\n[4/4] Verifying Configuration...")
    print("-" * 50)
    
    try:
        from utils import load_config
        
        config = load_config("config/config.yaml")
        
        assert "env" in config, "Missing 'env' section"
        assert "training" in config, "Missing 'training' section"
        assert "model" in config, "Missing 'model' section"
        assert "early_stopping" in config, "Missing 'early_stopping' section"
        assert "wandb" in config, "Missing 'wandb' section"
        
        print(f"  ✓ Environment: {config['env'].get('name', 'N/A')}")
        print(f"  ✓ Algorithm: {config.get('algorithm', 'N/A')}")
        print(f"  ✓ Ensemble size: {config['model'].get('ensemble_size', 'N/A')}")
        print(f"  ✓ Target reward: {config['early_stopping'].get('target_reward', 'N/A')}")
        
        print("✓ Configuration verification PASSED")
    except Exception as e:
        print(f"✗ Configuration verification FAILED: {e}")
        return False
    
    # Summary
    print("\n" + "=" * 70)
    print("ALL VERIFICATIONS PASSED!")
    print("=" * 70)
    print("""
Next Steps:
-----------
1. Install Wandb and login:
   $ pip install wandb
   $ wandb login

2. Run a quick test (5 minutes):
   $ python src/main.py --no-wandb

3. Start full training with Wandb:
   $ python src/main.py

4. Or run a hyperparameter sweep:
   $ wandb sweep config/sweep_config.yaml
   $ wandb agent <sweep-id>
""")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
