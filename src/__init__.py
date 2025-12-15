"""
MBPO-Breakout Source Package
============================

Model-Based Policy Optimization for BreakoutNoFrameskip-v4

Modules:
    - env_factory: Environment construction with Nature DQN wrappers
    - models: Neural network architectures (NatureCNN, DynamicsEnsemble)
    - agents: MBPO and DDQN agent implementations
    - utils: Replay buffers, early stopping, logging utilities
    - main: Training script with Wandb integration
    - eval: Evaluation script with video logging

Author: CMPS458 RL Project
"""

from .env_factory import make_env, make_vec_env, make_eval_env, verify_env_setup
from .models import (
    NatureCNN,
    PolicyNetwork,
    ValueNetwork,
    QNetwork,
    DynamicsModel,
    DynamicsEnsemble,
    ActorCritic,
    create_models,
)
from .agents import MBPOAgent, DDQNAgent, create_agent
from .utils import (
    ReplayBuffer,
    ModelReplayBuffer,
    MixedReplayBuffer,
    EarlyStopping,
    load_config,
    save_checkpoint,
    load_checkpoint,
    set_seed,
)

__version__ = "0.1.0"
__author__ = "CMPS458 RL Project"
__all__ = [
    # Environment
    "make_env",
    "make_vec_env",
    "make_eval_env",
    "verify_env_setup",
    # Models
    "NatureCNN",
    "PolicyNetwork",
    "ValueNetwork",
    "QNetwork",
    "DynamicsModel",
    "DynamicsEnsemble",
    "ActorCritic",
    "create_models",
    # Agents
    "MBPOAgent",
    "DDQNAgent",
    "create_agent",
    # Utilities
    "ReplayBuffer",
    "ModelReplayBuffer",
    "MixedReplayBuffer",
    "EarlyStopping",
    "load_config",
    "save_checkpoint",
    "load_checkpoint",
    "set_seed",
]
