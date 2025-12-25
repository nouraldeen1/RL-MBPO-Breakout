
from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
from gymnasium.wrappers import TimeLimit

class CenteredFloatFrame(gym.ObservationWrapper):
    """
    Normalize observations to [-0.5, 0.5] for Dreamer.
    """
    def __init__(self, env):
        super().__init__(env)
        old_shape = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=-0.5,
            high=0.5,
            shape=old_shape,
            dtype=np.float32
        )
    def observation(self, observation):
        return observation.astype(np.float32) / 255.0 - 0.5

"""
Environment Factory for MBPO-Breakout
=====================================

This module provides the environment construction utilities following the
standard "Nature DQN" preprocessing suite. This setup is critical for MBRL:

1. FrameStack(4): Allows dynamics model to learn velocity/acceleration
2. EpisodicLifeEnv: Clear reward signal when agent fails
3. Delta state prediction: Easier for CNN to learn motion than full frames

Author: CMPS458 RL Project
"""

import os
from typing import Callable, Optional, Sequence

import gymnasium as gym
import numpy as np
import ale_py  # Required to register Atari environments
from gymnasium import spaces
try:
    from gymnasium.wrappers.record_video import RecordVideo  # type: ignore
except Exception:
    try:
        from gymnasium.wrappers import RecordVideo  # type: ignore
    except Exception:
        # Minimal fallback RecordVideo wrapper that writes .mp4 files using OpenCV.
        class RecordVideo(gym.Wrapper):
            """
            Minimal replacement for Gymnasium's RecordVideo. This wrapper will
            attempt to call `env.render()` each step when recording is enabled
            and save frames to an .mp4 file at episode end. It's intentionally
            lightweight and only used when the native wrapper is unavailable.
            """

            def __init__(
                self,
                env: gym.Env,
                video_folder: str = "videos",
                episode_trigger=lambda episode_id: True,
                name_prefix: str = "rl-video",
            ) -> None:
                super().__init__(env)
                self.video_folder = video_folder
                os.makedirs(self.video_folder, exist_ok=True)
                self.episode_trigger = episode_trigger
                self.name_prefix = name_prefix
                self.episode_id = 0
                self.recording = False
                self.frames = []

            def reset(self, **kwargs):
                obs, info = self.env.reset(**kwargs)
                # Decide whether to record this episode
                try:
                    self.recording = bool(self.episode_trigger(self.episode_id))
                except Exception:
                    self.recording = False
                self.frames = []

                # Capture initial rendered frame if possible
                if self.recording:
                    try:
                        frame = self.env.render()
                        if frame is not None:
                            self.frames.append(frame)
                    except Exception:
                        pass

                return obs, info

            def step(self, action):
                obs, reward, terminated, truncated, info = self.env.step(action)

                if self.recording:
                    try:
                        frame = self.env.render()
                        if frame is not None:
                            self.frames.append(frame)
                    except Exception:
                        pass

                done = terminated or truncated
                if done and self.recording and self.frames:
                    # Try to write video
                    try:
                        fname = os.path.join(
                            self.video_folder, f"{self.name_prefix}-{self.episode_id:06d}.mp4"
                        )
                        h, w = self.frames[0].shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        out = cv2.VideoWriter(fname, fourcc, 30.0, (w, h))
                        for f in self.frames:
                            # Convert RGB to BGR for OpenCV
                            out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
                        out.release()
                    except Exception:
                        pass

                if done:
                    self.episode_id += 1

                return obs, reward, terminated, truncated, info

# FrameStack import fallback for compatibility across Gymnasium versions
try:
    # Newer Gymnasium versions expose FrameStack in a submodule
    from gymnasium.wrappers.frame_stack import FrameStack  # type: ignore
except Exception:
    try:
        # Older versions may export it directly
        from gymnasium.wrappers import FrameStack  # type: ignore
    except Exception:
        # Fallback: lightweight local implementation of FrameStack
        from collections import deque

        class FrameStack(gym.Wrapper):
            """
            Simple FrameStack replacement that stacks the last `num_stack`
            observations along the channel dimension. This is intentionally
            lightweight and only supports numpy-array observations.
            """

            def __init__(self, env: gym.Env, num_stack: int) -> None:
                super().__init__(env)
                self.num_stack = num_stack
                self.frames = deque(maxlen=num_stack)

                obs_space = env.observation_space
                shape = obs_space.shape
                # Expect shape to be (H, W) or (H, W, C)
                if len(shape) == 2:
                    c = 1
                    h, w = shape
                elif len(shape) == 3:
                    h, w, c = shape
                else:
                    raise ValueError("Unsupported observation shape for FrameStack: %s" % (shape,))

                self.observation_space = spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.num_stack * c, h, w) if c > 1 else (self.num_stack, h, w),
                    dtype=np.float32,
                )

            def _get_ob(self) -> np.ndarray:
                arrs = list(self.frames)
                # If frames are 2D, expand channel dim
                proc = []
                for fr in arrs:
                    if fr.ndim == 2:
                        proc.append(np.expand_dims(fr, 0))
                    else:
                        # assume (H, W, C) -> transpose to (C, H, W)
                        if fr.shape[-1] != fr.shape[0]:
                            proc.append(np.transpose(fr, (2, 0, 1)))
                        else:
                            proc.append(fr)
                return np.concatenate(proc, axis=0)

            def reset(self, **kwargs):
                obs, info = self.env.reset(**kwargs)
                # Fill deque with initial observation
                for _ in range(self.num_stack):
                    self.frames.append(obs)
                return self._get_ob(), info

            def step(self, action):
                obs, reward, terminated, truncated, info = self.env.step(action)
                self.frames.append(obs)
                return self._get_ob(), reward, terminated, truncated, info
from gymnasium.vector import SyncVectorEnv
import cv2


# =============================================================================
# Custom Atari Wrappers (Nature DQN Suite)
# =============================================================================

class NoopResetEnv(gym.Wrapper):
    """
    Sample initial states by taking random number of no-ops on reset.
    No-op is assumed to be action 0.
    
    This provides initial state variety, which is crucial for learning
    robust policies that don't overfit to specific starting positions.
    """
    
    def __init__(self, env: gym.Env, noop_max: int = 30) -> None:
        super().__init__(env)
        self.noop_max = noop_max
        self.override_num_noops: Optional[int] = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"
    
    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        obs, info = self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
        
        assert noops > 0
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        
        return obs, info


class MaxAndSkipEnv(gym.Wrapper):
    """
    Return only every `skip`-th frame with max pooling over last 2 frames.
    
    This reduces flickering artifacts in Atari games where sprites may not
    appear on every frame, and speeds up training by reducing temporal redundancy.
    """
    
    def __init__(self, env: gym.Env, skip: int = 4) -> None:
        super().__init__(env)
        # Most recent raw observations for max pooling
        self._obs_buffer = np.zeros(
            (2,) + env.observation_space.shape, 
            dtype=env.observation_space.dtype
        )
        self._skip = skip
    
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        total_reward = 0.0
        terminated = truncated = False
        
        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            # Store last two observations for max pooling
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if terminated or truncated:
                break
        
        # Take element-wise max over last 2 frames
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, terminated, truncated, info
    
    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        return self.env.reset(**kwargs)


class ClipRewardEnv(gym.RewardWrapper):
    """
    Clip rewards to {-1, 0, +1} range.
    
    CRITICAL FOR BREAKOUT: Raw rewards are 1, 4, or 7 points per brick.
    These varied magnitudes make training unstable. Clipping to [-1, +1]
    gives consistent reward scale and faster convergence.
    """
    def reward(self, reward: float) -> float:
        return np.sign(reward)


class EpisodicLifeEnv(gym.Wrapper):
    """
    Make end-of-life == end-of-episode, but only reset on true game over.
    
    CRITICAL FOR BREAKOUT: Without this wrapper, losing a life doesn't signal
    a terminal state. This confuses the value function because the agent sees
    rewards continue after "bad" states. With this wrapper:
    - Each life is treated as an episode for learning
    - The agent gets clear feedback when it fails
    - Training converges faster with higher quality gradients
    """
    
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True
    
    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = terminated or truncated
        
        # Check current lives
        lives = self.env.unwrapped.ale.lives()
        if 0 < lives < self.lives:
            # Lost a life - treat as terminal for learning but not for env reset
            terminated = True
        self.lives = lives
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        """
        Reset only when lives are exhausted.
        This allows a reset to be called after each "life" episode.
        """
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            # No-op step to advance from terminal/lost life state
            obs, _, _, _, info = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info


class FireResetEnv(gym.Wrapper):
    """
    Take action on reset for environments that require firing to start.
    
    In Breakout, the agent must press FIRE to launch the ball. This wrapper
    automatically handles this so the agent doesn't need to learn this trivial action.
    """
    
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3
    
    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(1)  # FIRE
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(2)  # RIGHT (to move paddle)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info


class WarpFrame(gym.ObservationWrapper):
    """
    Convert to grayscale and resize to 84x84.
    
    This is the standard preprocessing for Atari that reduces the observation
    space from (210, 160, 3) to (84, 84), making it computationally feasible
    while retaining essential game information.
    """
    
    def __init__(self, env: gym.Env, width: int = 84, height: int = 84) -> None:
        super().__init__(env)
        self._width = width
        self._height = height
        
        # Update observation space
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(self._height, self._width),
            dtype=np.uint8
        )
    
    def observation(self, frame: np.ndarray) -> np.ndarray:
        # Convert to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # Resize to target dimensions
        frame = cv2.resize(
            frame, 
            (self._width, self._height), 
            interpolation=cv2.INTER_AREA
        )
        return frame


class ScaledFloatFrame(gym.ObservationWrapper):
    """
    Normalize observations to [0, 1] range.
    
    Neural networks train better with normalized inputs. This converts
    uint8 pixel values [0, 255] to float32 values [0, 1].
    """
    
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=env.observation_space.shape,
            dtype=np.float32
        )
    
    def observation(self, observation: np.ndarray) -> np.ndarray:
        return np.asarray(observation, dtype=np.float32) / 255.0


class ChannelFirstWrapper(gym.ObservationWrapper):
    """
    Convert observation from (H, W, C) to (C, H, W) for PyTorch.
    """
    
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        old_shape = self.observation_space.shape
        
        # Handle different input shapes
        if len(old_shape) == 3:
            new_shape = (old_shape[2], old_shape[0], old_shape[1])
        else:
            new_shape = old_shape
            
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=new_shape,
            dtype=np.float32
        )
    
    def observation(self, observation: np.ndarray) -> np.ndarray:
        if len(observation.shape) == 3:
            return np.transpose(observation, (2, 0, 1))
        return observation


# =============================================================================
# Environment Factory Functions
# =============================================================================

def make_env(
    env_id: str = "BreakoutNoFrameskip-v4",
    seed: int = 0,
    idx: int = 0,
    capture_video: bool = False,
    video_dir: str = "videos",
    video_trigger_freq: int = 50,
    noop_max: int = 30,
    frame_skip: int = 4,
    frame_stack: int = 4,
    dreamer_norm: bool = False,
) -> Callable[[], gym.Env]:
    """
    Create a single Atari environment with Nature DQN preprocessing.
    
    Args:
        env_id: Gymnasium environment ID
        seed: Random seed for reproducibility
        idx: Environment index (for video naming)
        capture_video: Whether to record episodes
        video_dir: Directory to save videos
        video_trigger_freq: Record every N episodes
        noop_max: Max no-op actions at reset
        frame_skip: Frames to skip (with max pooling)
        frame_stack: Number of frames to stack
    
    Returns:
        A callable that creates the wrapped environment
    
    Example:
        >>> env_fn = make_env("BreakoutNoFrameskip-v4", seed=42)
        >>> env = env_fn()
        >>> obs, info = env.reset()
        >>> print(obs.shape)  # (4, 84, 84)
    """
    
    def _thunk() -> gym.Env:
        # Create base environment with rgb_array rendering for video capture
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
        else:
            env = gym.make(env_id)
        
        # Video recording (must be applied before other wrappers for correct frames)
        if capture_video and idx == 0:
            os.makedirs(video_dir, exist_ok=True)
            env = RecordVideo(
                env,
                video_folder=video_dir,
                episode_trigger=lambda episode_id: episode_id % video_trigger_freq == 0,
                name_prefix=f"breakout-{idx}",
            )
        
        # Apply Nature DQN preprocessing stack
        env = NoopResetEnv(env, noop_max=noop_max)
        env = MaxAndSkipEnv(env, skip=frame_skip)
        env = ClipRewardEnv(env)     # Clip rewards to [-1, +1]
        env = EpisodicLifeEnv(env)  # CRITICAL for Breakout
        env = FireResetEnv(env)      # Auto-start ball
        env = WarpFrame(env, width=84, height=84)  # Grayscale 84x84
        # For Dreamer, use CenteredFloatFrame for [-0.5, 0.5] normalization
        if dreamer_norm:
            env = CenteredFloatFrame(env)
        else:
            env = ScaledFloatFrame(env)  # Normalize to [0, 1]
        
        # Frame stacking - CRITICAL for velocity sensing
        # The dynamics model needs to see motion, not just static frames
        env = FrameStack(env, num_stack=frame_stack)
        
        # Add time limit to prevent infinite episodes
        env = TimeLimit(env, max_episode_steps=1000)
        
        # Set seed for reproducibility
        env.action_space.seed(seed + idx)
        
        return env
    
    return _thunk


def make_vec_env(
    env_id: str = "BreakoutNoFrameskip-v4",
    num_envs: int = 4,
    seed: int = 0,
    capture_video: bool = True,
    video_dir: str = "videos",
    video_trigger_freq: int = 50,
    **kwargs
) -> SyncVectorEnv:
    """
    Create a vectorized environment with multiple parallel instances.
    
    Using multiple environments allows for efficient data collection,
    which is crucial for MBRL where we need diverse trajectories.
    
    Args:
        env_id: Gymnasium environment ID
        num_envs: Number of parallel environments
        seed: Base random seed
        capture_video: Whether to record videos (only env 0)
        video_dir: Directory for video output
        video_trigger_freq: Record every N episodes
        **kwargs: Additional arguments passed to make_env
    
    Returns:
        SyncVectorEnv with all preprocessing applied
    
    Example:
        >>> vec_env = make_vec_env(num_envs=4, capture_video=True)
        >>> obs, info = vec_env.reset()
        >>> print(obs.shape)  # (4, 4, 84, 84) - (num_envs, frames, H, W)
    """
    env_fns = [
        make_env(
            env_id=env_id,
            seed=seed,
            idx=i,
            capture_video=capture_video if i == 0 else False,
            video_dir=video_dir,
            video_trigger_freq=video_trigger_freq,
            **kwargs
        )
        for i in range(num_envs)
    ]
    
    return SyncVectorEnv(env_fns)


def make_eval_env(
    env_id: str = "BreakoutNoFrameskip-v4",
    seed: int = 0,
    video_dir: str = "videos/eval",
    record_every: int = 1,
    **kwargs
) -> gym.Env:
    """
    Create a single evaluation environment with video recording.
    
    This environment records all episodes for evaluation purposes.
    Used in eval.py for final model assessment.
    
    Args:
        env_id: Gymnasium environment ID  
        seed: Random seed
        video_dir: Directory for evaluation videos
        record_every: Record every N episodes (default: all)
        **kwargs: Additional wrapper arguments
    
    Returns:
        Single wrapped environment for evaluation
    """
    env_fn = make_env(
        env_id=env_id,
        seed=seed,
        idx=0,
        capture_video=True,
        video_dir=video_dir,
        video_trigger_freq=record_every,
        **kwargs
    )
    return env_fn()


# =============================================================================
# Verification Utilities
# =============================================================================

def verify_env_setup(env_id: str = "BreakoutNoFrameskip-v4") -> dict:
    """
    Verify environment setup and print diagnostic information.
    
    Run this function to ensure your environment is correctly configured
    before starting training.
    
    Returns:
        Dictionary with environment specifications
    """
    print("=" * 60)
    print("Environment Verification")
    print("=" * 60)
    
    # Create single environment
    env = make_env(env_id, seed=42, capture_video=False, dreamer_norm=False)()
    
    obs, info = env.reset()
    
    # Get observation info
    obs_shape = obs.shape
    obs_dtype = obs.dtype
    
    # Get action info
    action_space = env.action_space
    num_actions = action_space.n
    
    # Get action meanings from base environment
    base_env = env.unwrapped
    action_meanings = base_env.get_action_meanings()
    
    # Print results
    print(f"\nEnvironment: {env_id}")
    print(f"Observation Shape: {obs_shape}")
    print(f"  Expected: (4, 84, 84) - 4 stacked grayscale frames")
    print(f"  Status: {'✓ CORRECT' if obs_shape == (4, 84, 84) else '✗ INCORRECT'}")
    
    print(f"\nObservation dtype: {obs_dtype}")
    print(f"  Expected: float32 (normalized to [0, 1])")
    print(f"  Status: {'✓ CORRECT' if obs_dtype == np.float32 else '✗ INCORRECT'}")
    
    print(f"\nObservation range: [{obs.min():.3f}, {obs.max():.3f}]")
    print(f"  Expected: [0.0, 1.0]")
    
    print(f"\nAction Space: {action_space}")
    print(f"  Number of actions: {num_actions}")
    print(f"  Expected: Discrete(4)")
    print(f"  Status: {'✓ CORRECT' if num_actions == 4 else '✗ INCORRECT'}")
    
    print(f"\nAction Meanings:")
    for i, meaning in enumerate(action_meanings):
        print(f"  {i}: {meaning}")
    
    # Test step
    action = env.action_space.sample()
    next_obs, reward, terminated, truncated, info = env.step(action)
    
    print(f"\nStep test:")
    print(f"  Next obs shape: {next_obs.shape}")
    print(f"  Reward: {reward}")
    print(f"  Terminated: {terminated}")
    
    env.close()
    
    print("\n" + "=" * 60)
    print("Verification Complete!")
    print("=" * 60)
    
    return {
        "env_id": env_id,
        "obs_shape": obs_shape,
        "obs_dtype": str(obs_dtype),
        "num_actions": num_actions,
        "action_meanings": action_meanings,
    }


if __name__ == "__main__":
    # Run verification when executed directly
    specs = verify_env_setup()
    
    # Also test vectorized environment
    print("\n\nTesting Vectorized Environment...")
    vec_env = make_vec_env(num_envs=2, capture_video=False)
    obs, info = vec_env.reset()
    print(f"Vectorized obs shape: {obs.shape}")
    print(f"Expected: (2, 4, 84, 84) - (num_envs, frames, H, W)")
    vec_env.close()
