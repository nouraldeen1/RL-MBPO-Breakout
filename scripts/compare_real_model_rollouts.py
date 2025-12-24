"""
Compare real environment frames vs. model-predicted frames (side-by-side video).

Usage:
  python scripts/compare_real_model_rollouts.py --checkpoint checkpoints/agent.pth --episodes 3

The script:
 - Loads config from `config/config.yaml`
 - Instantiates `MBPOAgent` and optionally loads a checkpoint
 - Runs `--episodes` environment episodes using the agent's policy
 - For each step, uses the dynamics ensemble to predict the next stacked frames
 - Writes a side-by-side video (Real | Model) per episode to `videos/compare`

Notes:
 - Requires OpenCV (`cv2`) and `imageio` (these are already used elsewhere in the repo).
 - The visualized frames are the last frame of the stacked 4-frame observation (grayscale).
"""

import os
import time
import argparse
from pathlib import Path
import sys

import numpy as np
import torch
import imageio
import cv2

# Ensure the project's `src` directory is on sys.path so imports like
# `from utils import ...` work when running this script from the repo root.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from typing import Optional

from utils import load_config
from env_factory import make_env
from agents import MBPOAgent


def frame_to_rgb_uint8(frame: np.ndarray) -> np.ndarray:
    """Convert single-channel float frame in [0,1] or uint8 to RGB uint8."""
    if frame.dtype != np.uint8:
        # assume float in [0,1]
        frame = (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
    # Convert to 3-channel RGB for video
    rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    # cv2 uses BGR ordering internally; ensure it's RGB for imageio
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    return rgb


def run_compare(checkpoint: Optional[str], episodes: int, max_steps: int, out_dir: str, device: str, record_episodes: Optional[str] = None):
    config = load_config("config/config.yaml")

    # Create agent and load checkpoint if provided
    agent = MBPOAgent(config=config, device=device)
    if checkpoint is not None and os.path.exists(checkpoint):
        print(f"Loading checkpoint: {checkpoint}")
        agent.load(checkpoint)
    else:
        if checkpoint is not None:
            print(f"Checkpoint not found: {checkpoint} — proceeding with uninitialized models")

    # Create a single evaluation environment with the same preprocessing
    env = make_env(
        env_id=config.get("env", {}).get("name", "BreakoutNoFrameskip-v4"),
        seed=config.get("seed", 42),
        idx=0,
        capture_video=False,
        video_dir=out_dir,
        video_trigger_freq=1,
        noop_max=config.get("env", {}).get("noop_max", 30),
        frame_skip=config.get("env", {}).get("frame_skip", 4),
        frame_stack=config.get("env", {}).get("frame_stack", 4),
    )()

    os.makedirs(out_dir, exist_ok=True)

    # Determine which episodes to record. By default record all episodes.
    record_all = True
    record_set = set()
    # Priority: CLI `record_episodes` argument (comma-separated ints), then env var RECORD_EPISODES.
    record_arg = record_episodes
    if record_arg is None:
        record_arg = os.environ.get("RECORD_EPISODES")
    if record_arg:
        try:
            record_idxs = [int(x) for x in str(record_arg).split(",") if x.strip()]
            record_all = False
            record_set.update(record_idxs)
        except Exception:
            print(f"Warning: failed to parse record episodes '{record_arg}' — recording all by default.")

    for ep in range(episodes):
        obs, info = env.reset()
        done = False
        step = 0

        should_record = record_all or (ep in record_set)
        writer = None
        fname = None
        if should_record:
            timestamp = int(time.time())
            fname = Path(out_dir) / f"compare-ep{ep:03d}-{timestamp}.mp4"
            writer = imageio.get_writer(str(fname), fps=30)
            print(f"Recording episode {ep} -> {fname}")
        else:
            print(f"Running episode {ep} (not recording)")

        while (not done) and (step < max_steps):
            # obs: stacked frames shape (4,84,84), float32 [0,1]
            state = obs

            # Get action from policy (deterministic=False to mimic training behavior)
            action, _info = agent.get_action(state, deterministic=False)
            if isinstance(action, np.ndarray):
                a = int(action[0]) if action.shape else int(action)
            else:
                a = int(action)

            # Step environment
            next_obs, reward, terminated, truncated, step_info = env.step(a)
            done = bool(terminated or truncated)

            # Model predicted next state
            with torch.no_grad():
                s_t = torch.from_numpy(np.expand_dims(state, 0)).float().to(agent.device)
                a_t = torch.from_numpy(np.array([a], dtype=np.int64)).to(agent.device)
                # Use a single model for speed (model_idx=0), use_mean=True
                pred_next_t, pred_reward, _ = agent.dynamics.predict_next_state(
                    s_t, a_t, model_idx=0, use_mean=True
                )

            pred_next = pred_next_t.squeeze(0).cpu().numpy()  # (4,84,84)

            # Visualize the last stacked frame for both real and model
            real_last = next_obs[-1] if next_obs.shape[0] >= 1 else next_obs.squeeze(0)
            model_last = pred_next[-1]

            real_rgb = frame_to_rgb_uint8(real_last)
            model_rgb = frame_to_rgb_uint8(model_last)

            # Put labels
            h, w, _ = real_rgb.shape
            pad = 4
            combined = np.concatenate([real_rgb, model_rgb], axis=1)
            # Draw labels using OpenCV (BGR expected) -> convert to BGR, draw, convert back
            combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
            cv2.putText(combined_bgr, "REAL", (10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2.putText(combined_bgr, "MODEL", (w + 10, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            # Convert back to RGB for imageio
            combined_rgb = cv2.cvtColor(combined_bgr, cv2.COLOR_BGR2RGB)

            # Pad combined frame to be divisible by 16 (macro block size) to avoid
            # ffmpeg resizing warnings. Also ensure contiguous uint8 array to
            # avoid alignment warnings from swscaler.
            h, w, _ = combined_rgb.shape
            mb = 16
            target_h = ((h + mb - 1) // mb) * mb
            target_w = ((w + mb - 1) // mb) * mb
            pad_h = target_h - h
            pad_w = target_w - w
            if pad_h > 0 or pad_w > 0:
                combined_rgb = np.pad(
                    combined_rgb,
                    ((0, pad_h), (0, pad_w), (0, 0)),
                    mode="constant",
                    constant_values=0,
                )

            combined_rgb = np.ascontiguousarray(combined_rgb)

            if writer is not None:
                writer.append_data(combined_rgb)

            # Advance
            obs = next_obs
            step += 1

        if writer is not None:
            writer.close()
            print(f"Saved: {fname}")
        else:
            print(f"Episode {ep} finished (not saved)")

    env.close()
    print("All done.")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, default=None, help="Path to agent checkpoint (optional)")
    p.add_argument("--episodes", type=int, default=3, help="Number of episodes to record")
    p.add_argument("--max-steps", type=int, default=1000, help="Max steps per episode")
    p.add_argument("--out-dir", type=str, default="videos/compare", help="Output directory for videos")
    p.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument(
        "--record-episodes",
        type=str,
        default=None,
        help="Comma-separated list of episode indices to record (e.g. '10,100'), or single index. If omitted records all episodes or use env var RECORD_EPISODES.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_compare(
        args.checkpoint,
        args.episodes,
        args.max_steps,
        args.out_dir,
        args.device,
        record_episodes=args.record_episodes,
    )
