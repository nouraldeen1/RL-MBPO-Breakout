import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym
import ale_py
from gymnasium.wrappers import AtariPreprocessing, RecordVideo
import cv2
import wandb

# Import your model architecture
# Ensure DreamerArch.py is in the same directory and has the fixes applied
try:
    from DreamerArch import DreamerAgent
except ImportError:
    print("Error: Could not import DreamerAgent from DreamerArch.py.")
    print("Please ensure DreamerArch.py is in the current directory.")
    exit(1)

# =============================================================================
# Environment Wrappers (Reused from Training)
# =============================================================================

class DreamerAtariWrapper(gym.Wrapper):
    """Atari wrapper for Dreamer (Resizes to 64x64)."""
    
    def __init__(self, env, size=64, grayscale=False):
        super().__init__(env)
        self.size = size
        self.grayscale = grayscale
        
        if grayscale:
            obs_shape = (1, size, size)
        else:
            obs_shape = (3, size, size)
        
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=obs_shape, dtype=np.float32
        )
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._process_obs(obs), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._process_obs(obs), reward, terminated, truncated, info
    
    def _process_obs(self, obs):
        if obs.ndim == 2:
            obs = cv2.resize(obs, (self.size, self.size), interpolation=cv2.INTER_AREA)
            if self.grayscale:
                obs = obs[np.newaxis, ...]
            else:
                obs = np.stack([obs] * 3, axis=0)
        else:
            obs = cv2.resize(obs, (self.size, self.size), interpolation=cv2.INTER_AREA)
            if self.grayscale:
                obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)[np.newaxis, ...]
            else:
                obs = obs.transpose(2, 0, 1)
        
        return obs.astype(np.float32) / 255.0

def make_eval_env(env_id, video_dir, run_name, record_every=1, seed=0, size=64):
    """Creates an environment that records video."""
    
    # 1. Initialize Base Environment
    env = gym.make(env_id, render_mode="rgb_array")
    
    # 2. Add Video Recording (Capture 84x84 or larger before resizing)
    env = RecordVideo(
        env, 
        video_folder=video_dir,
        episode_trigger=lambda ep_id: True, # Record ALL episodes requested
        name_prefix=f"eval_{run_name}"
    )
    
    # 3. Atari Preprocessing (Grayscale, Frame Skip)
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        grayscale_obs=False, 
        scale_obs=False,
        terminal_on_life_loss=False # IMPORTANT: Play full game for eval
    )
    
    # 4. Dreamer Formatting (64x64)
    env = DreamerAtariWrapper(env, size=size, grayscale=False)
    
    env.action_space.seed(seed)
    return env

# =============================================================================
# Evaluation Logic
# =============================================================================

def load_checkpoint(path, device):
    """Loads model and config from checkpoint."""
    print(f"Loading checkpoint from {path}...")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    config = checkpoint['config']
    
    # Reconstruct Agent
    agent = DreamerAgent(
        obs_shape=(3, config['obs_size'], config['obs_size']),
        action_dim=6, # SpaceInvaders
        embed_dim=config['embed_dim'],
        stoch_dim=config['stoch_dim'],
        deter_dim=config['deter_dim'],
        hidden_dim=config['hidden_dim'],
        num_classes=config['num_classes'],
    ).to(device)
    
    agent.load_state_dict(checkpoint['agent'])
    agent.eval()
    
    return agent, config

def evaluate(args):
    # 1. Setup WandB
    if not args.no_wandb:
        run_name = f"eval_{args.env_id}_{args.seed}"
        wandb.init(
            project="dreamer-spaceinvaders",
            name=run_name,
            job_type="evaluation",
            config=args
        )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Load Model
    agent, config = load_checkpoint(args.checkpoint, device)
    
    # 3. Setup Env
    video_dir = os.path.join("videos", "eval")
    os.makedirs(video_dir, exist_ok=True)
    
    env = make_eval_env(
        args.env_id, 
        video_dir, 
        run_name=f"seed{args.seed}", 
        record_every=1,
        seed=args.seed,
        size=config['obs_size']
    )
    
    print(f"\nStarting evaluation for {args.episodes} episodes...")
    
    total_rewards = []
    total_steps = 0
    
    for episode in range(args.episodes):
        seed = args.seed + episode
        obs, _ = env.reset(seed=seed)
        done = False
        episode_reward = 0
        episode_steps = 0
        
        # Initialize RSSM State (h=0, z=0)
        state = agent.rssm.initial_state(1, device)
        
        # Initial Action (zeros)
        action = torch.zeros(1, env.action_space.n).to(device)
        
        while not done:
            # Prepare observation
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
            
            with torch.no_grad():
                # 1. Embed Observation
                embed = agent.encoder(obs_tensor)
                
                # 2. Observe Step (Update Posterior)
                # We need the PREVIOUS action here. 
                # For first step, it's 0. For subsequent steps, it's what we just did.
                state, _, _ = agent.rssm.observe(embed, action, state)
                
                # 3. Get Feature for Policy
                feature = agent.rssm.get_feature(state)
                
                # 4. Select Action (Deterministic for Evaluation)
                # Dreamer uses mode() for eval, sample() for train
                action_idx, action_onehot, _ = agent.actor.get_action(feature, deterministic=True)
                
                # Update action tensor for next step's RSSM update
                action = action_onehot
            
            # Step Environment
            obs, reward, terminated, truncated, info = env.step(action_idx.item())
            done = terminated or truncated
            
            episode_reward += reward
            episode_steps += 1
            total_steps += 1
            
            # (Optional) Log per-step reward if highly requested
            # if not args.no_wandb:
            #     wandb.log({"eval/step_reward": reward}, step=total_steps)

        print(f"Episode {episode+1}: Reward = {episode_reward:.1f}, Length = {episode_steps}")
        total_rewards.append(episode_reward)
        
        # Log to WandB
        if not args.no_wandb:
            # Log metrics
            wandb.log({
                "eval/episode_reward": episode_reward,
                "eval/episode_length": episode_steps,
                "eval/global_step": total_steps
            })
            
            # Upload Video
            vid_path = os.path.join(video_dir, f"eval_seed{args.seed}-episode-{episode}.mp4")
            if os.path.exists(vid_path):
                wandb.log({f"video/ep_{episode}": wandb.Video(vid_path, fps=30, format="mp4")})

    avg_reward = sum(total_rewards) / len(total_rewards)
    print(f"\nEvaluation Complete.")
    print(f"Average Reward: {avg_reward:.2f}")
    
    if not args.no_wandb:
        wandb.log({"eval/avg_reward": avg_reward})
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pt")
    parser.add_argument("--env-id", type=str, default="SpaceInvadersNoFrameskip-v4")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--no-wandb", action="store_true")
    
    args = parser.parse_args()
    
    evaluate(args)