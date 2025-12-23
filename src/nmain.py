"""
Main Training Script
"""
import sys
import os
import torch
import numpy as np
from env_factory import make_vec_env
from agents import create_agent
from utils import load_config, set_seed, EarlyStopping, MetricsLogger
import argparse

def train(config):
    set_seed(config['seed'])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Ensure checkpoint dir exists
    ckpt_dir = config['logging']['checkpoint_dir']
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {os.path.abspath(ckpt_dir)}")
    # Env
    env = make_vec_env(
        env_id=config['env']['name'],
        num_envs=config['env']['num_envs'],
        seed=config['seed'],
        video_dir=config['env']['video_dir']
    )
    
    # Agent
    agent = create_agent(
        "world_models", 
        env.single_observation_space.shape,
        env.single_action_space.n,
        config,
        device
    )
    
    # Training Loop
    total_steps = config['training']['total_timesteps']
    obs, _ = env.reset()
    
    logger = MetricsLogger()
    stopper = EarlyStopping(**config['early_stopping'])
    
    episode_rewards = np.zeros(config['env']['num_envs'])
    
    print(f"Starting training on {device}...")
    
    for step in range(0, total_steps, config['env']['num_envs']):
        # Act
        actions, _ = agent.get_action(obs)
        
        # Step
        next_obs, rewards, terms, truncs, _ = env.step(actions)
        dones = np.logical_or(terms, truncs)
        
        # Store
        for i in range(len(obs)):
            agent.store_transition(obs[i], actions[i], rewards[i], next_obs[i], dones[i])
            episode_rewards[i] += rewards[i]
            
            if dones[i]:
                if stopper.update(episode_rewards[i]):
                    print(f"Early Stop: {stopper.stop_reason}")
                    agent.save("final_model.pt")
                    return
                episode_rewards[i] = 0
                
        obs = next_obs
        
        # Train
        if step > config['training']['learning_starts'] and step % config['training']['train_freq'] == 0:
            metrics = agent.update()
            logger.update(metrics)
            
        # Log
        if step % config['logging']['log_freq'] == 0:
            means = logger.get_means()
            r_mean = stopper.get_mean_reward()
            print(f"Step {step} | Reward: {r_mean:.2f} | "
                  f"VAE: {means.get('vae/recon', 0):.4f} | "
                  f"MDN: {means.get('mdnrnn/nll', 0):.4f}")
        # SAVE CHECKPOINT
        if step > 0 and step % config['logging']['save_freq'] == 0:
            path = os.path.join(ckpt_dir, f"ckpt_{step}.pt")
            agent.save(path)
            print(f"Saved checkpoint: {path}")
            
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/configWorldModels.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg)
    # cfg = load_config("config/configWorldModels.yaml")
    # train(cfg)