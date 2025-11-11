"""
DQN Model Evaluation Script for Splendor Solo Environment
Evaluates a trained model on multiple episodes and reports statistics.
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from typing import List, Dict
import json

# Import the environment
import splendor_solo_env

# Register the environment
gym.register(
    id="SplendorSolo-v0",
    entry_point="splendor_solo_env:SplendorSoloEnv",
)


class QNetwork(nn.Module):
    """Q-Network architecture matching the training script."""
    
    def __init__(self, env):
        super().__init__()
        # Get flattened observation space size
        # After FlattenObservation wrapper, obs_space is a Box with shape (n,)
        if hasattr(env.observation_space, 'shape') and env.observation_space.shape:
            obs_shape = int(np.prod(env.observation_space.shape))
        else:
            # Fallback: calculate from original Dict space
            obs_shape = 115  # Known size for Splendor: 6+6+5+1+21+63+12+1
        
        n_actions = env.action_space.n
        
        self.network = nn.Sequential(
            nn.Linear(obs_shape, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )

    def forward(self, x):
        return self.network(x)


def evaluate_model(
    model_path: str,
    env_id: str = "SplendorSolo-v0",
    n_eval_episodes: int = 100,
    seed: int = 42,
    device: str = "cuda"
):
    """
    Evaluate a trained DQN model.
    
    Args:
        model_path: Path to the saved model (.pth file)
        env_id: Environment ID
        n_eval_episodes: Number of episodes to evaluate
        seed: Random seed
        device: Device to run evaluation on
    
    Returns:
        Dictionary with evaluation statistics
    """
    
    # Setup device
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create environment with FlattenObservation wrapper
    env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.FlattenObservation(env)
    
    # Load model
    print(f"\nLoading model from: {model_path}")
    q_network = QNetwork(env).to(device)
    q_network.load_state_dict(torch.load(model_path, map_location=device))
    q_network.eval()
    print("Model loaded successfully!")
    
    # Evaluation metrics
    episode_returns = []
    episode_lengths = []
    episode_points = []
    win_count = 0  # Episodes where agent achieved >=15 points
    
    print(f"\nEvaluating for {n_eval_episodes} episodes...")
    print("-" * 60)
    
    for episode in range(n_eval_episodes):
        obs, info = env.reset(seed=seed + episode)
        done = False
        episode_return = 0
        episode_length = 0
        
        while not done:
            # Get action mask from legal_actions
            legal_actions = info.get('legal_actions', np.ones(env.action_space.n, dtype=np.int8))
            
            # Observation is already flattened by FlattenObservation wrapper
            # Get Q-values
            with torch.no_grad():
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
                q_values = q_network(obs_tensor).cpu().numpy()[0]
            
            # Apply action mask
            masked_q_values = np.where(legal_actions, q_values, -np.inf)
            
            # Select best valid action
            action = int(np.argmax(masked_q_values))
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_return += reward
            episode_length += 1
        
        # Record episode statistics
        # Get final player points from the unwrapped environment
        final_points = env.unwrapped.player_points
        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)
        episode_points.append(final_points)
        
        if final_points >= 15:
            win_count += 1
        
        # Print progress every 10 episodes
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{n_eval_episodes} - "
                  f"Avg Return: {np.mean(episode_returns):.2f}, "
                  f"Avg Points: {np.mean(episode_points):.2f}, "
                  f"Win Rate: {win_count/(episode+1)*100:.1f}%")
    
    # Calculate statistics
    stats = {
        'n_episodes': n_eval_episodes,
        'mean_return': float(np.mean(episode_returns)),
        'std_return': float(np.std(episode_returns)),
        'min_return': float(np.min(episode_returns)),
        'max_return': float(np.max(episode_returns)),
        'mean_length': float(np.mean(episode_lengths)),
        'std_length': float(np.std(episode_lengths)),
        'mean_points': float(np.mean(episode_points)),
        'std_points': float(np.std(episode_points)),
        'min_points': float(np.min(episode_points)),
        'max_points': float(np.max(episode_points)),
        'win_rate': float(win_count / n_eval_episodes * 100),
        'win_count': win_count,
    }
    
    # Print detailed results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nEpisodes Evaluated: {stats['n_episodes']}")
    print(f"\nReturn Statistics:")
    print(f"  Mean Return:     {stats['mean_return']:>8.2f} ± {stats['std_return']:.2f}")
    print(f"  Min Return:      {stats['min_return']:>8.2f}")
    print(f"  Max Return:      {stats['max_return']:>8.2f}")
    print(f"\nEpisode Length Statistics:")
    print(f"  Mean Length:     {stats['mean_length']:>8.1f} ± {stats['std_length']:.1f} steps")
    print(f"\nPoints Statistics:")
    print(f"  Mean Points:     {stats['mean_points']:>8.2f} ± {stats['std_points']:.2f}")
    print(f"  Min Points:      {stats['min_points']:>8.0f}")
    print(f"  Max Points:      {stats['max_points']:>8.0f}")
    print(f"\nWin Rate (≥15 points):")
    print(f"  Wins:            {stats['win_count']:>8d} / {stats['n_episodes']}")
    print(f"  Win Rate:        {stats['win_rate']:>8.1f}%")
    print("=" * 60)
    
    env.close()
    return stats


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained DQN model on Splendor Solo")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the model checkpoint (.pth file)")
    parser.add_argument("--env-id", type=str, default="SplendorSolo-v0",
                        help="Environment ID")
    parser.add_argument("--n-eval-episodes", type=int, default=100,
                        help="Number of episodes to evaluate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to run evaluation on (cuda/cpu)")
    parser.add_argument("--save-results", type=str, default=None,
                        help="Path to save evaluation results (JSON)")
    
    args = parser.parse_args()
    
    # Run evaluation
    stats = evaluate_model(
        model_path=args.model_path,
        env_id=args.env_id,
        n_eval_episodes=args.n_eval_episodes,
        seed=args.seed,
        device=args.device
    )
    
    # Save results if requested
    if args.save_results:
        with open(args.save_results, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\nResults saved to: {args.save_results}")


if __name__ == "__main__":
    main()
