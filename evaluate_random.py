"""
Random baseline evaluation for comparison
"""

import numpy as np
import gymnasium as gym
import splendor_solo_env

# Register the environment
gym.register(
    id="SplendorSolo-v0",
    entry_point="splendor_solo_env:SplendorSoloEnv",
)

def evaluate_random_policy(n_episodes=100, seed=42):
    """Evaluate a random policy."""
    
    env = gym.make("SplendorSolo-v0")
    
    episode_returns = []
    episode_lengths = []
    episode_points = []
    win_count = 0
    
    print(f"Evaluating random policy for {n_episodes} episodes...")
    print("-" * 60)
    
    for episode in range(n_episodes):
        obs, info = env.reset(seed=seed + episode)
        done = False
        episode_return = 0
        episode_length = 0
        
        while not done:
            # Get legal actions
            legal_actions = info.get('legal_actions', np.ones(env.action_space.n, dtype=np.int8))
            
            # Select random legal action
            legal_action_indices = np.where(legal_actions)[0]
            if len(legal_action_indices) == 0:
                print(f"Warning: No legal actions available in episode {episode}")
                break
            action = np.random.choice(legal_action_indices)
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_return += reward
            episode_length += 1
        
        # Record episode statistics
        final_points = env.unwrapped.player_points
        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)
        episode_points.append(final_points)
        
        if final_points >= 15:
            win_count += 1
        
        # Print progress every 10 episodes
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{n_episodes} - "
                  f"Avg Return: {np.mean(episode_returns):.2f}, "
                  f"Avg Points: {np.mean(episode_points):.2f}, "
                  f"Win Rate: {win_count/(episode+1)*100:.1f}%")
    
    # Print results
    print("\n" + "=" * 60)
    print("RANDOM BASELINE RESULTS")
    print("=" * 60)
    print(f"\nEpisodes Evaluated: {n_episodes}")
    print(f"\nReturn Statistics:")
    print(f"  Mean Return:     {np.mean(episode_returns):>8.2f} ± {np.std(episode_returns):.2f}")
    print(f"  Min Return:      {np.min(episode_returns):>8.2f}")
    print(f"  Max Return:      {np.max(episode_returns):>8.2f}")
    print(f"\nEpisode Length Statistics:")
    print(f"  Mean Length:     {np.mean(episode_lengths):>8.1f} ± {np.std(episode_lengths):.1f} steps")
    print(f"\nPoints Statistics:")
    print(f"  Mean Points:     {np.mean(episode_points):>8.2f} ± {np.std(episode_points):.2f}")
    print(f"  Min Points:      {np.min(episode_points):>8.0f}")
    print(f"  Max Points:      {np.max(episode_points):>8.0f}")
    print(f"\nWin Rate (≥15 points):")
    print(f"  Wins:            {win_count:>8d} / {n_episodes}")
    print(f"  Win Rate:        {win_count/n_episodes*100:>8.1f}%")
    print("=" * 60)
    
    env.close()

if __name__ == "__main__":
    evaluate_random_policy(n_episodes=100, seed=42)
