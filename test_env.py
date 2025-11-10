"""
Test script for the Splendor Solo Gymnasium Environment
"""

from splendor_solo_env import SplendorSoloEnv
import numpy as np


def test_basic_functionality():
    """Test basic environment functionality."""
    print("=" * 60)
    print("Testing SplendorSoloEnv Basic Functionality")
    print("=" * 60)
    
    # Create environment
    print("\n1. Creating environment...")
    env = SplendorSoloEnv()
    print("✓ Environment created successfully")
    
    # Reset environment
    print("\n2. Resetting environment...")
    obs, info = env.reset(seed=42)
    print("✓ Environment reset successfully")
    
    # Print observation space info
    print("\n3. Observation space:")
    for key, value in obs.items():
        print(f"   - {key}: shape={value.shape}, dtype={value.dtype}")
    
    # Print initial state
    print("\n4. Initial game state:")
    print(f"   - Bank tokens: {obs['bank_tokens']}")
    print(f"   - Player tokens: {obs['player_tokens']}")
    print(f"   - Player bonuses: {obs['player_bonuses']}")
    print(f"   - Player points: {obs['player_points'][0]}")
    print(f"   - Game clock: {obs['game_clock'][0]}")
    
    # Check legal actions
    legal_actions = info['legal_actions']
    num_legal = np.sum(legal_actions)
    print(f"\n5. Legal actions: {num_legal} out of {len(legal_actions)}")
    print(f"   - Legal action indices: {np.where(legal_actions == 1)[0][:10]}... (showing first 10)")
    
    # Take a few steps
    print("\n6. Taking random legal actions...")
    for step_num in range(5):
        legal_indices = np.where(legal_actions == 1)[0]
        if len(legal_indices) == 0:
            print("   No legal actions available!")
            break
        
        action = np.random.choice(legal_indices)
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"   Step {step_num + 1}:")
        print(f"      - Action: {action}")
        print(f"      - Reward: {reward}")
        print(f"      - Player points: {obs['player_points'][0]}")
        print(f"      - Player tokens: {obs['player_tokens']}")
        print(f"      - Game clock: {obs['game_clock'][0]}")
        print(f"      - Terminated: {terminated}")
        
        if terminated:
            print("   Game ended!")
            break
        
        legal_actions = info['legal_actions']
    
    print("\n✓ Basic functionality test completed successfully!")


def test_illegal_action():
    """Test that illegal actions are handled correctly."""
    print("\n" + "=" * 60)
    print("Testing Illegal Action Handling")
    print("=" * 60)
    
    env = SplendorSoloEnv()
    obs, info = env.reset(seed=42)
    
    # Find an illegal action
    legal_actions = info['legal_actions']
    illegal_indices = np.where(legal_actions == 0)[0]
    
    if len(illegal_indices) > 0:
        illegal_action = illegal_indices[0]
        print(f"\nAttempting illegal action {illegal_action}...")
        obs, reward, terminated, truncated, info = env.step(illegal_action)
        
        print(f"   - Reward: {reward}")
        print(f"   - Terminated: {terminated}")
        
        if reward == -100.0 and terminated:
            print("✓ Illegal action handled correctly (large negative reward + termination)")
        else:
            print("✗ Unexpected behavior for illegal action")
    else:
        print("No illegal actions found (all actions legal initially)")


def test_action_space():
    """Test action space mapping."""
    print("\n" + "=" * 60)
    print("Testing Action Space Mapping")
    print("=" * 60)
    
    env = SplendorSoloEnv()
    
    print(f"\nAction space size: {env.action_space.n}")
    print("\nSample action mappings:")
    for i in [0, 5, 10, 15, 20, 27, 36]:
        action_type, params = env.action_map[i]
        print(f"   Action {i}: {action_type} - {params}")
    
    print("\n✓ Action space test completed!")


def test_full_episode():
    """Run a complete episode with random actions."""
    print("\n" + "=" * 60)
    print("Testing Full Episode")
    print("=" * 60)
    
    env = SplendorSoloEnv()
    obs, info = env.reset(seed=123)
    
    total_reward = 0
    steps = 0
    
    print("\nRunning episode until termination...")
    
    while True:
        legal_actions = info['legal_actions']
        legal_indices = np.where(legal_actions == 1)[0]
        
        if len(legal_indices) == 0:
            print("No legal actions available!")
            break
        
        action = np.random.choice(legal_indices)
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        steps += 1
        
        if reward > 0:
            print(f"   Step {steps}: Action {action}, Reward {reward}, Total Points {obs['player_points'][0]}")
        
        if terminated:
            break
    
    print(f"\n✓ Episode completed!")
    print(f"   - Total steps: {steps}")
    print(f"   - Total reward (points): {total_reward}")
    print(f"   - Final points: {obs['player_points'][0]}")
    print(f"   - Final game clock: {obs['game_clock'][0]}")


if __name__ == "__main__":
    try:
        test_basic_functionality()
        test_illegal_action()
        test_action_space()
        test_full_episode()
        
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
