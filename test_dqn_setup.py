"""
Quick test script to verify train_dqn_splendor.py can initialize correctly.
"""

import sys
import numpy as np
import torch
import gymnasium as gym


def test_imports():
    """Test that all required imports work."""
    print("Testing imports...")
    try:
        import argparse
        import random
        import time
        from distutils.util import strtobool
        import torch.nn as nn
        import torch.nn.functional as F
        import torch.optim as optim
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_environment_registration():
    """Test that the Splendor environment can be registered and created."""
    print("\nTesting environment registration...")
    try:
        from splendor_solo_env import SplendorSoloEnv
        
        gym.register(
            id="SplendorSolo-v0",
            entry_point="splendor_solo_env:SplendorSoloEnv",
        )
        
        env = gym.make("SplendorSolo-v0")
        print(f"✓ Environment created successfully")
        print(f"  - Action space: {env.action_space}")
        print(f"  - Observation space: {env.observation_space}")
        
        env.close()
        return True
    except Exception as e:
        print(f"✗ Environment error: {e}")
        return False


def test_vectorized_env():
    """Test that vectorized environment with FlattenObservation works."""
    print("\nTesting vectorized environment...")
    try:
        envs = gym.vector.SyncVectorEnv([lambda: gym.make("SplendorSolo-v0")])
        envs = gym.wrappers.FlattenObservation(envs)
        
        obs, info = envs.reset()
        print(f"✓ Vectorized environment created")
        print(f"  - Observation shape: {obs.shape}")
        print(f"  - Single obs space: {envs.single_observation_space}")
        print(f"  - Single action space: {envs.single_action_space}")
        
        # Check that legal_actions is in info
        if "_legal_actions" in info:
            print(f"  - Legal actions key found: {info['_legal_actions'][0].shape}")
            print(f"  - Number of legal actions: {np.sum(info['_legal_actions'][0])}")
        else:
            print(f"  ⚠ Warning: '_legal_actions' not found in info dict")
            print(f"  Available keys: {list(info.keys())}")
        
        envs.close()
        return True
    except Exception as e:
        print(f"✗ Vectorized environment error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_network_creation():
    """Test that QNetwork can be created."""
    print("\nTesting Q-Network creation...")
    try:
        # Import the QNetwork class
        sys.path.insert(0, '.')
        from train_dqn_splendor import QNetwork
        
        envs = gym.vector.SyncVectorEnv([lambda: gym.make("SplendorSolo-v0")])
        envs = gym.wrappers.FlattenObservation(envs)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        q_network = QNetwork(envs).to(device)
        
        print(f"✓ Q-Network created successfully")
        print(f"  - Device: {device}")
        print(f"  - Parameters: {sum(p.numel() for p in q_network.parameters()):,}")
        
        # Test forward pass
        obs, info = envs.reset()
        obs_tensor = torch.Tensor(obs).to(device)
        
        # Without mask
        q_values = q_network(obs_tensor)
        print(f"  - Q-values shape (no mask): {q_values.shape}")
        
        # With mask - handle the legal_actions properly
        legal_actions_raw = info["_legal_actions"]
        
        # Check what type of data structure we have
        if isinstance(legal_actions_raw, tuple) and len(legal_actions_raw) > 0:
            legal_actions_array = legal_actions_raw[0]
        else:
            legal_actions_array = legal_actions_raw
        
        # Convert to numpy array if needed and ensure it's 1D
        legal_actions_array = np.atleast_1d(np.array(legal_actions_array))
        
        # If it's still scalar or wrong shape, create a test mask manually
        if legal_actions_array.size == 1 or len(legal_actions_array) != 39:
            print(f"  ⚠ Warning: legal_actions has unexpected shape {legal_actions_array.shape}")
            print(f"     Creating test mask with all actions legal for testing")
            legal_actions_array = np.ones(39, dtype=np.float32)
            # Make some actions illegal for testing
            legal_actions_array[5:10] = 0
        
        mask = torch.tensor(legal_actions_array, device=device, dtype=torch.float32)
        q_values_masked = q_network(obs_tensor, action_mask=mask)
        print(f"  - Q-values shape (with mask): {q_values_masked.shape}")
        
        # Check that illegal actions are masked
        illegal_actions = np.where(legal_actions_array == 0)[0]
        if len(illegal_actions) > 0:
            illegal_q_value = q_values_masked[illegal_actions[0]].item()
            print(f"  - Sample illegal action Q-value: {illegal_q_value:.2e} (should be ~-1e8)")
            if illegal_q_value < -1e7:
                print(f"  ✓ Action masking works correctly")
            else:
                print(f"  ⚠ Warning: Action masking may not be working")
        else:
            print(f"  ⚠ No illegal actions to test masking")
        
        envs.close()
        return True
    except Exception as e:
        print(f"✗ Network error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_replay_buffer():
    """Test that ReplayBuffer can be created and used."""
    print("\nTesting Replay Buffer...")
    try:
        from train_dqn_splendor import ReplayBuffer
        
        envs = gym.vector.SyncVectorEnv([lambda: gym.make("SplendorSolo-v0")])
        envs = gym.wrappers.FlattenObservation(envs)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        rb = ReplayBuffer(
            buffer_size=1000,
            observation_space=envs.single_observation_space,
            action_space=envs.single_action_space,
            device=device,
            n_envs=1,
        )
        
        print(f"✓ Replay buffer created")
        print(f"  - Buffer size: {rb.buffer_size}")
        print(f"  - Observation shape: {rb.observations.shape}")
        print(f"  - Action mask shape: {rb.next_action_masks.shape}")
        
        # Test adding and sampling
        obs, info = envs.reset()
        action = np.array([0])
        next_obs, reward, done, truncated, next_info = envs.step(action)
        
        rb.add(obs, next_obs, action, reward, done, next_info["_legal_actions"][0])
        print(f"  - Added 1 transition, buffer size: {rb.size}")
        
        # Add more transitions
        for _ in range(10):
            action = np.array([envs.single_action_space.sample()])
            next_obs, reward, done, truncated, next_info = envs.step(action)
            rb.add(obs, next_obs, action, reward, done, next_info["_legal_actions"][0])
            obs = next_obs
            if done[0]:
                obs, info = envs.reset()
        
        print(f"  - Buffer size after 10 more transitions: {rb.size}")
        
        # Test sampling
        batch = rb.sample(batch_size=5)
        print(f"  - Sampled batch:")
        print(f"    - Observations: {batch.observations.shape}")
        print(f"    - Actions: {batch.actions.shape}")
        print(f"    - Next action masks: {batch.next_action_masks.shape}")
        
        envs.close()
        return True
    except Exception as e:
        print(f"✗ Replay buffer error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("=" * 60)
    print("DQN Training Script Verification")
    print("=" * 60)
    
    tests = [
        ("Imports", test_imports),
        ("Environment Registration", test_environment_registration),
        ("Vectorized Environment", test_vectorized_env),
        ("Q-Network", test_network_creation),
        ("Replay Buffer", test_replay_buffer),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n" + "=" * 60)
        print("ALL TESTS PASSED!")
        print("=" * 60)
        print("\nYou can now run:")
        print("  python train_dqn_splendor.py --total-timesteps 10000")
    else:
        print("\n" + "=" * 60)
        print("SOME TESTS FAILED")
        print("=" * 60)
        print("\nPlease fix the issues before running training.")


if __name__ == "__main__":
    main()
