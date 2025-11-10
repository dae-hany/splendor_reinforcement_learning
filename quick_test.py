"""
Quick test to verify training setup works
"""
import gymnasium as gym
import numpy as np
import torch

# Register environment
gym.register(
    id="SplendorSolo-v0",
    entry_point="splendor_solo_env:SplendorSoloEnv",
)

print("Testing basic training setup...")

# Create environment
envs = gym.vector.SyncVectorEnv([lambda: gym.make("SplendorSolo-v0")])
envs = gym.wrappers.FlattenObservation(envs)

obs, info = envs.reset(seed=42)

print(f"\n1. Initial observation shape: {obs.shape}")
print(f"2. Info dict keys: {list(info.keys())}")

# Check legal actions
if "legal_actions" in info:
    la = info["legal_actions"]
    print(f"3. Found 'legal_actions': type={type(la)}")
    if isinstance(la, (tuple, list)) and len(la) > 0:
        print(f"   First element: type={type(la[0])}, value={la[0]}")
        if isinstance(la[0], np.ndarray):
            print(f"   Shape: {la[0].shape}, sum: {np.sum(la[0])}")

# Try stepping
print(f"\n4. Taking action 0...")
try:
    next_obs, reward, terminated, truncated, info = envs.step(np.array([0]))
    print(f"   ✓ Step successful")
    print(f"   Reward: {reward}")
    print(f"   Terminated: {terminated}")
    print(f"   Info keys: {list(info.keys())}")
    
    if "legal_actions" in info:
        la = info["legal_actions"]
        if isinstance(la, (tuple, list)) and len(la) > 0 and isinstance(la[0], np.ndarray):
            print(f"   Legal actions sum: {np.sum(la[0])}")
except Exception as e:
    print(f"   ✗ Step failed: {e}")
    import traceback
    traceback.print_exc()

envs.close()
print("\nTest complete!")
