"""
Debug script to check info dict structure with FlattenObservation wrapper
"""
import gymnasium as gym
import numpy as np

# Register environment
gym.register(
    id="SplendorSolo-v0",
    entry_point="splendor_solo_env:SplendorSoloEnv",
)

print("=" * 60)
print("Testing Info Dict Structure")
print("=" * 60)

# Test without wrapper
print("\n1. WITHOUT FlattenObservation wrapper:")
env = gym.make("SplendorSolo-v0")
obs, info = env.reset(seed=42)
print(f"   Info keys: {list(info.keys())}")
if 'legal_actions' in info:
    print(f"   legal_actions shape: {np.array(info['legal_actions']).shape}")
    print(f"   legal_actions type: {type(info['legal_actions'])}")
    print(f"   Number of legal actions: {np.sum(info['legal_actions'])}")
env.close()

# Test with vectorized environment
print("\n2. WITH SyncVectorEnv:")
envs = gym.vector.SyncVectorEnv([lambda: gym.make("SplendorSolo-v0")])
obs, info = envs.reset(seed=42)
print(f"   Info keys: {list(info.keys())}")
for key in info.keys():
    val = info[key]
    print(f"   {key}: type={type(val)}, ", end="")
    if isinstance(val, (tuple, list)):
        print(f"len={len(val)}, ", end="")
        if len(val) > 0:
            print(f"first_elem_type={type(val[0])}, ", end="")
            if isinstance(val[0], np.ndarray):
                print(f"first_elem_shape={val[0].shape}")
            else:
                print(f"first_elem={val[0]}")
        else:
            print()
    elif isinstance(val, np.ndarray):
        print(f"shape={val.shape}")
    else:
        print(f"value={val}")
envs.close()

# Test with FlattenObservation
print("\n3. WITH SyncVectorEnv + FlattenObservation:")
envs = gym.vector.SyncVectorEnv([lambda: gym.make("SplendorSolo-v0")])
envs = gym.wrappers.FlattenObservation(envs)
obs, info = envs.reset(seed=42)
print(f"   Info keys: {list(info.keys())}")
for key in info.keys():
    val = info[key]
    print(f"   {key}: type={type(val)}, ", end="")
    if isinstance(val, (tuple, list)):
        print(f"len={len(val)}, ", end="")
        if len(val) > 0:
            print(f"first_elem_type={type(val[0])}, ", end="")
            if isinstance(val[0], np.ndarray):
                print(f"first_elem_shape={val[0].shape}, sum={np.sum(val[0])}")
            else:
                print(f"first_elem={val[0]}")
        else:
            print()
    elif isinstance(val, np.ndarray):
        print(f"shape={val.shape}, sum={np.sum(val)}")
    else:
        print(f"value={val}")

# Try to extract legal actions
print("\n4. Extracting legal actions:")
if "_legal_actions" in info:
    print("   Found '_legal_actions'")
    la = info["_legal_actions"]
    print(f"   Type: {type(la)}")
    if isinstance(la, tuple) and len(la) > 0:
        la_array = la[0]
        print(f"   First element type: {type(la_array)}")
        if isinstance(la_array, np.ndarray):
            print(f"   Shape: {la_array.shape}")
            print(f"   Sum: {np.sum(la_array)}")
            print(f"   Sample values: {la_array[:10]}")
        else:
            print(f"   Value: {la_array}")
elif "legal_actions" in info:
    print("   Found 'legal_actions'")
    print(f"   Value: {info['legal_actions']}")
else:
    print("   ⚠ No legal_actions key found!")

# Test a step
print("\n5. After taking a step:")
action = np.array([0])  # Take first action
obs, reward, terminated, truncated, info = envs.step(action)
print(f"   Info keys: {list(info.keys())}")
if "_legal_actions" in info:
    la = info["_legal_actions"]
    if isinstance(la, tuple) and len(la) > 0:
        la_array = la[0]
        if isinstance(la_array, np.ndarray):
            print(f"   legal_actions shape: {la_array.shape}, sum: {np.sum(la_array)}")

envs.close()

print("\n" + "=" * 60)
print("Debug Complete")
print("=" * 60)
