"""
Minimal test of DQN training with just 100 steps
"""
import subprocess
import sys

print("=" * 60)
print("Running minimal DQN training test (100 steps)")
print("=" * 60)

result = subprocess.run([
    sys.executable,
    "train_dqn_splendor.py",
    "--total-timesteps", "100",
    "--learning-starts", "10",
    "--seed", "42"
], capture_output=False, text=True)

print("\n" + "=" * 60)
if result.returncode == 0:
    print("✓ Training completed successfully!")
else:
    print(f"✗ Training failed with exit code {result.returncode}")
print("=" * 60)
