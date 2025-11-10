# DQN Training Script for Splendor Solo Environment

## Overview

This script (`train_dqn_splendor.py`) implements a Deep Q-Network (DQN) agent using the CleanRL methodology with proper **action masking** support for the Splendor Solo environment.

## Features

✅ **Action Masking**: Properly handles illegal actions using the `legal_actions` mask from the environment  
✅ **CleanRL Methodology**: Follows CleanRL best practices for reproducibility  
✅ **Two Modes**: Training mode (default) and Evaluation mode  
✅ **TensorBoard Logging**: Track training progress with TensorBoard  
✅ **Model Saving**: Automatically saves trained models  
✅ **W&B Integration**: Optional Weights & Biases tracking  

## Installation

```bash
pip install gymnasium numpy pandas torch tensorboard
```

## Usage

### 1. Training Mode (Default)

Train a DQN agent on the Splendor Solo environment:

```bash
python train_dqn_splendor.py
```

**Common training arguments:**

```bash
# Train for 1M timesteps with custom learning rate
python train_dqn_splendor.py --total-timesteps 1000000 --learning-rate 1e-4

# Train with W&B tracking
python train_dqn_splendor.py --track --wandb-project-name my-splendor-project

# Train with custom hyperparameters
python train_dqn_splendor.py \
    --total-timesteps 500000 \
    --learning-rate 2.5e-4 \
    --buffer-size 50000 \
    --batch-size 128 \
    --gamma 0.99 \
    --start-e 1.0 \
    --end-e 0.05 \
    --exploration-fraction 0.5
```

**Key training parameters:**
- `--total-timesteps`: Total training steps (default: 500,000)
- `--learning-rate`: Optimizer learning rate (default: 2.5e-4)
- `--buffer-size`: Replay buffer size (default: 10,000)
- `--batch-size`: Batch size for training (default: 128)
- `--gamma`: Discount factor (default: 0.99)
- `--start-e`: Starting epsilon for exploration (default: 1.0)
- `--end-e`: Ending epsilon for exploration (default: 0.05)
- `--exploration-fraction`: Fraction of training for epsilon decay (default: 0.5)
- `--learning-starts`: Steps before training starts (default: 10,000)
- `--train-frequency`: Steps between training updates (default: 10)

### 2. Evaluation Mode

Evaluate a trained model:

```bash
python train_dqn_splendor.py \
    --evaluate \
    --model-path runs/SplendorSolo-v0__train_dqn_splendor__1__1234567890/SplendorSolo-v0.pth \
    --num-eval-episodes 20
```

**Evaluation parameters:**
- `--evaluate`: Enable evaluation mode
- `--model-path`: Path to saved model `.pth` file (required in eval mode)
- `--num-eval-episodes`: Number of episodes to evaluate (default: 10)

### 3. Monitor Training with TensorBoard

```bash
tensorboard --logdir runs
```

Then open http://localhost:6006 in your browser.

## Implementation Details

### Action Masking

The script implements proper action masking at three critical points:

1. **Action Selection During Training:**
   - **Epsilon-greedy exploration**: Random selection from legal actions only
   - **Greedy exploitation**: Q-network forward pass with action mask

2. **Target Q-Value Calculation:**
   - Uses `next_action_masks` stored in replay buffer
   - Ensures target network only considers legal actions

3. **Evaluation:**
   - Deterministic greedy policy with action masking
   - No illegal actions taken during evaluation

### Q-Network Architecture

```
Input (Flattened Observation) 
    ↓
Linear(obs_dim, 256) + ReLU
    ↓
Linear(256, 256) + ReLU
    ↓
Linear(256, 39) → Q-values
    ↓
[Optional] Apply Action Mask (set illegal actions to -1e8)
    ↓
Output Q-values
```

### Replay Buffer

Custom replay buffer that stores:
- Observations
- Actions
- Rewards
- Next observations
- Done flags
- **Next action masks** (for proper target Q-value calculation)

## File Structure

After training, you'll have:

```
runs/
└── SplendorSolo-v0__train_dqn_splendor__1__1234567890/
    ├── events.out.tfevents.xyz  # TensorBoard logs
    └── SplendorSolo-v0.pth      # Saved model
```

## Example Training Session

```bash
# Train for 500k steps
python train_dqn_splendor.py --total-timesteps 500000 --seed 42

# Output:
# global_step=1000, episodic_return=12.0
# global_step=2000, episodic_return=15.0
# global_step=3000, episodic_return=18.0
# ...
# Model saved to runs/SplendorSolo-v0__train_dqn_splendor__42__1234567890/SplendorSolo-v0.pth
```

## Example Evaluation Session

```bash
# Evaluate the trained model
python train_dqn_splendor.py \
    --evaluate \
    --model-path runs/SplendorSolo-v0__train_dqn_splendor__42__1234567890/SplendorSolo-v0.pth \
    --num-eval-episodes 20

# Output:
# ============================================================
# Evaluating model: runs/.../SplendorSolo-v0.pth
# ============================================================
#
# Episode 1/20: Return = 25.00, Steps = 17
# Episode 2/20: Return = 22.00, Steps = 17
# ...
# Episode 20/20: Return = 28.00, Steps = 17
#
# ============================================================
# Evaluation Results:
# ============================================================
# Mean Return: 24.50 +/- 2.34
# Mean Episode Length: 17.00
# Min/Max Return: 20.00 / 28.00
# ============================================================
```

## Troubleshooting

### Issue: "No legal actions available"
This shouldn't happen if the environment is correct, but the script has a fallback to random action selection.

### Issue: Model not saving
Make sure the `runs/` directory exists and you have write permissions.

### Issue: CUDA out of memory
The default network is small, but if you encounter issues:
```bash
python train_dqn_splendor.py --cuda False
```

### Issue: Training too slow
- Reduce `--total-timesteps`
- Increase `--train-frequency`
- Reduce `--buffer-size`

## Advanced Usage

### Custom Network Architecture

Edit the `QNetwork` class in `train_dqn_splendor.py` to modify the architecture:

```python
self.network = nn.Sequential(
    nn.Linear(obs_shape, 512),  # Larger hidden layer
    nn.ReLU(),
    nn.Linear(512, 512),
    nn.ReLU(),
    nn.Linear(512, n_actions),
)
```

### Hyperparameter Tuning

Key hyperparameters to tune:
1. **Learning Rate** (`--learning-rate`): Try [1e-4, 2.5e-4, 5e-4]
2. **Buffer Size** (`--buffer-size`): Try [10000, 50000, 100000]
3. **Batch Size** (`--batch-size`): Try [64, 128, 256]
4. **Exploration Fraction** (`--exploration-fraction`): Try [0.3, 0.5, 0.7]

## Citation

If you use this code, please cite:

- **CleanRL**: Huang et al., "CleanRL: High-quality Single-file Implementations of Deep Reinforcement Learning Algorithms"
- **Splendor**: Space Cowboys (original board game)

## License

This implementation follows the CleanRL methodology and is provided for educational purposes.
