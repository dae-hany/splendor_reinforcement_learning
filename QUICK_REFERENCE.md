# Quick Reference: DQN Training for Splendor

## 🚀 Quick Start

### 1. Verify Setup
```bash
python test_dqn_setup.py
```

### 2. Train (Quick Test)
```bash
python train_dqn_splendor.py --total-timesteps 10000
```

### 3. Train (Full)
```bash
python train_dqn_splendor.py --total-timesteps 500000
```

### 4. Evaluate
```bash
python train_dqn_splendor.py --evaluate --model-path runs/[RUN_NAME]/SplendorSolo-v0.pth
```

## 📊 Monitor Training

```bash
tensorboard --logdir runs
```

## 🔑 Key Features

### Action Masking Implementation

**1. During Epsilon-Greedy Selection:**
```python
if random.random() < epsilon:
    # Random LEGAL action
    legal_actions = np.where(mask == 1)[0]
    action = np.random.choice(legal_actions)
else:
    # Best LEGAL action
    q_values = q_network(obs, action_mask=mask)
    action = q_values.argmax()
```

**2. During Training (Target Q-Values):**
```python
# Use next_action_masks from replay buffer
target_q_values = target_network(
    data.next_observations, 
    action_mask=data.next_action_masks
)
target_max, _ = target_q_values.max(dim=1)
```

**3. During Evaluation:**
```python
# Deterministic greedy policy with mask
q_values = q_network(obs, action_mask=mask)
action = q_values.argmax()
```

## 📁 File Structure

```
splendor_reinforcement_learning/
├── splendor_solo_env.py          # Environment
├── train_dqn_splendor.py         # Training script ⭐
├── test_dqn_setup.py              # Verification script
├── DQN_TRAINING_GUIDE.md          # Full documentation
├── cards.csv                      # Game data
├── noble_tiles.csv                # Game data
└── runs/                          # Training outputs
    └── [RUN_NAME]/
        ├── events.out.tfevents    # TensorBoard logs
        └── SplendorSolo-v0.pth    # Saved model
```

## ⚙️ Common Commands

### Training Variations

```bash
# Short training run (testing)
python train_dqn_splendor.py --total-timesteps 10000

# Standard training
python train_dqn_splendor.py --total-timesteps 500000

# Long training with tracking
python train_dqn_splendor.py --total-timesteps 1000000 --track

# Custom hyperparameters
python train_dqn_splendor.py \
    --total-timesteps 500000 \
    --learning-rate 1e-4 \
    --buffer-size 50000 \
    --batch-size 256 \
    --gamma 0.99

# CPU only (if GPU issues)
python train_dqn_splendor.py --cuda False
```

### Evaluation Commands

```bash
# Evaluate with 10 episodes (default)
python train_dqn_splendor.py \
    --evaluate \
    --model-path runs/[RUN_NAME]/SplendorSolo-v0.pth

# Evaluate with 50 episodes
python train_dqn_splendor.py \
    --evaluate \
    --model-path runs/[RUN_NAME]/SplendorSolo-v0.pth \
    --num-eval-episodes 50
```

## 🎯 Expected Results

### Training Progress
- **Initial episodes**: 5-15 points (mostly random actions)
- **After 100k steps**: 15-25 points (learning basic strategies)
- **After 500k steps**: 20-30 points (competent play)

### Game Clock
- All episodes end at exactly 17 steps (game clock limit)
- Focus is on maximizing points within 17 turns

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Import errors | `pip install gymnasium torch tensorboard pandas numpy` |
| Environment not found | Check `splendor_solo_env.py` is in same directory |
| CUDA memory error | Add `--cuda False` |
| No legal actions error | Verify environment's `legal_actions` in info dict |
| Training too slow | Reduce `--total-timesteps` or increase `--train-frequency` |

## 📈 Hyperparameter Tuning Guide

### Learning Rate (`--learning-rate`)
- **Too high**: Training unstable, loss oscillates
- **Too low**: Training too slow
- **Try**: [1e-4, 2.5e-4, 5e-4]

### Buffer Size (`--buffer-size`)
- **Too small**: Not enough diversity, overfitting
- **Too large**: Slower training, more memory
- **Try**: [10000, 50000, 100000]

### Exploration Fraction (`--exploration-fraction`)
- **Too short**: Not enough exploration
- **Too long**: Suboptimal policy for too long
- **Try**: [0.3, 0.5, 0.7]

### Batch Size (`--batch-size`)
- **Too small**: Noisy gradients
- **Too large**: Less frequent updates
- **Try**: [64, 128, 256]

## 💡 Tips

1. **Start small**: Test with `--total-timesteps 10000` first
2. **Monitor TensorBoard**: Watch episodic returns and TD loss
3. **Save models**: Default is `--save-model True`
4. **Multiple seeds**: Train with different `--seed` values
5. **Action masking is critical**: Without it, illegal actions crash the game

## 📝 Important Notes

- **Action Space**: 39 discrete actions
- **Observation Space**: Flattened Dict (122 features total)
- **Game Length**: Always 17 steps (game clock)
- **Max Tokens**: Player can have max 10 tokens
- **Legal Actions**: Always provided in `info["_legal_actions"]`

## 🎓 Next Steps

After successful training:

1. **Analyze results**: Check TensorBoard for learning curves
2. **Evaluate multiple times**: Run evaluation with different seeds
3. **Compare strategies**: Train with different hyperparameters
4. **Extend**: Try PPO, A2C, or other algorithms
5. **Optimize**: Experiment with network architecture

---

**Ready to train?**

```bash
python test_dqn_setup.py && python train_dqn_splendor.py --total-timesteps 10000
```
