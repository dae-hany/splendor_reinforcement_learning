"""
DQN Training Script for Splendor Solo Environment
Based on CleanRL methodology with Action Masking support.
"""

import argparse
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="splendor-rl",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="whether to save model into the `runs/{run_name}` folder")
    parser.add_argument("--upload-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to upload the saved model to huggingface")
    parser.add_argument("--hf-entity", type=str, default="",
        help="the user or org name of the model repository from the Hugging Face Hub")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="SplendorSolo-v0",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=500000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--buffer-size", type=int, default=10000,
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=0.99,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=1.,
        help="the target network update rate")
    parser.add_argument("--target-network-frequency", type=int, default=500,
        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=128,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, default=1,
        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.05,
        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.5,
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=10000,
        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=10,
        help="the frequency of training")
    
    # Evaluation arguments
    parser.add_argument("--evaluate", action="store_true", default=False,
        help="if toggled, run evaluation mode instead of training")
    parser.add_argument("--model-path", type=str, default="",
        help="path to the model file to load for evaluation")
    parser.add_argument("--num-eval-episodes", type=int, default=10,
        help="number of episodes to run during evaluation")
    
    args = parser.parse_args()
    return args


def make_env(env_id, seed, idx, capture_video, run_name):
    """Create a single environment instance."""
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk


class QNetwork(nn.Module):
    """Q-Network with action masking support."""
    
    def __init__(self, env):
        super().__init__()
        # Get flattened observation space size
        # After FlattenObservation wrapper, obs_space is a Box with shape (n,)
        if hasattr(env.single_observation_space, 'shape') and env.single_observation_space.shape:
            obs_shape = int(np.prod(env.single_observation_space.shape))
        else:
            # Fallback: calculate from original Dict space
            obs_shape = 115  # Known size for Splendor: 6+6+5+1+21+63+12+1
        
        n_actions = env.single_action_space.n
        
        self.network = nn.Sequential(
            nn.Linear(obs_shape, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, n_actions),
        )
    
    def forward(self, x, action_mask=None):
        """
        Forward pass with optional action masking.
        
        Args:
            x: Observations
            action_mask: Binary mask of legal actions (1=legal, 0=illegal)
        
        Returns:
            Q-values (with illegal actions masked to -inf if mask provided)
        """
        q_values = self.network(x)
        
        # Apply action mask if provided
        if action_mask is not None:
            q_values = q_values.masked_fill(action_mask == 0, -1e8)
        
        return q_values


class ReplayBuffer:
    """Replay buffer with action mask storage."""
    
    def __init__(self, buffer_size, observation_space, action_space, device, n_envs=1):
        self.buffer_size = buffer_size
        self.device = device
        self.n_envs = n_envs
        self.ptr = 0
        self.size = 0
        
        # Handle observation shape properly (after FlattenObservation, it's a tuple)
        if hasattr(observation_space, 'shape') and observation_space.shape:
            obs_shape = observation_space.shape
        else:
            obs_shape = (115,)  # Fallback for Splendor environment
        
        # Determine dtype
        if hasattr(observation_space, 'dtype'):
            obs_dtype = observation_space.dtype
        else:
            obs_dtype = np.float32
        
        self.observations = np.zeros((buffer_size, n_envs, *obs_shape), dtype=obs_dtype)
        self.next_observations = np.zeros((buffer_size, n_envs, *obs_shape), dtype=obs_dtype)
        self.actions = np.zeros((buffer_size, n_envs), dtype=np.int64)
        self.rewards = np.zeros((buffer_size, n_envs), dtype=np.float32)
        self.dones = np.zeros((buffer_size, n_envs), dtype=np.float32)
        
        # Store action masks (39 actions for Splendor)
        self.next_action_masks = np.zeros((buffer_size, n_envs, action_space.n), dtype=np.float32)
    
    def add(self, obs, next_obs, action, reward, done, next_action_mask):
        """Add a transition to the buffer."""
        self.observations[self.ptr] = np.array(obs).copy()
        self.next_observations[self.ptr] = np.array(next_obs).copy()
        self.actions[self.ptr] = np.array(action).copy()
        self.rewards[self.ptr] = np.array(reward).copy()
        self.dones[self.ptr] = np.array(done).copy()
        self.next_action_masks[self.ptr] = np.array(next_action_mask).copy()
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
    
    def sample(self, batch_size):
        """Sample a batch of transitions."""
        batch_inds = np.random.randint(0, self.size, size=batch_size)
        
        return SimpleNamespace(
            observations=torch.as_tensor(self.observations[batch_inds, 0], dtype=torch.float32, device=self.device),
            next_observations=torch.as_tensor(self.next_observations[batch_inds, 0], dtype=torch.float32, device=self.device),
            actions=torch.as_tensor(self.actions[batch_inds, 0], dtype=torch.int64, device=self.device),
            rewards=torch.as_tensor(self.rewards[batch_inds, 0], dtype=torch.float32, device=self.device),
            dones=torch.as_tensor(self.dones[batch_inds, 0], dtype=torch.float32, device=self.device),
            next_action_masks=torch.as_tensor(self.next_action_masks[batch_inds, 0], dtype=torch.float32, device=self.device),
        )


class SimpleNamespace:
    """Simple namespace for storing batch data."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    """Linear epsilon decay schedule."""
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def extract_legal_actions(info, expected_size=39, debug=False):
    """
    Extract legal actions mask from info dict in a robust way.
    
    Args:
        info: Info dict from environment step
        expected_size: Expected size of action mask (default 39 for Splendor)
        debug: If True, print debug information
    
    Returns:
        numpy array of shape (expected_size,) with legal action mask
    """
    # Try multiple possible keys (environment uses "legal_actions", VectorEnv may prefix with "_")
    legal_actions_raw = None
    for key in ["legal_actions", "_legal_actions", "action_mask"]:
        if key in info:
            legal_actions_raw = info[key]
            if debug:
                print(f"Found legal_actions with key: {key}")
                print(f"  Type: {type(legal_actions_raw)}")
                print(f"  Shape/Len: {legal_actions_raw.shape if isinstance(legal_actions_raw, np.ndarray) else len(legal_actions_raw) if hasattr(legal_actions_raw, '__len__') else 'N/A'}")
                if isinstance(legal_actions_raw, np.ndarray):
                    print(f"  Dtype: {legal_actions_raw.dtype}")
            break
    
    if legal_actions_raw is None:
        if debug:
            print(f"WARNING: No legal_actions found in info. Keys: {list(info.keys())}")
        # Fallback: assume all actions are legal
        return np.ones(expected_size, dtype=np.float32)
    
    # Handle VectorEnv wrapping: it wraps into numpy array of shape (1,) with dtype=object
    if isinstance(legal_actions_raw, np.ndarray) and legal_actions_raw.dtype == object:
        if legal_actions_raw.size == 1:
            # Extract the actual array from the wrapper
            legal_actions_array = legal_actions_raw[0]
            if debug:
                print(f"  Unwrapped from object array: {type(legal_actions_array)}")
        else:
            if debug:
                print(f"WARNING: Object array with size {legal_actions_raw.size}")
            legal_actions_array = legal_actions_raw
    # Handle tuple (from VectorEnv)
    elif isinstance(legal_actions_raw, tuple):
        if len(legal_actions_raw) > 0:
            legal_actions_array = legal_actions_raw[0]
        else:
            if debug:
                print(f"WARNING: Empty tuple for legal_actions")
            return np.ones(expected_size, dtype=np.float32)
    elif isinstance(legal_actions_raw, list):
        if len(legal_actions_raw) > 0:
            legal_actions_array = legal_actions_raw[0]
        else:
            if debug:
                print(f"WARNING: Empty list for legal_actions")
            return np.ones(expected_size, dtype=np.float32)
    else:
        legal_actions_array = legal_actions_raw
    
    # Debug before conversion
    if debug:
        print(f"  Before conversion - Type: {type(legal_actions_array)}")
        if isinstance(legal_actions_array, np.ndarray):
            print(f"    Shape: {legal_actions_array.shape}, Dtype: {legal_actions_array.dtype}")
    
    # Convert to numpy array
    if not isinstance(legal_actions_array, np.ndarray):
        try:
            legal_actions_array = np.array(legal_actions_array, dtype=np.float32)
        except Exception as e:
            if debug:
                print(f"WARNING: Could not convert to numpy array: {type(legal_actions_array)}, error: {e}")
            return np.ones(expected_size, dtype=np.float32)
    else:
        # Already numpy array, check if it needs flattening first
        if legal_actions_array.ndim > 1:
            if debug:
                print(f"  Flattening from shape {legal_actions_array.shape}")
            legal_actions_array = legal_actions_array.flatten()
        
        # Now ensure correct dtype
        if legal_actions_array.dtype != np.float32:
            legal_actions_array = legal_actions_array.astype(np.float32)
    
    # Ensure it's 1D (should already be handled above, but double-check)
    if legal_actions_array.ndim == 0:
        # Scalar value
        if debug:
            print(f"WARNING: Scalar legal_actions value: {legal_actions_array}")
        return np.ones(expected_size, dtype=np.float32)
    elif legal_actions_array.ndim > 1:
        if debug:
            print(f"WARNING: Still multi-dimensional after conversion: {legal_actions_array.shape}")
        legal_actions_array = legal_actions_array.flatten()
    
    # Verify size
    if legal_actions_array.size != expected_size:
        if debug:
            print(f"WARNING: legal_actions size mismatch: got {legal_actions_array.size}, expected {expected_size}")
        
        if legal_actions_array.size == 1:
            # Scalar value in array form
            return np.ones(expected_size, dtype=np.float32)
        elif legal_actions_array.size > expected_size:
            # Truncate
            return legal_actions_array[:expected_size].astype(np.float32)
        else:
            # Pad with ones
            padded = np.ones(expected_size, dtype=np.float32)
            padded[:legal_actions_array.size] = legal_actions_array
            return padded
    
    result = legal_actions_array.astype(np.float32)
    
    # Sanity check: ensure at least one legal action
    if debug and result.sum() == 0:
        print(f"WARNING: No legal actions in mask! Sum={result.sum()}")
        print(f"  Mask: {result}")
    
    if debug:
        num_legal = int(result.sum())
        legal_indices = np.where(result == 1)[0]
        print(f"  Extracted {num_legal} legal actions: {legal_indices[:10]}{'...' if num_legal > 10 else ''}")
    
    # Sanity check: should have at least one legal action
    if np.sum(result) == 0:
        if debug:
            print(f"WARNING: All actions marked as illegal! Defaulting to all legal.")
        return np.ones(expected_size, dtype=np.float32)
    
    return result


def run_evaluation(args, q_network, envs, device):
    """
    Run evaluation on a trained model.
    
    Args:
        args: Parsed arguments
        q_network: Q-network to evaluate
        envs: Vectorized environment
        device: Device to run on
    """
    print(f"\n{'='*60}")
    print(f"Evaluating model: {args.model_path}")
    print(f"{'='*60}\n")
    
    # Load model
    q_network.load_state_dict(torch.load(args.model_path, map_location=device))
    q_network.eval()
    
    episode_returns = []
    episode_lengths = []
    
    for episode in range(args.num_eval_episodes):
        obs, info = envs.reset()
        terminated = False
        truncated = False
        total_reward = 0
        steps = 0
        
        while not (terminated or truncated):
            # Get legal action mask from info
            legal_actions = extract_legal_actions(info)
            
            # Prepare observation tensor
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            if obs_tensor.dim() == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
            
            # Prepare mask tensor
            mask_tensor = torch.as_tensor(legal_actions, dtype=torch.float32, device=device)
            if mask_tensor.dim() == 1:
                mask_tensor = mask_tensor.unsqueeze(0)
            
            # Get Q-values with mask
            with torch.no_grad():
                q_values = q_network(obs_tensor, action_mask=mask_tensor)
            
            # Take deterministic (greedy) action
            if q_values.dim() == 1:
                action = np.array([q_values.argmax().item()])
            else:
                action = q_values.argmax(dim=1).cpu().numpy()
            
            # Step environment
            obs, reward, terminated, truncated, info = envs.step(action)
            
            # Handle termination flags (they come as arrays from VectorEnv)
            terminated = terminated[0] if isinstance(terminated, np.ndarray) else terminated
            truncated = truncated[0] if isinstance(truncated, np.ndarray) else truncated
            
            total_reward += reward[0]
            steps += 1
        
        episode_returns.append(total_reward)
        episode_lengths.append(steps)
        print(f"Episode {episode + 1}/{args.num_eval_episodes}: "
              f"Return = {total_reward:.2f}, Steps = {steps}")
    
    # Calculate and print statistics
    mean_return = np.mean(episode_returns)
    std_return = np.std(episode_returns)
    mean_length = np.mean(episode_lengths)
    
    print(f"\n{'='*60}")
    print(f"Evaluation Results:")
    print(f"{'='*60}")
    print(f"Mean Return: {mean_return:.2f} +/- {std_return:.2f}")
    print(f"Mean Episode Length: {mean_length:.2f}")
    print(f"Min/Max Return: {np.min(episode_returns):.2f} / {np.max(episode_returns):.2f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    # Register the Splendor environment
    gym.register(
        id="SplendorSolo-v0",
        entry_point="splendor_solo_env:SplendorSoloEnv",
    )
    
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    
    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )
    
    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    
    # Environment setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    envs = gym.wrappers.FlattenObservation(envs)
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"
    
    # Network setup
    q_network = QNetwork(envs).to(device)
    
    # Evaluation mode
    if args.evaluate:
        if not args.model_path:
            print("Error: --model-path must be specified for evaluation mode")
            exit(1)
        run_evaluation(args, q_network, envs, device)
        exit(0)
    
    # Training mode - setup optimizer and target network
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())
    
    # Replay buffer
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
    )
    
    # Training loop
    start_time = time.time()
    obs, info = envs.reset(seed=args.seed)
    
    for global_step in range(args.total_timesteps):
        # Calculate epsilon
        epsilon = linear_schedule(
            args.start_e,
            args.end_e,
            args.exploration_fraction * args.total_timesteps,
            global_step,
        )
        
        # Action selection with action masking
        # Enable debug for first few steps
        debug_mode = (global_step < 5)
        legal_actions_mask = extract_legal_actions(info, debug=debug_mode)
        
        if random.random() < epsilon:
            # Epsilon: Take a RANDOM LEGAL action
            legal_action_indices = np.where(legal_actions_mask == 1)[0]
            if len(legal_action_indices) == 0:
                # Fallback: no legal actions (shouldn't happen)
                print(f"WARNING: No legal actions at step {global_step}! Using random action.")
                actions = np.array([envs.single_action_space.sample()])
            else:
                actions = np.array([np.random.choice(legal_action_indices)])
        else:
            # Greedy: Take the BEST LEGAL action
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device)
            # Add batch dimension if needed
            if obs_tensor.dim() == 1:
                obs_tensor = obs_tensor.unsqueeze(0)
            
            mask_tensor = torch.as_tensor(legal_actions_mask, dtype=torch.float32, device=device)
            # Add batch dimension to mask if needed
            if mask_tensor.dim() == 1:
                mask_tensor = mask_tensor.unsqueeze(0)
            
            with torch.no_grad():
                q_values = q_network(obs_tensor, action_mask=mask_tensor)
            
            # Handle different q_values shapes
            if q_values.dim() == 1:
                actions = np.array([q_values.argmax().item()])
            else:
                actions = q_values.argmax(dim=1).cpu().numpy()
        
        # Debug action selection
        if debug_mode:
            print(f"  Selected action: {actions[0]}, epsilon={epsilon:.3f}")
        
        # Execute action
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        
        # Debug rewards
        if debug_mode:
            print(f"  Reward: {rewards[0]}, done: {terminations[0] or truncations[0]}")
        
        # Handle episode end
        if "final_info" in infos:
            for info_item in infos["final_info"]:
                if info_item and "episode" in info_item:
                    print(f"global_step={global_step}, episodic_return={info_item['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info_item["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info_item["episode"]["l"], global_step)
        
        # Store transition with next action mask
        next_action_mask = extract_legal_actions(infos)
        rb.add(obs, next_obs, actions, rewards, terminations, next_action_mask)
        
        # Update observation and info
        obs = next_obs
        info = infos
        
        # Training
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                
                # Compute target Q-values with action masking
                with torch.no_grad():
                    target_q_values = target_network(data.next_observations, action_mask=data.next_action_masks)
                    target_max, _ = target_q_values.max(dim=1)
                    td_target = data.rewards + args.gamma * target_max * (1 - data.dones)
                
                # Compute current Q-values
                current_q_values = q_network(data.observations)
                old_val = current_q_values.gather(1, data.actions.unsqueeze(-1)).squeeze()
                
                # Compute loss
                loss = F.mse_loss(old_val, td_target)
                
                # Logging
                if global_step % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)
                
                # Optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # Update target network
            if global_step % args.target_network_frequency == 0:
                for target_network_param, q_network_param in zip(target_network.parameters(), q_network.parameters()):
                    target_network_param.data.copy_(
                        args.tau * q_network_param.data + (1.0 - args.tau) * target_network_param.data
                    )
    
    # Save model
    if args.save_model:
        model_path = f"runs/{run_name}/{args.env_id}.pth"
        torch.save(q_network.state_dict(), model_path)
        print(f"\nModel saved to {model_path}")
    
    envs.close()
    writer.close()
