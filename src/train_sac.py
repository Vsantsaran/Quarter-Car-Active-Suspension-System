"""
SAC Training Script for Active Suspension Control
All hyperparameters passed via command line arguments
"""

import os
import sys
import argparse
import time
import json
import torch
import numpy as np
from pathlib import Path
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from env.suspension_env import ActiveSuspensionEnv
from utils.rewards import RewardFunctions, RewardWrapper
from utils.logger import ExperimentLogger
from utils.metrics import SuspensionMetrics


class TrainingCallback(BaseCallback):
    """Custom callback for tracking training metrics"""
    
    def __init__(self, eval_env, logger, eval_freq=10000, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.logger = logger
        self.eval_freq = eval_freq
        self.episode_rewards = []
        self.episode_rms_values = []
        self.best_rms = float('inf')
        self.training_data = {
            'timesteps': [],
            'rewards': [],
            'rms_values': []
        }
    
    def _on_step(self) -> bool:
        # Evaluate periodically
        if self.n_calls % self.eval_freq == 0:
            self._evaluate()
        return True
    
    def _evaluate(self):
        """Evaluate current policy"""
        obs, _ = self.eval_env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = self.eval_env.step(action)
            episode_reward += reward
            done = terminated or truncated
        
        rms = info.get('acceleration_rms', 0)
        self.episode_rewards.append(episode_reward)
        self.episode_rms_values.append(rms)
        
        # Store training data
        self.training_data['timesteps'].append(self.n_calls)
        self.training_data['rewards'].append(episode_reward)
        self.training_data['rms_values'].append(rms)
        
        # Log
        self.logger.info(
            f"Step {self.n_calls:>10,} | Reward: {episode_reward:>8.2f} | "
            f"RMS: {rms:.6f} m/s²"
        )
        
        # Check for best model
        if rms < self.best_rms:
            self.best_rms = rms
            self.logger.info(f"  → New best RMS: {rms:.6f} m/s²")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train SAC for Active Suspension')
    
    # Environment parameters
    parser.add_argument('--road_class', type=str, default='D',
                       help='ISO 8608 road class (A-H)')
    parser.add_argument('--vehicle_speed', type=float, default=20.0,
                       help='Vehicle speed (m/s)')
    parser.add_argument('--dt', type=float, default=0.01,
                       help='Time step (s)')
    parser.add_argument('--max_steps', type=int, default=1500,
                       help='Maximum steps per episode')
    
    # SAC hyperparameters
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--buffer_size', type=int, default=100000,
                       help='Replay buffer size')
    parser.add_argument('--learning_starts', type=int, default=1000,
                       help='Steps before training starts')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--tau', type=float, default=0.005,
                       help='Soft update coefficient')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--train_freq', type=int, default=1,
                       help='Training frequency')
    parser.add_argument('--gradient_steps', type=int, default=1,
                       help='Gradient steps per update')
    parser.add_argument('--ent_coef', type=str, default='auto',
                       help='Entropy coefficient (auto or float)')
    parser.add_argument('--target_entropy', type=str, default='auto',
                       help='Target entropy (auto or float)')
    
    # Network architecture
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden layer dimension')
    parser.add_argument('--n_layers', type=int, default=2,
                       help='Number of hidden layers')
    
    # Training parameters
    parser.add_argument('--total_timesteps', type=int, default=10000000,
                       help='Total training timesteps')
    parser.add_argument('--eval_freq', type=int, default=10000,
                       help='Evaluation frequency')
    parser.add_argument('--checkpoint_freq', type=int, default=4000000,
                       help='Checkpoint frequency')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed')
    
    # Reward function
    parser.add_argument('--reward_id', type=int, default=1,
                       help='Reward function ID (1-8)')
    
    # GPU settings
    parser.add_argument('--gpu', type=int, default=None,
                       help='GPU ID to use (4-7)')
    
    # Output directories
    parser.add_argument('--log_dir', type=str, default='experiments/logs',
                       help='Log directory')
    parser.add_argument('--model_dir', type=str, default='experiments/models',
                       help='Model save directory')
    parser.add_argument('--result_dir', type=str, default='experiments/results',
                       help='Results directory')
    
    # Experiment name
    parser.add_argument('--exp_name', type=str, default=None,
                       help='Experiment name (auto-generated if None)')
    
    return parser.parse_args()


def setup_gpu(gpu_id):
    """Setup GPU device"""
    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        device = 'cuda'
    else:
        device = 'cpu'
    return device


def create_experiment_name(args):
    """Create experiment name from parameters"""
    if args.exp_name:
        return args.exp_name
    
    name = f"sac_road{args.road_class}_reward{args.reward_id}"
    name += f"_lr{args.learning_rate:.0e}"
    name += f"_buf{args.buffer_size//1000}k"
    name += f"_bs{args.batch_size}"
    
    if args.seed is not None:
        name += f"_seed{args.seed}"
    
    return name


def train_sac(args):
    """Main training function"""
    
    # Setup GPU
    device = setup_gpu(args.gpu)
    
    # Create experiment name
    exp_name = create_experiment_name(args)
    
    # Setup logger
    logger = ExperimentLogger(args.log_dir, exp_name)
    logger.info("="*80)
    logger.info("SAC TRAINING FOR ACTIVE SUSPENSION CONTROL")
    logger.info("="*80)
    
    # Log hyperparameters
    hyperparams = vars(args)
    hyperparams['device'] = device
    logger.log_hyperparameters(hyperparams)
    
    # Set random seeds
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
    
    # Create environment
    logger.info("Creating environment...")
    env = ActiveSuspensionEnv(
        road_class=args.road_class,
        dt=args.dt,
        max_steps=args.max_steps,
        vehicle_speed=args.vehicle_speed,
        seed=args.seed
    )
    
    # Get reward function
    reward_fn = RewardFunctions.get_reward_function(args.reward_id)
    reward_desc = RewardFunctions.get_reward_description(args.reward_id)
    logger.info(f"Using reward function {args.reward_id}: {reward_desc}")
    
    # Wrap environment with custom reward
    env = RewardWrapper(env, reward_fn)
    
    # Store physical parameters before wrapping
    m_s = env.env.m_s
    m_us = env.env.m_us
    k_s = env.env.k_s
    b_s = env.env.b_s
    
    # Wrap with Monitor and vectorize
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])
    
    # Evaluation environment
    eval_env = ActiveSuspensionEnv(
        road_class=args.road_class,
        dt=args.dt,
        max_steps=args.max_steps,
        vehicle_speed=args.vehicle_speed,
        seed=42  # Fixed seed for evaluation
    )
    eval_env = RewardWrapper(eval_env, reward_fn)
    
    logger.info(f"Environment: Road class {args.road_class} ({eval_env.env.get_road_roughness_description()})")
    logger.info(f"Physical parameters:")
    logger.info(f"  Sprung mass (m_s):       {m_s} kg")
    logger.info(f"  Unsprung mass (m_us):    {m_us} kg")
    logger.info(f"  Spring stiffness (k_s):  {k_s} N/m")
    logger.info(f"  Damping (b_s):           {b_s} N·s/m")
    
    # Create SAC model
    logger.info("Creating SAC model...")
    
    # Parse entropy coefficient
    if args.ent_coef == 'auto':
        ent_coef = 'auto'
    else:
        ent_coef = float(args.ent_coef)
    
    # Parse target entropy
    if args.target_entropy == 'auto':
        target_entropy = 'auto'
    else:
        target_entropy = float(args.target_entropy)
    
    # Network architecture
    net_arch = [args.hidden_dim] * args.n_layers
    
    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        tau=args.tau,
        gamma=args.gamma,
        train_freq=args.train_freq,
        gradient_steps=args.gradient_steps,
        ent_coef=ent_coef,
        target_entropy=target_entropy,
        policy_kwargs=dict(
            net_arch=net_arch,
            activation_fn=torch.nn.ReLU,
            log_std_init=-3
        ),
        verbose=0,
        device=device,
        seed=args.seed
    )
    
    logger.info(f"SAC model created on device: {device}")
    logger.info(f"Network architecture: {net_arch}")
    
    # Setup callbacks
    logger.info("Setting up callbacks...")
    
    # Training callback
    training_callback = TrainingCallback(
        eval_env=eval_env,
        logger=logger,
        eval_freq=args.eval_freq,
        verbose=1
    )
    
    # Checkpoint callback
    checkpoint_dir = Path(args.model_dir) / exp_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(checkpoint_dir),
        name_prefix='sac_checkpoint',
        save_replay_buffer=True,
        save_vecnormalize=True
    )
    
    callbacks = [training_callback, checkpoint_callback]
    
    # Start training
    logger.log_training_start(args.total_timesteps)
    start_time = time.time()
    
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callbacks,
            log_interval=100,
            progress_bar=True
        )
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise
    
    # Training completed
    total_time = time.time() - start_time
    logger.log_training_end(total_time)
    
    # Save final model
    final_model_path = checkpoint_dir / 'sac_final.zip'
    model.save(final_model_path)
    logger.info(f"Final model saved to: {final_model_path}")
    
    # Save training data
    result_dir = Path(args.result_dir) / exp_name
    result_dir.mkdir(parents=True, exist_ok=True)
    
    training_data_path = result_dir / 'training_data.json'
    with open(training_data_path, 'w') as f:
        json.dump(training_callback.training_data, f, indent=2)
    logger.info(f"Training data saved to: {training_data_path}")
    
    # Final evaluation
    logger.info("Running final evaluation...")
    obs, _ = eval_env.reset()
    done = False
    
    accelerations = []
    displacements = []
    forces = []
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = eval_env.step(action)
        done = terminated or truncated
        
        accelerations.append(info['body_acceleration'])
        displacements.append(info['suspension_deflection'])
        forces.append(info['control_force'])
    
    # Calculate final metrics
    final_metrics = SuspensionMetrics.calculate_metrics(
        accelerations, displacements, forces, args.road_class
    )
    
    logger.log_evaluation(final_metrics)
    
    # Save final metrics
    metrics_path = result_dir / 'final_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(final_metrics, f, indent=2)
    logger.info(f"Final metrics saved to: {metrics_path}")
    
    # Compare with paper
    paper_comparison = SuspensionMetrics.calculate_paper_comparison(
        final_metrics['rms_acceleration']
    )
    logger.info("-" * 80)
    logger.info("COMPARISON WITH PAPER:")
    logger.info(f"  Target RMS:    {paper_comparison['paper_target']:.6f} m/s²")
    logger.info(f"  Achieved RMS:  {paper_comparison['achieved_rms']:.6f} m/s²")
    logger.info(f"  Status:        {paper_comparison['status']}")
    logger.info(f"  {paper_comparison['message']}")
    logger.info("-" * 80)
    
    logger.info("Training completed successfully!")
    logger.close()
    
    return model, final_metrics, training_callback.training_data


if __name__ == '__main__':
    args = parse_args()
    model, metrics, training_data = train_sac(args)
