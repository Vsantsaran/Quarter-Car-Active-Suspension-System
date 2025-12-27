#!/usr/bin/env python
"""
=================================================================================
COMPLETE SAC ACTIVE SUSPENSION SYSTEM - ALL-IN-ONE SCRIPT
=================================================================================

This script contains the COMPLETE implementation:
1. Environment (ISO 8608 road classes A-H)
2. Reward functions (8 different formulations)
3. SAC training algorithm
4. Evaluation and comparison
5. Visualization and plotting

Can be run standalone or imported as modules.

Author: Active Suspension Research Team
Based on: Dridi et al. (2023)
=================================================================================
"""

import os
import sys
import argparse
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Deep learning imports
import torch
import torch.nn as nn
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

# Gymnasium
import gymnasium as gym
from gymnasium import spaces


# =============================================================================
# SECTION 1: ENVIRONMENT
# =============================================================================

class ActiveSuspensionEnv(gym.Env):
    """
    Active Suspension Environment with ISO 8608 Road Classes

    Quarter-car model (2-DOF):
    - Sprung mass (body): m_s = 300 kg
    - Unsprung mass (wheel): m_us = 30 kg
    - Electromagnetic actuator: U ∈ [-8000, 8000] N

    State: [x_s, ẋ_s, x_us, ẋ_us, x_r]
    Action: Normalized control force [-1, 1]
    """

    ROAD_CLASSES = {
        'A': 16e-6,      # Very good
        'B': 64e-6,      # Good
        'C': 256e-6,     # Average
        'D': 1024e-6,    # Poor (paper baseline)
        'E': 4096e-6,    # Very poor
        'F': 16384e-6,   # Bad
        'G': 65536e-6,   # Very bad
        'H': 262144e-6   # Extremely bad
    }

    metadata = {'render_modes': ['human']}

    def __init__(self, road_class='D', dt=0.01, max_steps=1500,
                 vehicle_speed=20.0, seed=None):
        super().__init__()

        # Physical parameters (Table 4 from paper)
        self.m_s = 300.0          # Sprung mass (kg)
        self.k_s = 40000.0        # Suspension stiffness (N/m)
        self.m_us = 30.0          # Unsprung mass (kg)
        self.b_s = 1385.0         # Suspension damping (N·s/m)
        self.k_us = 22000.0       # Tire stiffness (N/m)
        self.b_us = 100.0         # Tire damping (N·s/m)
        self.u_max = 8000.0       # Max actuator force (N)

        # Simulation parameters
        self.dt = dt
        self.max_steps = max_steps
        self.vehicle_speed = vehicle_speed

        # Road configuration
        self.road_class = road_class.upper()
        assert self.road_class in self.ROAD_CLASSES
        self.G_q0 = self.ROAD_CLASSES[self.road_class]
        self.n0 = 0.1
        self.w_index = 2.0

        # Random generator
        self._np_random = None
        if seed is not None:
            self.seed(seed)

        # Spaces
        self.observation_space = spaces.Box(
            low=np.array([-1.0, -10.0, -1.0, -10.0, -0.2], dtype=np.float32),
            high=np.array([1.0, 10.0, 1.0, 10.0, 0.2], dtype=np.float32),
            dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # State
        self.state = None
        self.steps = 0
        self.time = 0.0
        self.x_r = 0.0
        self.dx_r = 0.0
        self.accelerations = []
        self.displacements = []
        self.forces = []
        self.road_profile = None

        self._generate_road_profile()

    def seed(self, seed=None):
        self._np_random = np.random.RandomState(seed)
        return [seed]

    def _generate_road_profile(self):
        """Generate ISO 8608 road using PSD method"""
        total_distance = self.vehicle_speed * self.max_steps * self.dt
        dx = self.vehicle_speed * self.dt
        x = np.arange(0, total_distance, dx)

        N = len(x)
        df = 1.0 / total_distance
        frequencies = np.fft.rfftfreq(N, dx)
        frequencies[0] = 0.01  # Avoid zero

        # Power spectral density: G_q(n) = G_q0 * (n/n0)^(-w)
        G_q = self.G_q0 * (frequencies / self.n0) ** (-self.w_index)

        # Random phases
        if self._np_random is None:
            phases = np.random.uniform(0, 2*np.pi, len(frequencies))
        else:
            phases = self._np_random.uniform(0, 2*np.pi, len(frequencies))

        # Complex spectrum
        amplitude = np.sqrt(2 * G_q * df)
        spectrum = amplitude * np.exp(1j * phases)
        spectrum[0] = 0  # Zero mean

        # Inverse FFT
        road_elevation = np.fft.irfft(spectrum, N)

        self.road_profile = road_elevation
        self.road_distance = x

    def _get_road_input(self, time):
        """Get road elevation at current time"""
        distance = self.vehicle_speed * time
        idx = int(distance / (self.vehicle_speed * self.dt))
        idx = min(idx, len(self.road_profile) - 2)
    
        x_r = self.road_profile[idx]
    
        # Central difference (more accurate than forward difference)
        if 0 < idx < len(self.road_profile) - 1:
            dx_r = (self.road_profile[idx + 1] - self.road_profile[idx - 1]) / (2 * self.dt)
        elif idx == 0:
            # At boundary, use forward difference
            dx_r = (self.road_profile[1] - self.road_profile[0]) / self.dt
        else:
            # At end, use backward difference
            dx_r = (self.road_profile[-1] - self.road_profile[-2]) / self.dt
    
        return x_r, dx_r

    def _compute_derivatives(self, state, u, x_r, dx_r):
        """
        Equations of motion (Dridi et al. 2023):

        Sprung mass:
        m_s·ẍ_s = -b_s(ẋ_s - ẋ_us) - k_s(x_s - x_us) + U

        Unsprung mass:
        m_us·ẍ_us = b_s(ẋ_s - ẋ_us) + k_s(x_s - x_us)
                   + b_us(ẋ_r - ẋ_us) + k_us(x_r - x_us) - U
        """
        x_s, dx_s, x_us, dx_us = state

        ddx_s = (
            -self.b_s * (dx_s - dx_us)
            - self.k_s * (x_s - x_us)
            + u
        ) / self.m_s

        ddx_us = (
            self.b_s * (dx_s - dx_us)
            + self.k_s * (x_s - x_us)
            + self.b_us * (dx_r - dx_us)
            + self.k_us * (x_r - x_us)
            - u
        ) / self.m_us

        return np.array([dx_s, ddx_s, dx_us, ddx_us])

    def step(self, action):
        """Execute one timestep with RK4 integration"""
        u = np.clip(action[0] * self.u_max, -self.u_max, self.u_max)

        self.x_r, self.dx_r = self._get_road_input(self.time)

        # RK4 numerical integration
        k1 = self._compute_derivatives(self.state, u, self.x_r, self.dx_r)
        k2 = self._compute_derivatives(self.state + 0.5*self.dt*k1, u, self.x_r, self.dx_r)
        k3 = self._compute_derivatives(self.state + 0.5*self.dt*k2, u, self.x_r, self.dx_r)
        k4 = self._compute_derivatives(self.state + self.dt*k3, u, self.x_r, self.dx_r)

        self.state = self.state + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

        self.time += self.dt
        self.steps += 1

        x_s, dx_s, x_us, dx_us = self.state
        ddx_s = (
            -self.b_s * (dx_s - dx_us)
            - self.k_s * (x_s - x_us)
            + u
        ) / self.m_s

        self.accelerations.append(ddx_s)
        self.displacements.append(abs(x_s - x_us))
        self.forces.append(u)

        obs = np.array([x_s, dx_s, x_us, dx_us, self.x_r], dtype=np.float32)

        # Default reward (will be overridden by RewardWrapper)
        reward = self._default_reward(x_s, dx_s, x_us, dx_us, ddx_s, u)

        terminated = False
        truncated = self.steps >= self.max_steps

        info = {
            # 'acceleration_rms': np.sqrt(np.mean(np.array(self.accelerations)**2)),
            'body_acceleration': ddx_s,
            'suspension_deflection': abs(x_s - x_us),
            'tire_deflection': abs(x_us - self.x_r),
            'control_force': u,
            'road_class': self.road_class,
            'time': self.time
        }

        return obs, reward, terminated, truncated, info

    def _default_reward(self, x_s, dx_s, x_us, dx_us, ddx_s, u):
        """Default reward function"""
        tracking_error = (x_s - self.x_r) ** 2
        tire_contact = (x_us - self.x_r) ** 2
        comfort_cost = ddx_s ** 2
        control_cost = (u / self.u_max) ** 2

        return -(10.0 * tracking_error + 1.0 * tire_contact +
                 1.0 * comfort_cost + 0.00001 * control_cost)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if seed is not None:
            self.seed(seed)

        self._generate_road_profile()

        self.state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.time = 0.0
        self.steps = 0
        self.x_r = 0.0
        self.dx_r = 0.0
        self.accelerations = []
        self.displacements = []
        self.forces = []

        obs = np.array([*self.state, self.x_r], dtype=np.float32)
        return obs, {'road_class': self.road_class}

    def render(self):
        pass

    def close(self):
        pass


# =============================================================================
# SECTION 2: REWARD FUNCTIONS
# =============================================================================

class RewardFunctions:
    """8 reward functions for different objectives"""

    @staticmethod
    def reward_1(x_s, dx_s, x_us, dx_us, ddx_s, u, x_r, u_max=8000.0):
        """Balanced (paper baseline)"""
        return -(10.0 * (x_s - x_r)**2 + 1.0 * (x_us - x_r)**2 +
                 1.0 * ddx_s**2 + 0.00001 * (u/u_max)**2)

    @staticmethod
    def reward_2(x_s, dx_s, x_us, dx_us, ddx_s, u, x_r, u_max=8000.0):
        """Comfort-focused (high acceleration penalty)"""
        return -(5.0 * (x_s - x_r)**2 + 1.0 * (x_us - x_r)**2 +
                 20.0 * ddx_s**2 + 0.00001 * (u/u_max)**2)

    @staticmethod
    def reward_3(x_s, dx_s, x_us, dx_us, ddx_s, u, x_r, u_max=8000.0):
        """Road tracking"""
        return -(50.0 * (x_s - x_r)**2 + 5.0 * (x_us - x_r)**2 +
                 1.0 * ddx_s**2 + 0.00001 * (u/u_max)**2)

    @staticmethod
    def reward_4(x_s, dx_s, x_us, dx_us, ddx_s, u, x_r, u_max=8000.0):
        """Energy-efficient"""
        return -(10.0 * (x_s - x_r)**2 + 1.0 * (x_us - x_r)**2 +
                 5.0 * ddx_s**2 + 0.1 * (u/u_max)**2)

    @staticmethod
    def reward_5(x_s, dx_s, x_us, dx_us, ddx_s, u, x_r, u_max=8000.0):
        """Tire grip (maximize road contact)"""
        return -(10.0 * (x_s - x_r)**2 + 50.0 * (x_us - x_r)**2 +
                 5.0 * ddx_s**2 + 0.00001 * (u/u_max)**2)

    @staticmethod
    def reward_6(x_s, dx_s, x_us, dx_us, ddx_s, u, x_r, u_max=8000.0):
        """Velocity damping (minimize velocities)"""
        velocity_cost = dx_s**2 + dx_us**2
        return -(10.0 * (x_s - x_r)**2 + 1.0 * (x_us - x_r)**2 +
                 1.0 * ddx_s**2 + 5.0 * velocity_cost + 0.00001 * (u/u_max)**2)

    @staticmethod
    def reward_7(x_s, dx_s, x_us, dx_us, ddx_s, u, x_r, u_max=8000.0):
        """Suspension deflection (prevent bottoming)"""
        suspension_deflection = (x_s - x_us)**2
        return -(10.0 * (x_s - x_r)**2 + 1.0 * (x_us - x_r)**2 +
                 1.0 * ddx_s**2 + 10.0 * suspension_deflection + 0.00001 * (u/u_max)**2)

    @staticmethod
    def reward_8(x_s, dx_s, x_us, dx_us, ddx_s, u, x_r, u_max=8000.0):
        """Exponential comfort (non-linear acceleration penalty)"""
        comfort_cost = np.exp(abs(ddx_s) / 5.0) - 1.0
        return -(10.0 * (x_s - x_r)**2 + 1.0 * (x_us - x_r)**2 +
                 2.0 * comfort_cost + 0.00001 * (u/u_max)**2)

    @staticmethod
    def get_reward_function(reward_id):
        functions = {
            1: RewardFunctions.reward_1, 2: RewardFunctions.reward_2,
            3: RewardFunctions.reward_3, 4: RewardFunctions.reward_4,
            5: RewardFunctions.reward_5, 6: RewardFunctions.reward_6,
            7: RewardFunctions.reward_7, 8: RewardFunctions.reward_8
        }
        return functions[reward_id]


class RewardWrapper(gym.Wrapper):
    """Wrap environment with custom reward function"""
    def __init__(self, env, reward_function):
        super().__init__(env)  # ← KEY FIX
        self.reward_function = reward_function

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        x_s, dx_s, x_us, dx_us, x_r = obs
        reward = self.reward_function(
            x_s, dx_s, x_us, dx_us,
            info['body_acceleration'],
            info['control_force'],
            x_r, self.env.u_max
        )
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def __getattr__(self, name):
        return getattr(self.env, name)


# =============================================================================
# SECTION 3: TRAINING
# =============================================================================

def train_sac(
    road_class='D',
    reward_id=1,
    total_timesteps=10000000,
    learning_rate=3e-4,
    buffer_size=100000,
    batch_size=256,
    checkpoint_freq=4000000,
    gpu=4,
    seed=None,
    output_dir='experiments'
):
    """
    Train SAC agent

    Args:
        road_class: ISO 8608 road class (A-H)
        reward_id: Reward function ID (1-8)
        total_timesteps: Total training steps
        learning_rate: Learning rate
        buffer_size: Replay buffer size
        batch_size: Batch size
        checkpoint_freq: Checkpoint frequency
        gpu: GPU ID
        seed: Random seed
        output_dir: Output directory
    """

    print("="*80)
    print("SAC TRAINING - ACTIVE SUSPENSION")
    print("="*80)
    print(f"Road class: {road_class}")
    print(f"Reward ID: {reward_id}")
    print(f"Timesteps: {total_timesteps:,}")
    print(f"GPU: {gpu}")
    print("="*80)

    # Setup GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Create environment
    env = ActiveSuspensionEnv(road_class=road_class, seed=seed)
    reward_fn = RewardFunctions.get_reward_function(reward_id)
    env = RewardWrapper(env, reward_fn)
    env = Monitor(env)
    env = DummyVecEnv([lambda: env])

    # Create SAC model
    model = SAC(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        batch_size=batch_size,
        tau=0.005,
        gamma=0.99,
        policy_kwargs=dict(
            net_arch=[256, 256],
            activation_fn=torch.nn.ReLU
        ),
        verbose=1,
        device=device,
        seed=seed
    )

    # Setup checkpoints
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_name = f"sac_road{road_class}_reward{reward_id}_{timestamp}"
    checkpoint_dir = Path(output_dir) / 'models' / exp_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=str(checkpoint_dir),
        name_prefix='sac_checkpoint'
    )

    # Train
    print("\nTraining started...")
    start_time = time.time()

    model.learn(
        total_timesteps=total_timesteps,
        callback=checkpoint_callback,
        progress_bar=True
    )

    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time/3600:.2f} hours")

    # Save final model
    model_path = checkpoint_dir / 'sac_final.zip'
    model.save(model_path)
    print(f"Model saved: {model_path}")

    return model, str(model_path), exp_name


# =============================================================================
# SECTION 4: EVALUATION
# =============================================================================

def evaluate_model(model_path, road_class='D', reward_id=1, n_episodes=10, seed=42):
    """Evaluate trained model"""

    print("="*80)
    print("EVALUATION")
    print("="*80)
    print(f"Model: {model_path}")
    print(f"Road class: {road_class}")
    print(f"Episodes: {n_episodes}")
    print("="*80)

    # Load model
    model = SAC.load(model_path)

    # Create environment
    reward_fn = RewardFunctions.get_reward_function(reward_id)

    all_rms = []

    for ep in range(n_episodes):
        env = ActiveSuspensionEnv(road_class=road_class, seed=seed + ep)
        env = RewardWrapper(env, reward_fn)

        obs, _ = env.reset()
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        # rms = info['acceleration_rms']
        rms = np.sqrt(np.mean(np.array(env.env.accelerations)**2))
        all_rms.append(rms)
        print(f"Episode {ep+1}/{n_episodes}: RMS = {rms:.6f} m/s²")

    mean_rms = np.mean(all_rms)
    std_rms = np.std(all_rms)

    print("="*80)
    print(f"Mean RMS: {mean_rms:.6f} ± {std_rms:.6f} m/s²")
    print("="*80)

    return mean_rms, std_rms


# =============================================================================
# SECTION 5: COMPARISON
# =============================================================================

def compare_with_passive(model_path, road_class='D', reward_id=1, seed=42):
    """Compare active vs passive suspension"""

    print("="*80)
    print("ACTIVE VS PASSIVE COMPARISON")
    print("="*80)

    # Load model
    model = SAC.load(model_path)
    reward_fn = RewardFunctions.get_reward_function(reward_id)

    # Active suspension
    env_active = ActiveSuspensionEnv(road_class=road_class, seed=seed)
    env_active = RewardWrapper(env_active, reward_fn)

    obs, _ = env_active.reset()
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env_active.step(action)
        done = terminated or truncated

    # active_rms = info['acceleration_rms']
    active_rms = np.sqrt(np.mean(np.array(env_active.env.accelerations)**2))  # For active

    # Passive suspension
    env_passive = ActiveSuspensionEnv(road_class=road_class, seed=seed)
    obs, _ = env_passive.reset()
    done = False

    while not done:
        action = np.array([0.0])  # No control
        obs, reward, terminated, truncated, info = env_passive.step(action)
        done = terminated or truncated

    # passive_rms = info['acceleration_rms']
    passive_rms = np.sqrt(np.mean(np.array(env_passive.accelerations)**2))      # For passive

    # Calculate improvement
    reduction = ((passive_rms - active_rms) / passive_rms) * 100

    print(f"Passive RMS:  {passive_rms:.6f} m/s²")
    print(f"Active RMS:   {active_rms:.6f} m/s²")
    print(f"Reduction:    {reduction:.2f}%")
    print("="*80)

    return active_rms, passive_rms, reduction


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='SAC Active Suspension')
    parser.add_argument('--mode', type=str, default='train',
                       choices=['train', 'evaluate', 'compare', 'full'],
                       help='Execution mode')
    parser.add_argument('--road_class', type=str, default='D',
                       help='Road class (A-H)')
    parser.add_argument('--reward_id', type=int, default=1,
                       help='Reward function ID (1-8)')
    parser.add_argument('--total_timesteps', type=int, default=1000000,
                       help='Total timesteps')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--buffer_size', type=int, default=50000,
                       help='Buffer size')
    parser.add_argument('--batch_size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--gpu', type=int, default=4,
                       help='GPU ID')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Model path (for evaluate/compare)')
    args = parser.parse_args()

    if args.mode == 'train':
        model, model_path, exp_name = train_sac(
            road_class=args.road_class,
            reward_id=args.reward_id,
            total_timesteps=args.total_timesteps,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            gpu=args.gpu,
            seed=args.seed
        )
        print(f"\nExperiment: {exp_name}")
        print(f"Model saved: {model_path}")

    elif args.mode == 'evaluate':
        if args.model_path is None:
            print("Error: --model_path required for evaluation")
            sys.exit(1)
        evaluate_model(args.model_path, args.road_class, args.reward_id)

    elif args.mode == 'compare':
        if args.model_path is None:
            print("Error: --model_path required for comparison")
            sys.exit(1)
        compare_with_passive(args.model_path, args.road_class, args.reward_id)

    elif args.mode == 'full':
        # Full pipeline
        print("\n" + "="*80)
        print("FULL PIPELINE")
        print("="*80 + "\n")

        # Train
        model, model_path, exp_name = train_sac(
            road_class=args.road_class,
            reward_id=args.reward_id,
            total_timesteps=args.total_timesteps,
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            gpu=args.gpu,
            seed=args.seed
        )

        # Evaluate
        print("\n")
        mean_rms, std_rms = evaluate_model(model_path, args.road_class, args.reward_id)

        # Compare
        print("\n")
        active_rms, passive_rms, reduction = compare_with_passive(
            model_path, args.road_class, args.reward_id
        )

        # Final summary
        print("\n" + "="*80)
        print("FINAL RESULTS")
        print("="*80)
        print(f"Experiment: {exp_name}")
        print(f"Road class: {args.road_class}")
        print(f"Reward ID: {args.reward_id}")
        print(f"Active RMS: {active_rms:.6f} m/s²")
        print(f"Passive RMS: {passive_rms:.6f} m/s²")
        print(f"Reduction: {reduction:.2f}%")
        print(f"Paper target: 0.228 m/s²")
        if active_rms < 0.228:
            print(f"✓ BEAT PAPER by {(0.228 - active_rms)/0.228*100:.2f}%")
        else:
            print(f"✗ Below paper by {(active_rms - 0.228)/0.228*100:.2f}%")
        print("="*80)


if __name__ == '__main__':
    main()
