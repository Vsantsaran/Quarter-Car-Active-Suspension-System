"""
Active vs Passive Suspension Comparison
Compares trained SAC controller against passive suspension (no control)
"""

import os
import sys
import argparse
import json
import numpy as np
from pathlib import Path
from stable_baselines3 import SAC

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from env.suspension_env import ActiveSuspensionEnv
from utils.rewards import RewardFunctions, RewardWrapper
from utils.metrics import SuspensionMetrics


def compare_with_passive(
    model_path,
    road_class='D',
    reward_id=1,
    seed=42,
    output_dir='experiments/results',
    exp_name=None
):
    """
    Compare active (SAC) vs passive suspension
    
    Args:
        model_path: Path to trained model
        road_class: ISO 8608 road class
        reward_id: Reward function ID
        seed: Random seed (ensures same road for both)
        output_dir: Output directory
        exp_name: Experiment name
    
    Returns:
        Dictionary with comparison results and time series
    """
    
    print("="*80)
    print("ACTIVE VS PASSIVE SUSPENSION COMPARISON")
    print("="*80)
    print(f"Model:       {model_path}")
    print(f"Road class:  {road_class}")
    print(f"Seed:        {seed}")
    print("="*80)
    
    # Load model
    print("\nLoading model...")
    model = SAC.load(model_path)
    print("✓ Model loaded")
    
    # Get reward function
    reward_fn = RewardFunctions.get_reward_function(reward_id)
    
    # =========================================================================
    # RUN ACTIVE SUSPENSION
    # =========================================================================
    
    print("\n" + "-"*80)
    print("Running ACTIVE suspension (SAC controller)...")
    print("-"*80)
    
    env_active = ActiveSuspensionEnv(road_class=road_class, seed=seed)
    env_active = RewardWrapper(env_active, reward_fn)
    
    obs, _ = env_active.reset()
    done = False
    
    active_accelerations = []
    active_displacements = []
    active_forces = []
    active_rewards = []
    
    step = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env_active.step(action)
        done = terminated or truncated
        
        active_accelerations.append(info['body_acceleration'])
        active_displacements.append(info['suspension_deflection'])
        active_forces.append(info['control_force'])
        active_rewards.append(reward)
        
        step += 1
    
    # Calculate RMS correctly
    active_rms = np.sqrt(np.mean(np.array(active_accelerations)**2))
    
    print(f"✓ Active completed: {step} steps")
    print(f"  RMS acceleration: {active_rms:.6f} m/s²")
    
    # Calculate full metrics
    active_metrics = SuspensionMetrics.calculate_metrics(
        active_accelerations,
        active_displacements,
        active_forces,
        road_class
    )
    
    # =========================================================================
    # RUN PASSIVE SUSPENSION
    # =========================================================================
    
    print("\n" + "-"*80)
    print("Running PASSIVE suspension (no control)...")
    print("-"*80)
    
    # CRITICAL: Use same seed for fair comparison (same road)
    env_passive = ActiveSuspensionEnv(road_class=road_class, seed=seed)
    
    obs, _ = env_passive.reset()
    done = False
    
    passive_accelerations = []
    passive_displacements = []
    passive_forces = []
    
    step = 0
    while not done:
        # Zero control force (passive)
        action = np.array([0.0])
        obs, reward, terminated, truncated, info = env_passive.step(action)
        done = terminated or truncated
        
        passive_accelerations.append(info['body_acceleration'])
        passive_displacements.append(info['suspension_deflection'])
        passive_forces.append(info['control_force'])  # Should be 0
        
        step += 1
    
    # Calculate RMS correctly
    passive_rms = np.sqrt(np.mean(np.array(passive_accelerations)**2))
    
    print(f"✓ Passive completed: {step} steps")
    print(f"  RMS acceleration: {passive_rms:.6f} m/s²")
    
    # Calculate full metrics
    passive_metrics = SuspensionMetrics.calculate_metrics(
        passive_accelerations,
        passive_displacements,
        passive_forces,
        road_class
    )
    
    # =========================================================================
    # COMPARISON
    # =========================================================================
    
    comparison = SuspensionMetrics.compare_active_passive(
        active_metrics,
        passive_metrics
    )
    
    print("\n" + "="*80)
    print("COMPARISON RESULTS")
    print("="*80)
    print(f"\nRMS Acceleration:")
    print(f"  Passive:         {passive_rms:.6f} m/s²")
    print(f"  Active:          {active_rms:.6f} m/s²")
    print(f"  Reduction:       {comparison['rms_acceleration_reduction']:.2f}%")
    
    print(f"\nComfort Level:")
    print(f"  Passive:         {passive_metrics['comfort_level']}")
    print(f"  Active:          {active_metrics['comfort_level']}")
    
    print(f"\nControl Effort:")
    print(f"  RMS Force:       {active_metrics['rms_force']:.2f} N")
    print(f"  Max Force:       {active_metrics['max_force']:.2f} N")
    print(f"  Force Limit:     8000.00 N")
    
    print(f"\nPaper Comparison:")
    print(f"  Paper target:    0.228 m/s²")
    print(f"  Active RMS:      {active_rms:.6f} m/s²")
    
    if active_rms < 0.228:
        improvement = ((0.228 - active_rms) / 0.228) * 100
        print(f"  Status:          ✓ BEATS PAPER by {improvement:.2f}%")
    else:
        degradation = ((active_rms - 0.228) / 0.228) * 100
        print(f"  Status:          ✗ Below paper by {degradation:.2f}%")
    
    print("="*80)
    
    # =========================================================================
    # SAVE RESULTS
    # =========================================================================
    
    if exp_name:
        result_dir = Path(output_dir) / exp_name
    else:
        result_dir = Path(output_dir)
    
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare time series data for plotting
    dt = env_active.env.dt
    time = [i * dt for i in range(len(active_accelerations))]
    
    results = {
        'model_path': str(model_path),
        'road_class': road_class,
        'reward_id': reward_id,
        'seed': seed,
        
        # Metrics
        'active_metrics': active_metrics,
        'passive_metrics': passive_metrics,
        'comparison': comparison,
        
        # Time series for plotting
        'time_series': {
            'time': time,
            'active': {
                'accelerations': [float(x) for x in active_accelerations],
                'displacements': [float(x) for x in active_displacements],
                'forces': [float(x) for x in active_forces]
            },
            'passive': {
                'accelerations': [float(x) for x in passive_accelerations],
                'displacements': [float(x) for x in passive_displacements],
                'forces': [float(x) for x in passive_forces]
            }
        }
    }
    
    output_file = result_dir / 'comparison_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    return results


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Compare active vs passive suspension'
    )
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model (.zip)')
    parser.add_argument('--road_class', type=str, default='D',
                       help='ISO 8608 road class (A-H)')
    parser.add_argument('--reward_id', type=int, default=1,
                       help='Reward function ID (1-8)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output_dir', type=str, default='experiments/results',
                       help='Output directory')
    parser.add_argument('--exp_name', type=str, default=None,
                       help='Experiment name')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    results = compare_with_passive(
        model_path=args.model_path,
        road_class=args.road_class,
        reward_id=args.reward_id,
        seed=args.seed,
        output_dir=args.output_dir,
        exp_name=args.exp_name
    )
