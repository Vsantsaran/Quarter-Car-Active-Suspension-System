"""
Evaluation Script for Trained SAC Models
Tests model on multiple episodes and calculates performance metrics
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


def evaluate_model(
    model_path,
    road_class='D',
    reward_id=1,
    n_episodes=10,
    seed=42,
    output_dir='experiments/results',
    exp_name=None
):
    """
    Evaluate trained SAC model
    
    Args:
        model_path: Path to saved model (.zip)
        road_class: ISO 8608 road class
        reward_id: Reward function ID
        n_episodes: Number of evaluation episodes
        seed: Random seed
        output_dir: Output directory for results
        exp_name: Experiment name
    
    Returns:
        Dictionary with evaluation results
    """
    
    print("="*80)
    print("MODEL EVALUATION")
    print("="*80)
    print(f"Model:       {model_path}")
    print(f"Road class:  {road_class}")
    print(f"Reward ID:   {reward_id}")
    print(f"Episodes:    {n_episodes}")
    print(f"Seed:        {seed}")
    print("="*80)
    
    # Load model
    print("\nLoading model...")
    model = SAC.load(model_path)
    print("✓ Model loaded")
    
    # Get reward function
    reward_fn = RewardFunctions.get_reward_function(reward_id)
    reward_desc = RewardFunctions.get_reward_description(reward_id)
    print(f"Reward function: {reward_desc}")
    
    # Storage for all episodes
    all_metrics = []
    all_rms = []
    
    print(f"\nRunning {n_episodes} evaluation episodes...")
    print("-" * 80)
    
    for ep in range(n_episodes):
        # Create fresh environment for each episode
        env = ActiveSuspensionEnv(road_class=road_class, seed=seed + ep)
        env = RewardWrapper(env, reward_fn)
        
        # Reset
        obs, _ = env.reset()
        done = False
        
        # Episode data
        episode_accelerations = []
        episode_displacements = []
        episode_forces = []
        episode_reward = 0.0
        
        # Run episode
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_accelerations.append(info['body_acceleration'])
            episode_displacements.append(info['suspension_deflection'])
            episode_forces.append(info['control_force'])
            episode_reward += reward
        
        # Calculate RMS correctly (at episode end)
        rms_acceleration = np.sqrt(np.mean(np.array(episode_accelerations)**2))
        
        # Calculate full metrics for this episode
        episode_metrics = SuspensionMetrics.calculate_metrics(
            episode_accelerations,
            episode_displacements,
            episode_forces,
            road_class
        )
        episode_metrics['episode_reward'] = float(episode_reward)
        
        all_metrics.append(episode_metrics)
        all_rms.append(rms_acceleration)
        
        print(f"Episode {ep+1:2d}/{n_episodes}: "
              f"RMS = {rms_acceleration:.6f} m/s²  "
              f"({episode_metrics['comfort_level']})")
    
    print("-" * 80)
    
    # Aggregate statistics
    mean_rms = np.mean(all_rms)
    std_rms = np.std(all_rms)
    min_rms = np.min(all_rms)
    max_rms = np.max(all_rms)
    
    # Aggregate all metrics
    aggregated_metrics = {
        'rms_acceleration_mean': float(mean_rms),
        'rms_acceleration_std': float(std_rms),
        'rms_acceleration_min': float(min_rms),
        'rms_acceleration_max': float(max_rms),
        'n_episodes': n_episodes,
        'road_class': road_class,
        'reward_id': reward_id,
        'seed': seed
    }
    
    # Calculate percentage that beat paper target
    paper_target = 0.228
    beats_paper = [rms < paper_target for rms in all_rms]
    pct_beat_paper = (sum(beats_paper) / len(beats_paper)) * 100
    
    aggregated_metrics['paper_target'] = paper_target
    aggregated_metrics['pct_beat_paper'] = float(pct_beat_paper)
    aggregated_metrics['beats_paper'] = mean_rms < paper_target
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"Mean RMS:        {mean_rms:.6f} ± {std_rms:.6f} m/s²")
    print(f"Min RMS:         {min_rms:.6f} m/s²")
    print(f"Max RMS:         {max_rms:.6f} m/s²")
    print(f"Paper target:    {paper_target:.6f} m/s²")
    print(f"Beat paper:      {pct_beat_paper:.1f}% of episodes")
    
    if mean_rms < paper_target:
        improvement = ((paper_target - mean_rms) / paper_target) * 100
        print(f"Status:          ✓ BEATS PAPER by {improvement:.2f}%")
    else:
        degradation = ((mean_rms - paper_target) / paper_target) * 100
        print(f"Status:          ✗ Below paper by {degradation:.2f}%")
    
    print("="*80)
    
    # Save results
    if exp_name:
        result_dir = Path(output_dir) / exp_name
    else:
        result_dir = Path(output_dir)
    
    result_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'model_path': str(model_path),
        'road_class': road_class,
        'reward_id': reward_id,
        'n_episodes': n_episodes,
        'seed': seed,
        'aggregated_metrics': aggregated_metrics,
        'per_episode_metrics': all_metrics,
        'rms_values': [float(x) for x in all_rms]
    }
    
    output_file = result_dir / 'evaluation_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    
    return results


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Evaluate trained SAC model')
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model (.zip)')
    parser.add_argument('--road_class', type=str, default='D',
                       help='ISO 8608 road class (A-H)')
    parser.add_argument('--reward_id', type=int, default=1,
                       help='Reward function ID (1-8)')
    parser.add_argument('--n_episodes', type=int, default=10,
                       help='Number of evaluation episodes')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output_dir', type=str, default='experiments/results',
                       help='Output directory')
    parser.add_argument('--exp_name', type=str, default=None,
                       help='Experiment name (for organizing results)')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    results = evaluate_model(
        model_path=args.model_path,
        road_class=args.road_class,
        reward_id=args.reward_id,
        n_episodes=args.n_episodes,
        seed=args.seed,
        output_dir=args.output_dir,
        exp_name=args.exp_name
    )
