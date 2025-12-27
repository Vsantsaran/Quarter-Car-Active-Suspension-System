"""
Main Pipeline Orchestrator for SAC Active Suspension
Runs: Train → Evaluate → Compare → Visualize
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.train_sac import train_sac, parse_args as parse_train_args
from src.evaluate import evaluate_model
from src.compare_passive import compare_with_passive
from src.visualize import visualize_all


def run_full_pipeline(args):
    """
    Run complete pipeline: train, evaluate, compare, visualize
    
    Args:
        args: Command line arguments
    """
    
    print("\n" + "="*80)
    print("SAC ACTIVE SUSPENSION - FULL PIPELINE")
    print("="*80)
    print(f"Road class:      {args.road_class}")
    print(f"Reward ID:       {args.reward_id}")
    print(f"Total timesteps: {args.total_timesteps:,}")
    print(f"GPU:             {args.gpu}")
    print("="*80 + "\n")
    
    # =========================================================================
    # STEP 1: TRAINING
    # =========================================================================
    
    print("STEP 1/4: TRAINING")
    print("-" * 80)
    
    model, final_metrics, training_data = train_sac(args)
    
    # Get experiment name from training
    exp_name = Path(model.logger.dir).name if hasattr(model, 'logger') else args.exp_name
    
    # Get model path
    model_path = Path(args.model_dir) / exp_name / 'sac_final.zip'
    
    print(f"\n✓ Training completed")
    print(f"  Model: {model_path}")
    print(f"  Results: {args.result_dir}/{exp_name}")
    
    # =========================================================================
    # STEP 2: EVALUATION
    # =========================================================================
    
    print("\n" + "="*80)
    print("STEP 2/4: EVALUATION")
    print("-" * 80)
    
    eval_results = evaluate_model(
        model_path=str(model_path),
        road_class=args.road_class,
        reward_id=args.reward_id,
        n_episodes=10,
        seed=42,
        output_dir=args.result_dir,
        exp_name=exp_name
    )
    
    print(f"\n✓ Evaluation completed")
    
    # =========================================================================
    # STEP 3: COMPARISON
    # =========================================================================
    
    print("\n" + "="*80)
    print("STEP 3/4: COMPARISON WITH PASSIVE")
    print("-" * 80)
    
    comparison_results = compare_with_passive(
        model_path=str(model_path),
        road_class=args.road_class,
        reward_id=args.reward_id,
        seed=42,
        output_dir=args.result_dir,
        exp_name=exp_name
    )
    
    print(f"\n✓ Comparison completed")
    
    # =========================================================================
    # STEP 4: VISUALIZATION
    # =========================================================================
    
    print("\n" + "="*80)
    print("STEP 4/4: VISUALIZATION")
    print("-" * 80)
    
    results_dir = Path(args.result_dir) / exp_name
    plot_dir = Path(args.plot_dir if hasattr(args, 'plot_dir') 
                   else 'experiments/plots') / exp_name
    
    visualize_all(
        results_dir=results_dir,
        output_dir=plot_dir,
        exp_name=exp_name
    )
    
    print(f"\n✓ Visualization completed")
    
    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*80)
    
    # Get final results
    active_rms = comparison_results['active_metrics']['rms_acceleration']
    passive_rms = comparison_results['passive_metrics']['rms_acceleration']
    reduction = comparison_results['comparison']['rms_acceleration_reduction']
    
    print(f"\nFinal Results:")
    print(f"  Experiment:      {exp_name}")
    print(f"  Road class:      {args.road_class}")
    print(f"  Reward ID:       {args.reward_id}")
    print(f"\n  Passive RMS:     {passive_rms:.6f} m/s²")
    print(f"  Active RMS:      {active_rms:.6f} m/s²")
    print(f"  Reduction:       {reduction:.2f}%")
    print(f"\n  Paper target:    0.228 m/s²")
    
    if active_rms < 0.228:
        improvement = ((0.228 - active_rms) / 0.228) * 100
        print(f"  Status:          ✓ BEATS PAPER by {improvement:.2f}%")
    else:
        degradation = ((active_rms - 0.228) / 0.228) * 100
        print(f"  Status:          ✗ Below paper by {degradation:.2f}%")
    
    print(f"\nOutput Directories:")
    print(f"  Models:          {args.model_dir}/{exp_name}")
    print(f"  Results:         {args.result_dir}/{exp_name}")
    print(f"  Plots:           {plot_dir}")
    
    print("="*80 + "\n")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='SAC Active Suspension - Full Pipeline'
    )
    
    # Mode
    parser.add_argument('--mode', type=str, default='full',
                       choices=['train', 'evaluate', 'compare', 'visualize', 'full'],
                       help='Pipeline mode')
    
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
                       help='Entropy coefficient')
    parser.add_argument('--target_entropy', type=str, default='auto',
                       help='Target entropy')
    
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
    
    # GPU
    parser.add_argument('--gpu', type=int, default=None,
                       help='GPU ID')
    
    # Directories
    parser.add_argument('--log_dir', type=str, default='experiments/logs',
                       help='Log directory')
    parser.add_argument('--model_dir', type=str, default='experiments/models',
                       help='Model directory')
    parser.add_argument('--result_dir', type=str, default='experiments/results',
                       help='Results directory')
    parser.add_argument('--plot_dir', type=str, default='experiments/plots',
                       help='Plots directory')
    
    # Experiment name
    parser.add_argument('--exp_name', type=str, default=None,
                       help='Experiment name')
    
    # For individual modes
    parser.add_argument('--model_path', type=str, default=None,
                       help='Model path (for evaluate/compare/visualize)')
    parser.add_argument('--results_dir', type=str, default=None,
                       help='Results directory (for visualize)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (for visualize)')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    
    if args.mode == 'full':
        run_full_pipeline(args)
    
    elif args.mode == 'train':
        from src.train_sac import train_sac
        train_sac(args)
    
    elif args.mode == 'evaluate':
        if args.model_path is None:
            print("Error: --model_path required for evaluation")
            sys.exit(1)
        evaluate_model(
            model_path=args.model_path,
            road_class=args.road_class,
            reward_id=args.reward_id,
            n_episodes=10,
            seed=42,
            output_dir=args.result_dir,
            exp_name=args.exp_name
        )
    
    elif args.mode == 'compare':
        if args.model_path is None:
            print("Error: --model_path required for comparison")
            sys.exit(1)
        compare_with_passive(
            model_path=args.model_path,
            road_class=args.road_class,
            reward_id=args.reward_id,
            seed=42,
            output_dir=args.result_dir,
            exp_name=args.exp_name
        )
    
    elif args.mode == 'visualize':
        if args.results_dir is None or args.output_dir is None:
            print("Error: --results_dir and --output_dir required for visualization")
            sys.exit(1)
        visualize_all(
            results_dir=args.results_dir,
            output_dir=args.output_dir,
            exp_name=args.exp_name
        )
