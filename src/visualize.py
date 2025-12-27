"""
Visualization Script - Generates All Plots from JSON Results
Reads training_data.json and comparison_results.json to create publication-quality plots
"""

import os
import sys
import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Seaborn style
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'


def plot_acceleration_comparison(time, active_accel, passive_accel,
                                 active_rms, passive_rms, road_class,
                                 save_path):
    """Plot body acceleration: active vs passive"""
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    ax.plot(time, active_accel, 'b-', linewidth=2.5, 
           label='Active (SAC)', alpha=0.9)
    ax.plot(time, passive_accel, 'r--', linewidth=2.0, 
           label='Passive', alpha=0.8)
    
    ax.set_xlabel('Time (s)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Body Acceleration (m/s²)', fontsize=14, fontweight='bold')
    ax.set_title(f'Acceleration Comparison - Road Class {road_class}',
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=12, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.4)
    
    # Statistics box
    reduction = ((passive_rms - active_rms) / passive_rms) * 100
    textstr = f'Passive RMS: {passive_rms:.4f} m/s²\n'
    textstr += f'Active RMS:  {active_rms:.4f} m/s²\n'
    textstr += f'Reduction:   {reduction:.2f}%'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {save_path.name}")


def plot_control_force(time, forces, save_path, u_max=8000):
    """Plot control force over time"""
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(time, forces, 'g-', linewidth=2.0, label='Control Force')
    ax.axhline(y=u_max, color='r', linestyle='--', linewidth=1.5,
              alpha=0.6, label='Force Limits')
    ax.axhline(y=-u_max, color='r', linestyle='--', linewidth=1.5, alpha=0.6)
    ax.fill_between(time, u_max, np.max(forces)+1000, alpha=0.1, color='red')
    ax.fill_between(time, -u_max, np.min(forces)-1000, alpha=0.1, color='red')
    
    ax.set_xlabel('Time (s)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Control Force (N)', fontsize=14, fontweight='bold')
    ax.set_title('Actuator Control Force', fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=12, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Statistics
    mean_force = np.mean(np.abs(forces))
    max_force = np.max(np.abs(forces))
    rms_force = np.sqrt(np.mean(np.array(forces)**2))
    
    textstr = f'Mean |Force|: {mean_force:.2f} N\n'
    textstr += f'Max |Force|:  {max_force:.2f} N\n'
    textstr += f'RMS Force:    {rms_force:.2f} N'
    
    props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {save_path.name}")


def plot_comfort_zones(active_rms, passive_rms, road_class, save_path):
    """Plot ISO 2631-5 comfort zones"""
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    comfort_zones = [
        (0, 0.315, 'Not uncomfortable', '#2ecc71'),
        (0.315, 0.63, 'A little uncomfortable', '#f1c40f'),
        (0.63, 1.0, 'Fairly uncomfortable', '#e67e22'),
        (1.0, 1.6, 'Uncomfortable', '#e74c3c'),
        (1.6, 2.5, 'Very uncomfortable', '#c0392b'),
        (2.5, 3.5, 'Extremely uncomfortable', '#8b0000')
    ]
    
    for i, (lower, upper, label, color) in enumerate(comfort_zones):
        ax.barh(i, upper - lower, left=lower, height=0.8,
               color=color, alpha=0.6, label=label, 
               edgecolor='black', linewidth=1)
    
    # Plot RMS values
    for i, (lower, upper, _, _) in enumerate(comfort_zones):
        if lower <= active_rms < upper:
            ax.plot(active_rms, i, 'b*', markersize=25, 
                   label='Active SAC', zorder=10)
        if lower <= passive_rms < upper:
            ax.plot(passive_rms, i, 'ro', markersize=20, 
                   label='Passive', zorder=10)
    
    ax.set_xlabel('RMS Acceleration (m/s²)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Comfort Level', fontsize=14, fontweight='bold')
    ax.set_title(f'ISO 2631-5 Comfort Assessment - Road Class {road_class}',
                fontsize=16, fontweight='bold', pad=20)
    ax.set_yticks(range(len(comfort_zones)))
    ax.set_yticklabels([label for _, _, label, _ in comfort_zones])
    ax.set_xlim(0, 3.0)
    ax.grid(True, alpha=0.3, axis='x', linestyle='--')
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(),
             loc='upper right', fontsize=11, framealpha=0.95)
    
    textstr = f'Active RMS:  {active_rms:.4f} m/s²\n'
    textstr += f'Passive RMS: {passive_rms:.4f} m/s²'
    props = dict(boxstyle='round', facecolor='white', 
                alpha=0.9, edgecolor='black')
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=12,
           verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {save_path.name}")


def plot_learning_curves(timesteps, rewards, rms_values, save_path):
    """Plot training learning curves"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Rewards
    ax1.plot(timesteps, rewards, 'b-', linewidth=2.0, 
            alpha=0.7, label='Episode Reward')
    
    if len(rewards) > 100:
        window = min(100, len(rewards) // 10)
        smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
        ax1.plot(timesteps[:len(smoothed)], smoothed, 'r-',
                linewidth=2.5, label='Smoothed', alpha=0.9)
    
    ax1.set_xlabel('Timesteps', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Episode Reward', fontsize=14, fontweight='bold')
    ax1.set_title('Training Reward Progression', fontsize=15, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, linestyle='--')
    
    # RMS values
    if rms_values:
        ax2.plot(timesteps, rms_values, 'g-', linewidth=2.0, 
                alpha=0.7, label='RMS Acceleration')
        ax2.axhline(y=0.228, color='red', linestyle='--',
                   linewidth=2, label='Paper Target (0.228)', alpha=0.8)
        
        ax2.set_xlabel('Timesteps', fontsize=14, fontweight='bold')
        ax2.set_ylabel('RMS Acceleration (m/s²)', fontsize=14, fontweight='bold')
        ax2.set_title('RMS Acceleration Convergence', fontsize=15, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {save_path.name}")


def plot_displacement_comparison(time, active_disp, passive_disp, save_path):
    """Plot suspension deflection comparison"""
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.plot(time, active_disp, 'b-', linewidth=2.0, 
           label='Active', alpha=0.8)
    ax.plot(time, passive_disp, 'r--', linewidth=1.5, 
           label='Passive', alpha=0.7)
    
    ax.set_xlabel('Time (s)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Suspension Deflection (m)', fontsize=14, fontweight='bold')
    ax.set_title('Suspension Deflection Comparison',
                fontsize=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', fontsize=12, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {save_path.name}")


def visualize_all(results_dir, output_dir, exp_name=None):
    """
    Generate all plots from JSON results
    
    Args:
        results_dir: Directory containing JSON files
        output_dir: Directory to save plots
        exp_name: Experiment name (for titles)
    """
    
    results_dir = Path(results_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("GENERATING VISUALIZATIONS")
    print("="*80)
    print(f"Results: {results_dir}")
    print(f"Output:  {output_dir}")
    print("="*80)
    
    # =========================================================================
    # COMPARISON PLOTS (from comparison_results.json)
    # =========================================================================
    
    comparison_file = results_dir / 'comparison_results.json'
    if comparison_file.exists():
        print("\nGenerating comparison plots...")
        
        with open(comparison_file, 'r') as f:
            data = json.load(f)
        
        active_metrics = data['active_metrics']
        passive_metrics = data['passive_metrics']
        time_series = data['time_series']
        road_class = active_metrics.get('road_class', 'D')
        
        time = time_series['time']
        
        # Plot 1: Acceleration comparison
        plot_acceleration_comparison(
            time,
            np.array(time_series['active']['accelerations']),
            np.array(time_series['passive']['accelerations']),
            active_metrics['rms_acceleration'],
            passive_metrics['rms_acceleration'],
            road_class,
            output_dir / 'acceleration_comparison.png'
        )
        
        # Plot 2: Control force
        plot_control_force(
            time,
            np.array(time_series['active']['forces']),
            output_dir / 'control_force.png'
        )
        
        # Plot 3: ISO comfort zones
        plot_comfort_zones(
            active_metrics['rms_acceleration'],
            passive_metrics['rms_acceleration'],
            road_class,
            output_dir / 'comfort_zones.png'
        )
        
        # Plot 4: Displacement comparison
        plot_displacement_comparison(
            time,
            np.array(time_series['active']['displacements']),
            np.array(time_series['passive']['displacements']),
            output_dir / 'displacement_comparison.png'
        )
        
    else:
        print(f"\n⚠ Comparison results not found: {comparison_file}")
    
    # =========================================================================
    # LEARNING CURVES (from training_data.json)
    # =========================================================================
    
    training_file = results_dir / 'training_data.json'
    if training_file.exists():
        print("\nGenerating learning curves...")
        
        with open(training_file, 'r') as f:
            training_data = json.load(f)
        
        plot_learning_curves(
            training_data['timesteps'],
            training_data['rewards'],
            training_data.get('rms_values', []),
            output_dir / 'learning_curves.png'
        )
        
    else:
        print(f"\n⚠ Training data not found: {training_file}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    
    print("\n" + "="*80)
    print("✓ ALL PLOTS GENERATED")
    print("="*80)
    print(f"Location: {output_dir}")
    
    # List all plots
    plots = sorted(output_dir.glob('*.png'))
    if plots:
        print("\nGenerated plots:")
        for plot in plots:
            print(f"  • {plot.name}")
    
    print("="*80)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Generate all plots')
    
    parser.add_argument('--results_dir', type=str, required=True,
                       help='Directory containing JSON results')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Directory to save plots')
    parser.add_argument('--exp_name', type=str, default=None,
                       help='Experiment name (optional)')
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    visualize_all(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
        exp_name=args.exp_name
    )
