"""
Suspension Performance Metrics
Calculates all relevant metrics according to ISO 2631-5 and paper benchmarks
"""

import numpy as np
from typing import Dict, List


class SuspensionMetrics:
    """Calculate and analyze suspension system performance metrics"""
    
    # ISO 2631-5 comfort thresholds (m/s²)
    COMFORT_ZONES = {
        'not_uncomfortable': (0.0, 0.315),
        'little_uncomfortable': (0.315, 0.63),
        'fairly_uncomfortable': (0.63, 1.0),
        'uncomfortable': (1.0, 1.6),
        'very_uncomfortable': (1.6, 2.5),
        'extremely_uncomfortable': (2.5, float('inf'))
    }
    
    # Paper benchmark (Dridi et al. 2023)
    PAPER_TARGET_RMS = 0.228  # m/s²
    
    @staticmethod
    def calculate_rms(data: np.ndarray) -> float:
        """
        Calculate Root Mean Square (ISO 2631-5 Equation 1)
        
        RMS = sqrt(1/T ∫₀ᵀ a²(t) dt)
        
        Args:
            data: Time series data (acceleration, force, etc.)
        
        Returns:
            RMS value
        """
        return np.sqrt(np.mean(data ** 2))
    
    @staticmethod
    def calculate_peak_to_peak(data: np.ndarray) -> float:
        """Calculate peak-to-peak amplitude"""
        return np.max(data) - np.min(data)
    
    @staticmethod
    def calculate_crest_factor(data: np.ndarray) -> float:
        """
        Calculate crest factor: peak / RMS
        Indicates impulsiveness of signal
        """
        rms = SuspensionMetrics.calculate_rms(data)
        peak = np.max(np.abs(data))
        return peak / rms if rms > 0 else 0
    
    @staticmethod
    def get_comfort_level(rms_acceleration: float) -> str:
        """
        Determine comfort level based on ISO 2631-5
        
        Args:
            rms_acceleration: RMS body acceleration (m/s²)
        
        Returns:
            Comfort level description
        """
        for level, (lower, upper) in SuspensionMetrics.COMFORT_ZONES.items():
            if lower <= rms_acceleration < upper:
                return level.replace('_', ' ').title()
        return "Extremely Uncomfortable"
    
    @staticmethod
    def calculate_metrics(
        accelerations: List[float],
        displacements: List[float],
        forces: List[float],
        road_class: str = 'D'
    ) -> Dict:
        """
        Calculate comprehensive suspension metrics
        
        Args:
            accelerations: Body acceleration time series (m/s²)
            displacements: Suspension deflection time series (m)
            forces: Control force time series (N)
            road_class: ISO 8608 road class
        
        Returns:
            Dictionary of all metrics
        """
        accel = np.array(accelerations)
        disp = np.array(displacements)
        force = np.array(forces)
        
        # Acceleration metrics (primary comfort indicator)
        rms_accel = SuspensionMetrics.calculate_rms(accel)
        max_accel = np.max(np.abs(accel))
        mean_accel = np.mean(np.abs(accel))
        std_accel = np.std(accel)
        pk2pk_accel = SuspensionMetrics.calculate_peak_to_peak(accel)
        crest_accel = SuspensionMetrics.calculate_crest_factor(accel)
        
        # Displacement metrics (suspension travel)
        rms_disp = SuspensionMetrics.calculate_rms(disp)
        max_disp = np.max(disp)
        mean_disp = np.mean(disp)
        std_disp = np.std(disp)
        
        # Force metrics (control effort)
        rms_force = SuspensionMetrics.calculate_rms(force)
        max_force = np.max(np.abs(force))
        mean_force = np.mean(np.abs(force))
        std_force = np.std(force)
        
        # Comfort assessment
        comfort_level = SuspensionMetrics.get_comfort_level(rms_accel)
        
        # Paper comparison
        paper_diff = rms_accel - SuspensionMetrics.PAPER_TARGET_RMS
        paper_diff_pct = (paper_diff / SuspensionMetrics.PAPER_TARGET_RMS) * 100
        beats_paper = rms_accel < SuspensionMetrics.PAPER_TARGET_RMS
        
        metrics = {
            # Primary metric
            'rms_acceleration': float(rms_accel),
            
            # Acceleration metrics
            'max_acceleration': float(max_accel),
            'mean_acceleration': float(mean_accel),
            'std_acceleration': float(std_accel),
            'peak_to_peak_acceleration': float(pk2pk_accel),
            'crest_factor_acceleration': float(crest_accel),
            
            # Displacement metrics
            'rms_displacement': float(rms_disp),
            'max_displacement': float(max_disp),
            'mean_displacement': float(mean_disp),
            'std_displacement': float(std_disp),
            
            # Force metrics
            'rms_force': float(rms_force),
            'max_force': float(max_force),
            'mean_force': float(mean_force),
            'std_force': float(std_force),
            
            # Comfort assessment
            'comfort_level': comfort_level,
            'iso_compliant': rms_accel < 0.315,  # "Not uncomfortable" threshold
            
            # Paper comparison
            'paper_target': SuspensionMetrics.PAPER_TARGET_RMS,
            'paper_difference': float(paper_diff),
            'paper_difference_percent': float(paper_diff_pct),
            'beats_paper': beats_paper,
            
            # Metadata
            'road_class': road_class,
            'num_samples': len(accelerations)
        }
        
        return metrics
    
    @staticmethod
    def calculate_paper_comparison(achieved_rms: float) -> Dict:
        """
        Calculate detailed comparison with paper results
        
        Args:
            achieved_rms: Achieved RMS acceleration (m/s²)
        
        Returns:
            Comparison metrics
        """
        target = SuspensionMetrics.PAPER_TARGET_RMS
        diff = achieved_rms - target
        diff_pct = (diff / target) * 100
        
        if achieved_rms < target:
            status = "BEATS PAPER"
            message = f"✓ {abs(diff_pct):.2f}% better than paper"
        elif achieved_rms < target * 1.1:
            status = "MATCHES PAPER"
            message = f"≈ Within 10% of paper target"
        else:
            status = "BELOW PAPER"
            message = f"✗ {diff_pct:.2f}% worse than paper"
        
        return {
            'paper_target': target,
            'achieved_rms': achieved_rms,
            'difference': diff,
            'difference_percent': diff_pct,
            'status': status,
            'message': message
        }
    
    @staticmethod
    def compare_active_passive(
        active_metrics: Dict,
        passive_metrics: Dict
    ) -> Dict:
        """
        Compare active vs passive suspension performance
        
        Args:
            active_metrics: Metrics from active suspension
            passive_metrics: Metrics from passive suspension
        
        Returns:
            Comparison results
        """
        # RMS acceleration reduction
        rms_reduction = (
            (passive_metrics['rms_acceleration'] - active_metrics['rms_acceleration']) /
            passive_metrics['rms_acceleration']
        ) * 100
        
        # Max acceleration reduction
        max_reduction = (
            (passive_metrics['max_acceleration'] - active_metrics['max_acceleration']) /
            passive_metrics['max_acceleration']
        ) * 100
        
        # Displacement comparison
        disp_reduction = (
            (passive_metrics['rms_displacement'] - active_metrics['rms_displacement']) /
            passive_metrics['rms_displacement']
        ) * 100
        
        # Comfort improvement
        active_comfort = active_metrics['comfort_level']
        passive_comfort = passive_metrics['comfort_level']
        
        comparison = {
            # Reductions
            'rms_acceleration_reduction': float(rms_reduction),
            'max_acceleration_reduction': float(max_reduction),
            'displacement_reduction': float(disp_reduction),
            
            # Absolute values
            'active_rms': active_metrics['rms_acceleration'],
            'passive_rms': passive_metrics['rms_acceleration'],
            
            # Comfort
            'active_comfort': active_comfort,
            'passive_comfort': passive_comfort,
            'comfort_improved': active_comfort != passive_comfort,
            
            # Control effort
            'control_rms_force': active_metrics['rms_force'],
            'control_max_force': active_metrics['max_force'],
            
            # Paper target
            'beats_paper': active_metrics['beats_paper'],
            'paper_target': SuspensionMetrics.PAPER_TARGET_RMS,
            
            # Summary
            'summary': f"{rms_reduction:.2f}% RMS reduction ({passive_comfort} → {active_comfort})"
        }
        
        return comparison
    
    @staticmethod
    def format_metrics_report(metrics: Dict) -> str:
        """
        Format metrics as readable report
        
        Args:
            metrics: Metrics dictionary
        
        Returns:
            Formatted string report
        """
        report = []
        report.append("=" * 80)
        report.append("SUSPENSION PERFORMANCE METRICS")
        report.append("=" * 80)
        
        # Primary metric
        report.append(f"\nPRIMARY METRIC (ISO 2631-5):")
        report.append(f"  RMS Acceleration:        {metrics['rms_acceleration']:.6f} m/s²")
        report.append(f"  Comfort Level:           {metrics['comfort_level']}")
        report.append(f"  ISO Compliant:           {'✓ Yes' if metrics['iso_compliant'] else '✗ No'}")
        
        # Acceleration metrics
        report.append(f"\nACCELERATION METRICS:")
        report.append(f"  Max:                     {metrics['max_acceleration']:.4f} m/s²")
        report.append(f"  Mean:                    {metrics['mean_acceleration']:.4f} m/s²")
        report.append(f"  Std Dev:                 {metrics['std_acceleration']:.4f} m/s²")
        report.append(f"  Peak-to-Peak:            {metrics['peak_to_peak_acceleration']:.4f} m/s²")
        report.append(f"  Crest Factor:            {metrics['crest_factor_acceleration']:.2f}")
        
        # Displacement metrics
        report.append(f"\nDISPLACEMENT METRICS:")
        report.append(f"  RMS:                     {metrics['rms_displacement']:.6f} m")
        report.append(f"  Max:                     {metrics['max_displacement']:.6f} m")
        report.append(f"  Mean:                    {metrics['mean_displacement']:.6f} m")
        
        # Force metrics
        report.append(f"\nCONTROL FORCE METRICS:")
        report.append(f"  RMS:                     {metrics['rms_force']:.2f} N")
        report.append(f"  Max:                     {metrics['max_force']:.2f} N")
        report.append(f"  Mean:                    {metrics['mean_force']:.2f} N")
        
        # Paper comparison
        report.append(f"\nPAPER COMPARISON:")
        report.append(f"  Target RMS:              {metrics['paper_target']:.6f} m/s²")
        report.append(f"  Difference:              {metrics['paper_difference']:+.6f} m/s²")
        report.append(f"  Difference %:            {metrics['paper_difference_percent']:+.2f}%")
        report.append(f"  Status:                  {'✓ BEATS PAPER' if metrics['beats_paper'] else '✗ BELOW PAPER'}")
        
        # Metadata
        report.append(f"\nMETADATA:")
        report.append(f"  Road Class:              {metrics['road_class']}")
        report.append(f"  Samples:                 {metrics['num_samples']:,}")
        
        report.append("=" * 80)
        
        return "\n".join(report)


class PerformanceTracker:
    """Track performance metrics during training"""
    
    def __init__(self):
        self.episodes = []
        self.rms_values = []
        self.rewards = []
        self.timesteps = []
        self.best_rms = float('inf')
        self.best_episode = 0
    
    def update(self, episode: int, timestep: int, rms: float, reward: float):
        """Update tracker with new episode data"""
        self.episodes.append(episode)
        self.timesteps.append(timestep)
        self.rms_values.append(rms)
        self.rewards.append(reward)
        
        if rms < self.best_rms:
            self.best_rms = rms
            self.best_episode = episode
    
    def get_statistics(self) -> Dict:
        """Get training statistics"""
        if not self.rms_values:
            return {}
        
        return {
            'total_episodes': len(self.episodes),
            'best_rms': float(self.best_rms),
            'best_episode': int(self.best_episode),
            'mean_rms': float(np.mean(self.rms_values)),
            'std_rms': float(np.std(self.rms_values)),
            'final_rms': float(self.rms_values[-1]),
            'mean_reward': float(np.mean(self.rewards)),
            'std_reward': float(np.std(self.rewards)),
            'final_reward': float(self.rewards[-1])
        }
    
    def get_convergence_info(self, window: int = 100) -> Dict:
        """Check if training has converged"""
        if len(self.rms_values) < window * 2:
            return {'converged': False, 'message': 'Insufficient data'}
        
        recent = np.array(self.rms_values[-window:])
        previous = np.array(self.rms_values[-2*window:-window])
        
        recent_mean = np.mean(recent)
        previous_mean = np.mean(previous)
        improvement = (previous_mean - recent_mean) / previous_mean * 100
        
        converged = improvement < 1.0  # Less than 1% improvement
        
        return {
            'converged': converged,
            'recent_mean_rms': float(recent_mean),
            'previous_mean_rms': float(previous_mean),
            'improvement_percent': float(improvement),
            'message': 'Converged' if converged else 'Still improving'
        }
