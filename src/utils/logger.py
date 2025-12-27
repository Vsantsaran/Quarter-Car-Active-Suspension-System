"""
Experiment Logger
Comprehensive logging for SAC training experiments
"""

import os
import json
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional


class ExperimentLogger:
    """
    Comprehensive logger for SAC experiments
    
    Features:
    - File and console logging
    - Hyperparameter tracking
    - Training metrics logging
    - JSON export for analysis
    """
    
    def __init__(
        self,
        log_dir: str,
        exp_name: str,
        log_level: int = logging.INFO
    ):
        """
        Args:
            log_dir: Directory for log files
            exp_name: Experiment name
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        self.log_dir = Path(log_dir)
        self.exp_name = exp_name
        self.log_level = log_level
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Log file path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{exp_name}_{timestamp}.log"
        
        # Setup logger
        self.logger = self._setup_logger()
        
        # Training metadata
        self.start_time = None
        self.end_time = None
        self.hyperparameters = {}
        self.training_log = []
        self.evaluation_log = []
        
        self.info(f"Logger initialized: {self.log_file}")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup Python logger with file and console handlers"""
        logger = logging.getLogger(self.exp_name)
        logger.setLevel(self.log_level)
        logger.handlers = []  # Clear existing handlers
        
        # File handler
        file_handler = logging.FileHandler(self.log_file, mode='w')
        file_handler.setLevel(self.log_level)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.log_level)
        console_formatter = logging.Formatter(
            '%(levelname)-8s | %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        return logger
    
    def info(self, message: str):
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message"""
        self.logger.error(message)
    
    def debug(self, message: str):
        """Log debug message"""
        self.logger.debug(message)
    
    def log_hyperparameters(self, hyperparams: Dict[str, Any]):
        """
        Log hyperparameters at start of training
        
        Args:
            hyperparams: Dictionary of hyperparameters
        """
        self.hyperparameters = hyperparams
        
        self.info("-" * 80)
        self.info("HYPERPARAMETERS:")
        self.info("-" * 80)
        
        # Group by category
        categories = {
            'Environment': ['road_class', 'vehicle_speed', 'dt', 'max_steps'],
            'SAC': ['learning_rate', 'buffer_size', 'batch_size', 'tau', 'gamma', 
                   'ent_coef', 'target_entropy'],
            'Network': ['hidden_dim', 'n_layers'],
            'Training': ['total_timesteps', 'eval_freq', 'checkpoint_freq', 'seed'],
            'Reward': ['reward_id'],
            'Hardware': ['device', 'gpu']
        }
        
        for category, keys in categories.items():
            self.info(f"\n{category}:")
            for key in keys:
                if key in hyperparams:
                    value = hyperparams[key]
                    self.info(f"  {key:.<30} {value}")
        
        self.info("-" * 80)
    
    def log_training_start(self, total_timesteps: int):
        """Log training start"""
        self.start_time = time.time()
        self.info("=" * 80)
        self.info("TRAINING STARTED")
        self.info("=" * 80)
        self.info(f"Total timesteps: {total_timesteps:,}")
        self.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.info("=" * 80)
    
    def log_training_end(self, total_time: float):
        """Log training completion"""
        self.end_time = time.time()
        
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = int(total_time % 60)
        
        self.info("=" * 80)
        self.info("TRAINING COMPLETED")
        self.info("=" * 80)
        self.info(f"Total time: {hours}h {minutes}m {seconds}s")
        self.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        self.info("=" * 80)
    
    def log_evaluation(self, metrics: Dict[str, Any]):
        """
        Log evaluation metrics
        
        Args:
            metrics: Dictionary of evaluation metrics
        """
        self.evaluation_log.append({
            'timestamp': time.time(),
            'metrics': metrics
        })
        
        self.info("-" * 80)
        self.info("EVALUATION METRICS:")
        self.info("-" * 80)
        
        # Primary metric
        rms = metrics.get('rms_acceleration', 0)
        comfort = metrics.get('comfort_level', 'Unknown')
        self.info(f"RMS Acceleration:     {rms:.6f} m/s² ({comfort})")
        
        # Acceleration
        self.info(f"\nAcceleration:")
        self.info(f"  Max:                {metrics.get('max_acceleration', 0):.4f} m/s²")
        self.info(f"  Mean:               {metrics.get('mean_acceleration', 0):.4f} m/s²")
        self.info(f"  Std:                {metrics.get('std_acceleration', 0):.4f} m/s²")
        
        # Displacement
        self.info(f"\nDisplacement:")
        self.info(f"  RMS:                {metrics.get('rms_displacement', 0):.6f} m")
        self.info(f"  Max:                {metrics.get('max_displacement', 0):.6f} m")
        
        # Force
        self.info(f"\nControl Force:")
        self.info(f"  RMS:                {metrics.get('rms_force', 0):.2f} N")
        self.info(f"  Max:                {metrics.get('max_force', 0):.2f} N")
        
        # Paper comparison
        paper_diff = metrics.get('paper_difference_percent', 0)
        status = "✓ BEATS" if metrics.get('beats_paper', False) else "✗ BELOW"
        self.info(f"\nPaper Target:         0.228 m/s²")
        self.info(f"Status:               {status} ({paper_diff:+.2f}%)")
        
        self.info("-" * 80)
    
    def log_comparison(self, active_metrics: Dict, passive_metrics: Dict, 
                      comparison: Dict):
        """
        Log active vs passive comparison
        
        Args:
            active_metrics: Active suspension metrics
            passive_metrics: Passive suspension metrics
            comparison: Comparison results
        """
        self.info("=" * 80)
        self.info("ACTIVE VS PASSIVE COMPARISON")
        self.info("=" * 80)
        
        self.info(f"\nRMS Acceleration:")
        self.info(f"  Passive:            {passive_metrics['rms_acceleration']:.6f} m/s²")
        self.info(f"  Active:             {active_metrics['rms_acceleration']:.6f} m/s²")
        self.info(f"  Reduction:          {comparison['rms_acceleration_reduction']:.2f}%")
        
        self.info(f"\nComfort Level:")
        self.info(f"  Passive:            {passive_metrics['comfort_level']}")
        self.info(f"  Active:             {active_metrics['comfort_level']}")
        
        self.info(f"\nControl Effort:")
        self.info(f"  RMS Force:          {comparison['control_rms_force']:.2f} N")
        self.info(f"  Max Force:          {comparison['control_max_force']:.2f} N")
        
        self.info(f"\nPaper Target:         {comparison['paper_target']:.6f} m/s²")
        self.info(f"Status:               {'✓ ACHIEVED' if comparison['beats_paper'] else '✗ NOT ACHIEVED'}")
        
        self.info("=" * 80)
    
    def log_checkpoint(self, timestep: int, metrics: Dict[str, float]):
        """
        Log checkpoint metrics
        
        Args:
            timestep: Current timestep
            metrics: Metrics dictionary
        """
        self.training_log.append({
            'timestep': timestep,
            'timestamp': time.time(),
            'metrics': metrics
        })
        
        msg = f"Step {timestep:>10,} |"
        for key, value in metrics.items():
            if isinstance(value, float):
                msg += f" {key}: {value:.6f} |"
            else:
                msg += f" {key}: {value} |"
        
        self.info(msg)
    
    def export_json(self, output_path: Optional[str] = None) -> str:
        """
        Export all logged data to JSON
        
        Args:
            output_path: Output file path (optional)
        
        Returns:
            Path to JSON file
        """
        if output_path is None:
            output_path = self.log_dir / f"{self.exp_name}_log.json"
        else:
            output_path = Path(output_path)
        
        data = {
            'experiment_name': self.exp_name,
            'log_file': str(self.log_file),
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration_seconds': self.end_time - self.start_time if self.end_time else None,
            'hyperparameters': self.hyperparameters,
            'training_log': self.training_log,
            'evaluation_log': self.evaluation_log
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        self.info(f"Exported log to JSON: {output_path}")
        return str(output_path)
    
    def close(self):
        """Close logger and export final data"""
        if self.end_time is None:
            self.end_time = time.time()
        
        # Export to JSON
        self.export_json()
        
        # Close handlers
        for handler in self.logger.handlers:
            handler.close()
        
        self.info("Logger closed")


class TensorboardLogger:
    """Optional Tensorboard logger (requires tensorboard package)"""
    
    def __init__(self, log_dir: str, exp_name: str):
        """
        Args:
            log_dir: Tensorboard log directory
            exp_name: Experiment name
        """
        self.log_dir = Path(log_dir) / exp_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
            self.enabled = True
        except ImportError:
            print("Warning: Tensorboard not available. Install with: pip install tensorboard")
            self.enabled = False
    
    def log_scalar(self, tag: str, value: float, step: int):
        """Log scalar value"""
        if self.enabled:
            self.writer.add_scalar(tag, value, step)
    
    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        """Log multiple scalars"""
        if self.enabled:
            self.writer.add_scalars(main_tag, tag_scalar_dict, step)
    
    def close(self):
        """Close writer"""
        if self.enabled:
            self.writer.close()


class MetricsLogger:
    """Simple CSV logger for metrics"""
    
    def __init__(self, output_file: str):
        """
        Args:
            output_file: CSV output path
        """
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.data = []
        self.columns = None
    
    def log(self, data_dict: Dict[str, Any]):
        """Log data row"""
        if self.columns is None:
            self.columns = list(data_dict.keys())
        self.data.append(data_dict)
    
    def save(self):
        """Save to CSV"""
        if not self.data:
            return
        
        import csv
        with open(self.output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            writer.writeheader()
            writer.writerows(self.data)
        
        print(f"Metrics saved to: {self.output_file}")
