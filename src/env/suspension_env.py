"""
Active Suspension Environment with ISO 8608 Road Classes
Supports road classes A through H with proper power spectral density
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class ActiveSuspensionEnv(gym.Env):
    """
    Active Suspension System Environment
    
    ISO 8608 Road Classes:
    A - Very good (G_q = 16e-6 m³)
    B - Good (G_q = 64e-6 m³)
    C - Average (G_q = 256e-6 m³)
    D - Poor (G_q = 1024e-6 m³)
    E - Very poor (G_q = 4096e-6 m³)
    F - Bad (G_q = 16384e-6 m³)
    G - Very bad (G_q = 65536e-6 m³)
    H - Extremely bad (G_q = 262144e-6 m³)
    """
    
    # ISO 8608 road class definitions
    ROAD_CLASSES = {
        'A': 16e-6,
        'B': 64e-6,
        'C': 256e-6,
        'D': 1024e-6,
        'E': 4096e-6,
        'F': 16384e-6,
        'G': 65536e-6,
        'H': 262144e-6
    }
    
    metadata = {'render_modes': ['human']}
    
    def __init__(
        self,
        road_class='D',
        dt=0.01,
        max_steps=1500,
        vehicle_speed=20.0,
        seed=None
    ):
        """
        Args:
            road_class: ISO 8608 class ('A' to 'H')
            dt: Time step (seconds)
            max_steps: Maximum steps per episode
            vehicle_speed: Vehicle speed (m/s)
            seed: Random seed for reproducibility
        """
        super().__init__()
        
        # Physical parameters (from paper Table 4)
        self.m_s = 300.0          # Sprung mass (kg)
        self.k_s = 40000.0        # Suspension spring stiffness (N/m)
        self.m_us = 30.0          # Unsprung mass (kg)
        self.b_s = 1385.0         # Suspension damping (N·s/m)
        self.k_us = 22000.0       # Tire stiffness (N/m)
        self.b_us = 100.0         # Tire damping (N·s/m)
        self.u_max = 8000.0       # Maximum actuator force (N)
        
        # Simulation parameters
        self.dt = dt
        self.max_steps = max_steps
        self.vehicle_speed = vehicle_speed
        
        # Road profile parameters
        self.road_class = road_class.upper()
        assert self.road_class in self.ROAD_CLASSES, f"Invalid road class. Must be A-H"
        self.G_q0 = self.ROAD_CLASSES[self.road_class]  # Road roughness coefficient
        self.n0 = 0.1  # Reference spatial frequency (1/m)
        self.w_index = 2.0  # Frequency index (waviness)
        
        # Random number generator
        self._np_random = None
        if seed is not None:
            self.seed(seed)
        
        # State space: [x_s, ẋ_s, x_us, ẋ_us, x_r]
        self.observation_space = spaces.Box(
            low=np.array([-1.0, -10.0, -1.0, -10.0, -0.2], dtype=np.float32),
            high=np.array([1.0, 10.0, 1.0, 10.0, 0.2], dtype=np.float32),
            dtype=np.float32
        )
        
        # Action space: normalized control force [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        
        # Episode state
        self.state = None
        self.steps = 0
        self.time = 0.0
        self.x_r = 0.0
        self.dx_r = 0.0
        self.accelerations = []
        self.displacements = []
        self.forces = []
        
        # Road profile generation
        self.road_profile = None
        self._generate_road_profile()
    
    def seed(self, seed=None):
        """Set random seed for reproducibility"""
        self._np_random = np.random.RandomState(seed)
        return [seed]
    
    def _generate_road_profile(self):
        """
        Generate ISO 8608 road profile using power spectral density
        
        Road elevation PSD: G_q(n) = G_q(n0) * (n/n0)^(-w)
        where n is spatial frequency (cycles/m)
        """
        # Total distance for episode
        total_distance = self.vehicle_speed * self.max_steps * self.dt
        
        # Spatial sampling
        dx = self.vehicle_speed * self.dt  # Spatial step
        x = np.arange(0, total_distance, dx)
        
        # Frequency range (avoid zero frequency)
        n_min = 0.01  # cycles/m
        n_max = 10.0  # cycles/m
        N = len(x)
        
        # Generate road using inverse FFT method
        df = 1.0 / total_distance
        frequencies = np.fft.rfftfreq(N, dx)
        frequencies[0] = n_min  # Avoid division by zero
        
        # Power spectral density
        G_q = self.G_q0 * (frequencies / self.n0) ** (-self.w_index)
        
        # Generate random phases
        if self._np_random is None:
            phases = np.random.uniform(0, 2*np.pi, len(frequencies))
        else:
            phases = self._np_random.uniform(0, 2*np.pi, len(frequencies))
        
        # Complex spectrum
        amplitude = np.sqrt(2 * G_q * df)
        spectrum = amplitude * np.exp(1j * phases)
        spectrum[0] = 0  # Zero mean
        
        # Inverse FFT to get road profile
        road_elevation = np.fft.irfft(spectrum, N)
        
        # Store road profile
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
        Compute state derivatives (Equations 2 & 3 from paper)
        
        Sprung mass:
        m_s·ẍ_s = -b_s(ẋ_s - ẋ_us) - k_s(x_s - x_us) + U
        
        Unsprung mass:
        m_us·ẍ_us = b_s(ẋ_s - ẋ_us) + k_s(x_s - x_us) 
                   + b_us(ẋ_r - ẋ_us) + k_us(x_r - x_us) - U
        """
        x_s, dx_s, x_us, dx_us = state
        
        # Sprung mass acceleration
        ddx_s = (
            -self.b_s * (dx_s - dx_us) 
            - self.k_s * (x_s - x_us) 
            + u
        ) / self.m_s
        
        # Unsprung mass acceleration
        ddx_us = (
            self.b_s * (dx_s - dx_us) 
            + self.k_s * (x_s - x_us)
            + self.b_us * (dx_r - dx_us)
            + self.k_us * (x_r - x_us)
            - u
        ) / self.m_us
        
        return np.array([dx_s, ddx_s, dx_us, ddx_us])
    
    def step(self, action):
        """Execute one timestep"""
        # Denormalize action to actual force
        u = np.clip(action[0] * self.u_max, -self.u_max, self.u_max)
        
        # Get road input at current time
        self.x_r, self.dx_r = self._get_road_input(self.time)
        
        # RK4 integration for numerical accuracy
        k1 = self._compute_derivatives(self.state, u, self.x_r, self.dx_r)
        k2 = self._compute_derivatives(self.state + 0.5*self.dt*k1, u, self.x_r, self.dx_r)
        k3 = self._compute_derivatives(self.state + 0.5*self.dt*k2, u, self.x_r, self.dx_r)
        k4 = self._compute_derivatives(self.state + self.dt*k3, u, self.x_r, self.dx_r)
        
        self.state = self.state + (self.dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        # Update time and step counter
        self.time += self.dt
        self.steps += 1
        
        # Extract state variables
        x_s, dx_s, x_us, dx_us = self.state
        
        # Compute body acceleration
        ddx_s = (
            -self.b_s * (dx_s - dx_us) 
            - self.k_s * (x_s - x_us) 
            + u
        ) / self.m_s
        
        # Track metrics
        self.accelerations.append(ddx_s)
        self.displacements.append(abs(x_s - x_us))
        self.forces.append(u)
        
        # Observation
        obs = np.array([x_s, dx_s, x_us, dx_us, self.x_r], dtype=np.float32)
        
        # Reward (computed by external reward function in training)
        # Default reward for environment
        reward = self._default_reward(x_s, dx_s, x_us, dx_us, ddx_s, u)
        
        # Episode termination
        terminated = False
        truncated = self.steps >= self.max_steps
        
        # Info dictionary
        info = {
            'acceleration_rms': np.sqrt(np.mean(np.array(self.accelerations)**2)),
            'body_acceleration': ddx_s,
            'suspension_deflection': abs(x_s - x_us),
            'tire_deflection': abs(x_us - self.x_r),
            'control_force': u,
            'road_class': self.road_class,
            'time': self.time
        }
        
        return obs, reward, terminated, truncated, info
    
    def _default_reward(self, x_s, dx_s, x_us, dx_us, ddx_s, u):
        """Default reward function (can be overridden)"""
        tracking_error = (x_s - self.x_r) ** 2
        tire_contact = (x_us - self.x_r) ** 2
        comfort_cost = ddx_s ** 2
        control_cost = (u / self.u_max) ** 2
        
        reward = -(
            10.0 * tracking_error +
            1.0 * tire_contact +
            1.0 * comfort_cost +
            0.00001 * control_cost
        )
        
        return reward
    
    def reset(self, seed=None, options=None):
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        if seed is not None:
            self.seed(seed)
        
        # Generate new road profile for this episode
        self._generate_road_profile()
        
        # Reset state
        self.state = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.time = 0.0
        self.steps = 0
        self.x_r = 0.0
        self.dx_r = 0.0
        
        # Reset metrics
        self.accelerations = []
        self.displacements = []
        self.forces = []
        
        obs = np.array([*self.state, self.x_r], dtype=np.float32)
        info = {'road_class': self.road_class}
        
        return obs, info
    
    def render(self):
        """Render environment (optional)"""
        pass
    
    def close(self):
        """Cleanup"""
        pass
    
    def get_road_roughness_description(self):
        """Get human-readable road class description"""
        descriptions = {
            'A': 'Very good',
            'B': 'Good',
            'C': 'Average',
            'D': 'Poor',
            'E': 'Very poor',
            'F': 'Bad',
            'G': 'Very bad',
            'H': 'Extremely bad'
        }
        return descriptions[self.road_class]
