"""
Reward Functions for Active Suspension Control
8 different reward formulations - easily editable
"""

import numpy as np


class RewardFunctions:
    """Collection of reward functions for active suspension control"""
    
    @staticmethod
    def reward_1(x_s, dx_s, x_us, dx_us, ddx_s, u, x_r, u_max=8000.0):
        """
        Reward 1: Balanced approach (Paper baseline)
        Focus: Equal weight on comfort, tracking, and tire contact
        """
        tracking_error = (x_s - x_r) ** 2
        tire_contact = (x_us - x_r) ** 2
        comfort_cost = ddx_s ** 2
        control_cost = (u / u_max) ** 2
        
        reward = -(
            10.0 * tracking_error +
            1.0 * tire_contact +
            1.0 * comfort_cost +
            0.00001 * control_cost
        )
        return reward
    
    @staticmethod
    def reward_2(x_s, dx_s, x_us, dx_us, ddx_s, u, x_r, u_max=8000.0):
        """
        Reward 2: Comfort-focused
        Focus: Heavily penalize body acceleration (passenger comfort)
        """
        tracking_error = (x_s - x_r) ** 2
        tire_contact = (x_us - x_r) ** 2
        comfort_cost = ddx_s ** 2
        control_cost = (u / u_max) ** 2
        
        reward = -(
            5.0 * tracking_error +
            1.0 * tire_contact +
            20.0 * comfort_cost +      # Much higher weight on comfort
            0.00001 * control_cost
        )
        return reward
    
    @staticmethod
    def reward_3(x_s, dx_s, x_us, dx_us, ddx_s, u, x_r, u_max=8000.0):
        """
        Reward 3: Road tracking focused
        Focus: Body should closely follow road contour
        """
        tracking_error = (x_s - x_r) ** 2
        tire_contact = (x_us - x_r) ** 2
        comfort_cost = ddx_s ** 2
        control_cost = (u / u_max) ** 2
        
        reward = -(
            50.0 * tracking_error +    # High tracking weight
            5.0 * tire_contact +
            1.0 * comfort_cost +
            0.00001 * control_cost
        )
        return reward
    
    @staticmethod
    def reward_4(x_s, dx_s, x_us, dx_us, ddx_s, u, x_r, u_max=8000.0):
        """
        Reward 4: Energy-efficient
        Focus: Minimize control effort while maintaining comfort
        """
        tracking_error = (x_s - x_r) ** 2
        tire_contact = (x_us - x_r) ** 2
        comfort_cost = ddx_s ** 2
        control_cost = (u / u_max) ** 2
        
        reward = -(
            10.0 * tracking_error +
            1.0 * tire_contact +
            5.0 * comfort_cost +
            0.1 * control_cost         # Much higher control penalty
        )
        return reward
    
    @staticmethod
    def reward_5(x_s, dx_s, x_us, dx_us, ddx_s, u, x_r, u_max=8000.0):
        """
        Reward 5: Tire grip focused
        Focus: Maximize tire-road contact (safety)
        """
        tracking_error = (x_s - x_r) ** 2
        tire_contact = (x_us - x_r) ** 2
        comfort_cost = ddx_s ** 2
        control_cost = (u / u_max) ** 2
        
        reward = -(
            10.0 * tracking_error +
            50.0 * tire_contact +      # Very high tire contact weight
            5.0 * comfort_cost +
            0.00001 * control_cost
        )
        return reward
    
    @staticmethod
    def reward_6(x_s, dx_s, x_us, dx_us, ddx_s, u, x_r, u_max=8000.0):
        """
        Reward 6: Velocity damping focused
        Focus: Minimize velocities (stability)
        """
        tracking_error = (x_s - x_r) ** 2
        tire_contact = (x_us - x_r) ** 2
        comfort_cost = ddx_s ** 2
        velocity_cost = dx_s ** 2 + dx_us ** 2  # Add velocity penalty
        control_cost = (u / u_max) ** 2
        
        reward = -(
            10.0 * tracking_error +
            1.0 * tire_contact +
            1.0 * comfort_cost +
            5.0 * velocity_cost +      # Penalize high velocities
            0.00001 * control_cost
        )
        return reward
    
    @staticmethod
    def reward_7(x_s, dx_s, x_us, dx_us, ddx_s, u, x_r, u_max=8000.0):
        """
        Reward 7: Suspension deflection focused
        Focus: Minimize suspension travel (prevent bottoming out)
        """
        tracking_error = (x_s - x_r) ** 2
        tire_contact = (x_us - x_r) ** 2
        comfort_cost = ddx_s ** 2
        suspension_deflection = (x_s - x_us) ** 2  # Suspension travel
        control_cost = (u / u_max) ** 2
        
        reward = -(
            10.0 * tracking_error +
            1.0 * tire_contact +
            1.0 * comfort_cost +
            10.0 * suspension_deflection +  # Penalize large deflection
            0.00001 * control_cost
        )
        return reward
    
    @staticmethod
    def reward_8(x_s, dx_s, x_us, dx_us, ddx_s, u, x_r, u_max=8000.0):
        """
        Reward 8: Exponential comfort (paper's advanced version)
        Focus: Non-linear penalty on acceleration (ISO 2631-5 inspired)
        """
        tracking_error = (x_s - x_r) ** 2
        tire_contact = (x_us - x_r) ** 2
        
        # Exponential comfort cost (more sensitive to large accelerations)
        comfort_cost = np.exp(abs(ddx_s) / 5.0) - 1.0
        
        control_cost = (u / u_max) ** 2
        
        reward = -(
            10.0 * tracking_error +
            1.0 * tire_contact +
            2.0 * comfort_cost +       # Exponential scaling
            0.00001 * control_cost
        )
        return reward
    
    @staticmethod
    def get_reward_function(reward_id):
        """
        Get reward function by ID
        
        Args:
            reward_id: Integer 1-8
        
        Returns:
            Reward function
        """
        reward_functions = {
            1: RewardFunctions.reward_1,
            2: RewardFunctions.reward_2,
            3: RewardFunctions.reward_3,
            4: RewardFunctions.reward_4,
            5: RewardFunctions.reward_5,
            6: RewardFunctions.reward_6,
            7: RewardFunctions.reward_7,
            8: RewardFunctions.reward_8
        }
        
        if reward_id not in reward_functions:
            raise ValueError(f"Reward ID must be 1-8, got {reward_id}")
        
        return reward_functions[reward_id]
    
    @staticmethod
    def get_reward_description(reward_id):
        """Get human-readable description of reward function"""
        descriptions = {
            1: "Balanced (baseline from paper)",
            2: "Comfort-focused (high acceleration penalty)",
            3: "Road tracking (follow road contour)",
            4: "Energy-efficient (minimize control effort)",
            5: "Tire grip (maximize road contact)",
            6: "Velocity damping (minimize velocities)",
            7: "Suspension deflection (prevent bottoming)",
            8: "Exponential comfort (non-linear acceleration penalty)"
        }
        return descriptions.get(reward_id, "Unknown")


# Wrapper class to use custom reward in environment
class RewardWrapper:
    """Wraps environment to use custom reward function"""
    
    def __init__(self, env, reward_function):
        """
        Args:
            env: Gymnasium environment
            reward_function: Callable reward function
        """
        self.env = env
        self.reward_function = reward_function
    
    def step(self, action):
        """Step with custom reward"""
        obs, _, terminated, truncated, info = self.env.step(action)
        
        # Extract state variables
        x_s, dx_s, x_us, dx_us, x_r = obs
        ddx_s = info['body_acceleration']
        u = info['control_force']
        
        # Compute custom reward
        reward = self.reward_function(
            x_s, dx_s, x_us, dx_us, ddx_s, u, x_r, self.env.u_max
        )
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        """Reset environment"""
        return self.env.reset(**kwargs)
    
    def __getattr__(self, name):
        """Forward all other attributes to wrapped environment"""
        return getattr(self.env, name)
