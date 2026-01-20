import jax.numpy as jnp
from typing import List, Callable
from dataclasses import dataclass
from .factor import Factor

@dataclass
class AgentConfig:
    """Configuration for an agent."""
    state_dim: int = 4
    ctrl_dim: int = 2
    dt: float = 0.1
    damping: float = 0.95


class AgentNode:
    """
    Variable node representing a robot agent.
    Contains:
    - State dynamics
    - List of factors
    - EKI parameters
    """
    
    def __init__(self, agent_id: int, init_state: jnp.ndarray, 
                 config: AgentConfig = AgentConfig()):
        self.agent_id = agent_id
        self.init_state = init_state
        self.config = config
        self.factors: List[Factor] = []
    
    def add_factor(self, factor: Factor):
        """Add a factor to this agent."""
        self.factors.append(factor)
        return self
    
    def dynamics(self, state: jnp.ndarray, control: jnp.ndarray, 
                 process_noise: jnp.ndarray) -> jnp.ndarray:
        """
        State transition function with input noise.
        
        Args:
            state: [x, y, vx, vy]
            control: [ax, ay]
            process_noise: [noise_ax, noise_ay]
        """
        x, y, vx, vy = state
        ax, ay = control
        n_ax, n_ay = process_noise
        
        # Noisy acceleration
        noisy_ax = ax + n_ax
        noisy_ay = ay + n_ay
        
        # Velocity update
        new_vx = vx + noisy_ax * self.config.dt
        new_vy = vy + noisy_ay * self.config.dt
        
        # Damping
        new_vx *= self.config.damping
        new_vy *= self.config.damping
        
        # Position update
        new_x = x + new_vx * self.config.dt
        new_y = y + new_vy * self.config.dt
        
        return jnp.array([new_x, new_y, new_vx, new_vy])
    
    def compose_observation_fn(self, **context) -> Callable:
        """
        Compose all factors into a single observation function.
        
        Returns:
            obs_fn: state -> combined_observation
        """
        def obs_fn(state):
            observations = []
            for factor in self.factors:
                obs = factor.observe(state, **context)
                observations.append(obs)
            return jnp.concatenate(observations)
        
        return obs_fn
    
    def compose_target(self, **context) -> jnp.ndarray:
        """Compose target values from all factors."""
        targets = []
        for factor in self.factors:
            targets.append(factor.target(**context))
        return jnp.concatenate(targets)
    
    def compose_weights(self) -> jnp.ndarray:
        """Compose weight matrix (diagonal R) from all factors."""
        weights = []
        for factor in self.factors:
            weights.extend([factor.weight] * factor.obs_dim())
        return jnp.array(weights)