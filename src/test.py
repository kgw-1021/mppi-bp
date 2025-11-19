import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle, Rectangle
import time
from typing import List, Tuple, Dict
from dataclasses import dataclass, field
from functools import partial
import pickle

from fg.factor_graph_mppi import SampleVNode, SampleFNode, SampleFactorGraph, SampleMessage

@dataclass
class MovingObstacle:
    """동적 장애물"""
    position: np.ndarray  # [x, y]
    velocity: np.ndarray  # [vx, vy]
    radius: float
    
    def update(self, dt: float):
        self.position += self.velocity * dt
    
    def get_future_position(self, t: float) -> np.ndarray:
        return self.position + self.velocity * t

@dataclass
class ExperimentConfig:
    """실험 설정"""
    # Environment
    env_size: Tuple[float, float] = (20.0, 20.0)
    
    # Agent
    start_pos: np.ndarray = field(default_factory=partial(np.array, [2.0, 2.0]))
    initial_goal: np.ndarray = field(default_factory=partial(np.array, [18.0, 18.0]))
    max_velocity: float = 2.0
    
    # Planning
    planning_horizon: int = 10  # timesteps
    dt: float = 0.2  # seconds per timestep
    replanning_frequency: float = 0.4  # replan every N seconds
    
    # Moving obstacles
    n_moving_obstacles: int = 3
    obstacle_radius: float = 1.0
    obstacle_speed_range: Tuple[float, float] = (0.5, 1.5)
    
    # Static obstacles
    static_obstacles: List[Tuple[float, float, float]] = None  # (x, y, radius)
    
    # Goal changes
    goal_change_times: List[float] = None  # times when goal changes
    new_goals: List[np.ndarray] = None
    
    # Particle settings
    N_particles: int = 500
    N_factor_particles: int = 800
    
    # BP settings
    bp_iterations: int = 3
    
    def __post_init__(self):
        if self.static_obstacles is None:
            self.static_obstacles = [
                (10.0, 10.0, 1.5),
                (15.0, 8.0, 1.2),
                (8.0, 15.0, 1.0),
            ]
        if self.goal_change_times is None:
            self.goal_change_times = [5.0, 10.0]
            self.new_goals = [
                np.array([18.0, 5.0]),
                np.array([5.0, 18.0])
            ]

class DynamicEnvironment:
    """동적 환경 시뮬레이터"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.current_time = 0.0
        self.current_goal = config.initial_goal.copy()
        self.goal_change_idx = 0
        
        # Initialize moving obstacles
        self.moving_obstacles = self._initialize_moving_obstacles()
        
    def _initialize_moving_obstacles(self) -> List[MovingObstacle]:
        obstacles = []
        cfg = self.config
        
        for i in range(cfg.n_moving_obstacles):
            # Random position (not too close to start/goal)
            while True:
                pos = np.random.uniform([3, 3], [cfg.env_size[0]-3, cfg.env_size[1]-3])
                if (np.linalg.norm(pos - cfg.start_pos) > 3.0 and 
                    np.linalg.norm(pos - cfg.initial_goal) > 3.0):
                    break
            
            # Random velocity
            speed = np.random.uniform(*cfg.obstacle_speed_range)
            angle = np.random.uniform(0, 2*np.pi)
            vel = speed * np.array([np.cos(angle), np.sin(angle)])
            
            obstacles.append(MovingObstacle(pos, vel, cfg.obstacle_radius))
        
        return obstacles
    
    def update(self, dt: float):
        """환경 업데이트"""
        self.current_time += dt
        
        # Update moving obstacles
        for obs in self.moving_obstacles:
            obs.update(dt)
            # Bounce off walls
            if obs.position[0] < obs.radius or obs.position[0] > self.config.env_size[0] - obs.radius:
                obs.velocity[0] *= -1
                obs.position[0] = np.clip(obs.position[0], obs.radius, self.config.env_size[0] - obs.radius)
            if obs.position[1] < obs.radius or obs.position[1] > self.config.env_size[1] - obs.radius:
                obs.velocity[1] *= -1
                obs.position[1] = np.clip(obs.position[1], obs.radius, self.config.env_size[1] - obs.radius)
        
        # Check for goal changes
        if (self.goal_change_idx < len(self.config.goal_change_times) and 
            self.current_time >= self.config.goal_change_times[self.goal_change_idx]):
            self.current_goal = self.config.new_goals[self.goal_change_idx].copy()
            self.goal_change_idx += 1
            print(f"[{self.current_time:.2f}s] Goal changed to {self.current_goal}")
    
    def get_current_state(self) -> Dict:
        """현재 환경 상태 반환"""
        return {
            'time': self.current_time,
            'goal': self.current_goal.copy(),
            'moving_obstacles': [(obs.position.copy(), obs.velocity.copy(), obs.radius) 
                                 for obs in self.moving_obstacles],
            'static_obstacles': self.config.static_obstacles.copy()
        }

class FactorGraphPlanner:
    """Factor Graph 기반 플래너"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.graph = None
        self.vnodes = []
        self.factors = {}
        self.last_plan_time = 0.0
        self.current_trajectory = None
        
        # Message exponents (can be tuned)
        self.exponents = {
            'dynamics': 0.3,  # Weak influence for smoothness
            'goal': 1.5,      # Strong influence for reaching goal
            'collision': 1.2, # Strong influence for safety
        }
    
    def initialize_graph(self, current_state: np.ndarray, env_state: Dict):
        """그래프 초기화"""
        T = self.config.planning_horizon
        dt = self.config.dt
        
        # Create variable nodes for each timestep
        self.vnodes = []
        for t in range(T):
            dims = [f'x{t}', f'y{t}', f'vx{t}', f'vy{t}']
            
            # Initialize with current state for t=0, forward prediction for t>0
            if t == 0:
                samples = np.tile(current_state, (self.config.N_particles, 1))
                samples += np.random.randn(self.config.N_particles, 4) * 0.1
            else:
                # Simple forward prediction
                prev_mean = np.mean(self.vnodes[t-1]._belief.samples, axis=0)
                pred_pos = prev_mean[:2] + prev_mean[2:] * dt
                pred_vel = prev_mean[2:]
                pred_state = np.concatenate([pred_pos, pred_vel])
                samples = np.tile(pred_state, (self.config.N_particles, 1))
                samples += np.random.randn(self.config.N_particles, 4) * 0.5
            
            vnode = SampleVNode(f'state_{t}', dims, 
                               SampleMessage(dims, samples),
                               N_particles=self.config.N_particles)
            self.vnodes.append(vnode)
        
        # Create factor graph
        self.graph = SampleFactorGraph()
        for v in self.vnodes:
            self.graph.add_node(v)
        
        # Add factors
        self._add_dynamics_factors(dt)
        self._add_goal_factor(env_state['goal'])
        self._add_collision_factors(env_state)
    
    def _add_dynamics_factors(self, dt: float):
        """Dynamics smoothness factors"""
        
        for t in range(len(self.vnodes) - 1):
            v_curr = self.vnodes[t]
            v_next = self.vnodes[t+1]
            
            dims = v_curr.dims + v_next.dims
            
            def make_dynamics_cost(t_idx, dt_val):
                def dynamics_cost(samples):
                    # samples: (K, 8) = [x_t, y_t, vx_t, vy_t, x_t+1, y_t+1, vx_t+1, vy_t+1]
                    x_t, y_t, vx_t, vy_t = samples[:, 0], samples[:, 1], samples[:, 2], samples[:, 3]
                    x_tp1, y_tp1, vx_tp1, vy_tp1 = samples[:, 4], samples[:, 5], samples[:, 6], samples[:, 7]
                    
                    # Position consistency: x_{t+1} ≈ x_t + vx_t * dt
                    dx_pred = x_t + vx_t * dt_val
                    dy_pred = y_t + vy_t * dt_val
                    pos_error = (x_tp1 - dx_pred)**2 + (y_tp1 - dy_pred)**2
                    
                    # Velocity smoothness: v_{t+1} ≈ v_t
                    vel_error = (vx_tp1 - vx_t)**2 + (vy_tp1 - vy_t)**2
                    
                    # Velocity magnitude constraint
                    vel_mag = np.sqrt(vx_t**2 + vy_t**2)
                    vel_penalty = np.maximum(0, vel_mag - self.config.max_velocity)**2 * 10.0
                    
                    return pos_error * 1.0 + vel_error * 0.5 + vel_penalty
                
                return dynamics_cost
            
            factor = SampleFNode(
                f'dynamics_{t}',
                [v_curr, v_next],
                mppi_params={'K': self.config.N_factor_particles, 'lambda': 1.0, 
                            'noise_std': 0.5, 'dt': dt},
                N_factor_particles=self.config.N_factor_particles,
                message_exponent=self.exponents['dynamics']
            )
            
            # Store cost function for later updates
            self.factors[f'dynamics_{t}'] = (factor, make_dynamics_cost(t, dt), {})
            
            self.graph.add_node(factor)
            self.graph.add_edge(v_curr, factor)
            self.graph.add_edge(v_next, factor)
    
    def _add_goal_factor(self, goal: np.ndarray):
        """Goal reaching factor (final timestep)"""
        
        v_final = self.vnodes[-1]
        
        def goal_cost(samples):
            # samples: (K, 4) = [x, y, vx, vy]
            x, y = samples[:, 0], samples[:, 1]
            dist_sq = (x - goal[0])**2 + (y - goal[1])**2
            
            # Also penalize high velocity at goal
            vx, vy = samples[:, 2], samples[:, 3]
            vel_sq = vx**2 + vy**2
            
            return dist_sq * 10.0 + vel_sq * 1.0
        
        factor = SampleFNode(
            'goal',
            [v_final],
            mppi_params={'K': self.config.N_factor_particles, 'lambda': 1.0, 
                        'noise_std': 0.3, 'dt': self.config.dt},
            N_factor_particles=self.config.N_factor_particles,
            message_exponent=self.exponents['goal']
        )
        
        self.factors['goal'] = (factor, goal_cost, {})
        
        self.graph.add_node(factor)
        self.graph.add_edge(v_final, factor)
    
    def _add_collision_factors(self, env_state: Dict):
        """Collision avoidance factors"""
        
        dt = self.config.dt
        
        for t, vnode in enumerate(self.vnodes):
            future_time = t * dt
            
            def make_collision_cost(t_idx, t_future, env):
                def collision_cost(samples):
                    # samples: (K, 4) = [x, y, vx, vy]
                    x, y = samples[:, 0], samples[:, 1]
                    cost = np.zeros(len(x))
                    
                    # Static obstacles
                    for ox, oy, r in env['static_obstacles']:
                        dist = np.sqrt((x - ox)**2 + (y - oy)**2)
                        safe_dist = r + 0.5  # safety margin
                        collision_penalty = np.maximum(0, safe_dist - dist)**2 * 100.0
                        cost += collision_penalty
                    
                    # Moving obstacles (predict their future position)
                    for obs_pos, obs_vel, obs_r in env['moving_obstacles']:
                        future_obs_pos = obs_pos + obs_vel * t_future
                        dist = np.sqrt((x - future_obs_pos[0])**2 + (y - future_obs_pos[1])**2)
                        safe_dist = obs_r + 0.5
                        collision_penalty = np.maximum(0, safe_dist - dist)**2 * 100.0
                        cost += collision_penalty
                    
                    # Boundary constraints
                    boundary_penalty = (
                        np.maximum(0, 1.0 - x)**2 * 50.0 +
                        np.maximum(0, x - (self.config.env_size[0] - 1.0))**2 * 50.0 +
                        np.maximum(0, 1.0 - y)**2 * 50.0 +
                        np.maximum(0, y - (self.config.env_size[1] - 1.0))**2 * 50.0
                    )
                    cost += boundary_penalty
                    
                    return cost
                
                return collision_cost
            
            factor = SampleFNode(
                f'collision_{t}',
                [vnode],
                mppi_params={'K': self.config.N_factor_particles, 'lambda': 1.0, 
                            'noise_std': 0.4, 'dt': dt},
                N_factor_particles=self.config.N_factor_particles,
                message_exponent=self.exponents['collision']
            )
            
            self.factors[f'collision_{t}'] = (factor, make_collision_cost(t, future_time, env_state), {})
            
            self.graph.add_node(factor)
            self.graph.add_edge(vnode, factor)
    
    def update_factors(self, env_state: Dict):
        """환경 변화에 따라 팩터 업데이트"""
        # Update goal factor
        goal = env_state['goal']
        
        def goal_cost(samples):
            x, y = samples[:, 0], samples[:, 1]
            dist_sq = (x - goal[0])**2 + (y - goal[1])**2
            vx, vy = samples[:, 2], samples[:, 3]
            vel_sq = vx**2 + vy**2
            return dist_sq * 10.0 + vel_sq * 1.0
        
        self.factors['goal'] = (self.factors['goal'][0], goal_cost, {})
        
        # Update collision factors
        dt = self.config.dt
        for t in range(len(self.vnodes)):
            future_time = t * dt
            
            def make_collision_cost(t_idx, t_future, env):
                def collision_cost(samples):
                    x, y = samples[:, 0], samples[:, 1]
                    cost = np.zeros(len(x))
                    
                    for ox, oy, r in env['static_obstacles']:
                        dist = np.sqrt((x - ox)**2 + (y - oy)**2)
                        safe_dist = r + 0.5
                        collision_penalty = np.maximum(0, safe_dist - dist)**2 * 100.0
                        cost += collision_penalty
                    
                    for obs_pos, obs_vel, obs_r in env['moving_obstacles']:
                        future_obs_pos = obs_pos + obs_vel * t_future
                        dist = np.sqrt((x - future_obs_pos[0])**2 + (y - future_obs_pos[1])**2)
                        safe_dist = obs_r + 0.5
                        collision_penalty = np.maximum(0, safe_dist - dist)**2 * 100.0
                        cost += collision_penalty
                    
                    boundary_penalty = (
                        np.maximum(0, 1.0 - x)**2 * 50.0 +
                        np.maximum(0, x - (self.config.env_size[0] - 1.0))**2 * 50.0 +
                        np.maximum(0, 1.0 - y)**2 * 50.0 +
                        np.maximum(0, y - (self.config.env_size[1] - 1.0))**2 * 50.0
                    )
                    cost += boundary_penalty
                    
                    return cost
                
                return collision_cost
            
            factor_key = f'collision_{t}'
            self.factors[factor_key] = (self.factors[factor_key][0], 
                                       make_collision_cost(t, future_time, env_state), {})
    
    def plan(self, current_state: np.ndarray, env_state: Dict, 
             force_replan: bool = False) -> np.ndarray:
        """계획 수립"""
        current_time = env_state['time']
        
        # Check if replanning is needed
        if (not force_replan and 
            self.current_trajectory is not None and
            current_time - self.last_plan_time < self.config.replanning_frequency):
            return self.current_trajectory
        
        print(f"[{current_time:.2f}s] Replanning...")
        start_time = time.time()
        
        # Initialize or update graph
        if self.graph is None:
            self.initialize_graph(current_state, env_state)
        else:
            # Update first vnode with current state
            dims = self.vnodes[0].dims
            samples = np.tile(current_state, (self.config.N_particles, 1))
            samples += np.random.randn(self.config.N_particles, 4) * 0.05
            self.vnodes[0]._belief = SampleMessage(dims, samples)
            
            # Update factors with new environment state
            self.update_factors(env_state)
        
        # Prepare factor costs for BP
        factor_costs = {}
        for name, (factor, cost_fn, kwargs) in self.factors.items():
            factor_costs[factor] = (cost_fn, kwargs)
        
        # Run belief propagation
        beliefs = self.graph.loopy_propagate(
            steps=self.config.bp_iterations,
            factor_costs=factor_costs,
            density_method='kde'
        )
        
        # Extract trajectory (mean of beliefs)
        trajectory = []
        for vnode in self.vnodes:
            belief = beliefs[vnode]
            mean_state = np.average(belief.samples, weights=belief.weights, axis=0)
            trajectory.append(mean_state[:2])  # [x, y]
        
        self.current_trajectory = np.array(trajectory)
        self.last_plan_time = current_time
        
        elapsed = time.time() - start_time
        print(f"  Planning took {elapsed:.3f}s")
        
        return self.current_trajectory
    
    def get_next_action(self, trajectory: np.ndarray, dt: float) -> np.ndarray:
        """다음 액션 추출 (MPC 스타일)"""
        if len(trajectory) < 2:
            return np.array([0.0, 0.0])
        
        # Use first step velocity
        velocity = (trajectory[1] - trajectory[0]) / dt
        
        # Clip to max velocity
        speed = np.linalg.norm(velocity)
        if speed > self.config.max_velocity:
            velocity = velocity / speed * self.config.max_velocity
        
        return velocity

class BaselinePlanner:
    """비교를 위한 베이스라인 플래너"""
    
    def __init__(self, config: ExperimentConfig, method: str = 'straight_line'):
        self.config = config
        self.method = method
        self.last_plan_time = 0.0
        self.current_trajectory = None
    
    def plan(self, current_state: np.ndarray, env_state: Dict, 
             force_replan: bool = False) -> np.ndarray:
        """계획 수립"""
        current_time = env_state['time']
        
        # Check if replanning is needed
        if (not force_replan and 
            self.current_trajectory is not None and
            current_time - self.last_plan_time < self.config.replanning_frequency):
            return self.current_trajectory
        
        start_time = time.time()
        
        if self.method == 'straight_line':
            trajectory = self._plan_straight_line(current_state, env_state)
        elif self.method == 'simple_avoid':
            trajectory = self._plan_simple_avoid(current_state, env_state)
        else:
            raise ValueError(f"Unknown baseline method: {self.method}")
        
        self.current_trajectory = trajectory
        self.last_plan_time = current_time
        
        elapsed = time.time() - start_time
        print(f"[{current_time:.2f}s] Baseline ({self.method}) planning took {elapsed:.3f}s")
        
        return self.current_trajectory
    
    def _plan_straight_line(self, current_state: np.ndarray, env_state: Dict) -> np.ndarray:
        """직선 경로"""
        current_pos = current_state[:2]
        goal = env_state['goal']
        
        trajectory = []
        for t in range(self.config.planning_horizon):
            alpha = (t + 1) / self.config.planning_horizon
            pos = current_pos * (1 - alpha) + goal * alpha
            trajectory.append(pos)
        
        return np.array(trajectory)
    
    def _plan_simple_avoid(self, current_state: np.ndarray, env_state: Dict) -> np.ndarray:
        """간단한 회피 알고리즘"""
        current_pos = current_state[:2]
        goal = env_state['goal']
        dt = self.config.dt
        
        trajectory = [current_pos.copy()]
        pos = current_pos.copy()
        
        for t in range(self.config.planning_horizon):
            # Direction to goal
            to_goal = goal - pos
            dist_to_goal = np.linalg.norm(to_goal)
            
            if dist_to_goal < 0.1:
                trajectory.append(goal.copy())
                continue
            
            direction = to_goal / dist_to_goal
            
            # Check obstacles and adjust direction
            repulsion = np.zeros(2)
            
            # Static obstacles
            for ox, oy, r in env_state['static_obstacles']:
                obs_pos = np.array([ox, oy])
                to_obs = pos - obs_pos
                dist = np.linalg.norm(to_obs)
                if dist < r + 2.0:  # influence radius
                    repulsion += to_obs / (dist**2 + 0.1)
            
            # Moving obstacles
            for obs_pos, obs_vel, obs_r in env_state['moving_obstacles']:
                future_obs_pos = obs_pos + obs_vel * t * dt
                to_obs = pos - future_obs_pos
                dist = np.linalg.norm(to_obs)
                if dist < obs_r + 2.0:
                    repulsion += to_obs / (dist**2 + 0.1)
            
            # Combine attraction and repulsion
            desired_direction = direction + repulsion * 0.5
            desired_direction = desired_direction / (np.linalg.norm(desired_direction) + 1e-6)
            
            # Move
            step = desired_direction * self.config.max_velocity * dt
            pos = pos + step
            trajectory.append(pos.copy())
        
        return np.array(trajectory[1:])
    
    def get_next_action(self, trajectory: np.ndarray, dt: float) -> np.ndarray:
        """다음 액션 추출"""
        if len(trajectory) < 1:
            return np.array([0.0, 0.0])
        
        # Use first step as target
        velocity = (trajectory[0] - trajectory[0]) / dt if len(trajectory) < 2 else (trajectory[1] - trajectory[0]) / dt
        
        speed = np.linalg.norm(velocity)
        if speed > self.config.max_velocity:
            velocity = velocity / speed * self.config.max_velocity
        
        return velocity

@dataclass
class ExperimentMetrics:
    """실험 메트릭"""
    method_name: str
    
    # Success metrics
    reached_goal: bool
    final_distance_to_goal: float
    
    # Efficiency metrics
    total_time: float
    path_length: float
    avg_velocity: float
    
    # Safety metrics
    min_obstacle_distance: float
    num_collisions: int
    collision_times: List[float]
    
    # Replanning metrics
    num_replans: int
    total_planning_time: float
    avg_planning_time: float
    
    # Trajectory smoothness
    trajectory_jerk: float  # sum of acceleration changes
    
    # Goal adaptability
    goal_switch_response_times: List[float]  # time to adapt after goal change

class ExperimentRunner:
    """실험 실행기"""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.env = DynamicEnvironment(config)
        
    def run_experiment(self, planner, method_name: str, 
                       max_time: float = 20.0, visualize: bool = True) -> Tuple[ExperimentMetrics, Dict]:
        """실험 실행"""
        print(f"\n{'='*60}")
        print(f"Running experiment: {method_name}")
        print(f"{'='*60}")
        
        # Reset environment
        self.env = DynamicEnvironment(self.config)
        
        # Initial state
        current_state = np.concatenate([self.config.start_pos, [0.0, 0.0]])
        
        # Tracking variables
        trajectory_history = [current_state[:2].copy()]
        time_history = [0.0]
        velocity_history = [[0.0, 0.0]]
        planning_times = []
        num_replans = 0
        collision_count = 0
        collision_times = []
        min_obs_dist = float('inf')
        goal_switch_times = []
        
        dt_sim = 0.05  # simulation timestep (finer than planning dt)
        sim_time = 0.0
        
        while sim_time < max_time:
            # Get current environment state
            env_state = self.env.get_current_state()
            
            # Check for goal changes
            if (len(goal_switch_times) < len(self.config.goal_change_times) and
                abs(sim_time - self.config.goal_change_times[len(goal_switch_times)]) < dt_sim):
                goal_switch_times.append(sim_time)
            
            # Plan
            start_plan = time.time()
            trajectory = planner.plan(current_state, env_state)
            planning_time = time.time() - start_plan
            
            if planning_time > 0.001:  # actual replan happened
                planning_times.append(planning_time)
                num_replans += 1
            
            # Get action
            velocity = planner.get_next_action(trajectory, self.config.dt)
            
            # Simulate
            current_state[:2] += velocity * dt_sim
            current_state[2:] = velocity
            
            # Update environment
            self.env.update(dt_sim)
            sim_time += dt_sim
            
            # Record
            trajectory_history.append(current_state[:2].copy())
            time_history.append(sim_time)
            velocity_history.append(velocity.copy())
            
            # Check collisions and distances
            pos = current_state[:2]
            
            # Static obstacles
            for ox, oy, r in self.config.static_obstacles:
                dist = np.linalg.norm(pos - np.array([ox, oy])) - r
                min_obs_dist = min(min_obs_dist, dist)
                if dist < 0:
                    collision_count += 1
                    collision_times.append(sim_time)
            
            # Moving obstacles
            for obs in self.env.moving_obstacles:
                dist = np.linalg.norm(pos - obs.position) - obs.radius
                min_obs_dist = min(min_obs_dist, dist)
                if dist < 0:
                    collision_count += 1
                    collision_times.append(sim_time)
            
            # Check if reached goal
            if np.linalg.norm(pos - env_state['goal']) < 0.5:
                print(f"Goal reached at t={sim_time:.2f}s!")
                break
        
        # Compute metrics
        trajectory_history = np.array(trajectory_history)
        velocity_history = np.array(velocity_history)
        
        final_distance = np.linalg.norm(current_state[:2] - self.env.current_goal)
        reached_goal = final_distance < 0.5
        
        # Path length
        path_length = np.sum(np.linalg.norm(np.diff(trajectory_history, axis=0), axis=1))
        
        # Average velocity
        avg_velocity = np.mean(np.linalg.norm(velocity_history, axis=1))
        
        # Trajectory smoothness (jerk)
        accelerations = np.diff(velocity_history, axis=0) / dt_sim
        jerk = np.sum(np.linalg.norm(np.diff(accelerations, axis=0), axis=1))
        
        # Goal switch response times
        response_times = []
        for switch_time in goal_switch_times:
            # Find when agent starts moving towards new goal (velocity direction change)
            idx = np.searchsorted(time_history, switch_time)
            if idx < len(velocity_history) - 10:
                # Measure time until velocity aligns with new goal direction
                for i in range(idx, min(idx + 100, len(velocity_history))):
                    vel = velocity_history[i]
                    pos = trajectory_history[i]
                    goal_dir = self.env.current_goal - pos
                    if np.linalg.norm(goal_dir) > 0.1 and np.linalg.norm(vel) > 0.1:
                        cos_angle = np.dot(vel, goal_dir) / (np.linalg.norm(vel) * np.linalg.norm(goal_dir))
                        if cos_angle > 0.7:  # roughly aligned
                            response_times.append(time_history[i] - switch_time)
                            break
        
        # Total planning time
        total_planning_time = sum(planning_times)
        avg_planning_time = np.mean(planning_times) if planning_times else 0.0
        
        metrics = ExperimentMetrics(
            method_name=method_name,
            reached_goal=reached_goal,
            final_distance_to_goal=final_distance,
            total_time=sim_time,
            path_length=path_length,
            avg_velocity=avg_velocity,
            min_obstacle_distance=min_obs_dist,
            num_collisions=collision_count,
            collision_times=list(set(collision_times[:10])),  # unique, limited
            num_replans=num_replans,
            total_planning_time=total_planning_time,
            avg_planning_time=avg_planning_time,
            trajectory_jerk=jerk,
            goal_switch_response_times=response_times
        )
        
        history = {
            'trajectory': trajectory_history,
            'time': np.array(time_history),
            'velocity': velocity_history,
            'env_states': []
        }
        
        print(f"\nResults for {method_name}:")
        print(f"  Reached goal: {reached_goal} (final dist: {final_distance:.2f}m)")
        print(f"  Path length: {path_length:.2f}m")
        print(f"  Total time: {sim_time:.2f}s")
        print(f"  Collisions: {collision_count}")
        print(f"  Min obstacle distance: {min_obs_dist:.2f}m")
        print(f"  Replans: {num_replans} (avg time: {avg_planning_time:.3f}s)")
        print(f"  Goal switch responses: {response_times}")
        
        return metrics, history
    
    def visualize_comparison(self, results: Dict[str, Tuple[ExperimentMetrics, Dict]], 
                            save_path: str = None):
        """결과 시각화"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Plot 1: Trajectories
        ax = axes[0, 0]
        ax.set_title('Trajectories', fontsize=14, fontweight='bold')
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_xlim(0, self.config.env_size[0])
        ax.set_ylim(0, self.config.env_size[1])
        ax.grid(True, alpha=0.3)
        
        # Draw obstacles
        for ox, oy, r in self.config.static_obstacles:
            circle = Circle((ox, oy), r, color='gray', alpha=0.5)
            ax.add_patch(circle)
        
        # Draw start and goals
        ax.plot(*self.config.start_pos, 'go', markersize=12, label='Start', zorder=10)
        ax.plot(*self.config.initial_goal, 'r*', markersize=15, label='Goal 1', zorder=10)
        for i, goal in enumerate(self.config.new_goals):
            ax.plot(*goal, 'r*', markersize=15, label=f'Goal {i+2}', zorder=10)
        
        # Draw trajectories
        colors = ['blue', 'orange', 'green', 'red']
        for idx, (method_name, (metrics, history)) in enumerate(results.items()):
            traj = history['trajectory']
            ax.plot(traj[:, 0], traj[:, 1], '-', color=colors[idx % len(colors)],
                   linewidth=2, alpha=0.7, label=method_name)
        
        ax.legend(loc='best')
        
        # Plot 2: Path Length Comparison
        ax = axes[0, 1]
        ax.set_title('Path Length', fontsize=14, fontweight='bold')
        methods = list(results.keys())
        path_lengths = [results[m][0].path_length for m in methods]
        ax.bar(methods, path_lengths, color=colors[:len(methods)])
        ax.set_ylabel('Length (m)')
        ax.grid(True, axis='y', alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Plot 3: Success Rate & Collisions
        ax = axes[0, 2]
        ax.set_title('Safety Metrics', fontsize=14, fontweight='bold')
        success_rates = [1 if results[m][0].reached_goal else 0 for m in methods]
        collision_counts = [results[m][0].num_collisions for m in methods]
        
        x = np.arange(len(methods))
        width = 0.35
        ax.bar(x - width/2, success_rates, width, label='Success', color='green', alpha=0.7)
        ax.bar(x + width/2, collision_counts, width, label='Collisions', color='red', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_ylabel('Count')
        ax.legend()
        ax.grid(True, axis='y', alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Plot 4: Planning Time
        ax = axes[1, 0]
        ax.set_title('Planning Performance', fontsize=14, fontweight='bold')
        avg_times = [results[m][0].avg_planning_time * 1000 for m in methods]  # ms
        num_replans = [results[m][0].num_replans for m in methods]
        
        ax2 = ax.twinx()
        bars1 = ax.bar(x - width/2, avg_times, width, label='Avg Planning Time', 
                      color='skyblue', alpha=0.7)
        bars2 = ax2.bar(x + width/2, num_replans, width, label='Num Replans', 
                       color='orange', alpha=0.7)
        
        ax.set_xticks(x)
        ax.set_xticklabels(methods)
        ax.set_ylabel('Avg Planning Time (ms)', color='skyblue')
        ax2.set_ylabel('Number of Replans', color='orange')
        ax.tick_params(axis='y', labelcolor='skyblue')
        ax2.tick_params(axis='y', labelcolor='orange')
        ax.grid(True, axis='y', alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        
        # Plot 5: Goal Switch Response Time
        ax = axes[1, 1]
        ax.set_title('Goal Adaptability', fontsize=14, fontweight='bold')
        
        response_data = []
        labels_data = []
        for method in methods:
            response_times = results[method][0].goal_switch_response_times
            if response_times:
                response_data.append(response_times)
                labels_data.append(method)
        
        if response_data:
            bp = ax.boxplot(response_data, labels=labels_data, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
        
        ax.set_ylabel('Response Time (s)')
        ax.grid(True, axis='y', alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Plot 6: Trajectory Smoothness
        ax = axes[1, 2]
        ax.set_title('Trajectory Smoothness', fontsize=14, fontweight='bold')
        jerk_values = [results[m][0].trajectory_jerk for m in methods]
        ax.bar(methods, jerk_values, color=colors[:len(methods)], alpha=0.7)
        ax.set_ylabel('Total Jerk (lower is better)')
        ax.grid(True, axis='y', alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        plt.show()
    
    def create_animation(self, results: Dict[str, Tuple[ExperimentMetrics, Dict]], 
                        save_path: str = None, fps: int = 20):
        """애니메이션 생성"""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Setup plot
        ax.set_xlim(0, self.config.env_size[0])
        ax.set_ylim(0, self.config.env_size[1])
        ax.set_xlabel('X (m)', fontsize=12)
        ax.set_ylabel('Y (m)', fontsize=12)
        ax.set_title('Dynamic Replanning Comparison', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Draw static obstacles
        for ox, oy, r in self.config.static_obstacles:
            circle = Circle((ox, oy), r, color='gray', alpha=0.5, zorder=1)
            ax.add_patch(circle)
        
        # Initialize moving obstacles
        moving_obs_patches = []
        for obs in self.env.moving_obstacles:
            circle = Circle(obs.position, obs.radius, color='red', alpha=0.3, zorder=2)
            ax.add_patch(circle)
            moving_obs_patches.append(circle)
        
        # Initialize agent markers and trails
        colors = ['blue', 'orange', 'green', 'purple']
        agent_markers = {}
        trail_lines = {}
        
        for idx, method_name in enumerate(results.keys()):
            marker, = ax.plot([], [], 'o', color=colors[idx % len(colors)], 
                            markersize=10, label=method_name, zorder=5)
            trail, = ax.plot([], [], '-', color=colors[idx % len(colors)], 
                           linewidth=2, alpha=0.5, zorder=3)
            agent_markers[method_name] = marker
            trail_lines[method_name] = trail
        
        # Goal marker
        goal_marker, = ax.plot([], [], 'r*', markersize=20, label='Current Goal', zorder=4)
        
        # Time text
        time_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                          fontsize=12, verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.legend(loc='upper right')
        
        # Get maximum time
        max_time = max([history['time'][-1] for _, history in results.values()])
        dt_anim = 1.0 / fps
        num_frames = int(max_time / dt_anim)
        
        # Recreate environment for obstacle positions
        env_anim = DynamicEnvironment(self.config)
        
        def init():
            for marker in agent_markers.values():
                marker.set_data([], [])
            for trail in trail_lines.values():
                trail.set_data([], [])
            goal_marker.set_data([], [])
            time_text.set_text('')
            return list(agent_markers.values()) + list(trail_lines.values()) + [goal_marker, time_text]
        
        def animate(frame):
            current_time = frame * dt_anim
            
            # Update environment (for moving obstacles)
            env_anim.current_time = current_time
            for obs in env_anim.moving_obstacles:
                obs.position = obs.position + obs.velocity * dt_anim
                # Bounce off walls
                if obs.position[0] < obs.radius or obs.position[0] > self.config.env_size[0] - obs.radius:
                    obs.velocity[0] *= -1
                    obs.position[0] = np.clip(obs.position[0], obs.radius, self.config.env_size[0] - obs.radius)
                if obs.position[1] < obs.radius or obs.position[1] > self.config.env_size[1] - obs.radius:
                    obs.velocity[1] *= -1
                    obs.position[1] = np.clip(obs.position[1], obs.radius, self.config.env_size[1] - obs.radius)
            
            # Update moving obstacle patches
            for obs, patch in zip(env_anim.moving_obstacles, moving_obs_patches):
                patch.center = obs.position
            
            # Update goal
            goal_idx = sum([1 for t in self.config.goal_change_times if t <= current_time])
            if goal_idx == 0:
                current_goal = self.config.initial_goal
            else:
                current_goal = self.config.new_goals[goal_idx - 1]
            goal_marker.set_data([current_goal[0]], [current_goal[1]])
            
            # Update agents
            for method_name, (metrics, history) in results.items():
                traj = history['trajectory']
                times = history['time']
                
                # Find current position
                idx = np.searchsorted(times, current_time)
                if idx >= len(traj):
                    idx = len(traj) - 1
                
                pos = traj[idx]
                agent_markers[method_name].set_data([pos[0]], [pos[1]])
                
                # Update trail
                trail_traj = traj[:idx+1]
                trail_lines[method_name].set_data(trail_traj[:, 0], trail_traj[:, 1])
            
            time_text.set_text(f'Time: {current_time:.2f}s')
            
            return (list(agent_markers.values()) + list(trail_lines.values()) + 
                   [goal_marker, time_text] + moving_obs_patches)
        
        anim = FuncAnimation(fig, animate, init_func=init, frames=num_frames,
                           interval=1000/fps, blit=True, repeat=True)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=fps)
            print(f"Animation saved to {save_path}")
        
        plt.show()
        
        return anim

def main():
    """메인 실험 실행"""
    
    # Configuration
    config = ExperimentConfig(
        planning_horizon=10,
        dt=0.3,
        replanning_frequency=0.6,
        n_moving_obstacles=3,
        N_particles=500,
        N_factor_particles=800,
        bp_iterations=3,
        goal_change_times=[5.0, 10.0],
        new_goals=[
            np.array([18.0, 5.0]),
            np.array([5.0, 18.0])
        ]
    )
    
    runner = ExperimentRunner(config)
    
    # Run experiments
    results = {}
    
    print("\n" + "="*80)
    print("DYNAMIC ENVIRONMENT REPLANNING EXPERIMENT")
    print("="*80)
    
    # 1. Factor Graph Planner
    print("\n[1/3] Testing Factor Graph Planner...")
    fg_planner = FactorGraphPlanner(config)
    fg_metrics, fg_history = runner.run_experiment(
        fg_planner, 
        "Factor Graph BP",
        max_time=15.0,
        visualize=False
    )
    results["Factor Graph BP"] = (fg_metrics, fg_history)
    
    # 2. Baseline: Straight Line
    print("\n[2/3] Testing Baseline: Straight Line...")
    baseline_straight = BaselinePlanner(config, method='straight_line')
    straight_metrics, straight_history = runner.run_experiment(
        baseline_straight,
        "Straight Line",
        max_time=15.0,
        visualize=False
    )
    results["Straight Line"] = (straight_metrics, straight_history)
    
    # 3. Baseline: Simple Avoidance
    print("\n[3/3] Testing Baseline: Simple Avoidance...")
    baseline_avoid = BaselinePlanner(config, method='simple_avoid')
    avoid_metrics, avoid_history = runner.run_experiment(
        baseline_avoid,
        "Simple Avoidance",
        max_time=15.0,
        visualize=False
    )
    results["Simple Avoidance"] = (avoid_metrics, avoid_history)
    
    # Visualize comparison
    print("\n" + "="*80)
    print("GENERATING COMPARISON PLOTS")
    print("="*80)
    runner.visualize_comparison(results, save_path='experiment_results.png')
    
    # Create animation
    print("\n" + "="*80)
    print("GENERATING ANIMATION")
    print("="*80)
    runner.create_animation(results, save_path='experiment_animation.gif', fps=20)
    
    # Save results
    with open('experiment_metrics.pkl', 'wb') as f:
        pickle.dump(results, f)
    print("\nMetrics saved to experiment_metrics.pkl")
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY TABLE")
    print("="*80)
    print(f"{'Method':<20} {'Success':<10} {'Path(m)':<10} {'Collis.':<10} {'Replans':<10} {'Avg.Plan(ms)':<12}")
    print("-"*80)
    for method_name, (metrics, _) in results.items():
        print(f"{method_name:<20} "
              f"{'✓' if metrics.reached_goal else '✗':<10} "
              f"{metrics.path_length:<10.2f} "
              f"{metrics.num_collisions:<10} "
              f"{metrics.num_replans:<10} "
              f"{metrics.avg_planning_time*1000:<12.2f}")
    print("="*80)

if __name__ == "__main__":
    main()