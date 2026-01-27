# obstacle.py
import numpy as np

class ObstacleMap:
    def __init__(self):
        self.circles = [] # (x, y, r)

    def add_circle(self, x, y, r):
        self.circles.append(np.array([x, y, r]))

    def get_obstacle_cost(self, samples: np.ndarray, safe_dist: float = 0.5) -> np.ndarray:
        """
        samples: (N, D) - only uses x, y (first 2 dims)
        returns: (N,) cost
        """
        if not self.circles:
            return np.zeros(samples.shape[0])

        total_cost = np.zeros(samples.shape[0])
        
        # Vectorized calculation for all circles
        # Obstacles: (M, 3), Samples: (N, 2)
        # Distance matrix: (N, M)
        
        obs_array = np.array(self.circles)
        centers = obs_array[:, :2]
        radii = obs_array[:, 2]
        
        # Broadcasting: (N, 1, 2) - (1, M, 2)
        diff = samples[:, np.newaxis, :2] - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2) # (N, M)
        
        # Signed Distance: dist - radius
        sdf = dists - radii[np.newaxis, :]
        
        # Cost Logic:
        # 1. Collision (sdf < 0): High Cost
        # 2. Warning (0 < sdf < safe_dist): Exponential Cost
        # 3. Safe (sdf > safe_dist): 0 Cost
        
        # Collision Mask
        collision_mask = sdf < 0.0
        total_cost += np.sum(collision_mask * 1000.0, axis=1) # Hard collision
        
        # Soft Constraint
        # exp(-alpha * dist)
        safe_mask = (sdf >= 0.0) & (sdf < safe_dist)
        soft_costs = np.exp(-2.0 * sdf) 
        # Apply mask
        soft_costs[~safe_mask] = 0.0
        soft_costs[collision_mask] = 0.0 # Already handled
        
        total_cost += np.sum(soft_costs, axis=1) * 10.0
        
        return total_cost