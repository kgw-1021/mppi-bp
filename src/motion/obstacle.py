from typing import Tuple, List, Dict
import numpy as np


class ObstacleMap:
    def __init__(self) -> None:
        self.objects = {}

    def set_circle(self, name: str, centerx, centery, radius):
        o = {'type': 'circle', 'name': name, 'centerx': centerx, 'centery': centery, 'radius': radius}
        self.objects[name] = o

    def get_d_grad(self, x, y) -> Tuple[float, float, float]:
        mindist = np.inf
        mino = None
        for o in self.objects.values():
            if o['type'] == 'circle':
                ox, oy, r = o['centerx'], o['centery'], o['radius']
                d = np.sqrt((x - ox)**2 + (y - oy)**2) - r
                if d < mindist:
                    mindist = d
                    mino = o
        if mino is None:
            return np.inf, 0, 0
        if mino['type'] == 'circle':
            ox, oy = mino['centerx'], mino['centery']
            dx, dy = x - ox, y - oy
            mag = np.sqrt(dx**2 + dy**2) + 1e-8
            return mindist, dx/mag, dy/mag

    def get_obstacle_cost(self, samples: np.ndarray, safe_dist: float) -> np.ndarray:
        """
        장애물 cost function for MPPI.
        samples: (K, 4) array [x, y, vx, vy]
        Returns: (K,) cost values
        """
        K = samples.shape[0]
        costs = np.zeros(K)
        for i in range(K):
            x, y = samples[i, 0], samples[i, 1]
            distance, _, _ = self.get_d_grad(x, y)
            if distance < safe_dist:
                # Exponential penalty when inside safe zone
                costs[i] = np.exp(-distance / safe_dist) * 100
            else:
                costs[i] = 0
        return costs