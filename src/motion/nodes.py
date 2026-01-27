# nodes.py
import numpy as np
from fg.factor_graph_mppi import SampleFNode
from .obstacle import ObstacleMap

class GoalSampleFNode(SampleFNode):
    def __init__(self, name: str, dims: list, goal: np.ndarray, strength: float = 2.0):
        super().__init__(name, dims, strength)
        self.goal = goal

    def update_factor(self):
        # Goal은 명확하므로 MPPI 없이 바로 Analytic Gaussian 메시지 생성
        # R -> Small (High confidence)
        msg_mean = self.goal
        # Generate covariance: I * 1e-2 (small uncertainty)
        # Scaled by strength (higher strength = smaller cov)
        msg_cov = np.eye(len(self.goal)) * (1e-2 / self.factor_strength)
        
        msg = {'mean': msg_mean, 'cov': msg_cov, 'type': 'goal'}
        
        # 메시지 전파
        for edge in self.edges:
            edge._messages[self] = msg

class ObstacleSampleFNode(SampleFNode):
    def __init__(self, name: str, dims: list, omap: ObstacleMap, strength: float = 1.0):
        super().__init__(name, dims, strength)
        self.omap = omap

    def _cost_fn(self, samples: np.ndarray):
        return self.omap.get_obstacle_cost(samples, safe_dist=1.0)

    def update_factor(self):
        # MPPI 수행 (탐색 노이즈 0.3, lambda 0.1)
        self.update_factor_with_mppi(self._cost_fn, lambda_val=0.1, exploration_sigma=0.3)

class DistSampleFNode(SampleFNode):
    """ 멀티 로봇 간 충돌 방지 """
    def __init__(self, name: str, dims: list, min_dist: float = 1.0, strength: float = 10.0):
        super().__init__(name, dims, strength)
        self.min_dist = min_dist
        self.remote_belief = None # (mean, cov)

    def set_remote_belief(self, mean, cov):
        self.remote_belief = (mean, cov)

    def _cost_fn(self, samples: np.ndarray):
        if self.remote_belief is None:
            return np.zeros(samples.shape[0])
        
        remote_mean, _ = self.remote_belief
        # samples: (N, 2)
        diff = samples[:, :2] - remote_mean[:2]
        dist = np.linalg.norm(diff, axis=1)
        
        costs = np.zeros_like(dist)
        
        # 충돌 시 매우 큰 비용
        collision = dist < self.min_dist
        costs[collision] = 500.0
        
        # 근접 시 소프트 비용
        near_mask = (dist >= self.min_dist) & (dist < self.min_dist * 2.0)
        costs[near_mask] = np.exp(-1.0 * (dist[near_mask] - self.min_dist)) * 20.0
        
        return costs

    def update_factor(self):
        if self.remote_belief is not None:
            self.update_factor_with_mppi(self._cost_fn, lambda_val=0.2, exploration_sigma=0.2)

class PriorFactor(SampleFNode):
    """ 이전 스텝과의 연결성을 유지 (Smoothness / Dynamics) """
    def __init__(self, name: str, dims: list, prev_mean: np.ndarray, strength: float = 1.0):
        super().__init__(name, dims, strength)
        self.prev_mean = prev_mean
        
    def update_factor(self):
        # 단순히 이전 위치 주변에 있어야 한다는 Gaussian 메시지
        # 실제로는 속도 등을 고려해야 하지만 여기선 단순 위치 연결성
        msg_mean = self.prev_mean
        msg_cov = np.eye(len(self.prev_mean)) * (0.1 / self.factor_strength)
        
        msg = {'mean': msg_mean, 'cov': msg_cov, 'type': 'prior'}
        for edge in self.edges:
            edge._messages[self] = msg