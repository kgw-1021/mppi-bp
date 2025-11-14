from typing import Tuple, List, Dict
import numpy as np
from fg.factor_graph_mppi import SampleMessage, SampleVNode, SampleFNode

from .obstacle import ObstacleMap


class RemoteSampleVNode(SampleVNode):
    """다른 에이전트의 변수 노드 (원격)"""
    def __init__(self, name: str, dims: List[str], belief: SampleMessage = None) -> None:
        super().__init__(name, dims, belief)
        self._msgs = {}

    def update_belief(self) -> SampleMessage:
        # 원격 노드는 자체적으로 belief 업데이트하지 않음
        return None
    
    def calc_msg(self, edge):
        # 원격에서 받은 메시지 사용
        return self._msgs.get(edge, None)


class DynaSampleFNode(SampleFNode):
    """동역학 제약 팩터 노드 (MPPI 기반)"""
    def __init__(self, name: str, vnodes: List[SampleVNode], 
                 dt: float = 0.1, pos_weight: float = 10, vel_weight: float = 2,
                 mppi_params: dict = None) -> None:
        assert len(vnodes) == 2
        if mppi_params is None:
            mppi_params = {'K': 400, 'lambda': 1.0, 'noise_std': 0.5}
        super().__init__(name, vnodes, factor_samples=None, mppi_params=mppi_params)
        self._dt = dt
        self._pos_weight = pos_weight
        self._vel_weight = vel_weight

    def dynamics_cost(self, samples: np.ndarray) -> np.ndarray:
        """
        동역학 제약 비용 함수.
        samples: (K, 8) [x0, y0, vx0, vy0, x1, y1, vx1, vy1]
        Returns: (K,) costs
        """
        K = samples.shape[0]
        costs = np.zeros(K)
        
        x0, y0, vx0, vy0 = samples[:, 0], samples[:, 1], samples[:, 2], samples[:, 3]
        x1, y1, vx1, vy1 = samples[:, 4], samples[:, 5], samples[:, 6], samples[:, 7]
        
        # 위치 제약: x1 ≈ x0 + vx0 * dt
        dx = x1 - (x0 + vx0 * self._dt)
        dy = y1 - (y0 + vy0 * self._dt)
        pos_cost = (dx**2 + dy**2) * self._pos_weight
        
        # 속도 제약: vx1 ≈ vx0 (부드러운 움직임)
        dvx = vx1 - vx0
        dvy = vy1 - vy0
        vel_cost = (dvx**2 + dvy**2) * self._vel_weight
        
        costs = pos_cost + vel_cost
        return costs

    def update_factor_with_mppi(self, cost_fn=None, base_trajectory: np.ndarray = None):
        """MPPI로 팩터 업데이트"""
        # cost_fn이 제공되지 않으면 자체 cost function 사용
        if cost_fn is None:
            cost_fn = self.dynamics_cost
        
        # 연결된 변수들의 현재 belief 평균으로 base 설정
        if base_trajectory is None:
            v0_belief = self._vnodes[0].belief
            v1_belief = self._vnodes[1].belief
            v0_mean = np.average(v0_belief.samples, weights=v0_belief.weights, axis=0)
            v1_mean = np.average(v1_belief.samples, weights=v1_belief.weights, axis=0)
            base = np.concatenate([v0_mean, v1_mean])
        else:
            base = base_trajectory
        
        super().update_factor_with_mppi(cost_fn, base_trajectory=base)


class ObstacleSampleFNode(SampleFNode):
    """장애물 회피 팩터 노드 (MPPI 기반)"""
    def __init__(self, name: str, vnodes: List[SampleVNode], 
                 omap: ObstacleMap = None, safe_dist: float = 5,
                 obstacle_weight: float = 50, mppi_params: dict = None) -> None:
        assert len(vnodes) == 1
        if mppi_params is None:
            mppi_params = {'K': 400, 'lambda': 1.0, 'noise_std': 0.3}
        super().__init__(name, vnodes, factor_samples=None, mppi_params=mppi_params)
        self._omap = omap
        self._safe_dist = safe_dist
        self._obstacle_weight = obstacle_weight

    def obstacle_cost(self, samples: np.ndarray) -> np.ndarray:
        """
        장애물 회피 비용 함수.
        samples: (K, 4) [x, y, vx, vy]
        Returns: (K,) costs
        """
        if self._omap is None:
            return np.zeros(samples.shape[0])
        
        costs = self._omap.get_obstacle_cost(samples, self._safe_dist)
        return costs * self._obstacle_weight

    def update_factor_with_mppi(self, cost_fn=None, base_trajectory: np.ndarray = None):
        """MPPI로 팩터 업데이트"""
        # cost_fn이 제공되지 않으면 자체 cost function 사용
        if cost_fn is None:
            cost_fn = self.obstacle_cost
        
        if base_trajectory is None:
            v_belief = self._vnodes[0].belief
            base = v_belief.samples.mean(axis=0)
        else:
            base = base_trajectory
        
        super().update_factor_with_mppi(cost_fn, base_trajectory=base)


class DistSampleFNode(SampleFNode):
    """에이전트 간 충돌 회피 팩터 노드 (MPPI 기반)"""
    def __init__(self, name: str, vnodes: List[SampleVNode], 
                 safe_dist: float = 20, distance_weight: float = 15,
                 mppi_params: dict = None) -> None:
        assert len(vnodes) == 2
        if mppi_params is None:
            mppi_params = {'K': 400, 'lambda': 1.0, 'noise_std': 0.3}
        super().__init__(name, vnodes, factor_samples=None, mppi_params=mppi_params)
        self._safe_dist = safe_dist
        self._distance_weight = distance_weight

    def distance_cost(self, samples: np.ndarray) -> np.ndarray:
        """
        에이전트 간 거리 제약 비용 함수.
        samples: (K, 8) [x0, y0, vx0, vy0, x1, y1, vx1, vy1]
        Returns: (K,) costs
        """
        K = samples.shape[0]
        costs = np.zeros(K)
        
        x0, y0 = samples[:, 0], samples[:, 1]
        x1, y1 = samples[:, 4], samples[:, 5]
        
        distance = np.sqrt((x0 - x1)**2 + (y0 - y1)**2) + 1e-8
        
        # safe_dist보다 가까우면 패널티
        violation = np.maximum(0, self._safe_dist - distance)
        costs = (violation / self._safe_dist)**2 * self._distance_weight * 100
        
        return costs

    def update_factor_with_mppi(self, cost_fn=None, base_trajectory: np.ndarray = None):
        """MPPI로 팩터 업데이트"""
        # cost_fn이 제공되지 않으면 자체 cost function 사용
        if cost_fn is None:
            cost_fn = self.distance_cost
        
        if base_trajectory is None:
            v0_belief = self._vnodes[0].belief
            v1_belief = self._vnodes[1].belief
            base = np.concatenate([v0_belief.samples.mean(axis=0), 
                                   v1_belief.samples.mean(axis=0)])
        else:
            base = base_trajectory
        
        super().update_factor_with_mppi(cost_fn, base_trajectory=base)