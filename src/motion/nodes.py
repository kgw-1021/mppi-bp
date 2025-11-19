from typing import Tuple, List, Dict
import numpy as np
from fg.factor_graph_mppi import SampleMessage, SampleVNode, SampleFNode

from .obstacle import ObstacleMap


class RemoteSampleVNode(SampleVNode):
    """다른 에이전트의 변수 노드 (원격)"""
    def __init__(self, name: str, dims: List[str], belief: SampleMessage = None) -> None:
        super().__init__(name, dims, belief)
        self._outgoing_msgs = {}  # edge -> SampleMessage 매핑

    def update_belief(self) -> SampleMessage:
        """원격 노드는 상대방이 push한 belief를 그대로 사용"""
        return self._belief
    
    def calc_msg(self, edge):
        """원격 노드의 메시지는 push_msg를 통해 미리 설정됨"""
        return self._outgoing_msgs.get(edge, None)
    
    def set_outgoing_msg(self, edge, msg: SampleMessage):
        """외부에서 메시지를 설정 (push_msg에서 호출)"""
        self._outgoing_msgs[edge] = msg


class DynaSampleFNode(SampleFNode):
    """동역학 제약 팩터 노드 (MPPI 기반)"""
    def __init__(self, name: str, vnodes: List[SampleVNode], 
                 dt: float = 0.1, pos_weight: float = 10, vel_weight: float = 2, 
                 limit_weight: float = 0.1, mppi_params: dict = None, _message_exponent=1.0) -> None:
        assert len(vnodes) == 2
        if mppi_params is None:
            mppi_params = {'K': 400, 'lambda': 1.0, 'noise_std': 0.5}
        
        mppi_params['dt'] = dt
        
        super().__init__(name, vnodes, factor_samples=None, mppi_params=mppi_params, message_exponent=_message_exponent)
        self._dt = dt
        self._pos_weight = pos_weight
        self._vel_weight = vel_weight
        self.limit_weight = limit_weight
        self.MAX_SPEED = 80.0

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
        # dvx = vx1 - vx0
        # dvy = vy1 - vy0
        # vel_cost = (dvx**2 + dvy**2) * self._vel_weight
        
        speed1 = np.sqrt(vx1**2 + vy1**2)
        violation = np.maximum(0, speed1 - self.MAX_SPEED)
        max_vel_cost = violation**2 * self.limit_weight

        # costs = pos_cost + vel_cost + max_vel_cost
        costs = pos_cost + max_vel_cost

        return costs

    def update_factor_with_mppi(self, cost_fn=None, base_trajectory: np.ndarray = None):
        """MPPI로 팩터 업데이트"""
        if cost_fn is None:
            cost_fn = self.dynamics_cost
        
        if base_trajectory is None:
            v0_belief = self._vnodes[0].belief
            v1_belief = self._vnodes[1].belief

            if v0_belief is None or v1_belief is None:
                return
            
            v0_mean = np.average(v0_belief.samples, weights=v0_belief.weights, axis=0)
            v1_mean = np.average(v1_belief.samples, weights=v1_belief.weights, axis=0)
            base = np.concatenate([v0_mean, v1_mean])
        else:
            base = base_trajectory
        # print("DynamicsSampleFNode cost:\n")
        super().update_factor_with_mppi(cost_fn, base_trajectory=base)


class ObstacleSampleFNode(SampleFNode):
    """장애물 회피 팩터 노드 (MPPI 기반)"""
    def __init__(self, name: str, vnodes: List[SampleVNode], 
                 omap: ObstacleMap = None, safe_dist: float = 5,
                 obstacle_weight: float = 50, mppi_params: dict = None,
                 dt: float = 0.1, _message_exponent = 1.0) -> None:
        assert len(vnodes) == 1
        if mppi_params is None:
            mppi_params = {'K': 400, 'lambda': 1.0, 'noise_std': 0.3}
        
        mppi_params['dt'] = dt
        
        super().__init__(name, vnodes, factor_samples=None, mppi_params=mppi_params, message_exponent=_message_exponent)
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
        if cost_fn is None:
            cost_fn = self.obstacle_cost
        
        if base_trajectory is None:
            v_belief = self._vnodes[0].belief
            if v_belief is None:
                return
            base = np.average(v_belief.samples, weights=v_belief.weights, axis=0)
        else:
            base = base_trajectory
        # print("ObstacleSampleFNode cost:\n")
        super().update_factor_with_mppi(cost_fn, base_trajectory=base)


class DistSampleFNode(SampleFNode):
    """에이전트 간 충돌 회피 팩터 노드 (MPPI 기반)"""
    def __init__(self, name: str, vnodes: List[SampleVNode], 
                 safe_dist: float = 20, distance_weight: float = 15,
                 mppi_params: dict = None, dt: float = 0.1, _message_exponent = 1.0) -> None:
        assert len(vnodes) == 2
        if mppi_params is None:
            mppi_params = {'K': 400, 'lambda': 1.0, 'noise_std': 0.3}
        
        mppi_params['dt'] = dt
        
        super().__init__(name, vnodes, factor_samples=None, mppi_params=mppi_params, message_exponent=_message_exponent)
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
        if cost_fn is None:
            cost_fn = self.distance_cost
        
        if base_trajectory is None:
            v0_belief = self._vnodes[0].belief
            v1_belief = self._vnodes[1].belief
            
            if v0_belief is None or v1_belief is None:
                return
            
            v0_mean = np.average(v0_belief.samples, weights=v0_belief.weights, axis=0)
            v1_mean = np.average(v1_belief.samples, weights=v1_belief.weights, axis=0)
            base = np.concatenate([v0_mean, v1_mean])
        else:
            base = base_trajectory
        # print("DistSampleFNode cost:\n")
        super().update_factor_with_mppi(cost_fn, base_trajectory=base)

class GoalSampleFNode(SampleFNode):
    """
    목표지점(anchor) 팩터 노드.
    - GBP의 anchor 역할과 동일
    - 마지막 변수노드를 목표 위치로 강하게 끌어당김
    """
    def __init__(
        self, 
        name: str, 
        vnodes: List[SampleVNode],
        goal: Tuple[float, float], 
        goal_pos_weight: float = 50.0,
        goal_vel_weight: float = 50.0,
        mppi_params: dict = None,
        dt: float = 0.1,
        _message_exponent = 1.0
    ) -> None:

        assert len(vnodes) == 1  # unary factor

        # 기본 MPPI anchor 설정 (강하게 끌어당기도록)
        if mppi_params is None:
            mppi_params = {
                'K': 800,
                'lambda': 10.0,       # 작을수록 anchor 효과 강함
                'noise_std': 0.2,     # 작은 noise로 goal 근처 탐색
            }

        mppi_params['dt'] = dt

        super().__init__(name, vnodes, factor_samples=None, mppi_params=mppi_params, message_exponent=_message_exponent)

        self._goal = np.array(goal, dtype=float)
        self._goal_pos_weight = goal_pos_weight
        self._goal_vel_weight = goal_vel_weight 

    def goal_cost(self, samples: np.ndarray) -> np.ndarray:
        """
        samples: (K,4) [x,y,vx,vy]
        목표점까지의 거리 비용
        """
        x = samples[:, 0]
        y = samples[:, 1]
        vx = samples[:, 2]
        vy = samples[:, 3]

        # 위치 비용 (L2 distance^2)
        dx = x - self._goal[0]
        dy = y - self._goal[1]
        pos_cost = dx**2 + dy**2
        vel_cost = vx**2 + vy**2

        scale_factor = 1.0 / (pos_cost + 1e-4)
        dynamic_vel_weight = self._goal_vel_weight * scale_factor

        goal_cost = (pos_cost * self._goal_pos_weight * 2) + (vel_cost * dynamic_vel_weight)

        return goal_cost

    def update_factor_with_mppi(self, cost_fn=None, base_trajectory: np.ndarray = None):
        """
        MPPI 업데이트. base trajectory는 항상 belief의 평균으로 설정.
        """
        if cost_fn is None:
            cost_fn = self.goal_cost

        v_belief = self._vnodes[0].belief
        if v_belief is None:
            return

        # base: 현재 belief의 평균
        base = np.average(v_belief.samples, weights=v_belief.weights, axis=0)
        # print("GoalSampleFNode cost:\n")
        super().update_factor_with_mppi(cost_fn, base_trajectory=base)