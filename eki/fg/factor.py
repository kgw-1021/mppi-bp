import jax.numpy as jnp
from abc import ABC, abstractmethod

# --- [Base Classes] ---

class Factor(ABC):
    """모든 팩터의 기본 클래스"""
    def __init__(self, weight=1.0):
        self.weight = weight

class UnaryFactor(Factor):
    """노드 하나(x_t)에 영향을 주는 팩터"""
    @abstractmethod
    def error(self, x_t):
        pass

class BinaryFactor(Factor):
    """두 노드(x_prev, x_curr) 사이의 관계를 정의하는 팩터"""
    @abstractmethod
    def error(self, x_prev, x_curr, dt):
        pass

# --- [Concrete Implementations] ---

class ObstacleFactor(UnaryFactor):
    """장애물 회피 팩터 (SDF 기반)"""
    def __init__(self, center, radius, safe_margin=0.5, weight=10.0):
        super().__init__(weight)
        self.center = jnp.array(center)
        self.radius = radius
        self.safe_margin = safe_margin

    def error(self, x_t):
        # x_t shape: (StateDim,)
        pos = x_t[:2]
        dist = jnp.linalg.norm(pos - self.center) - self.safe_margin
        # 장애물 안쪽이면 페널티 (양수), 바깥이면 0
        # EKI는 Error=0을 목표로 하므로, 침범 깊이를 에러로 반환
        return jnp.maximum(0.0, self.radius - dist) * self.weight

class GoalFactor(UnaryFactor):
    """목표 지점 도달 팩터"""
    def __init__(self, goal_state, weight=5.0):
        super().__init__(weight)
        self.goal_state = jnp.array(goal_state)

    def error(self, x_t):
        # 위치 오차만 고려 (필요시 속도 오차도 추가 가능)
        return jnp.linalg.norm(x_t[:2] - self.goal_state[:2]) * self.weight

class SpeedLimitFactor(UnaryFactor):
    """(예시) 속도 제한 팩터"""
    def __init__(self, max_speed, weight=5.0):
        super().__init__(weight)
        self.max_speed = max_speed

    def error(self, x_t):
        speed = jnp.linalg.norm(x_t[2:4])
        return jnp.maximum(0.0, speed - self.max_speed) * self.weight

class DynamicsFactor(BinaryFactor):
    """물리 법칙 연결 팩터 (Forward Euler)"""
    def __init__(self, weight=2.0):
        super().__init__(weight)

    def error(self, x_prev, x_curr, dt):
        # 예측: x_prev에서 물리 법칙대로 흘러갔을 때의 위치
        x, y, vx, vy = x_prev
        pred_x = x + vx * dt
        pred_y = y + vy * dt
        pred_vx = vx
        pred_vy = vy
        pred_state = jnp.array([pred_x, pred_y, pred_vx, pred_vy])
        
        # 오차: 실제 x_curr와 예측된 상태의 차이
        # 이 값이 0이어야 물리적으로 타당함
        return (x_curr - pred_state) * self.weight
    
class InterRobotFactor(BinaryFactor):
    """두 로봇(x_a, x_b) 사이의 충돌 방지 팩터"""
    def __init__(self, min_dist, weight=20.0):
        super().__init__(weight)
        self.min_dist = min_dist # 로봇 A 반지름 + 로봇 B 반지름 + 안전거리

    def error(self, x_a, x_b):
        # x_a, x_b shape: (StateDim,)
        pos_a = x_a[:2]
        pos_b = x_b[:2]
        
        dist = jnp.linalg.norm(pos_a - pos_b)
        
        # 거리가 min_dist보다 가까우면 페널티 부여
        # EKI는 Error=0을 지향하므로 침범한 만큼을 에러로 반환
        return jnp.maximum(0.0, self.min_dist - dist) * self.weight