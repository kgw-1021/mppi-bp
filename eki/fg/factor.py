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
    """두 노드(x_a, x_b) 사이의 관계를 정의하는 팩터"""
    @abstractmethod
    def error(self, x_a, x_b):
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

class AgentGoalFactor(UnaryFactor):
    """목표 지점 유도 팩터"""
    def __init__(self, weight=5.0):
        super().__init__(weight)

    def error(self, x_curr, **kwargs):
        agent_idx = kwargs.get('agent_idx')
        all_goals = kwargs.get('goal_states') # (M, D)
        
        if agent_idx is None or all_goals is None:
            return 0.0
            
        my_goal = all_goals[agent_idx]
        return jnp.linalg.norm(x_curr[:2] - my_goal[:2]) * self.weight
    

class SpeedLimitFactor(UnaryFactor):
    """(예시) 속도 제한 팩터"""
    def __init__(self, max_speed, weight=5.0):
        super().__init__(weight)
        self.max_speed = max_speed

    def error(self, x_t):
        speed = jnp.linalg.norm(x_t[2:4])
        return jnp.maximum(0.0, speed - self.max_speed) * self.weight

class DynamicsFactor(BinaryFactor):
    """
    개선된 물리 법칙 팩터 (Trapezoidal Integration)
    - 위치 적분: (v_prev + v_curr) / 2 를 사용하여 곡선 주행을 부드럽게 만듦
    - 가중치 분리: 위치 오차(엄격함)와 속도 변화(부드러움)를 따로 제어
    """
    def __init__(self, pos_weight=10.0, vel_weight=1.0):
        # 부모 클래스에는 대표 weight만 넘김 (실제로는 안 씀)
        super().__init__(weight=1.0) 
        self.pos_weight = pos_weight  # 물리 법칙 위반 패널티 (텔레포트 방지, 커야 함)
        self.vel_weight = vel_weight  # 급가속 패널티 (경로 평활화, 작을수록 급격히 움직임)

    def error(self, x_prev, x_curr, dt, **kwargs):
        # State: [x, y, vx, vy]
        p_prev, v_prev = x_prev[:2], x_prev[2:]
        p_curr, v_curr = x_curr[:2], x_curr[2:]

        # 1. 위치 예측 (Trapezoidal Integration) 
        # 단순 v * dt 보다 훨씬 정확하고 부드러움
        # "이동 거리는 평균 속도 * 시간이다"
        pred_pos = p_prev + 0.5 * (v_prev + v_curr) * dt

        # 2. 오차 계산
        # (1) 위치 오차: 물리적으로 말이 안 되는 이동 (텔레포트) -> 강하게 처벌
        pos_error = (p_curr - pred_pos) * self.pos_weight
        
        # (2) 속도 오차 (가속도): 급격한 속도 변화 -> 약하게 처벌 (부드러움 유도)
        # v_curr - v_prev는 사실상 가속도(Acceleration * dt)임
        vel_error = (v_curr - v_prev) * self.vel_weight
        
        # 4차원 벡터로 합쳐서 리턴 (Solver가 알아서 제곱합으로 처리)
        return jnp.concatenate([pos_error, vel_error])
    
class InterRobotFactor(BinaryFactor):
    """
    모든 에이전트의 충돌을 관리하는 팩터.
    Agent 객체에서 추출된 radius와 priority 배열을 사용함.
    """
    def __init__(self, safety_margin=0.5, weight=50.0):
        super().__init__(weight)
        self.safety_margin = safety_margin

    def error(self, x_self, x_other, dt=None, **kwargs):
        
        # 1. 메타 데이터 추출
        agent_idx = kwargs.get('agent_idx')     # 나의 인덱스
        other_idx = kwargs.get('other_idx')     # 상대방 인덱스 (graph.py에서 넘겨줌)
        radii = kwargs.get('radii')             # (M,) 배열
        priorities = kwargs.get('priorities')   # (M,) 배열
        
        # 정보가 없으면 계산 불가
        if agent_idx is None or other_idx is None or radii is None:
            return 0.0

        # 2. 위치 및 거리 계산
        pos_self = x_self[:2]
        pos_other = x_other[:2]
        dist = jnp.linalg.norm(pos_self - pos_other)
        
        # 3. 반지름 합 계산 (배열에서 인덱스로 가져옴)
        r_self = radii[agent_idx]
        r_other = radii[other_idx]
        limit_dist = r_self + r_other + self.safety_margin
        
        # 4. 충돌 에러 (Hinge Loss)
        raw_error = jnp.maximum(0.0, limit_dist - dist)
        
        # 5. 우선순위(Priority) 적용
        scale = 1.0
        if priorities is not None:
            p_self = priorities[agent_idx]
            p_other = priorities[other_idx]
            
            # 상대방 점수 - 내 점수 (상대가 높으면 내가 더 크게 반응)
            p_diff = p_other - p_self
            scale = jnp.exp(p_diff)
            # 수치 안정성을 위해 클리핑
            scale = jnp.clip(scale, 0.1, 100.0)
            
        return raw_error * scale * self.weight
