import jax.numpy as jnp
from .factor import UnaryFactor, BinaryFactor

class FactorGraph:
    def __init__(self, dt):
        self.dt = dt
        # 타임스텝에 상관없이 공통적으로 적용되는 팩터들 (예: 모든 구간에 장애물 존재)
        self.unary_factors = [] 
        self.binary_factors = [] # 주로 Dynamics
        
    def add_factor(self, factor):
        if isinstance(factor, UnaryFactor):
            self.unary_factors.append(factor)
        elif isinstance(factor, BinaryFactor):
            self.binary_factors.append(factor)
        else:
            raise ValueError("Unknown factor type")

    def evaluate_node_error(self, x_curr, x_prev, x_next, is_start=False, is_end=False):
        """
        단일 노드(x_curr) 입장에서의 모든 에러를 합산하여 벡터로 반환.
        Solver의 'obs_fn' 역할을 수행함.
        """
        errors = []

        # 1. Unary Factors (장애물, 속도제한 등)
        for factor in self.unary_factors:
            # 특수 로직: GoalFactor는 보통 마지막 노드에만 적용할 수도 있음
            # 여기서는 편의상 모든 노드에 적용하되 가중치로 조절 가능하다고 가정
            # 혹은 Factor 내부에 적용 시간(idx) 로직을 넣을 수도 있음
            err = factor.error(x_curr)
            # 스칼라 에러를 1차원 배열로 변환하여 추가
            errors.append(jnp.atleast_1d(err))

        # 2. Forward Dynamics (x_prev -> x_curr)
        # 시작 노드(is_start)는 이전 노드가 없으므로 계산 제외 (혹은 0)
        if not is_start:
            for factor in self.binary_factors:
                err = factor.error(x_prev, x_curr, self.dt)
                errors.append(jnp.atleast_1d(err))
        else:
            # 차원 맞추기용 0 채우기 (State Dim만큼)
            # 실제로는 Solver 레벨에서 Boundary 처리를 하므로 여기는 dummy
            pass 

        # 3. Backward Dynamics (x_curr -> x_next)
        # 마지막 노드(is_end)는 다음 노드가 없으므로 제외
        if not is_end:
            for factor in self.binary_factors:
                # 주의: BinaryFactor.error(from, to)
                err = factor.error(x_curr, x_next, self.dt)
                errors.append(jnp.atleast_1d(err))
        
        # 모든 에러를 하나의 긴 벡터로 연결 (Concatenate)
        # JAX vmap 호환성을 위해 리스트 언패킹 사용
        if not errors:
            return jnp.zeros(1)
            
        return jnp.concatenate(errors)