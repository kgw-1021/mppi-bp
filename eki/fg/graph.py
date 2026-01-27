import jax
import jax.numpy as jnp
from .factor import UnaryFactor, BinaryFactor, InterRobotFactor

class FactorGraph:
    def __init__(self, dt):
        self.dt = dt
        self.factors = [] 

    def add_factor(self, factor):
        """팩터 추가"""
        self.factors.append(factor)

    def remove_factor(self, factor):
        """팩터 제거 (특정 조건 만족 시)"""
        if factor in self.factors:
            self.factors.remove(factor)

    def clear_factors_of_type(self, factor_type):
        """특정 타입의 팩터 일괄 제거 (예: 모든 장애물 제거)"""
        self.factors = [f for f in self.factors if not isinstance(f, factor_type)]

    def evaluate_node_error(self, x_curr, x_prev, x_next, **kwargs):
        """
        x_curr: 현재 노드 상태
        kwargs: 'global_msgs', 'agent_idx', 'priorities' 등 포함
        """
        errors = []
        
        # 컨텍스트 추출
        global_msgs = kwargs.get('global_msgs') # (M, D)
        my_idx = kwargs.get('agent_idx')
        
        for factor in self.factors:
            
            # [Case A] 로봇 간 충돌 팩터 (InterRobotFactor)
            if isinstance(factor, InterRobotFactor):
                if global_msgs is None or my_idx is None:
                    continue # 정보 없으면 패스

                # 다른 모든 에이전트에 대해 BinaryFactor 계산
                # vmap을 사용하여 효율적으로 계산 (One-to-Many)
                
                def compute_pair_error(other_state, other_idx):
                    # 자기 자신과의 충돌은 계산하지 않음 (Error = 0)
                    is_me = (other_idx == my_idx)
                    
                    # BinaryFactor 호출 (dt는 무시됨)
                    # 필요한 정보(인덱스 등)를 kwargs로 전달
                    err = factor.error(
                        x_curr, 
                        other_state, 
                        dt=self.dt, 
                        my_idx=my_idx, 
                        other_idx=other_idx, 
                        **kwargs
                    )
                    
                    return err * (1.0 - is_me) # 내가 아니면 에러 반영

                # 모든 에이전트 인덱스 생성
                all_indices = jnp.arange(global_msgs.shape[0])
                
                # vmap으로 일괄 계산 후 합산 (Sum of constraints)
                # 이것이 곧 Sigma(BinaryFactors)가 됨
                pair_errors = jax.vmap(compute_pair_error)(global_msgs, all_indices)
                total_collision_error = jnp.sum(pair_errors)
                
                errors.append(jnp.atleast_1d(total_collision_error))

            # [Case B] 일반적인 동역학 팩터 (Dynamics - Temporal Binary)
            elif isinstance(factor, BinaryFactor):
                # x_prev와 x_curr 연결
                err = factor.error(x_prev, x_curr, self.dt)
                errors.append(jnp.atleast_1d(err))
                
            # [Case C] 단항 팩터 (Unary - Obstacle, Goal)
            elif isinstance(factor, UnaryFactor):
                err = factor.error(x_curr)
                errors.append(jnp.atleast_1d(err))
                
        if not errors:
            return jnp.zeros(1)
        return jnp.concatenate(errors)