import jax
import jax.numpy as jnp
from jax import vmap, random

class FactorGraphEKISolver:
    def __init__(self, graph, config):
        self.graph = graph
        self.config = config
        
        # JIT Compile
        self.parallel_update = jax.jit(self.parallel_update_step)

    def initialize_belief(self, key, start, goal):
        T, N, D = self.config.horizon, self.config.num_particles, self.config.state_dim
        
        # 선형 보간 초기화
        alpha = jnp.linspace(0, 1, T)[:, None]
        mean_traj = (1 - alpha) * start + alpha * goal
        
        # 노이즈 추가
        noise = random.normal(key, (T, N, D)) * 0.5
        belief = mean_traj[:, None, :] + noise
        
        # 시작점 고정
        belief = belief.at[0].set(start)
        return belief

    def parallel_update_step(self, current_belief, start_state, goal_state):
        T, N, D = current_belief.shape

        # --- [1. Message Preparation (Shift)] ---
        nodes_prev = jnp.roll(current_belief, shift=1, axis=0)
        nodes_prev = nodes_prev.at[0].set(start_state) # Boundary

        nodes_next = jnp.roll(current_belief, shift=-1, axis=0)
        nodes_next = nodes_next.at[-1].set(goal_state) # Boundary

        # --- [2. Define Observation Function per Node] ---
        # 이 함수가 'vmap' 될 대상입니다.
        def node_obs_fn(x_c, x_p, x_n):
            # Graph에게 에러 계산 위임
            # 여기서는 Boundary Flag 처리가 까다로우므로, 
            # 모든 노드에 대해 Fwd/Bwd 에러를 다 계산하고 
            # 나중에 마스킹(Masking)하거나, 그냥 다 계산해도 수렴에 큰 지장 없음.
            # (t=0의 Fwd 에러는 0으로, t=T의 Bwd 에러는 0으로 나오게 설계됨)
            
            # 여기서 Graph의 메서드를 호출!
            return self.graph.evaluate_node_error(x_c, x_p, x_n)

        # --- [3. EKI Update Core] ---
        def single_particle_update(p_c, p_p, p_n):
            # 관측값(에러) 계산
            h_x = node_obs_fn(p_c, p_p, p_n) 
            
            # --- EKI Statistics ---
            mean_param = jnp.mean(p_c, axis=0)
            mean_h = jnp.mean(h_x, axis=0)
            
            diff_param = p_c - mean_param
            diff_h = h_x - mean_h
            
            # Covariances
            C_xy = (diff_param.T @ diff_h) / (N - 1)
            C_yy = (diff_h.T @ diff_h) / (N - 1)
            
            # Regularization
            R = jnp.eye(h_x.shape[1]) * self.config.r_diag
            
            # Kalman Gain
            K = C_xy @ jnp.linalg.inv(C_yy + R)
            
            # Update (Target y is always 0 for error minimization)
            residual = -h_x # (0 - h_x)
            shift = (K @ residual.T).T
            
            return p_c + shift

        # --- [4. Vectorized Execution] ---
        # vmap over Time(0)
        # vmap over Particles는 single_particle_update 내부에서 처리하는게 아니라
        # 위 함수는 (N, D)를 받아서 (N, D)를 뱉는 구조여야 함.
        # 따라서, 위 single_particle_update는 사실 "single_node_update"임.
        
        def single_node_update(node_particles_c, node_particles_p, node_particles_n):
            # vmap over particles to get h_x for all particles
            h_x_all = vmap(node_obs_fn)(node_particles_c, node_particles_p, node_particles_n)
            
            # EKI 통계 계산은 입자 전체(N)를 보고 함
            mean_param = jnp.mean(node_particles_c, axis=0)
            mean_h = jnp.mean(h_x_all, axis=0)
            
            diff_param = node_particles_c - mean_param
            diff_h = h_x_all - mean_h
            
            N_part = node_particles_c.shape[0]
            C_xy = (diff_param.T @ diff_h) / (N_part - 1) + 1e-6 * jnp.eye(D, h_x_all.shape[1])
            C_yy = (diff_h.T @ diff_h) / (N_part - 1) + 1e-6 * jnp.eye(h_x_all.shape[1])
            
            R = jnp.eye(h_x_all.shape[1]) * self.config.r_diag
            K = C_xy @ jnp.linalg.inv(C_yy + R)
            
            residual = -h_x_all
            shift = (K @ residual.T).T
            return node_particles_c + shift

        # 최종 vmap: 시간축(Time)에 대해 병렬 실행
        new_belief = vmap(single_node_update)(current_belief, nodes_prev, nodes_next)
        
        # Boundary Fix
        new_belief = new_belief.at[0].set(start_state)
        
        return new_belief

    def solve(self, key, start, goal, iterations=20):
        belief = self.initialize_belief(key, start, goal)
        
        for i in range(iterations):
            updated = self.parallel_update(belief, start, goal)
            # Damping
            belief = (1 - self.config.damping) * belief + self.config.damping * updated
            
        return belief