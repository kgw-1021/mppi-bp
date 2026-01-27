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
    

class MultiAgentEKISolver:
    def __init__(self, graph, config):
        self.graph = graph
        self.config = config
        self.update_all_agents = jax.jit(self._update_step_logic)

    def _update_step_logic(self, current_beliefs, start_states, goal_states, 
                           agent_indices, priorities, radii):
        
        # [Step 1] 통신: 메시지(평균 궤적) 생성
        global_msgs = jnp.mean(current_beliefs, axis=2) # (M, T, D)
        
        # [Step 2] 개별 에이전트 업데이트
        def single_agent_update(my_belief, my_start, my_goal, my_idx):
            
            # --- (A) 관측 함수 ---
            def particle_obs_fn(x_c, x_p, x_n, t_idx):
                # 모든 메타 데이터를 Context로 포장해서 Graph로 던짐
                context = {
                    'global_msgs': global_msgs[t_idx],
                    'agent_idx': my_idx,
                    'goal_states': goal_states, # GoalFactor용
                    'priorities': priorities,   # InterRobotFactor용
                    'radii': radii              # InterRobotFactor용
                }
                return self.graph.evaluate_node_error(x_c, x_p, x_n, **context)

            # --- (B) 시간축 병렬화 ---
            nodes_prev = jnp.roll(my_belief, 1, axis=0).at[0].set(my_start)
            nodes_next = jnp.roll(my_belief, -1, axis=0).at[-1].set(my_goal)
            time_indices = jnp.arange(self.config.horizon)

            # --- (C) 파티클 병렬화 ---
            def update_node_at_t(p_c, p_p, p_n, t):
                # vmap으로 파티클별 에러 계산 시 context(t)가 같이 들어감
                h_x = vmap(lambda x, xp, xn: particle_obs_fn(x, xp, xn, t))(p_c, p_p, p_n)
                
                # EKI Statistics & Update (기존과 동일)
                mean_param = jnp.mean(p_c, axis=0)
                mean_h = jnp.mean(h_x, axis=0)
                diff_param = p_c - mean_param
                diff_h = h_x - mean_h
                
                N_part = p_c.shape[0]
                C_xy = (diff_param.T @ diff_h) / (N_part - 1) + 1e-6 * jnp.eye(self.config.state_dim, h_x.shape[1])
                C_yy = (diff_h.T @ diff_h) / (N_part - 1) + 1e-6 * jnp.eye(h_x.shape[1])
                
                R = jnp.eye(h_x.shape[1]) * self.config.r_diag
                K = C_xy @ jnp.linalg.inv(C_yy + R)
                
                shift = (K @ (-h_x).T).T
                return p_c + shift

            new_traj = vmap(update_node_at_t)(my_belief, nodes_prev, nodes_next, time_indices)
            new_traj = new_traj.at[0].set(my_start)
            return new_traj

        # [Step 3] 에이전트 축 병렬화
        updated_beliefs = vmap(single_agent_update, in_axes=(0, 0, 0, 0))(
            current_beliefs, start_states, goal_states, agent_indices
        )
        
        return updated_beliefs

    def solve(self, key, agents, iterations=20):
        M = len(agents)
        T, N, D = self.config.horizon, self.config.num_particles, self.config.state_dim
        dt = self.config.dt # dt 가져오기

        # 1. Agent 객체 -> JAX Array 변환
        start_states = jnp.stack([a.start_pose for a in agents]) # (M, D)
        goal_states = jnp.stack([a.goal_pose for a in agents])   # (M, D)
        priorities = jnp.array([a.priority for a in agents])     
        radii = jnp.array([a.radius for a in agents])            
        agent_indices = jnp.arange(M)

        # 2. [핵심 변경] 초기화 로직: Velocity Rollout
        def init_one(s, g, k):
            """
            s: Start State (D,)
            g: Goal State (D,)
            k: PRNG Key
            """
            # (A) 기준 속도 계산 (Start -> Goal)
            # 목표까지 등속도로 간다고 가정했을 때의 평균 속도
            dist_vec = g[:2] - s[:2]
            avg_vel = dist_vec / (T * dt) # (2,)
            
            # (B) 속도에 노이즈 주입 (Exploration)
            # 모든 타임스텝, 모든 파티클에 대해 속도 노이즈 생성
            # (T, N, 2)
            # 노이즈 크기(1.0~2.0)를 조절하여 탐색 범위를 결정
            vel_noise = random.normal(k, (T, N, 2)) * 1.5 
            
            # 초기 속도 궤적 = 기준 속도 + 노이즈
            # (T, N, 2)
            velocities = avg_vel[None, None, :] + vel_noise
            
            # (C) 위치 적분 (Rollout) - jax.lax.scan 사용
            # x_{t+1} = x_t + v_t * dt
            
            def integrate_step(curr_pos, vel):
                # curr_pos: (N, 2), vel: (N, 2)
                next_pos = curr_pos + vel * dt
                return next_pos, next_pos # carry, output

            # 초기 위치는 모든 파티클이 동일하게 Start Position
            init_pos = jnp.tile(s[:2], (N, 1)) # (N, 2)
            
            # v_0 ~ v_{T-2}를 사용하여 x_1 ~ x_{T-1} 계산
            # velocities[:-1] shape: (T-1, N, 2)
            _, pos_traj_rest = jax.lax.scan(integrate_step, init_pos, velocities[:-1])
            
            # x_0 (Start)와 나머지 궤적 합치기
            # x_0 shape: (1, N, 2)
            pos_traj_0 = init_pos[None, :, :]
            pos_traj = jnp.concatenate([pos_traj_0, pos_traj_rest], axis=0) # (T, N, 2)
            
            # (D) 상태 합치기 [Pos, Vel]
            # (T, N, 4)
            belief_init = jnp.concatenate([pos_traj, velocities], axis=-1)
            
            return belief_init
        
        # vmap으로 모든 에이전트 동시 초기화
        keys = random.split(key, M)
        belief = vmap(init_one)(start_states, goal_states, keys)
        
        # 안전장치: 0번 타임스텝(Start)은 물리적으로 완벽하게 고정
        belief = belief.at[:, 0].set(start_states[:, None, :])

        # 3. 최적화 루프
        for i in range(iterations):
            updated = self.update_all_agents(
                belief, start_states, goal_states, agent_indices, priorities, radii
            )
            belief = (1 - self.config.damping) * belief + self.config.damping * updated
            
            # 루프마다 시작점 고정
            belief = belief.at[:, 0].set(start_states[:, None, :])
            
        return belief