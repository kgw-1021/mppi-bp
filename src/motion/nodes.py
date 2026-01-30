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
    # 생성자에 dt 추가 (기본값 0.1)
    def __init__(self, name: str, dims: list, omap: ObstacleMap, dt: float = 0.1, strength: float = 5.0):
        super().__init__(name, dims, strength)
        self.omap = omap
        self.dt = dt

    def _cost_fn(self, samples: np.ndarray):
        # dt를 넘겨주어 선분 충돌 검사 수행
        return self.omap.get_obstacle_cost(samples, safe_dist=1.0, dt=self.dt)

    def update_factor(self):
        self.update_factor_with_mppi(self._cost_fn, lambda_val=0.01, exploration_sigma=0.1)

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
        costs[collision] = 1000.0
        
        # 근접 시 소프트 비용
        near_mask = (dist >= self.min_dist) & (dist < self.min_dist * 2.0)
        costs[near_mask] = np.exp(-1.0 * (dist[near_mask] - self.min_dist)) * 20.0
        
        return costs

    def update_factor(self):
        if self.remote_belief is not None:
            self.update_factor_with_mppi(self._cost_fn, lambda_val=0.01, exploration_sigma=0.1)

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

class KinematicsFNode(SampleFNode):
    """
    운동학적 제약(Dynamics Constraint)을 처리하는 팩터.
    x, y, vx, vy의 커플링을 고려하여, 속도에만 노이즈를 주입하고 위치는 적분합니다.
    """
    def __init__(self, name: str, dims: list, dt: float = 0.1, strength: float = 5.0):
        super().__init__(name, dims, strength)
        self.dt = dt

    def update_factor(self):
        """
        연결된 두 변수(prev_node, next_node) 사이의 
        Forward 및 Backward 메시지를 생성합니다.
        """
        # 연결된 노드 확인 (순서가 중요: 시간순으로 연결되었다고 가정)
        # edges[0] -> prev_node (t), edges[1] -> next_node (t+1)
        # 만약 순서가 보장되지 않는다면 이름 등을 통해 정렬 필요
        node_a = self.edges[0].get_other(self)
        node_b = self.edges[1].get_other(self)
        
        # 이름 기반 정렬 (예: "agent_v0", "agent_v1" -> v0이 prev)
        # 실제 구현시에는 graph 구성 단계에서 순서를 보장하거나 별도 속성 사용 권장
        if node_a.name > node_b.name:
            node_prev, node_next = node_b, node_a
        else:
            node_prev, node_next = node_a, node_b

        # ---------------------------------------------------
        # 1. Forward Message (Prev -> Next)
        # "이전 상태가 이렇다면, 다음 상태는 물리적으로 여기 있어야 해"
        # ---------------------------------------------------
        particles_prev = node_prev.particles # (N, 4) [x, y, vx, vy]
        N, D = particles_prev.shape
        
        # (1) 속도 노이즈 주입 (가속도 불확실성 모델링)
        # 위치에는 직접 노이즈를 주지 않음!
        acc_noise_sigma = 0.5
        dv = np.random.randn(N, 2) * acc_noise_sigma
        
        # (2) 운동학적 적분 (Kinematic Integration)
        # v_next = v_prev + noise
        # p_next = p_prev + v_next * dt
        v_prev = particles_prev[:, 2:4]
        p_prev = particles_prev[:, 0:2]
        
        v_next_pred = v_prev + dv
        p_next_pred = p_prev + v_next_pred * self.dt
        
        samples_fwd = np.hstack([p_next_pred, v_next_pred])
        
        # (3) 통계량(Mean, Cov) 계산 및 메시지 전송
        self._send_gaussian_msg(samples_fwd, target_edge=self.edges[1]) # node_next로 보냄


        # ---------------------------------------------------
        # 2. Backward Message (Next -> Prev)
        # "다음 상태가 저기라면, 이전 상태는 물리적으로 여기 있었어야 해"
        # ---------------------------------------------------
        particles_next = node_next.particles
        
        # (1) 역방향 적분 (Inverse Kinematics)
        # p_prev = p_next - v_next * dt (단순화된 역학)
        v_next = particles_next[:, 2:4]
        p_next = particles_next[:, 0:2]
        
        # 속도의 역방향 추정은 노이즈를 뺌 (v_prev = v_next - noise)
        dv_back = np.random.randn(N, 2) * acc_noise_sigma
        v_prev_pred = v_next - dv_back
        
        # 위치 역추정
        p_prev_pred = p_next - v_next * self.dt 
        
        samples_bwd = np.hstack([p_prev_pred, v_prev_pred])
        
        # (3) 메시지 전송
        self._send_gaussian_msg(samples_bwd, target_edge=self.edges[0]) # node_prev로 보냄

    def _send_gaussian_msg(self, samples, target_edge):
        """샘플들로부터 Mean, Cov를 계산하여 EKI용 메시지 전송"""
        N, D = samples.shape
        msg_mean = np.mean(samples, axis=0)
        
        diff = samples - msg_mean
        msg_cov = (diff.T @ diff) / (N - 1 + 1e-9)
        
        # 공분산 정규화 및 Factor Strength 적용
        msg_cov = (msg_cov + np.eye(D) * 1e-6) / self.factor_strength
        
        msg = {
            'mean': msg_mean,
            'cov': msg_cov,
            'type': 'dynamics_target' # kinematics
        }
        target_edge._messages[self] = msg