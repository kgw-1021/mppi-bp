# factor_graph_mppi.py
import numpy as np
from scipy.linalg import block_diag 
from typing import List, Dict, Callable, Tuple
from .graph import Node, Edge, Graph

class SampleFNode(Node):
    """
    [MPPI Factor Node] - "The Judge"
    역할: 현재 파티클 상태에서 Cost를 평가하고,
    'Cost가 낮은 영역'을 대변하는 가우시안 통계(Message)를 생성하여 변수 노드에 전달함.
    """
    def __init__(self, name: str, dims: list, strength: float = 1.0):
        super().__init__(name, dims)
        self.factor_strength = strength  # 값이 클수록 R(관측 노이즈)이 작아짐 -> 강한 제약

    def update_factor_with_mppi(self, cost_fn: Callable[[np.ndarray], np.ndarray], 
                                lambda_val: float = 1.0, 
                                num_samples: int = None, 
                                exploration_sigma: float = 1.0):
        """
        MPPI 프로세스:
        1. Exploration (Noise) -> 2. Evaluation (Cost) -> 3. Weighting (Softmax)
        -> 4. Statistics (Gaussian Approximation of Good Regions)
        """
        # 1. 타겟 변수(Variable Node)의 파티클 가져오기
        target_edge = self.edges[0]
        target_var = target_edge.get_other(self)
        
        current_particles = target_var.particles # (N, D)
        N, D = current_particles.shape
        
        # 2. 탐색 (Exploration): 현재 위치 주변을 정찰
        noise = np.random.randn(N, D) * exploration_sigma
        perturbed_samples = current_particles + noise
        
        # 3. 평가 (Evaluation): Cost 계산
        costs = cost_fn(perturbed_samples) # (N,)
        
        # 4. 가중치 계산 (Weighting): MPPI의 핵심
        # Cost가 높은(나쁜) 파티클의 가중치는 0에 수렴 -> 통계에서 배제됨 
        min_cost = np.min(costs)
        weights_unnorm = np.exp(-(costs - min_cost) / lambda_val)
        weights = weights_unnorm / (np.sum(weights_unnorm) + 1e-10)
        
        # 5. 메시지 생성 (Weighted Statistics)
        # "Cost가 낮은 영역은 어디인가?"에 대한 통계적 답변
        msg_mean = np.average(perturbed_samples, axis=0, weights=weights)
        
        diff = perturbed_samples - msg_mean
        # Weighted Covariance: 좋은 파티클들이 모여있는 분포의 형상
        msg_cov = (diff.T @ (diff * weights[:, None])) / (1.0 - np.sum(weights**2) + 1e-9)
        
        # Factor Strength 적용:
        # 강한 팩터일수록 불확실성(Covariance)을 줄여서 EKI가 더 강하게 끌어당기게 함
        msg_cov = (msg_cov + np.eye(D) * 1e-6) / self.factor_strength
        
        msg = {
            'mean': msg_mean, # "여기로 오세요"
            'cov': msg_cov,   # "이만큼의 확실성으로"
            'type': 'gaussian_target'
        }
        
        # 메시지 전송
        target_edge._messages[self] = msg


class SampleVNode(Node):
    """
    [EKI Variable Node] - "The Executor"
    역할: 여러 팩터(판사)들이 보낸 메시지를 한 번에 취합(Joint Update)하여,
    파티클들을 '오차를 줄이는 방향'으로 물리적 이동(Shift) 시킴.
    """
    def __init__(self, name: str, dims: list, num_particles: int = 50):
        super().__init__(name, dims)
        self.num_particles = num_particles
        self.particles = np.random.randn(num_particles, dims[0]) * 0.1

    def propagate(self, step_size: float = 1.0):
        """
        Joint EKI Update Step
        모든 팩터의 제약조건을 쌓아서(Stacking) 한 번에 업데이트.
        상충되는 팩터(Conflict)가 있을 때 중앙 붕괴를 막고 최적해로 이동하는 핵심 로직.
        """
        incoming_data = [] 
        
        # 1. 메시지 수집
        for edge in self.edges:
            sender = edge.get_other(self)
            if sender in edge._messages:
                incoming_data.append(edge._messages[sender])
        
        if not incoming_data:
            return

        # -----------------------------------------------------------
        # Joint Update Logic (Stacking Observations)
        # -----------------------------------------------------------

        
        Y_list = []
        R_list = []
        
        for msg in incoming_data:
            Y_list.append(msg['mean'])
            R_list.append(msg['cov'])
            
        y_joint = np.concatenate(Y_list) # (M * D, )
        
        # Block Diagonal Matrix 생성 (팩터 간의 독립적 노이즈 가정)
        R_joint = block_diag(*R_list) # (M*D, M*D)
        
        # 2. Prior Statistics (현재 파티클의 상태)
        X = self.particles
        N, D = X.shape
        mu_x = np.mean(X, axis=0)
        
        # H(X): 관측 함수. 여기서는 팩터들이 State(X)를 직접 관측한다고 가정 (Identity)
        # 따라서 H(X)는 X를 팩터 개수만큼 반복한 형태
        # Shape: (N, M * D)
        HX = np.tile(X, (1, len(incoming_data)))
        mu_hx = np.mean(HX, axis=0)
        
        # 3. Covariance Calculation (Global Statistics)
        dx = X - mu_x       # (N, D)
        dhx = HX - mu_hx    # (N, M*D)
        
        # Cross-Covariance (입력과 모든 관측 간의 상관관계)
        # "어떤 방향으로 움직여야 전체 팩터들의 오차 합이 줄어드는가?"
        C_xy = (dx.T @ dhx) / (N - 1) # (D, M*D)
        
        # Innovation Covariance
        C_yy = (dhx.T @ dhx) / (N - 1) + R_joint # (M*D, M*D)
        
        # 4. Kalman Gain (Global Direction)
        # K = C_xy @ C_yy^-1
        # 상충되는 팩터가 있을 때, R(불확실성)이 작은 쪽(강한 팩터)으로 K가 유도됨
        try:
            K = C_xy @ np.linalg.inv(C_yy)
        except np.linalg.LinAlgError:
            K = C_xy @ np.linalg.pinv(C_yy)

        # 5. Update (Shift) with Perturbation
        # Perturbation: 파티클이 하나의 점으로 붕괴되는 것을 막는 EKI의 필수 테크닉
        noise_vec = np.random.multivariate_normal(
            np.zeros(len(y_joint)), R_joint, N
        )
        y_perturbed = y_joint + noise_vec
        
        # Innovation: (Target - Current)
        innovation = y_perturbed - HX
        
        # x_new = x + alpha * K * (y - h(x))
        displacement = (K @ innovation.T).T
        
        self.particles = self.particles + step_size * displacement

    def get_belief_stats(self):
        return np.mean(self.particles, axis=0), np.cov(self.particles, rowvar=False)