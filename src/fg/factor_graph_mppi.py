# factor_graph_mppi.py
import numpy as np
from typing import List, Dict, Callable
from .graph import Node, Edge, Graph

class SampleFNode(Node):
    """
    MPPI Factor Node:
    비용 함수를 샘플링하여 '이상적인 분포(Gaussian Moment)'를 제안함.
    """
    def __init__(self, name: str, dims: list, strength: float = 1.0):
        super().__init__(name, dims)
        self.factor_strength = strength  # 값이 클수록 강한 제약 (작은 공분산)

    def update_factor_with_mppi(self, cost_fn: Callable[[np.ndarray], np.ndarray], 
                                lambda_val: float = 1.0, 
                                num_samples: int = 100, 
                                exploration_sigma: float = 0.5):
        """
        MPPI를 수행하여 연결된 변수들에게 보낼 메시지(Mean, Cov)를 생성
        """
        # 1. 연결된 변수의 현재 샘플 가져오기 (단일 변수 연결 가정)
        # 다중 변수 연결 시에는 joint state로 확장 필요
        target_edge = self.edges[0]
        target_var = target_edge.get_other(self)
        
        current_particles = target_var.particles # (N, D)
        N, D = current_particles.shape
        
        # 2. 탐색 (Exploration): 노이즈 주입
        noise = np.random.randn(N, D) * exploration_sigma
        perturbed_samples = current_particles + noise
        
        # 3. 평가 (Evaluation): 비용 계산
        costs = cost_fn(perturbed_samples) # (N,)
        
        # 4. 가중치 계산 (Softmax)
        min_cost = np.min(costs)
        # 수치적 안정성을 위해 min_cost 뺌
        weights_unnorm = np.exp(-(costs - min_cost) / lambda_val)
        weights = weights_unnorm / (np.sum(weights_unnorm) + 1e-10)
        
        # 5. 메시지 생성 (Moment Matching)
        msg_mean = np.average(perturbed_samples, axis=0, weights=weights)
        
        diff = perturbed_samples - msg_mean
        # Weighted Covariance
        msg_cov = (diff.T @ (diff * weights[:, None])) / (1.0 - np.sum(weights**2) + 1e-9)
        
        # Factor Strength 적용: 강할수록 공분산을 작게 만듦
        msg_cov = (msg_cov + np.eye(D) * 1e-6) / self.factor_strength
        
        msg = {
            'mean': msg_mean,
            'cov': msg_cov,
            'type': 'gaussian_target'
        }
        
        # 메시지 전송
        target_edge._messages[self] = msg

class SampleVNode(Node):
    """
    EKI Variable Node:
    여러 팩터의 메시지를 Ensemble Kalman Inversion으로 융합하여 입자 이동.
    """
    def __init__(self, name: str, dims: list, num_particles: int = 50):
        super().__init__(name, dims)
        self.num_particles = num_particles
        # 초기화는 0 또는 외부에서 설정
        self.particles = np.random.randn(num_particles, dims[0]) * 0.1

    def propagate(self, step_size: float = 0.4):
        """
        EKI Update Step
        step_size: Damping factor (0.0 ~ 1.0) to prevent oscillation
        """
        incoming_msgs = []
        for edge in self.edges:
            sender = edge.get_other(self)
            if sender in edge._messages:
                incoming_msgs.append(edge._messages[sender])
        
        if not incoming_msgs:
            return

        # 1. Prior Statistics (현재 입자들의 분포)
        prior_mean = np.mean(self.particles, axis=0)
        # Covariance calculation (D x D)
        diff = self.particles - prior_mean
        prior_cov = (diff.T @ diff) / (self.num_particles - 1) + np.eye(self.dims[0]) * 1e-6

        total_displacement = np.zeros_like(self.particles)
        
        # 2. Calculate Displacement for each message (EKI)
        for msg in incoming_msgs:
            target_mean = msg['mean']
            target_cov = msg['cov'] # R matrix (Factor Uncertainty)

            # Kalman Gain: K = C_xx * (C_xx + R)^-1
            innovation_cov = prior_cov + target_cov
            
            # Solve for K.T: (C_xx + R) * K.T = C_xx
            K_T = np.linalg.solve(innovation_cov, prior_cov).T 
            
            # Generate Virtual Observations (y) with perturbation
            # y ~ N(target_mean, target_cov)
            obs_noise = np.random.multivariate_normal(
                np.zeros(len(target_mean)), target_cov, self.num_particles
            )
            y_samples = target_mean + obs_noise
            
            # Analysis Step: x_new = x + K * (y - x)
            # Implementation: displacement = (y - x) @ K.T
            innovation = y_samples - self.particles
            displacement = innovation @ K_T
            
            total_displacement += displacement

        # 3. Batch Update (Averaging)
        # 여러 팩터의 힘을 평균내어 진동 방지
        avg_displacement = total_displacement / len(incoming_msgs)
        
        # Apply update with step size
        self.particles = self.particles + step_size * avg_displacement

    def get_belief_stats(self):
        return np.mean(self.particles, axis=0), np.cov(self.particles, rowvar=False)