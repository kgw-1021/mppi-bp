# factor_graph_mppi.py
import numpy as np
from scipy.linalg import block_diag 
from typing import List, Dict, Callable, Tuple
from .graph import Node, Edge, Graph

class SampleFNode(Node):

    def __init__(self, name: str, dims: list, strength: float = 1.0):
        super().__init__(name, dims)
        self.factor_strength = strength 

    def update_factor_with_mppi(self, cost_fn: Callable[[np.ndarray], np.ndarray], 
                                lambda_val: float = 1.0, 
                                num_samples: int = None, 
                                exploration_sigma: float = 1.0):

        # target variable Node 
        target_edge = self.edges[0]
        target_var = target_edge.get_other(self)
        
        current_particles = target_var.particles # (N, D)
        N, D = current_particles.shape
        

        noise = np.random.randn(N, D) * exploration_sigma
        perturbed_samples = current_particles + noise
        

        costs = cost_fn(perturbed_samples) # (N,)

        min_cost = np.min(costs)
        weights_unnorm = np.exp(-(costs - min_cost) / lambda_val)
        weights = weights_unnorm / (np.sum(weights_unnorm) + 1e-10)
        
        msg_mean = np.average(perturbed_samples, axis=0, weights=weights)
        
        diff = perturbed_samples - msg_mean

        msg_cov = (diff.T @ (diff * weights[:, None])) / (1.0 - np.sum(weights**2) + 1e-9)
        
        msg_cov = (msg_cov + np.eye(D) * 1e-6) / self.factor_strength
        
        msg = {
            'mean': msg_mean, 
            'cov': msg_cov,   
            'type': 'gaussian_target'
        }
        
        target_edge._messages[self] = msg


class SampleVNode(Node):

    def __init__(self, name: str, dims: list, num_particles: int = 50):
        super().__init__(name, dims)
        self.num_particles = num_particles
        self.particles = np.random.randn(num_particles, dims[0]) * 0.1

    def propagate(self, step_size: float = 1.0):

        incoming_data = [] 
        
        for edge in self.edges:
            sender = edge.get_other(self)
            if sender in edge._messages:
                incoming_data.append(edge._messages[sender])
        
        if not incoming_data:
            return
        
        Y_list = []
        R_list = []
        
        for msg in incoming_data:
            Y_list.append(msg['mean'])
            R_list.append(msg['cov'])
            
        y_joint = np.concatenate(Y_list) # (M * D, )
        
        # Block Diagonal Matrix 
        R_joint = block_diag(*R_list) # (M*D, M*D)
        
        # Prior Statistics 
        X = self.particles
        N, D = X.shape
        mu_x = np.mean(X, axis=0)
        
        # Shape: (N, M * D)
        HX = np.tile(X, (1, len(incoming_data)))
        mu_hx = np.mean(HX, axis=0)
        
        # Covariance Calculation (Global Statistics)
        dx = X - mu_x       # (N, D)
        dhx = HX - mu_hx    # (N, M*D)
        
        # Cross-Covariance 
        C_xy = (dx.T @ dhx) / (N - 1) # (D, M*D)
        
        # Innovation Covariance
        C_yy = (dhx.T @ dhx) / (N - 1) + R_joint # (M*D, M*D)
        
        # 4. Kalman Gain (Global Direction)
        # K = C_xy @ C_yy^-1
        try:
            K = C_xy @ np.linalg.inv(C_yy)
        except np.linalg.LinAlgError:
            K = C_xy @ np.linalg.pinv(C_yy)

        # 5. Update (Shift) with Perturbation
        # Perturbation 
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