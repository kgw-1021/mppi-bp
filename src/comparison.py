import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ==============================================================================
# 1. Joint EKI Solver Implementation
# ==============================================================================

class MPPIEKIJointSolver:
    def __init__(self, num_particles, init_mean, init_std=2.0):
        # 초기화: 0 근처에서 넓게 퍼뜨림
        self.particles = np.random.normal(init_mean, init_std, (num_particles, 1))
        self.history = [np.mean(self.particles)]
        
    def step_joint(self):
        """
        모든 팩터를 한 번에 모아서 업데이트 (Batch / Joint Update)
        Target: [x^2=4, x=-2]
        """
        
        # 1. Joint Observation Function: h(x) = [x^2, x]
        def h_joint_func(x):
            # x shape: (N, 1)
            # return shape: (N, 2)
            f1 = x**2
            f2 = x
            return np.hstack([f1, f2])

        # 2. Joint Target: y = [4.0, -2.0]
        y_joint = np.array([[4.0, -2.0]])
        
        # 3. Joint Noise Covariance (R matrix)
        # Factor 1 (x^2) std=1.0, Factor 2 (x) std=1.0
        R_diag = np.array([1.0**2, 1.0**2]) 
        R_mat = np.diag(R_diag)
        
        # EKI Update (One Shot)
        self.particles = self.eki_update_general(self.particles, h_joint_func, y_joint, R_mat)
        
        self.history.append(np.mean(self.particles))
        
    def eki_update_general(self, X, h_func, y, R_mat):
        """
        Generalized EKI Update for multi-dimensional observations
        """
        N = X.shape[0]
        obs_dim = y.shape[1]
        
        # 1. Forward Pass (Prediction)
        HX = h_func(X) # Shape: (N, obs_dim)
        
        # 2. Means
        mu_x = np.mean(X, axis=0)   # (dim_x,)
        mu_hx = np.mean(HX, axis=0) # (obs_dim,)
        
        # 3. Covariances
        dx = X - mu_x       # (N, dim_x)
        dhx = HX - mu_hx    # (N, obs_dim)
        
        # Cross-Covariance C_xy (Input-Output correlation)
        C_xy = (dx.T @ dhx) / (N - 1)
        
        # Output Covariance C_yy (Innovation covariance)
        C_yy = (dhx.T @ dhx) / (N - 1) + R_mat
        
        # 4. Kalman Gain: K = C_xy * C_yy^-1
        # Shape: (dim_x, obs_dim)
        K = C_xy @ np.linalg.inv(C_yy)
        
        # 5. Perturbed Observation (Add noise to target for ensemble diversity)
        # y: (1, obs_dim) -> y_noise: (N, obs_dim)
        noise = np.random.multivariate_normal(np.zeros(obs_dim), R_mat, N)
        y_perturbed = y + noise
        
        # 6. Update State
        # x_new = x + K * (y_perturbed - h(x))
        innovation = y_perturbed - HX
        shift = (K @ innovation.T).T
        
        X_new = X + shift
        return X_new

# ==============================================================================
# 2. GaBP Solver (For Comparison) - Same as before
# ==============================================================================
# GaBP는 구조상 Factor를 하나씩 메시지 패싱하므로 기존 로직 유지 (Iterative fusion)
class GaBPSolver:
    def __init__(self, init_mean, init_prec=1.0):
        # [수정] 초기값을 2차원 (1, 1)로 만들어서 업데이트 후의 shape과 일치시킴
        self.mu = np.array([[init_mean]]) 
        self.prec = np.eye(1) * init_prec
        self.history = [self.mu.copy()]

    def step(self):
        x = self.mu
        
        # Factor 1 (x^2=4)
        J1 = 2 * x 
        h1 = x**2 
        R1 = np.eye(1) * 1.0
        
        # J1이 (1,1)이면 행렬곱 결과도 (1,1) 유지
        Info_1 = J1.T @ np.linalg.inv(R1) @ J1
        Vec_1 = J1.T @ np.linalg.inv(R1) @ (np.array([[4.0]]) - h1 + J1 @ x)
        
        # Factor 2 (x=-2)
        J2 = np.eye(1)
        h2 = x
        R2 = np.eye(1) * 1.0
        
        Info_2 = J2.T @ np.linalg.inv(R2) @ J2
        Vec_2 = J2.T @ np.linalg.inv(R2) @ (np.array([[-2.0]]) - h2 + J2 @ x)
        
        # Update
        new_prec = Info_1 + Info_2 + np.eye(1) * 0.1
        new_vec = Vec_1 + Vec_2
        
        self.mu = np.linalg.inv(new_prec) @ new_vec
        self.prec = new_prec
        
        # 이제 항상 (1,1) 형태가 저장됨
        self.history.append(self.mu.copy())

# ==============================================================================
# 3. Experiment
# ==============================================================================

iters = 10
num_particles = 1000

# Setup Solvers
# Start from 0.0 (Perfectly Ambiguous for GaBP)
mppi_joint = MPPIEKIJointSolver(num_particles, init_mean=2.0, init_std=2.0)
gabp = GaBPSolver(init_mean=2.0)

for _ in range(iters):
    mppi_joint.step_joint()
    gabp.step()

# ==============================================================================
# 4. Visualization
# ==============================================================================

plt.figure(figsize=(12, 6))

# Plot 1: Convergence
plt.subplot(1, 2, 1)
mppi_hist = np.array(mppi_joint.history).flatten()
gabp_hist = np.array(gabp.history).flatten()

plt.plot(mppi_hist, 'b-s', linewidth=2, label='MPPI (Joint Update)')
plt.plot(gabp_hist, 'r-o', linewidth=2, label='GaBP (Sequential)')
plt.axhline(-2.0, color='g', linestyle='--', label='Target (-2)')
plt.axhline(2.0, color='gray', linestyle=':', label='False Target (+2)')

plt.title("Convergence Speed: Joint Update", fontsize=14)
plt.xlabel("Iteration")
plt.ylabel("Mean Value")
plt.legend()
plt.grid(True)

# Plot 2: Final Particle Distribution
plt.subplot(1, 2, 2)
plt.hist(mppi_joint.particles.flatten(), bins=50, density=True, color='b', alpha=0.6, label='MPPI Particles')
plt.axvline(-2.0, color='g', linestyle='--', linewidth=3, label='Ground Truth')
plt.title("Final Distribution (Joint EKI)", fontsize=14)
plt.xlabel("x value")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print Statistics
print(f"MPPI Joint Final Mean: {mppi_hist[-1]:.4f}")
print(f"GaBP Final Mean      : {gabp_hist[-1]:.4f}")