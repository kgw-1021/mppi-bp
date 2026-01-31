import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.linalg import block_diag

# ==============================================================================
# 1. MPPI-BP Solver Implementation (Modified)
# ==============================================================================

class MPPIEKIJointSolver:
    def __init__(self, num_particles, init_mean, init_std=2.0):
        # 초기화: 파티클 생성
        self.particles = np.random.normal(init_mean, init_std, (num_particles, 1))
        self.history = [np.mean(self.particles)]
        
    def get_mppi_message(self, cost_func, exploration_sigma=0.01, lambda_val=0.1):
        """
        [Factor Node Logic]
        파티클 주변을 탐색(Perturbation)하고, Cost가 낮은 샘플들의
        가중 평균(Mean)과 공분산(Cov)을 계산하여 메시지로 반환
        """
        N, D = self.particles.shape
        
        # 1. Exploration: 현재 파티클 주변 탐색
        noise = np.random.randn(N, D) * exploration_sigma
        samples = self.particles + noise
        
        # 2. Evaluation: Cost 계산
        costs = cost_func(samples) # Shape: (N, )
        
        # 3. Weighting: Softmax (Cost가 낮을수록 가중치 높음)
        min_cost = np.min(costs)
        # lambda_val: 온도 파라미터 (낮을수록 최적해에 민감하게 반응)
        weights_unnorm = np.exp(-(costs - min_cost) / lambda_val)
        weights = weights_unnorm / (np.sum(weights_unnorm) + 1e-10)
        
        # 4. Statistics: 가중 평균 및 공분산 (Gaussian Message 생성)
        msg_mean = np.average(samples, axis=0, weights=weights)
        
        diff = samples - msg_mean
        # Weighted Covariance
        msg_cov = (diff.T @ (diff * weights[:, None])) / (1.0 - np.sum(weights**2) + 1e-10)
        
        # 공분산 정규화 (너무 0이 되지 않도록 안전장치)
        msg_cov = msg_cov + np.eye(D) * 1e-2
        
        return msg_mean, msg_cov

    def step_joint(self):
        # --- Factor 1: x^2 = 4 ---
        # Cost Function: (x^2 - 4)^2
        def cost_f1(x):
            residual = x**2 - 4.0
            return (residual**2).flatten()
        
        # MPPI를 통해 "x^2=4를 만족하는 x의 분포"를 추정
        mu1, cov1 = self.get_mppi_message(cost_f1, exploration_sigma=0.1, lambda_val=0.1)
        
        # --- Factor 2: x = -2 ---
        # Cost Function: (x - (-2))^2
        def cost_f2(x):
            residual = x - (-2.0)
            return (residual**2).flatten() # L2 Norm
            
        # MPPI를 통해 "x=-2를 만족하는 x의 분포"를 추정
        mu2, cov2 = self.get_mppi_message(cost_f2, exploration_sigma=0.1, lambda_val=0.1)
        
        # --- Joint EKI Update ---
        
        # 1. Joint Target (y): 팩터들이 제안한 평균값들
        y_joint = np.hstack([mu1, mu2]).reshape(1, -1) # Shape (1, 2)
        # 2. Joint Covariance (R): 블록 대각 행렬
        R_mat = block_diag(cov1, cov2)
        
        # 3. Observation Function h(x):
        def h_joint_func(x):
            # x: (N, 1) -> return: (N, 2) [x, x]
            return np.hstack([x, x])

        # EKI Update 실행
        self.particles = self.eki_update_general(self.particles, h_joint_func, y_joint, R_mat)
        
        self.history.append(np.mean(self.particles))
        
    def eki_update_general(self, X, h_func, y, R_mat):
        """
        Standard EKI Update (변수 노드의 업데이트 로직)
        """
        N = X.shape[0]
        obs_dim = y.shape[1]
        
        HX = h_func(X) # (N, 2)
        
        mu_x = np.mean(X, axis=0)
        mu_hx = np.mean(HX, axis=0)
        
        dx = X - mu_x
        dhx = HX - mu_hx
        
        C_xy = (dx.T @ dhx) / (N - 1)
        C_yy = (dhx.T @ dhx) / (N - 1) + R_mat
        
        # Kalman Gain
        K = C_xy @ np.linalg.inv(C_yy)
        
        # Perturbed Observation (다양성 유지)
        noise = np.random.multivariate_normal(np.zeros(obs_dim), R_mat, N)
        y_perturbed = y + noise
        
        # Update
        innovation = y_perturbed - HX
        shift = (K @ innovation.T).T
        
        return X + shift

# ==============================================================================
# 2. GaBP Solver (Existing)
# ==============================================================================
class GaBPSolver:
    def __init__(self, init_mean, init_prec=1.0):
        self.mu = np.array([[init_mean]]) 
        self.prec = np.eye(1) * init_prec
        self.history = [self.mu.copy()]

    def step(self):
        x = self.mu
        
        # Factor 1 (x^2=4)
        J1 = 2 * x 
        h1 = x**2 
        R1 = np.eye(1) * 1.0
        
        Info_1 = J1.T @ np.linalg.inv(R1) @ J1
        Vec_1 = J1.T @ np.linalg.inv(R1) @ (np.array([[4.0]]) - h1 + J1 @ x)
        
        # Factor 2 (x=-2)
        J2 = np.eye(1)
        h2 = x
        R2 = np.eye(1) * 0.5 
        
        Info_2 = J2.T @ np.linalg.inv(R2) @ J2
        Vec_2 = J2.T @ np.linalg.inv(R2) @ (np.array([[-2.0]]) - h2 + J2 @ x)
        
        # Update
        new_prec = Info_1 + Info_2 + np.eye(1) * 0.1
        new_vec = Vec_1 + Vec_2
        
        self.mu = np.linalg.inv(new_prec) @ new_vec
        self.prec = new_prec
        
        self.history.append(self.mu.copy())

# ==============================================================================
# 3. Experiment
# ==============================================================================

iters = 10
num_particles = 1000

# Setup Solvers
# Start from 6.0 (Bad initial guess for GaBP)
mppi_joint = MPPIEKIJointSolver(num_particles, init_mean=6.0, init_std=3.0)
gabp = GaBPSolver(init_mean=6.0)

for _ in range(iters):
    mppi_joint.step_joint()
    gabp.step()

# ==============================================================================
# 4. Visualization
# ==============================================================================

plt.figure(1, figsize=(6, 6))

# --- Plot 1: Convergence (Mean Values) ---
mppi_hist = np.array(mppi_joint.history).flatten()
gabp_hist = np.array(gabp.history).flatten()

plt.plot(mppi_hist, 'b-s', markersize=8, linewidth=2, label='MPPI-BP (Mean)')
plt.plot(gabp_hist, 'r-o', markersize=8, linewidth=2, label='GaBP (Mean)')
plt.axhline(-2.0, color='green', linestyle='--', alpha=0.7, label='Target (-2)')
plt.title("Mean Convergence Comparison", fontsize=14)
plt.xlabel("Iteration")
plt.ylabel("Value")
plt.legend()
plt.grid(True, alpha=0.3)

# --- Plot 2: Detailed Distribution Evolution (Subplots) ---
plot_steps = [0, 1, 2, 3, 4, 9] 
fig, axes = plt.subplots(2, len(plot_steps)//2, figsize=(12, 4), sharey=True)
axes = axes.flatten()
x_range = np.linspace(-8, 8, 500)

for i, step in enumerate(plot_steps):
    ax = axes[i]
    
    mppi_mean = mppi_joint.history[step]
    current_std = 5.0 / (step + 1) # 수렴 가정
    mppi_sim_samples = np.random.normal(mppi_mean, current_std, 1000)
    
    ax.hist(mppi_sim_samples, bins=100, density=True, color='blue', alpha=0.4, label='MPPI approx')
    
    # GaBP PDF
    if step < len(gabp.history):
        gabp_mu = gabp.history[step][0, 0]
        # Prec가 커질수록 std 작아짐
        gabp_prec = 1.0 + step * 2.0 # 시각화용 근사
        gabp_std = np.sqrt(1.0/gabp_prec)
        gabp_pdf = norm.pdf(x_range, gabp_mu, gabp_std)
        ax.plot(x_range, gabp_pdf, 'r-', linewidth=2, label='GaBP PDF')

    ax.axvline(-2.0, color='green', linestyle='--')
    ax.set_title(f"Iter {step+1}")
    ax.set_xlim(-8, 8)
    ax.grid(True, alpha=0.3)
    if i == 0: ax.legend()

plt.tight_layout()
plt.show()

# 통계 출력
print(f"--- 최종 결과 ---")
print(f"MPPI-BP Final Mean : {mppi_hist[-1]:.4f}")
print(f"GaBP Final Mean    : {gabp_hist[-1]:.4f}")