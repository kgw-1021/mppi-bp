import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from typing import List, Dict, Tuple, Callable
import networkx as nx
from fg.factor_graph import FactorGraph, VNode, FNode
from fg.gaussian import Gaussian
from fg.factor_graph_mppi import SampleFactorGraph, SampleVNode, SampleFNode, SampleMessage

# ==================== GBP용 비선형 팩터 (Jittering이 잘 발생하도록 수정) ====================
class NonlinearFNodeGBP(FNode):
    """GBP용 비선형 팩터 - 선형화로 근사"""
    def __init__(self, name: str, vnodes: List[VNode], 
                 nonlinear_fn: Callable, gt_value: float, precision: float = 100):
        super().__init__(name, vnodes)
        self._nonlinear_fn = nonlinear_fn
        self._gt_value = gt_value
        self._precision = precision
    
    def update_factor(self):
        """선형화를 통해 팩터 업데이트 (비선형 함수를 사용하는 팩터)"""
        if len(self._vnodes) == 2:
            try:
                # v1: (x, y) / v2: (x, y)
                v1_x, v1_y = self._vnodes[0].belief.mean[0, 0], self._vnodes[0].belief.mean[1, 0]
                v2_x, v2_y = self._vnodes[1].belief.mean[0, 0], self._vnodes[1].belief.mean[1, 0]
                
                # 비선형 함수 값 (현재 평균 위치에서의 값)
                h_val = self._nonlinear_fn(v1_x, v1_y, v2_x, v2_y)
                
                # 수치 미분 (Jacobian 계산)
                eps = 1e-4
                
                # J = [dh/dv1_x, dh/dv1_y, dh/dv2_x, dh/dv2_y]
                dh_dv1_x = (self._nonlinear_fn(v1_x + eps, v1_y, v2_x, v2_y) - h_val) / eps
                dh_dv1_y = (self._nonlinear_fn(v1_x, v1_y + eps, v2_x, v2_y) - h_val) / eps
                dh_dv2_x = (self._nonlinear_fn(v1_x, v1_y, v2_x + eps, v2_y) - h_val) / eps
                dh_dv2_y = (self._nonlinear_fn(v1_x, v1_y, v2_x, v2_y + eps) - h_val) / eps
                
                jacob = np.array([[dh_dv1_x, dh_dv1_y, dh_dv2_x, dh_dv2_y]])
                precision_mat = np.array([[self._precision]]) # 팩터의 정밀도
                
                # 정보 행렬 (정밀도 행렬) 및 정보 벡터 계산
                prec = jacob.T @ precision_mat @ jacob
                v = np.array([[v1_x], [v1_y], [v2_x], [v2_y]])
                
                # 선형화된 측정치: z = h(v) + J * (v_true - v)  => J * v_true = z - h(v) + J*v
                residual = self._gt_value - h_val + jacob @ v
                info = jacob.T @ precision_mat @ residual
                
                # 팩터 갱신 (선형 근사된 가우시안 팩터)
                self._factor = Gaussian.from_info(self.dims, info, prec)
            except Exception as e:
                # print(f"GBP Update Error: {e}")
                pass

# ==================== 2D Factor Graph 생성 (GBP) - 비선형 강화 ====================
def create_2d_factor_graph_gbp(n_vars: int = 7) -> Tuple[FactorGraph, List[VNode], np.ndarray, List]:
    """
    2D 평면에 배치된 변수들로 구성된 팩터 그래프 생성 (GBP) - 비선형 강화 버전
    """
    angles = np.linspace(0, 2*np.pi, n_vars, endpoint=False)
    radius = 10.0
    gt_values = np.column_stack([
        radius * np.cos(angles),
        radius * np.sin(angles)
    ])
    
    graph = FactorGraph()
    vnodes = []
    
    # 변수 노드 생성 (초기값은 GT에서 노이즈 추가)
    for i in range(n_vars):
        init_mean = gt_values[i:i+1].T + np.random.randn(2, 1) * 2.0
        init_cov = np.eye(2) * 5.0
        v = VNode(f'x{i}', [f'x{i}_x', f'x{i}_y'], 
                  belief=Gaussian([f'x{i}_x', f'x{i}_y'], init_mean, init_cov))
        vnodes.append(v)
        graph.add_node(v)
    
    fnodes = []
    
    # 비선형 함수 정의: 다소 복잡한 형태
    def make_nonlinear_fn(is_loop):
        # Loop: sin(x_ax * PI/2) * cos(x_by * PI/2)
        # Diag: x_ay * x_bx
        if is_loop:
            return lambda v1_x, v1_y, v2_x, v2_y: np.sin(v1_x * np.pi/2) * np.cos(v2_y * np.pi/2)
        else:
            return lambda v1_x, v1_y, v2_x, v2_y: v1_y * v2_x # 간단한 2차식
    
    # 1. 원형 루프 연결 (인접 노드끼리)
    for i in range(n_vars):
        v_a = vnodes[i]
        v_b = vnodes[(i + 1) % n_vars]
        gt_a = gt_values[i]
        gt_b = gt_values[(i + 1) % n_vars]
        
        nonlinear_fn = make_nonlinear_fn(is_loop=True)
        # GT 값에 대한 제약 값
        gt_constraint_val = nonlinear_fn(gt_a[0], gt_a[1], gt_b[0], gt_b[1]) 
        
        joint_dims = v_a.dims + v_b.dims
        
        # 팩터의 정밀도 강화
        f = NonlinearFNodeGBP(f'f_loop{i}', [v_a, v_b], 
                              nonlinear_fn=nonlinear_fn, gt_value=gt_constraint_val, 
                              precision=100) # 정밀도 10배 증가
        fnodes.append(f)
        graph.connect(v_a, f)
        graph.connect(v_b, f)
    
    # 2. 대각선 연결 (비선형 팩터)
    for i in range(n_vars // 2):
        v_a = vnodes[i]
        v_b = vnodes[i + n_vars // 2]
        gt_a = gt_values[i]
        gt_b = gt_values[i + n_vars // 2]
        
        nonlinear_fn = make_nonlinear_fn(is_loop=False)
        gt_constraint_val = nonlinear_fn(gt_a[0], gt_a[1], gt_b[0], gt_b[1])
        
        joint_dims = v_a.dims + v_b.dims
        
        f = NonlinearFNodeGBP(f'f_diag{i}', [v_a, v_b],
                              nonlinear_fn=nonlinear_fn, gt_value=gt_constraint_val,
                              precision=100)
        fnodes.append(f)
        graph.connect(v_a, f)
        graph.connect(v_b, f)
    
    return graph, vnodes, gt_values, fnodes


# ==================== 2D Factor Graph 생성 (MPPI-BP) - 비선형 강화 ====================
def create_2d_factor_graph_sample(n_vars: int = 7, n_particles: int = 5000) -> Tuple[SampleFactorGraph, List[SampleVNode], np.ndarray, List]:
    """
    2D 평면에 배치된 변수들로 구성된 팩터 그래프 생성 (MPPI-BP) - 비선형 강화 버전
    """
    angles = np.linspace(0, 2*np.pi, n_vars, endpoint=False)
    radius = 10.0
    gt_values = np.column_stack([
        radius * np.cos(angles),
        radius * np.sin(angles)
    ])
    
    graph = SampleFactorGraph()
    vnodes = []
    
    # 변수 노드 생성
    for i in range(n_vars):
        # 초기 샘플도 더 넓게 퍼지도록 설정
        init_samples = gt_values[i:i+1] + np.random.randn(n_particles, 2) * 3.0
        v = SampleVNode(f'x{i}', [f'x{i}_x', f'x{i}_y'],
                        belief=SampleMessage([f'x{i}_x', f'x{i}_y'], init_samples,
                                            np.ones(n_particles) / n_particles))
        vnodes.append(v)
        graph.add_node(v)
    
    fnodes = []
    # MPPI 파라미터 조정
    mppi_K = 100 # 더 많은 샘플
    mppi_lambda = 0.05 # 람다 증가 (더 많은 탐색)
    mppi_noise_std = 0.5 # 노이즈 증가 (더 많은 탐색)
    
    # 비선형 함수 정의: GBP와 동일하게 설정
    def make_nonlinear_fn(is_loop):
        if is_loop:
            return lambda v1_x, v1_y, v2_x, v2_y: np.sin(v1_x * np.pi/2) * np.cos(v2_y * np.pi/2)
        else:
            return lambda v1_x, v1_y, v2_x, v2_y: v1_y * v2_x

    # 1. 원형 루프 연결
    for i in range(n_vars):
        v_a = vnodes[i]
        v_b = vnodes[(i + 1) % n_vars]
        gt_a = gt_values[i]
        gt_b = gt_values[(i + 1) % n_vars]
        
        nonlinear_fn = make_nonlinear_fn(is_loop=True)
        gt_constraint_val = nonlinear_fn(gt_a[0], gt_a[1], gt_b[0], gt_b[1]) 
        
        def make_nonlinear_cost(target, weight=1000):
            def cost_fn(samples):  # samples: (K, 4) -> [v1_x, v1_y, v2_x, v2_y]
                v1_x, v1_y, v2_x, v2_y = samples[:, 0], samples[:, 1], samples[:, 2], samples[:, 3]
                h_val = np.sin(v1_x * np.pi/2) * np.cos(v2_y * np.pi/2)
                return (h_val - target)**2 * weight
            return cost_fn
        
        joint_samples = np.hstack([v_a.belief.samples, v_b.belief.samples])
        joint_dims = v_a.dims + v_b.dims
        
        f = SampleFNode(f'f_loop{i}', [v_a, v_b],
                        factor_samples=SampleMessage(joint_dims, joint_samples,
                                                     np.ones(n_particles) / n_particles),
                        mppi_params={'K': mppi_K, 'lambda': mppi_lambda, 'noise_std': mppi_noise_std})
        f._cost_fn = make_nonlinear_cost(gt_constraint_val, weight=1000)
        fnodes.append(f)
        graph.connect(v_a, f)
        graph.connect(v_b, f)
    
    # 2. 대각선 연결
    for i in range(n_vars // 2):
        v_a = vnodes[i]
        v_b = vnodes[i + n_vars // 2]
        gt_a = gt_values[i]
        gt_b = gt_values[i + n_vars // 2]
        
        nonlinear_fn = make_nonlinear_fn(is_loop=False)
        gt_constraint_val = nonlinear_fn(gt_a[0], gt_a[1], gt_b[0], gt_b[1])
        
        def make_nonlinear_cost(target, weight=500):
            def cost_fn(samples):
                v1_y, v2_x = samples[:, 1], samples[:, 2]
                h_val = v1_y * v2_x
                return (h_val - target)**2 * weight
            return cost_fn
        
        joint_samples = np.hstack([v_a.belief.samples, v_b.belief.samples])
        joint_dims = v_a.dims + v_b.dims
        
        f = SampleFNode(f'f_diag{i}', [v_a, v_b],
                        factor_samples=SampleMessage(joint_dims, joint_samples,
                                                     np.ones(n_particles) / n_particles),
                        mppi_params={'K': mppi_K, 'lambda': mppi_lambda, 'noise_std': mppi_noise_std})
        f._cost_fn = make_nonlinear_cost(gt_constraint_val, weight=500)
        fnodes.append(f)
        graph.connect(v_a, f)
        graph.connect(v_b, f)
    
    return graph, vnodes, gt_values, fnodes


# ==================== 시각화 (변경 없음) ====================
def get_positions_gbp(vnodes: List[VNode]) -> np.ndarray:
    """GBP belief의 평균값 추출"""
    positions = []
    for v in vnodes:
        try:
            mean = v.belief.mean  # (2, 1)
            positions.append([mean[0, 0], mean[1, 0]])
        except:
            positions.append([0.0, 0.0])
    return np.array(positions)


def get_positions_sample(vnodes: List[SampleVNode]) -> np.ndarray:
    """Sample belief의 가중 평균값 추출"""
    positions = []
    for v in vnodes:
        mean = np.average(v.belief.samples, axis=0, weights=v.belief.weights)
        positions.append(mean)
    return np.array(positions)


def create_animation(n_vars: int = 7, max_iters: int = 10, n_particles: int = 5000,
                     save_path: str = '2d_convergence_highly_nonlinear2.gif'):
    """
    GBP vs MPPI-BP 수렴 과정을 나란히 비교하는 애니메이션 생성
    """
    print(f"Creating 2D Factor Graph with {n_vars} variables (Highly Nonlinear)...")
    
    # 두 그래프 생성
    graph_gbp, vnodes_gbp, gt_values, fnodes_gbp = create_2d_factor_graph_gbp(n_vars)
    graph_sample, vnodes_sample, _, fnodes_sample = create_2d_factor_graph_sample(n_vars, n_particles)
    
    # 초기 위치 기록
    history_gbp = [get_positions_gbp(vnodes_gbp)]
    history_sample = [get_positions_sample(vnodes_sample)]
    
    print("Running inference...")
    for it in range(max_iters):
        if it % 1 == 0:
            print(f"  Iteration {it+1}/{max_iters}")
        
        # GBP 업데이트
        graph_gbp.loopy_propagate(steps=1)
        history_gbp.append(get_positions_gbp(vnodes_gbp))
        
        # MPPI-BP 업데이트
        factor_costs = {f: (f._cost_fn, {}) for f in fnodes_sample}
        graph_sample.loopy_propagate(steps=1, factor_costs=factor_costs)
        history_sample.append(get_positions_sample(vnodes_sample))
    
    print("Creating animation...")
    
    # Figure 설정
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 축 범위 설정
    all_pos = np.vstack([gt_values])
    margin = 10
    x_min, x_max = all_pos[:, 0].min() - margin, all_pos[:, 0].max() + margin
    y_min, y_max = all_pos[:, 1].min() - margin, all_pos[:, 1].max() + margin
    
    def init():
        for ax in [ax1, ax2]:
            ax.clear()
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
        
        ax1.set_title('GBP (Linearized)', fontsize=14, fontweight='bold')
        ax2.set_title('MPPI-BP (Ours)', fontsize=14, fontweight='bold')
        
        return []
    
    def update(frame):
        for ax in [ax1, ax2]:
            ax.clear()
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
        
        # GT 위치 (초록색 별)
        ax1.scatter(gt_values[:, 0], gt_values[:, 1], 
                   c='green', s=200, marker='*', label='Ground Truth', 
                   edgecolors='darkgreen', linewidths=2, zorder=5)
        ax2.scatter(gt_values[:, 0], gt_values[:, 1],
                   c='green', s=200, marker='*', label='Ground Truth',
                   edgecolors='darkgreen', linewidths=2, zorder=5)
        
        # 현재 추정 위치
        pos_gbp = history_gbp[frame]
        pos_sample = history_sample[frame]
        
        # GBP
        ax1.scatter(pos_gbp[:, 0], pos_gbp[:, 1],
                   c='blue', s=150, marker='o', label='Current Estimate',
                   edgecolors='darkblue', linewidths=2, alpha=0.7, zorder=4)
        
        # 궤적 표시
        for i in range(n_vars):
            traj = np.array([history_gbp[f][i] for f in range(frame + 1)])
            ax1.plot(traj[:, 0], traj[:, 1], 'b-', alpha=0.3, linewidth=1)
        
        # 연결선 (루프)
        for i in range(n_vars):
            j = (i + 1) % n_vars
            ax1.plot([pos_gbp[i, 0], pos_gbp[j, 0]], 
                    [pos_gbp[i, 1], pos_gbp[j, 1]],
                    'gray', alpha=0.3, linewidth=1)
        
        # MPPI-BP
        ax2.scatter(pos_sample[:, 0], pos_sample[:, 1],
                   c='red', s=150, marker='s', label='Current Estimate',
                   edgecolors='darkred', linewidths=2, alpha=0.7, zorder=4)
        
        for i in range(n_vars):
            traj = np.array([history_sample[f][i] for f in range(frame + 1)])
            ax2.plot(traj[:, 0], traj[:, 1], 'r-', alpha=0.3, linewidth=1)
        
        for i in range(n_vars):
            j = (i + 1) % n_vars
            ax2.plot([pos_sample[i, 0], pos_sample[j, 0]],
                    [pos_sample[i, 1], pos_sample[j, 1]],
                    'gray', alpha=0.3, linewidth=1)
        
        # 오차 계산
        error_gbp = np.sqrt(np.mean(np.sum((pos_gbp - gt_values)**2, axis=1)))
        error_sample = np.sqrt(np.mean(np.sum((pos_sample - gt_values)**2, axis=1)))
        
        ax1.set_title(f'GBP (Linearized)\nIteration: {frame}, RMSE: {error_gbp:.4f}',
                     fontsize=14, fontweight='bold')
        ax2.set_title(f'MPPI-BP (Ours)\nIteration: {frame}, RMSE: {error_sample:.4f}',
                     fontsize=14, fontweight='bold')
        
        ax1.legend(loc='upper right', fontsize=10)
        ax2.legend(loc='upper right', fontsize=10)
        
        return []
    
    # 애니메이션 생성 (매 1프레임마다)
    frames = list(range(0, len(history_gbp), 1))
    anim = FuncAnimation(fig, update, frames=frames, init_func=init,
                        blit=False, repeat=True, interval=200) # 인터벌을 늘려 관찰 용이하게
    
    # GIF 저장
    print(f"Saving animation to {save_path}...")
    writer = PillowWriter(fps=3)
    anim.save(save_path, writer=writer)
    print(f"Animation saved successfully!")
    
    plt.close()
    
    # 최종 결과 플롯
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    final_gbp = history_gbp[-1]
    final_sample = history_sample[-1]
    
    for ax, pos, title, color, marker in [
        (ax1, final_gbp, 'GBP (Linearized) - Final', 'blue', 'o'),
        (ax2, final_sample, 'MPPI-BP (Ours) - Final', 'red', 's')
    ]:
        ax.scatter(gt_values[:, 0], gt_values[:, 1],
                  c='green', s=200, marker='*', label='Ground Truth',
                  edgecolors='darkgreen', linewidths=2, zorder=5)
        ax.scatter(pos[:, 0], pos[:, 1],
                  c=color, s=150, marker=marker, label='Final Estimate',
                  edgecolors='dark' + color, linewidths=2, alpha=0.7, zorder=4)
        
        # 오차선
        for i in range(n_vars):
            ax.plot([gt_values[i, 0], pos[i, 0]],
                   [gt_values[i, 1], pos[i, 1]],
                   'k--', alpha=0.5, linewidth=1)
        
        # 루프 연결
        for i in range(n_vars):
            j = (i + 1) % n_vars
            ax.plot([pos[i, 0], pos[j, 0]],
                   [pos[i, 1], pos[j, 1]],
                   'gray', alpha=0.3, linewidth=1)
        
        # 대각선 연결 (시각화 추가)
        for i in range(n_vars // 2):
            j = i + n_vars // 2
            ax.plot([pos[i, 0], pos[j, 0]],
                   [pos[i, 1], pos[j, 1]],
                   'gray', linestyle='--', alpha=0.3, linewidth=1)
        
        error = np.sqrt(np.mean(np.sum((pos - gt_values)**2, axis=1)))
        ax.set_title(f'{title}\nFinal RMSE: {error:.4f}',
                    fontsize=14, fontweight='bold')
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('2d_convergence_highly_nonlinear.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nExperiment completed with highly nonlinear factors!")


if __name__ == '__main__':
    create_animation(
        n_vars=20,
        max_iters=5, # 반복 횟수 증가
        n_particles=100,
        save_path='2d_convergence_highly_nonlinear.gif'
    )