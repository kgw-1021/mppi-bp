"""
GBP vs Sample-based MPPI 수렴성 비교 실험
(비선형 "이항" 팩터 및 루프 포함)

[실험 케이스 (총 8개)]
- 팩터 타입 (Binary Factor):
  - linear: f(a,b) = a - b
  - sin:    f(a,b) = sin(a) + sin(b)
  - quad:   f(a,b) = a^2 + b^2
  - exp:    f(a,b) = exp(a/3) + exp(b/3)
- 구조 타입 (Structure):
  - _chain (루프 없음)
  - _loop (루프 있음)
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Callable
import time

# GBP 방식
from fg.gaussian import Gaussian
from fg.factor_graph import VNode, FNode, FactorGraph

# Sample-based MPPI 방식
from fg.factor_graph_mppi import SampleMessage, SampleVNode, SampleFNode, SampleFactorGraph


# ==================== GBP용 비선형 팩터 (선형화 사용) ====================
class NonlinearFNodeGBP(FNode):
    """GBP용 비선형 팩터 - 선형화로 근사 (오류 처리 추가)"""
    def __init__(self, name: str, vnodes: List[VNode], 
                 nonlinear_fn: Callable, gt_value: float, precision: float = 100):
        super().__init__(name, vnodes)
        self._nonlinear_fn = nonlinear_fn  # h(x)
        self._gt_value = gt_value  # z (측정값/목표값)
        self._precision = precision
    
    def update_factor(self):
        """선형화를 통해 팩터 업데이트: h(x) ≈ z"""
        
        try:
            # --- 이 try 블록이 추가되었습니다 ---
            
            if len(self._vnodes) == 1:
                # Unary factor: h(x) = z
                
                # v.mean 접근 시 Singular Matrix 오류가 발생할 수 있음
                v = self._vnodes[0].mean  # (1, 1) 
                x = v[0, 0]
                
                # 현재 점에서 h(x) 계산 및 미분
                h_x = self._nonlinear_fn(x)
                eps = 1e-6
                h_x_plus = self._nonlinear_fn(x + eps)
                jacob = (h_x_plus - h_x) / eps  # dh/dx
                
                # 선형화: h(x) ≈ h(x0) + J*(x - x0) = z
                # J*x = z - h(x0) + J*x0
                # 정보 형태로 변환
                jacob_array = np.array([[jacob]])  # (1, 1)
                precision_mat = np.array([[self._precision]])
                
                prec = jacob_array.T @ precision_mat @ jacob_array
                residual = self._gt_value - h_x + jacob * x
                info = jacob_array.T @ precision_mat @ np.array([[residual]])
                
                self._factor = Gaussian.from_info(self.dims, info, prec)
            
            elif len(self._vnodes) == 2:
                # Binary factor: h(x1, x2) = z
                
                # v.mean 접근 시 Singular Matrix 오류가 발생할 수 있음
                v1 = self._vnodes[0].mean[0, 0] 
                v2 = self._vnodes[1].mean[0, 0]
                
                h_val = self._nonlinear_fn(v1, v2)
                
                # 수치 미분
                eps = 1e-6
                dh_dv1 = (self._nonlinear_fn(v1 + eps, v2) - h_val) / eps
                dh_dv2 = (self._nonlinear_fn(v1, v2 + eps) - h_val) / eps
                
                jacob = np.array([[dh_dv1, dh_dv2]])  # (1, 2)
                precision_mat = np.array([[self._precision]])
                
                prec = jacob.T @ precision_mat @ jacob
                v = np.array([[v1], [v2]])
                residual = self._gt_value - h_val + jacob @ v
                info = jacob.T @ precision_mat @ residual
                
                self._factor = Gaussian.from_info(self.dims, info, prec)
                
        except np.linalg.LinAlgError:
            # 특이 행렬 오류로 인해 v.mean에 접근 실패
            # 이 팩터의 업데이트를 이번 스텝에서는 건너뜁니다(pass).
            # self._factor는 이전 값을 유지하게 됩니다.
            return


# ==================== 그래프 생성 함수들 ====================
def create_nonlinear_graph_gbp(graph_type: str = 'sin_loop', n_vars: int = 5, 
                               gt_values: np.ndarray = None) -> Tuple[FactorGraph, List[VNode], np.ndarray]:
    """
    비선형 '이항' 팩터 및 루프를 포함한 그래프 생성 (GBP 버전)
    """
    
    # GT 값 생성을 위한 기본 타입 추출
    base_type = graph_type.split('_')[0]
    
    if gt_values is None:
        if base_type == 'linear':
            gt_values = np.arange(n_vars, dtype=float).reshape(-1, 1) * 0.5
        elif base_type == 'sin':
            # sin(a)+sin(b) 제약이므로 GT 값 범위 조절
            gt_values = np.linspace(0, 1.0, n_vars).reshape(-1, 1)
        elif base_type == 'quadratic':
            gt_values = np.sqrt(np.arange(n_vars, dtype=float)).reshape(-1, 1)
        elif base_type == 'exp':
            # exp(a/3)+exp(b/3) 제약이므로 GT 값 범위 조절
            gt_values = np.linspace(0, 2.0, n_vars).reshape(-1, 1)
    
    graph = FactorGraph()
    vnodes = []
    
    # 1. 변수 노드 생성
    for i in range(n_vars):
        init_mean = np.random.randn(1, 1) * 2 + gt_values[i]
        init_cov = np.array([[5.0]])
        v = VNode(f'x{i}', [f'x{i}'], belief=Gaussian([f'x{i}'], init_mean, init_cov))
        vnodes.append(v)
        graph.add_node(v)
    
    # 2. 이항 팩터 추가 (Chain 또는 Loop)
    
    # 루프/체인에 따라 반복 범위 설정
    is_loop = 'loop' in graph_type
    num_factors = n_vars if is_loop else n_vars - 1
    
    for i in range(num_factors):
        v_a = vnodes[i]
        v_b = vnodes[(i + 1) % n_vars] # % 연산 (루프면 마지막->처음, 체인이면 마지막은 사용안됨)
        
        gt_a = gt_values[i, 0]
        gt_b = gt_values[(i + 1) % n_vars, 0]
        
        f = None
        
        if base_type == 'linear':
            # 선형: f(a,b) = b - a = (gt_b - gt_a)
            # FNode.from_linear_constraint(dims, A[[-1, 1]], b[[gt_b-gt_a]], cov)
            # 여기서는 기존 방식대로 Joint Gaussian으로 근사
            joint_dims = v_a.dims + v_b.dims
            joint_mean = np.vstack([gt_a, gt_b])
            joint_cov = np.eye(2) * 0.01 
            f = FNode(f'f_binary{i}', [v_a, v_b], 
                      factor=Gaussian(joint_dims, joint_mean, joint_cov))
        
        elif base_type == 'sin':
            # 비선형: f(a,b) = sin(a) + sin(b)
            fn = lambda v1, v2: np.sin(v1) + np.sin(v2)
            target = np.sin(gt_a) + np.sin(gt_b)
            f = NonlinearFNodeGBP(f'f_binary{i}', [v_a, v_b], fn, target, precision=50)

        elif base_type == 'quadratic':
            # 비선형: f(a,b) = a^2 + b^2
            fn = lambda v1, v2: v1**2 + v2**2
            target = gt_a**2 + gt_b**2
            f = NonlinearFNodeGBP(f'f_binary{i}', [v_a, v_b], fn, target, precision=50)

        elif base_type == 'exp':
            # 비선형: f(a,b) = exp(a/3) + exp(b/3)
            fn = lambda v1, v2: np.exp(v1/3) + np.exp(v2/3)
            target = np.exp(gt_a/3) + np.exp(gt_b/3)
            f = NonlinearFNodeGBP(f'f_binary{i}', [v_a, v_b], fn, target, precision=50)

        if f:
            graph.connect(v_a, f)
            graph.connect(v_b, f)
    
    return graph, vnodes, gt_values


def create_nonlinear_graph_sample(graph_type: str = 'sin_loop', n_vars: int = 5, 
                                  gt_values: np.ndarray = None, 
                                  n_particles: int = 200) -> Tuple[SampleFactorGraph, List[SampleVNode], np.ndarray, List]:
    """
    비선형 '이항' 팩터 및 루프를 포함한 그래프 생성 (Sample-based MPPI 버전)
    """
    
    base_type = graph_type.split('_')[0]
    
    if gt_values is None:
        if base_type == 'linear':
            gt_values = np.arange(n_vars, dtype=float).reshape(-1, 1) * 0.5
        elif base_type == 'sin':
            gt_values = np.linspace(0, 1.0, n_vars).reshape(-1, 1)
        elif base_type == 'quadratic':
            gt_values = np.sqrt(np.arange(n_vars, dtype=float)).reshape(-1, 1)
        elif base_type == 'exp':
            gt_values = np.linspace(0, 2.0, n_vars).reshape(-1, 1)
    
    graph = SampleFactorGraph()
    vnodes = []
    
    # 1. 변수 노드 생성
    for i in range(n_vars):
        init_samples = np.random.randn(n_particles, 1) * 2 + gt_values[i]
        v = SampleVNode(f'x{i}', [f'x{i}'], 
                        belief=SampleMessage([f'x{i}'], init_samples, 
                                            np.ones(n_particles)/n_particles))
        vnodes.append(v)
        graph.add_node(v)
    
    fnodes = []
    mppi_K = 400
    mppi_lambda = 1.0
    
    # 2. 이항 팩터 (비용함수) 추가 (Chain 또는 Loop)
    
    is_loop = 'loop' in graph_type
    num_factors = n_vars if is_loop else n_vars - 1

    for i in range(num_factors):
        v_a = vnodes[i]
        v_b = vnodes[(i + 1) % n_vars]
        
        gt_a = gt_values[i, 0]
        gt_b = gt_values[(i + 1) % n_vars, 0]
        
        cost_fn_builder = None
        target_val = 0

        if base_type == 'linear':
            target_val = gt_b - gt_a
            def make_linear_cost(td):
                def cost_fn(samples): # (K, 2)
                    pred = samples[:, 1] - samples[:, 0] # b - a
                    return (pred - td)**2 * 50
                return cost_fn
            cost_fn_builder = make_linear_cost

        elif base_type == 'sin':
            target_val = np.sin(gt_a) + np.sin(gt_b)
            def make_sin_cost(tgt):
                def cost_fn(samples): # (K, 2)
                    pred = np.sin(samples[:, 0]) + np.sin(samples[:, 1]) # sin(a) + sin(b)
                    return (pred - tgt)**2 * 50
                return cost_fn
            cost_fn_builder = make_sin_cost
        
        elif base_type == 'quadratic':
            target_val = gt_a**2 + gt_b**2
            def make_quad_cost(tgt):
                def cost_fn(samples): # (K, 2)
                    pred = samples[:, 0]**2 + samples[:, 1]**2 # a^2 + b^2
                    return (pred - tgt)**2 * 50
                return cost_fn
            cost_fn_builder = make_quad_cost

        elif base_type == 'exp':
            target_val = np.exp(gt_a/3) + np.exp(gt_b/3)
            def make_exp_cost(tgt):
                def cost_fn(samples): # (K, 2)
                    pred = np.exp(samples[:, 0]/3) + np.exp(samples[:, 1]/3)
                    return (pred - tgt)**2 * 50
                return cost_fn
            cost_fn_builder = make_exp_cost

        joint_samples = np.hstack([v_a.belief.samples, v_b.belief.samples])
        joint_dims = v_a.dims + v_b.dims
        
        f = SampleFNode(f'f_binary{i}', [v_a, v_b],
                        factor_samples=SampleMessage(joint_dims, joint_samples, 
                                                     np.ones(n_particles)/n_particles),
                        mppi_params={'K': mppi_K, 'lambda': mppi_lambda, 'noise_std': 0.5})
        
        f._cost_fn = cost_fn_builder(target_val)
        fnodes.append(f)
        graph.connect(v_a, f)
        graph.connect(v_b, f)
    
    return graph, vnodes, gt_values, fnodes

def compute_error_gbp(vnodes: List[VNode], gt_values: np.ndarray) -> float:
    """GBP belief와 ground truth 간의 RMSE (오류 처리 추가)"""
    errors = []
    for v, gt in zip(vnodes, gt_values):
        try:
            # 이 부분에서 Singular matrix 오류가 발생할 수 있음
            belief_mean = v.belief.mean[0, 0]
            errors.append((belief_mean - gt[0])**2)
        except np.linalg.LinAlgError:
            # 특이 행렬 오류(수렴 실패) 발생 시, 계산이 불가능하므로
            # RMSE가 매우 크게 나오도록 큰 제곱 오차 값을 대신 추가합니다.
            # (sqrt(1e18) = 1e9)
            errors.append(1e18) 
    
    if not errors:
        # vnodes가 비어있는 극단적인 경우
        return 0.0
        
    return np.sqrt(np.mean(errors))


def compute_error_sample(vnodes: List[SampleVNode], gt_values: np.ndarray) -> float:
    """Sample belief와 ground truth 간의 RMSE"""
    errors = []
    for v, gt in zip(vnodes, gt_values):
        belief_mean = np.average(v.belief.samples, axis=0, weights=v.belief.weights)[0]
        errors.append((belief_mean - gt[0])**2)
    return np.sqrt(np.mean(errors))


def run_experiment_single_type(graph_type: str, n_vars: int = 5, max_iters: int = 50, 
                               n_particles: int = 200, n_trials: int = 5):
    """단일 그래프 타입에 대한 실험"""
    print(f"\n{'='*60}")
    print(f"Graph Type: {graph_type.upper()}")
    print(f"{'='*60}")
    
    gbp_errors_all = []
    sample_errors_all = []
    
    for trial in range(n_trials):
        print(f"  Trial {trial+1}/{n_trials}...", end=' ')
        
        # GBP
        graph_gbp, vnodes_gbp, gt_values = create_nonlinear_graph_gbp(graph_type, n_vars)
        gbp_errors = [compute_error_gbp(vnodes_gbp, gt_values)]
        
        for it in range(max_iters):
            graph_gbp.loopy_propagate(steps=1)
            error = compute_error_gbp(vnodes_gbp, gt_values)
            gbp_errors.append(error)
        
        gbp_errors_all.append(gbp_errors)
        
        # Sample-based MPPI
        # (GBP에서 생성한 GT를 재사용하여 동일 조건에서 비교)
        graph_sample, vnodes_sample, _, fnodes_sample = create_nonlinear_graph_sample(
            graph_type, n_vars, gt_values, n_particles)
        sample_errors = [compute_error_sample(vnodes_sample, gt_values)]
        
        for it in range(max_iters):
            factor_costs = {f: (f._cost_fn, {}) for f in fnodes_sample}
            graph_sample.loopy_propagate(steps=1, factor_costs=factor_costs)
            error = compute_error_sample(vnodes_sample, gt_values)
            sample_errors.append(error)
        
        sample_errors_all.append(sample_errors)
        print(f"Done (GBP final: {gbp_errors[-1]:.4f}, Sample: {sample_errors[-1]:.4f})")
    
    gbp_mean = np.mean(gbp_errors_all, axis=0)
    gbp_std = np.std(gbp_errors_all, axis=0)
    sample_mean = np.mean(sample_errors_all, axis=0)
    sample_std = np.std(sample_errors_all, axis=0)
    
    print(f"\n  Final RMSE: GBP={gbp_mean[-1]:.6f}±{gbp_std[-1]:.6f}, "
          f"Sample={sample_mean[-1]:.6f}±{sample_std[-1]:.6f}")
    
    return gbp_mean, gbp_std, sample_mean, sample_std


def run_full_comparison(n_vars: int = 5, max_iters: int = 50, 
                        n_particles: int = 200, n_trials: int = 5):
    """모든 그래프 타입(8개)에 대한 비교 실험"""
    print(f"=== 비선형 이항 팩터/루프 수렴성 비교 실험 ===")
    print(f"변수: {n_vars}, Iterations: {max_iters}, Particles: {n_particles}, Trials: {n_trials}")
    
    # *** 수정: 8개 그래프 타입 (Nonlinear Binary Factors) ***
    graph_types = [
        'linear_chain', 'linear_loop',
        'sin_chain', 'sin_loop',
        'quadratic_chain', 'quadratic_loop',
        'exp_chain', 'exp_loop'
    ]
    results = {}
    
    for gtype in graph_types:
        gbp_mean, gbp_std, sample_mean, sample_std = run_experiment_single_type(
            gtype, n_vars, max_iters, n_particles, n_trials)
        results[gtype] = {
            'gbp_mean': gbp_mean,
            'gbp_std': gbp_std,
            'sample_mean': sample_mean,
            'sample_std': sample_std
        }
    
    # 4x2 그리드로 변경 (총 8개 플롯)
    fig, axes = plt.subplots(4, 2, figsize=(14, 20)) 
    axes = axes.flatten()
    
    for idx, gtype in enumerate(graph_types):
        ax = axes[idx]
        res = results[gtype]
        iters = np.arange(max_iters + 1)
        
        # GBP
        ax.plot(iters, res['gbp_mean'], 'b-', linewidth=2.5, label='GBP (Linearized)', marker='o', 
                markevery=max_iters//5, markersize=6)
        ax.fill_between(iters, res['gbp_mean'] - res['gbp_std'], 
                        res['gbp_mean'] + res['gbp_std'], alpha=0.2, color='b')
        
        # Sample-MPPI
        ax.plot(iters, res['sample_mean'], 'r-', linewidth=2.5, label='Sample-MPPI', 
                marker='s', markevery=max_iters//5, markersize=6)
        ax.fill_between(iters, res['sample_mean'] - res['sample_std'],
                        res['sample_mean'] + res['sample_std'], alpha=0.2, color='r')
        
        ax.set_xlabel('Iteration', fontsize=11, fontweight='bold')
        ax.set_ylabel('RMSE', fontsize=11, fontweight='bold')
        ax.set_title(f'{gtype.upper()} Constraints', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10, loc='upper right')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_yscale('log')
        
        # 최종 값 텍스트 표시
        final_gbp = res['gbp_mean'][-1]
        final_sample = res['sample_mean'][-1]
        winner = 'GBP' if final_gbp < final_sample else 'Sample-MPPI'
        ax.text(0.05, 0.05, f'Winner: {winner}', transform=ax.transAxes,
                fontsize=10, fontweight='bold', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
    plt.tight_layout()
    plt.savefig('nonlinear_binary_loop_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n그래프가 'nonlinear_binary_loop_comparison.png'로 저장되었습니다.")
    plt.show()
    
    # 요약 테이블
    print(f"\n{'='*70}")
    print(f"{'Graph Type':<20} {'GBP Final RMSE':<20} {'Sample Final RMSE':<20} {'Winner':<10}")
    print(f"{'='*70}")
    for gtype in graph_types:
        res = results[gtype]
        gbp_final = f"{res['gbp_mean'][-1]:.6f}±{res['gbp_std'][-1]:.6f}"
        sample_final = f"{res['sample_mean'][-1]:.6f}±{res['sample_std'][-1]:.6f}"
        winner = 'GBP' if res['gbp_mean'][-1] < res['sample_mean'][-1] else 'Sample-MPPI'
        print(f"{gtype.upper():<20} {gbp_final:<20} {sample_final:<20} {winner:<10}")
    print(f"{'='*70}")


if __name__ == '__main__':
    run_full_comparison(
        n_vars=5,
        max_iters=50,
        n_particles=200,
        n_trials=5
    )