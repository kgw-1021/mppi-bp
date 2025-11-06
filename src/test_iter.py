"""
GBP vs Sample-based MPPI 수렴성 비교 실험 (비선형 팩터 포함)
- 선형 제약: GBP에 유리
- 비선형 제약 (sin, exp, quadratic): Sample-MPPI에 유리
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
    """GBP용 비선형 팩터 - 선형화로 근사"""
    def __init__(self, name: str, vnodes: List[VNode], 
                 nonlinear_fn: Callable, gt_value: float, precision: float = 100):
        super().__init__(name, vnodes)
        self._nonlinear_fn = nonlinear_fn  # h(x)
        self._gt_value = gt_value  # z (측정값/목표값)
        self._precision = precision
    
    def update_factor(self):
        """선형화를 통해 팩터 업데이트: h(x) ≈ z"""
        if len(self._vnodes) == 1:
            # Unary factor: h(x) = z
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


# ==================== 그래프 생성 함수들 ====================
def create_nonlinear_graph_gbp(graph_type: str = 'sin', n_vars: int = 5, 
                                gt_values: np.ndarray = None) -> Tuple[FactorGraph, List[VNode], np.ndarray]:
    """
    비선형 팩터를 포함한 그래프 생성 (GBP 버전)
    
    graph_type:
        'linear': x_{i+1} = x_i + 1 (기준선)
        'sin': sin(x_i) = i * 0.3 (사인 함수 제약)
        'quadratic': x_i^2 = i (제곱 제약)
        'exp': exp(x_i/3) = i + 1 (지수 제약)
    """
    if gt_values is None:
        if graph_type == 'linear':
            gt_values = np.arange(n_vars, dtype=float).reshape(-1, 1)
        elif graph_type == 'sin':
            gt_values = np.arcsin(np.linspace(0, 0.9, n_vars)).reshape(-1, 1)
        elif graph_type == 'quadratic':
            gt_values = np.sqrt(np.arange(n_vars, dtype=float)).reshape(-1, 1)
        elif graph_type == 'exp':
            gt_values = np.log(np.arange(1, n_vars+1, dtype=float)) * 3
            gt_values = gt_values.reshape(-1, 1)
    
    graph = FactorGraph()
    vnodes = []
    
    # 변수 노드 생성 (무작위 초기화)
    for i in range(n_vars):
        init_mean = np.random.randn(1, 1) * 2 + gt_values[i]
        init_cov = np.array([[5.0]])
        v = VNode(f'x{i}', [f'x{i}'], belief=Gaussian([f'x{i}'], init_mean, init_cov))
        vnodes.append(v)
        graph.add_node(v)
    
    # 팩터 생성
    if graph_type == 'linear':
        for i in range(n_vars - 1):
            joint_dims = vnodes[i].dims + vnodes[i+1].dims
            joint_mean = np.vstack([gt_values[i], gt_values[i+1]])
            joint_cov = np.eye(2) * 0.01
            f = FNode(f'f{i}', [vnodes[i], vnodes[i+1]], 
                      factor=Gaussian(joint_dims, joint_mean, joint_cov))
            graph.connect(vnodes[i], f)
            graph.connect(vnodes[i+1], f)
    
    elif graph_type == 'sin':
        # Unary factors: sin(x_i) = target_i
        for i in range(n_vars):
            target = np.sin(gt_values[i, 0])
            f = NonlinearFNodeGBP(f'f{i}', [vnodes[i]], 
                                  lambda x: np.sin(x), target, precision=50)
            graph.connect(vnodes[i], f)
    
    elif graph_type == 'quadratic':
        # Unary factors: x_i^2 = target_i
        for i in range(n_vars):
            target = gt_values[i, 0] ** 2
            f = NonlinearFNodeGBP(f'f{i}', [vnodes[i]], 
                                  lambda x: x**2, target, precision=50)
            graph.connect(vnodes[i], f)
    
    elif graph_type == 'exp':
        # Unary factors: exp(x_i/3) = target_i
        for i in range(n_vars):
            target = np.exp(gt_values[i, 0] / 3)
            f = NonlinearFNodeGBP(f'f{i}', [vnodes[i]], 
                                  lambda x: np.exp(x/3), target, precision=50)
            graph.connect(vnodes[i], f)
    
    return graph, vnodes, gt_values


def create_nonlinear_graph_sample(graph_type: str = 'sin', n_vars: int = 5, 
                                   gt_values: np.ndarray = None, 
                                   n_particles: int = 200) -> Tuple[SampleFactorGraph, List[SampleVNode], np.ndarray, List]:
    """
    비선형 팩터를 포함한 그래프 생성 (Sample-based MPPI 버전)
    """
    if gt_values is None:
        if graph_type == 'linear':
            gt_values = np.arange(n_vars, dtype=float).reshape(-1, 1)
        elif graph_type == 'sin':
            gt_values = np.arcsin(np.linspace(0, 0.9, n_vars)).reshape(-1, 1)
        elif graph_type == 'quadratic':
            gt_values = np.sqrt(np.arange(n_vars, dtype=float)).reshape(-1, 1)
        elif graph_type == 'exp':
            gt_values = np.log(np.arange(1, n_vars+1, dtype=float)) * 3
            gt_values = gt_values.reshape(-1, 1)
    
    graph = SampleFactorGraph()
    vnodes = []
    
    # 변수 노드 생성
    for i in range(n_vars):
        init_samples = np.random.randn(n_particles, 1) * 2 + gt_values[i]
        v = SampleVNode(f'x{i}', [f'x{i}'], 
                        belief=SampleMessage([f'x{i}'], init_samples, 
                                           np.ones(n_particles)/n_particles))
        vnodes.append(v)
        graph.add_node(v)
    
    fnodes = []
    
    if graph_type == 'linear':
        for i in range(n_vars - 1):
            target_diff = gt_values[i+1, 0] - gt_values[i, 0]
            
            def make_cost_fn(td):
                def cost_fn(samples):
                    diff = samples[:, 1] - samples[:, 0]
                    return (diff - td)**2 * 50
                return cost_fn
            
            joint_samples = np.hstack([vnodes[i].belief.samples, vnodes[i+1].belief.samples])
            joint_dims = vnodes[i].dims + vnodes[i+1].dims
            f = SampleFNode(f'f{i}', [vnodes[i], vnodes[i+1]],
                           factor_samples=SampleMessage(joint_dims, joint_samples, 
                                                       np.ones(n_particles)/n_particles),
                           mppi_params={'K': 400, 'lambda': 1.0, 'noise_std': 0.5})
            f._cost_fn = make_cost_fn(target_diff)
            fnodes.append(f)
            graph.connect(vnodes[i], f)
            graph.connect(vnodes[i+1], f)
    
    elif graph_type == 'sin':
        for i in range(n_vars):
            target = np.sin(gt_values[i, 0])
            
            def make_cost_fn(tgt):
                def cost_fn(samples):
                    # samples: (K, 1)
                    pred = np.sin(samples[:, 0])
                    return (pred - tgt)**2 * 50
                return cost_fn
            
            f = SampleFNode(f'f{i}', [vnodes[i]],
                           factor_samples=vnodes[i].belief.copy(),
                           mppi_params={'K': 400, 'lambda': 1.0, 'noise_std': 0.3})
            f._cost_fn = make_cost_fn(target)
            fnodes.append(f)
            graph.connect(vnodes[i], f)
    
    elif graph_type == 'quadratic':
        for i in range(n_vars):
            target = gt_values[i, 0] ** 2
            
            def make_cost_fn(tgt):
                def cost_fn(samples):
                    pred = samples[:, 0] ** 2
                    return (pred - tgt)**2 * 50
                return cost_fn
            
            f = SampleFNode(f'f{i}', [vnodes[i]],
                           factor_samples=vnodes[i].belief.copy(),
                           mppi_params={'K': 400, 'lambda': 1.0, 'noise_std': 0.3})
            f._cost_fn = make_cost_fn(target)
            fnodes.append(f)
            graph.connect(vnodes[i], f)
    
    elif graph_type == 'exp':
        for i in range(n_vars):
            target = np.exp(gt_values[i, 0] / 3)
            
            def make_cost_fn(tgt):
                def cost_fn(samples):
                    pred = np.exp(samples[:, 0] / 3)
                    return (pred - tgt)**2 * 50
                return cost_fn
            
            f = SampleFNode(f'f{i}', [vnodes[i]],
                           factor_samples=vnodes[i].belief.copy(),
                           mppi_params={'K': 400, 'lambda': 1.0, 'noise_std': 0.3})
            f._cost_fn = make_cost_fn(target)
            fnodes.append(f)
            graph.connect(vnodes[i], f)
    
    return graph, vnodes, gt_values, fnodes


def compute_error_gbp(vnodes: List[VNode], gt_values: np.ndarray) -> float:
    """GBP belief와 ground truth 간의 RMSE"""
    errors = []
    for v, gt in zip(vnodes, gt_values):
        belief_mean = v.belief.mean[0, 0]
        errors.append((belief_mean - gt[0])**2)
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
        graph_sample, vnodes_sample, gt_values, fnodes_sample = create_nonlinear_graph_sample(
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
    """모든 그래프 타입에 대한 비교 실험"""
    print(f"=== 비선형 팩터 수렴성 비교 실험 ===")
    print(f"변수: {n_vars}, Iterations: {max_iters}, Particles: {n_particles}, Trials: {n_trials}")
    
    graph_types = ['linear', 'sin', 'quadratic', 'exp']
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
    
    # 시각화
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, gtype in enumerate(graph_types):
        ax = axes[idx]
        res = results[gtype]
        iters = np.arange(max_iters + 1)
        
        # GBP
        ax.plot(iters, res['gbp_mean'], 'b-', linewidth=2.5, label='GBP', marker='o', 
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
    plt.savefig('nonlinear_convergence_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n그래프가 'nonlinear_convergence_comparison.png'로 저장되었습니다.")
    plt.show()
    
    # 요약 테이블
    print(f"\n{'='*70}")
    print(f"{'Graph Type':<15} {'GBP Final RMSE':<20} {'Sample Final RMSE':<20} {'Winner':<10}")
    print(f"{'='*70}")
    for gtype in graph_types:
        res = results[gtype]
        gbp_final = f"{res['gbp_mean'][-1]:.6f}±{res['gbp_std'][-1]:.6f}"
        sample_final = f"{res['sample_mean'][-1]:.6f}±{res['sample_std'][-1]:.6f}"
        winner = 'GBP' if res['gbp_mean'][-1] < res['sample_mean'][-1] else 'Sample-MPPI'
        print(f"{gtype.upper():<15} {gbp_final:<20} {sample_final:<20} {winner:<10}")
    print(f"{'='*70}")


if __name__ == '__main__':
    run_full_comparison(
        n_vars=5,
        max_iters=50,
        n_particles=200,
        n_trials=5
    )