import numpy as np
from typing import List, Dict, Iterable, Tuple, Literal
import itertools
from scipy.stats import gaussian_kde
from sklearn.mixture import GaussianMixture
from .graph import Node, Edge, Graph # 원본과 동일

class SampleMessage:
    """
    samples: (N, D) np.ndarray
    dims: list of dim names (length D)
    weights: (N,) non-negative, not necessarily normalized
    """
    def __init__(self, dims: List[str], samples: np.ndarray, weights: np.ndarray = None):
        self._dims = list(dims)
        self.samples = np.asarray(samples).copy()
        
        # 샘플이 1D 배열로 들어온 경우 (N,) -> (N, 1)로 변환
        if self.samples.ndim == 1:
            self.samples = self.samples.reshape(-1, 1)

        assert self.samples.ndim == 2 and self.samples.shape[1] == len(self._dims), \
            f"Samples shape {self.samples.shape} mismatch with dims {self._dims}"
            
        self.N = self.samples.shape[0]
        
        if weights is None:
            self.weights = np.ones(self.N) / self.N
        else:
            w = np.asarray(weights).astype(float)
            assert w.shape == (self.N,), \
                f"Weights shape {w.shape} mismatch with N={self.N}"
            
            # REFACTOR: 가중치 합이 0일 때의 안정성 강화 (모두 0이거나 음수인 경우)
            s = w.sum()
            if s <= 0 or not np.isfinite(s):
                w = np.ones_like(w) / len(w)
            self.weights = w / w.sum()

    @property
    def dims(self):
        return self._dims.copy()

    def copy(self):
        return SampleMessage(self._dims.copy(), self.samples.copy(), self.weights.copy())

    def normalize(self):
        s = self.weights.sum()
        if s <= 0 or not np.isfinite(s):
            self.weights = np.ones_like(self.weights) / len(self.weights)
        else:
            self.weights = self.weights / s
        return self

    def project(self, dims_out: List[str]) -> 'SampleMessage':
        """지정 dims_out으로 사영 — 단순히 차원 선택, 동일한 입자 수 유지."""
        try:
            idxs = [self._dims.index(d) for d in dims_out]
        except ValueError as e:
            raise ValueError(f"Cannot project: Dim '{e.args[0].split()[0]}' not in message dims {self._dims}") from e
            
        samples_out = self.samples[:, idxs]
        return SampleMessage(dims_out, samples_out.copy(), self.weights.copy())

    # ... evaluate_kernel, evaluate_gmm, evaluate_gaussian (원본과 동일) ...
    # (원본 코드의 evaluate_kernel, evaluate_gmm, evaluate_gaussian 메서드 위치)
    def evaluate_kernel(self, pts: np.ndarray, bandwidth: float = None) -> np.ndarray:
        """
        커널 밀도 추정으로 이 메시지의 밀도값을 pts에 대해 평가.
        pts: (M, D) where D == len(self._dims)
        Returns: (M,) estimated density (not normalized by bandwidth constant)
        """
        pts = np.asarray(pts)
        if pts.ndim == 1:
            pts = pts[None, :]
        assert pts.shape[1] == len(self._dims)
        M = pts.shape[0]
        N = self.N
        if bandwidth is None:
            # Silverman's rule-of-thumb scalar bandwidth per-dim
            stds = np.std(self.samples, axis=0, ddof=1)
            avg_std = np.mean(stds) if np.any(stds > 0) else 1.0
            bandwidth = 1.06 * avg_std * (N ** (-1/5)) + 1e-6
        # Gaussian kernel
        diffs = pts[:, None, :] - self.samples[None, :, :]  # (M, N, D)
        sq = np.sum(diffs * diffs, axis=2)  # (M, N)
        ker = np.exp(-0.5 * sq / (bandwidth**2))
        # weigh kernels by particle weights
        vals = ker @ self.weights
        # Note: Not dividing by (sqrt(2pi)*bandwidth)^D -- constant factor not needed for reweighting
        return vals

    def evaluate_gmm(self, pts: np.ndarray, n_components: int = None) -> np.ndarray:
        """
        GMM으로 이 메시지의 밀도값을 pts에 대해 평가.
        pts: (M, D) where D == len(self._dims)
        Returns: (M,) estimated density
        """
        pts = np.asarray(pts)
        if pts.ndim == 1:
            pts = pts[None, :]
        assert pts.shape[1] == len(self._dims)
        
        if n_components is None:
            n_components = min(10, max(1, self.N // 20))
        
        # Fit GMM with weighted samples
        gmm = GaussianMixture(n_components=n_components, covariance_type='full', 
                              max_iter=100, random_state=42)
        # Use sample_weight parameter for weighted fitting
        try:
            gmm.fit(self.samples, sample_weight=self.weights)
            vals = np.exp(gmm.score_samples(pts))
        except:
            # Fallback to unweighted if weighted fitting fails
            gmm.fit(self.samples)
            vals = np.exp(gmm.score_samples(pts))
        
        return vals

    def evaluate_gaussian(self, pts: np.ndarray) -> np.ndarray:
        """
        가중치 샘플셋을 단일 가우시안으로 근사하여 밀도값을 pts에 대해 평가.
        pts: (M, D) where D == len(self._dims)
        Returns: (M,) estimated density
        """
        pts = np.asarray(pts)
        if pts.ndim == 1:
            pts = pts[None, :]
        assert pts.shape[1] == len(self._dims)
        
        # Compute weighted mean and covariance
        mean = np.average(self.samples, weights=self.weights, axis=0)
        diffs = self.samples - mean[None, :]
        cov = np.average(diffs[:, :, None] * diffs[:, None, :], 
                        weights=self.weights, axis=0)
        
        # Add small regularization for numerical stability
        cov += np.eye(cov.shape[0]) * 1e-6
        
        # Compute Gaussian density
        D = len(self._dims)
        try:
            cov_inv = np.linalg.inv(cov)
            diffs_pts = pts - mean[None, :]
            mahal = np.sum(diffs_pts @ cov_inv * diffs_pts, axis=1)
            vals = np.exp(-0.5 * mahal) / np.sqrt((2 * np.pi) ** D * np.linalg.det(cov))
        except:
            # If singular, fall back to diagonal approximation
            var = np.diag(cov) + 1e-6
            diffs_pts = pts - mean[None, :]
            mahal = np.sum((diffs_pts ** 2) / var[None, :], axis=1)
            vals = np.exp(-0.5 * mahal) / np.sqrt((2 * np.pi) ** D * np.prod(var))
        
        return vals

    def resample(self, N_out: int, jitter: float = 1e-3) -> 'SampleMessage':
        """가중치에 따라 재샘플링(복원추출) 하고 작은 가우시안 노이즈 추가"""
        idx = np.random.choice(self.N, size=N_out, p=self.weights)
        samples = self.samples[idx, :].copy()
        if jitter > 0:
            samples += np.random.randn(*samples.shape) * jitter
        weights = np.ones(N_out) / N_out
        return SampleMessage(self._dims.copy(), samples, weights)

    # -----------------------------------------------------------------
    # REFACTOR 1: (가장 중요) 차원 불일치를 허용하는 유연한 곱셈 연산
    # -----------------------------------------------------------------
    def multiply_with(self, other: 'SampleMessage', 
                     method: Literal['kde', 'gmm', 'gaussian'] = 'kde',
                     bandwidth: float = None,
                     n_components: int = None) -> 'SampleMessage':
        """
        두 메시지의 곱(정보 결합)을 근사.
        - p_result(X) ∝ p_self(X) * p_other(Y)
        - p_self(X)의 샘플(self.samples)을 제안 분포로 사용.
        - p_other(Y)를 공통 차원 (X ∩ Y)에 대해 평가하여 재가중치.

        Args:
            other: 곱할 다른 메시지 p_other(Y)
            method: other 메시지의 밀도 추정 방법
        """
        
        # 1. 제안 샘플 = self.samples
        base_samples = self.samples
        new_weights = self.weights.copy()
        base_dims = self.dims
        
        # 2. 공통 차원 찾기
        common_dims = [d for d in base_dims if d in other.dims]
        
        # 3. 공통 차원이 없으면, other는 self에 대해 상수 취급. self 반환.
        if not common_dims:
            return self.copy() 

        # 4. 공통 차원에 대해 other 메시지 평가
        
        # 4a. self.samples에서 공통 차원 부분(X ∩ Y) 추출
        try:
            self_idxs = [base_dims.index(d) for d in common_dims]
            samples_for_eval = base_samples[:, self_idxs]
        except ValueError:
            # (이론상) 일어날 수 없음
            return self.copy() 

        # 4b. other 메시지가 공통 차원 외에 다른 차원도 가지는가?
        # 예: self(x,y) * other(x,z)
        # -> other(x,z)를 p(x) = ∫ p(x,z) dz 로 사영(marginalize)해야 함.
        if set(other.dims) != set(common_dims):
            other_projected = other.project(common_dims)
        else:
            other_projected = other
            
        # 5. 사영된 other 메시지(p(X ∩ Y))를 평가
        if method == 'kde':
            other_vals = other_projected.evaluate_kernel(samples_for_eval, bandwidth=bandwidth)
        elif method == 'gmm':
            other_vals = other_projected.evaluate_gmm(samples_for_eval, n_components=n_components)
        elif method == 'gaussian':
            other_vals = other_projected.evaluate_gaussian(samples_for_eval)
        else:
            raise ValueError(f"Unknown method: {method}. Choose from 'kde', 'gmm', 'gaussian'")
        
        # 6. 가중치 업데이트 및 정규화
        new_weights *= other_vals
        
        # self.dims와 self.samples를 그대로 사용하고 가중치만 업데이트
        return SampleMessage(base_dims, base_samples, new_weights).normalize()


# ---------------------------
# Sample-based nodes
# ---------------------------
class SampleVNode(Node):
    def __init__(self, name: str, dims: List[str], belief: SampleMessage = None, N_particles: int = 200, init_std: float = 1.0):
        super().__init__(name, dims)
        self.N_particles = N_particles
        if belief is None:
            # REFACTOR 2: (0,0,...) 대신 비정보성 가우시안 Prior로 초기화
            samples = np.random.randn(self.N_particles, len(dims)) * init_std
            belief = SampleMessage(dims, samples, np.ones(self.N_particles)/self.N_particles)
        self._belief: SampleMessage = belief
        # VNode가 메시지 곱셈 시 사용할 밀도 추정 방식 (설정 가능)
        self.density_method = 'kde' 

    @property
    def belief(self) -> SampleMessage:
        return self._belief.copy()

    # -----------------------------------------------------------------
    # REFACTOR 3: (가장 중요) BP 원칙에 맞는 신념 업데이트
    # -----------------------------------------------------------------
    def update_belief(self) -> SampleMessage:
        """
        표준 BP: b(s) ∝ Prod[ m_{f->s} ]
        (참고: Prior b_0(s)를 곱할 수도 있지만, 여기서는 생략)
        """
        msgs = []
        for e in self.edges:
            m = e.get_message_to(self)
            if m is not None:
                # BP 원칙: VNode로 들어오는 모든 메시지는 VNode의 차원과 일치해야 함
                # (FNode가 이미 사영해서 보냈어야 함)
                if m.dims != self.dims:
                    # 차원이 다른 메시지를 사영 (방어 코드)
                    try:
                        m = m.project(self.dims)
                    except ValueError:
                        continue # 이 변수와 관련 없는 메시지
                msgs.append(m)
        
        if len(msgs) == 0:
            return self._belief.copy() # 수신 메시지 없으면 현재 신념 유지

        # REFACTOR: b_{k-1}가 아닌, 첫 번째 메시지를 base로 시작
        base = msgs[0].copy()
        for m in msgs[1:]:
            # Refactor 1에서 수정된 multiply_with 사용
            base = base.multiply_with(m, method=self.density_method)
        
        # 재샘플링 (입자 수 유지)
        base = base.resample(self.N_particles, jitter=1e-4)
        self._belief = base
        return self._belief.copy()

    # -----------------------------------------------------------------
    # REFACTOR 4: (가장 중요) BP 원칙에 맞는 메시지 계산
    # -----------------------------------------------------------------
    def calc_msg(self, edge: Edge) -> SampleMessage:
        """
        표준 BP: m_{s->f} ∝ Prod[ m_{g->s} ] for g != f
        """
        msgs = []
        for e in self.edges:
            if e is edge: # 메시지를 보낼 엣지(f)는 제외
                continue
            
            m = e.get_message_to(self)
            if m is not None:
                if m.dims != self.dims:
                    try:
                        m = m.project(self.dims)
                    except ValueError:
                        continue
                msgs.append(m)

        if len(msgs) == 0:
            # REFACTOR: self._belief (f의 정보 포함) 대신 
            # 초기 신념(Prior)과 유사한 비정보성 메시지 전송
            # 여기서는 편의상 초기화와 동일한 방식으로 생성
            samples = np.random.randn(self.N_particles, len(self.dims)) * 1.0 # init_std
            return SampleMessage(self.dims, samples)

        # 첫 번째 메시지를 base로 사용
        base = msgs[0].copy()
        for m in msgs[1:]:
            base = base.multiply_with(m, method=self.density_method)

        # 메시지 전파 시 재샘플링 (입자 수 고정)
        base = base.resample(min(500, self.N_particles), jitter=1e-4)
        return base

    def propagate(self):
        for e in self.edges:
            msg = self.calc_msg(e)
            if msg is not None:
                e.set_message_from(self, msg)

class SampleFNode(Node):
    def __init__(self, name: str, vnodes: List[SampleVNode], factor_samples: SampleMessage = None, mppi_params: dict = None, N_factor_particles: int = 400):
        dims = list(dict.fromkeys(itertools.chain(*[v.dims for v in vnodes]))) # 중복 제거
        super().__init__(name, dims)
        self._vnodes = vnodes
        self._dims = dims
        self.N_factor_particles = N_factor_particles
        
        if factor_samples is None:
            # REFACTOR: 0 대신 가우시안 초기화
            samples = np.random.randn(self.N_factor_particles, len(dims)) * 1.0
            factor_samples = SampleMessage(dims, samples)
        self._factor: SampleMessage = factor_samples
        
        self.mppi_params = mppi_params if mppi_params is not None else {'K': self.N_factor_particles, 'lambda': 1.0, 'noise_std': 1.0}
        
        # FNode가 메시지 곱셈 시 사용할 밀도 추정 방식 (설정 가능)
        self.density_method = 'kde' 

    def update_factor_with_mppi(self, cost_fn, base_trajectory: np.ndarray = None):
        """
        cost_fn(samples) -> (K,) cost values.
        samples shape should be (K, D) where D == len(self._dims)
        base_trajectory: optional (D,) center to sample around
        """
        K = int(self.mppi_params.get('K', self.N_factor_particles))
        lam = float(self.mppi_params.get('lambda', 1.0))
        noise_std = float(self.mppi_params.get('noise_std', 1.0))
        D = len(self._dims)
        
        if base_trajectory is None:
            # REFACTOR: 0 대신 현재 팩터 샘플의 가중 평균 사용
            base = np.average(self._factor.samples, weights=self._factor.weights, axis=0)
        else:
            base = np.asarray(base_trajectory).reshape(-1)
            assert base.shape[0] == D
            
        # 샘플 생성 (가우시안 노이즈)
        noises = np.random.randn(K, D) * noise_std
        samples = base[None, :] + noises
        costs = cost_fn(samples)  # (K,)
        
        # MPPI 가중치
        weights = np.exp(-costs / lam)
        
        # REFACTOR: 가중치 안정성 (inf, 0, nan 처리)
        if np.all(np.isinf(weights)) or np.all(weights == 0) or not np.all(np.isfinite(weights)):
            weights = np.ones_like(weights)
        weights = weights / np.sum(weights)
        
        self._factor = SampleMessage(self._dims.copy(), samples, weights)
        return self._factor.copy()

    # -----------------------------------------------------------------
    # REFACTOR 5: 유연한 밀도 추정 방식 적용
    # -----------------------------------------------------------------
    def calc_msg(self, edge: Edge) -> SampleMessage:
        """
        [표준 BP 방식]
        m_{f -> s}(x_s) propto Sum_{~x_s} [ f(x_f) * Prod_{t!=s} m_{t -> f}(x_t) ]
        
        이를 입자 기반으로 근사:
        1. 제안 분포로 self._factor의 입자 (samples, weights)를 사용. (f(x_f) 부분)
        2. 이 입자들의 가중치를 다른 모든 수신 메시지(m_{t->f})로 재가중치.
        3. 목표 변수(x_s) 차원으로 사영(marginalize)하고 재샘플링.
        """
        var_node = edge.get_other(self)
        var_dims = var_node.dims

        # 1. 제안 분포 (팩터)
        joint_samples = self._factor.samples
        new_weights = self._factor.weights.copy()

        # 2. 다른 모든 수신 메시지로 재가중치
        for e in self.edges:
            if e is edge:
                continue  

            msg_in = e.get_message_to(self)
            if msg_in is None:
                continue

            in_dims = msg_in.dims 

            # 이 메시지를 평가하기 위해 joint_samples에서 해당 차원의 열을 추출
            try:
                idxs_in = [self._dims.index(d) for d in in_dims]
            except ValueError:
                # 이 팩터가 해당 차원을 다루지 않으면 메시지 무시
                continue
            
            samples_for_msg = joint_samples[:, idxs_in]

            # REFACTOR: 하드코딩된 KDE 대신 설정된 density_method 사용
            if self.density_method == 'kde':
                msg_vals = msg_in.evaluate_kernel(samples_for_msg)
            elif self.density_method == 'gmm':
                msg_vals = msg_in.evaluate_gmm(samples_for_msg)
            elif self.density_method == 'gaussian':
                msg_vals = msg_in.evaluate_gaussian(samples_for_msg)
            else:
                raise ValueError(f"Unknown density method: {self.density_method}")

            new_weights *= msg_vals

        # 3. 목표 변수 차원으로 사영(Marginalize)
        try:
            idxs_out = [self._dims.index(d) for d in var_dims]
        except ValueError:
            return None # 이 팩터가 목표 변수 차원을 모름

        samples_out = joint_samples[:, idxs_out]
        
        out_msg = SampleMessage(var_dims, samples_out, new_weights)
        out_msg.normalize() # 정규화 필수

        # 4. 재샘플링
        out_msg = out_msg.resample(min(500, out_msg.N), jitter=1e-4)

        return out_msg

    def propagate(self):
        for e in self.edges:
            msg = self.calc_msg(e)
            if msg is not None:
                e.set_message_from(self, msg)

# ---------------------------
# MPPI-based FactorGraph (sample-based loopy BP)
# ---------------------------
class SampleFactorGraph(Graph):
    def __init__(self):
        super().__init__()

    def get_vnodes(self) -> List[SampleVNode]:
        return [n for n in self._nodes if isinstance(n, SampleVNode)]

    def get_fnodes(self) -> List[SampleFNode]:
        return [n for n in self._nodes if isinstance(n, SampleFNode)]

    # -----------------------------------------------------------------
    # REFACTOR 6: 밀도 추정 방식을 전역적으로 설정 가능하게 변경
    # -----------------------------------------------------------------
    def loopy_propagate(self, steps: int = 1, 
                        factor_costs: Dict[SampleFNode, Tuple[callable, dict]] = None,
                        density_method: Literal['kde', 'gmm', 'gaussian'] = 'kde'):
        """
        factor_costs: MPPI 업데이트를 수행할 팩터 노드와 (cost_fn, kwargs) 딕셔너리
        density_method: 메시지 곱셈 및 팩터 메시지 계산에 사용할 밀도 추정 방식
        """
        vnodes = self.get_vnodes()
        fnodes = self.get_fnodes()

        # 전역 밀도 추정 방식 설정
        for node in vnodes + fnodes:
            node.density_method = density_method

        for istep in range(steps):
            # 1. Var -> Factor
            for v in vnodes:
                v.propagate()

            # 2. Factor update (MPPI)
            if factor_costs is not None:
                for f in fnodes:
                    if f in factor_costs:
                        cost_fn, kwargs = factor_costs[f]
                        base = kwargs.get('base', None)
                        f.update_factor_with_mppi(cost_fn, base_trajectory=base)

            # 3. Factor -> Var
            for f in fnodes:
                f.propagate()

            # 4. Var beliefs
            beliefs = {}
            for v in vnodes:
                beliefs[v] = v.update_belief()
        
        return beliefs