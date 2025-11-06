import numpy as np
from typing import List, Dict, Iterable, Tuple
import itertools
from .graph import Node, Edge, Graph

class SampleMessage:
    """
    samples: (N, D) np.ndarray
    dims: list of dim names (length D)
    weights: (N,) non-negative, not necessarily normalized
    """
    def __init__(self, dims: List[str], samples: np.ndarray, weights: np.ndarray = None):
        self._dims = list(dims)
        self.samples = np.asarray(samples).copy()
        assert self.samples.ndim == 2 and self.samples.shape[1] == len(self._dims)
        self.N = self.samples.shape[0]
        if weights is None:
            self.weights = np.ones(self.N) / self.N
        else:
            w = np.asarray(weights).astype(float)
            assert w.shape == (self.N,)
            if w.sum() == 0:
                w = np.ones_like(w) / len(w)
            self.weights = w / (w.sum())

    @property
    def dims(self):
        return self._dims.copy()

    def copy(self):
        return SampleMessage(self._dims.copy(), self.samples.copy(), self.weights.copy())

    def normalize(self):
        s = self.weights.sum()
        if s == 0:
            self.weights = np.ones_like(self.weights) / len(self.weights)
        else:
            self.weights = self.weights / s
        return self

    def project(self, dims_out: List[str]) -> 'SampleMessage':
        """지정 dims_out으로 사영 — 단순히 차원 선택, 동일한 입자 수 유지."""
        idxs = [self._dims.index(d) for d in dims_out]
        samples_out = self.samples[:, idxs]
        return SampleMessage(dims_out, samples_out.copy(), self.weights.copy())

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

    def resample(self, N_out: int, jitter: float = 1e-3) -> 'SampleMessage':
        """가중치에 따라 재샘플링(복원추출) 하고 작은 가우시안 노이즈 추가"""
        idx = np.random.choice(self.N, size=N_out, p=self.weights)
        samples = self.samples[idx, :].copy()
        if jitter > 0:
            samples += np.random.randn(*samples.shape) * jitter
        weights = np.ones(N_out) / N_out
        return SampleMessage(self._dims.copy(), samples, weights)

    def multiply_with(self, other: 'SampleMessage', target_samples: np.ndarray = None, bandwidth: float = None) -> 'SampleMessage':
        """
        두 메시지의 곱(정보 결합)을 근사.
        - 결과 표본은 target_samples(예: self.samples) 위에서 재가중치.
        - p_result(x) ∝ p_self(x) * p_other(x)
        """
        # 선택된 표본 위에서 other의 밀도 추정치를 계산
        if target_samples is None:
            target_samples = self.samples
            target_weights = self.weights.copy()
            dims = self._dims
            # evaluate other on self.samples (align dims)
            # if dims differ, require same dims
            assert self._dims == other._dims, "multiply_with: dims must match for simple multiply"
            other_vals = other.evaluate_kernel(target_samples, bandwidth=bandwidth)
            new_weights = target_weights * other_vals
            msg = SampleMessage(self._dims.copy(), target_samples.copy(), new_weights)
            msg.normalize()
            return msg
        else:
            # evaluate both at provided target_samples
            raise NotImplementedError("multiply_with with arbitrary target_samples not implemented")

# ---------------------------
# Sample-based nodes
# ---------------------------
class SampleVNode(Node):
    def __init__(self, name: str, dims: List[str], belief: SampleMessage = None):
        super().__init__(name, dims)
        if belief is None:
            # 기본 비정보성 prior: 표준 정규 분포(입자) -- 나중에 팩터로부터 보강됨
            N = 200
            samples = np.zeros((N, len(dims)))
            belief = SampleMessage(dims, samples, np.ones(N)/N)
        self._belief: SampleMessage = belief

    @property
    def belief(self) -> SampleMessage:
        return self._belief.copy()

    def update_belief(self) -> SampleMessage:
        """
        모든 들어오는 엣지로부터 온 메시지를 곱해서 신념을 근사.
        순차적으로 메시지들을 self._belief 위에서 재가중치.
        """
        msgs = []
        for e in self.edges:
            m = e.get_message_to(self)
            if m is not None:
                msgs.append(m)
        if len(msgs) == 0:
            return self._belief.copy()
        # 시작표본: 현재 belief의 표본 (or first message)
        base = self._belief.copy()
        for m in msgs:
            # m와 base가 동일한 dims라면 곱 연산 간단
            if base.dims == m.dims:
                base = base.multiply_with(m)
            else:
                # dims mismatch: try to project m to base.dims if possible
                common = [d for d in base.dims if d in m.dims]
                if common == base.dims:
                    # m contains superset of dims -> project m to base dims first
                    mp = m.project(common)
                    base = base.multiply_with(mp)
                else:
                    continue

        base = base.resample(base.N, jitter=1e-4)
        self._belief = base
        return self._belief.copy()

    def calc_msg(self, edge: Edge) -> SampleMessage:
        """
        edge 방향으로 보낼 메시지: (모든 들어오는 메시지 중 edge을 제외한 것들의 곱)
        """
        msgs = []
        for e in self.edges:
            if e is edge:
                continue
            m = e.get_message_to(self)
            if m is not None:
                msgs.append(m)
        if len(msgs) == 0:
            return self._belief.copy()
        base = msgs[0].copy()
        for m in msgs[1:]:
            if base.dims == m.dims:
                base = base.multiply_with(m)
            else:
                # attempt to align dims
                common = [d for d in base.dims if d in m.dims]
                if common == base.dims:
                    mp = m.project(common)
                    base = base.multiply_with(mp)
                else:
                    continue
        # send message as resampled fixed-size message
        base = base.resample(min(500, base.N), jitter=1e-4)
        return base

    def propagate(self):
        for e in self.edges:
            msg = self.calc_msg(e)
            if msg is not None:
                e.set_message_from(self, msg)

class SampleFNode(Node):
    def __init__(self, name: str, vnodes: List[SampleVNode], factor_samples: SampleMessage = None, mppi_params: dict = None):
        dims = list(itertools.chain(*[v.dims for v in vnodes]))
        super().__init__(name, dims)
        self._vnodes = vnodes
        self._dims = dims
        # factor_samples는 joint samples over dims
        if factor_samples is None:
            N = 400
            samples = np.zeros((N, len(dims)))
            factor_samples = SampleMessage(dims, samples, np.ones(N)/N)
        self._factor: SampleMessage = factor_samples
        # MPPI 파라미터 (lambda, K 등)
        self.mppi_params = mppi_params if mppi_params is not None else {'K': 400, 'lambda': 1.0, 'noise_std': 1.0}

    def update_factor_with_mppi(self, cost_fn, base_trajectory: np.ndarray = None):
        """
        cost_fn(samples) -> (K,) cost values.
        samples shape should be (K, D) where D == len(self._dims)
        base_trajectory: optional (D,) center to sample around
        """
        K = int(self.mppi_params.get('K', 400))
        lam = float(self.mppi_params.get('lambda', 1.0))
        noise_std = float(self.mppi_params.get('noise_std', 1.0))
        D = len(self._dims)
        if base_trajectory is None:
            base = np.zeros(D)
        else:
            base = np.asarray(base_trajectory).reshape(-1)
            assert base.shape[0] == D
        # 샘플 생성 (가우시안 노이즈)
        noises = np.random.randn(K, D) * noise_std
        samples = base[None, :] + noises
        costs = cost_fn(samples)  # (K,)
        # MPPI 가중치
        weights = np.exp(-costs / lam)
        if np.all(np.isinf(weights)) or np.all(weights == 0):
            weights = np.ones_like(weights)
        weights = weights / np.sum(weights)
        self._factor = SampleMessage(self._dims.copy(), samples, weights)
        return self._factor.copy()

    def calc_msg(self, edge: Edge) -> SampleMessage:
        """
        팩터가 특정 variable edge로 보낼 메시지:
        - joint factor samples를 대상 변수 차원으로 사영하고,
          동일 샘플 값이 있으면 가중치 합산(경향)
        """
        var_node = edge.get_other(self)
        var_dims = var_node.dims
        # project joint samples to var dims
        proj = self._factor.project(var_dims)
        # 합쳐진 동일 샘플은 그대로 두고 정규화
        proj.normalize()
        # 재샘플링(원하면)
        proj = proj.resample(min(500, proj.N), jitter=1e-4)
        return proj

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

    def loopy_propagate(self, steps: int = 1, factor_costs: Dict[SampleFNode, Tuple[callable, dict]] = None):
        """
        factor_costs: optional dictionary mapping SampleFNode -> (cost_fn, kwargs)
        cost_fn should accept (samples: (K, D)) and return (K,) costs.
        """
        vnodes = self.get_vnodes()
        fnodes = self.get_fnodes()

        for istep in range(steps):
            # Var -> Factor
            for v in vnodes:
                v.propagate()

            # Factor update: if cost fns provided, run MPPI to refresh factor joint samples
            if factor_costs is not None:
                for f in fnodes:
                    if f in factor_costs:
                        cost_fn, kwargs = factor_costs[f]
                        base = kwargs.get('base', None)
                        f.update_factor_with_mppi(cost_fn, base_trajectory=base)

            # Factor -> Var
            for f in fnodes:
                f.propagate()

            # Var beliefs
            beliefs = {}
            for v in vnodes:
                beliefs[v] = v.update_belief()
        return beliefs
