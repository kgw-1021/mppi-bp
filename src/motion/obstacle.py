import numpy as np
from typing import Tuple, List, Dict

class ObstacleMap:
    def __init__(self) -> None:
        self.objects = {}
        # 객체가 추가될 때마다 JAX/GPU가 처리하기 쉬운 형태로 캐싱
        self._cache_obstacles()

    def set_circle(self, name: str, centerx, centery, radius):
        o = {'type': 'circle', 'name': name, 'centerx': centerx, 'centery': centery, 'radius': radius}
        self.objects[name] = o
        self._cache_obstacles() # 장애물이 변경될 때마다 캐시 업데이트

    def _cache_obstacles(self):
        """장애물 데이터를 NumPy 배열로 변환 (JAX/GPU 연산에 최적화)"""
        self._circles = []
        for o in self.objects.values():
            if o['type'] == 'circle':
                self._circles.append([o['centerx'], o['centery'], o['radius']])
        
        if self._circles:
            # (M, 3) 배열. M은 원형 장애물 수
            self._circle_params = np.array(self._circles)
        else:
            self._circle_params = np.empty((0, 3))

    def get_obstacle_cost(self, samples: np.ndarray, safe_dist: float) -> np.ndarray:
        """
        [VECTORIZED - NO GRADIENT] 장애물 cost function for MPPI.
        samples: (K, 4) array [x, y, vx, vy]
        Returns: (K,) cost values
        """
        if self._circle_params.shape[0] == 0:
            return np.zeros(samples.shape[0])

        # (K, 2) 샘플 위치
        samples_xy = samples[:, :2]
        
        # (M, 2) 장애물 중심
        centers = self._circle_params[:, :2]
        # (M,) 장애물 반지름
        radii = self._circle_params[:, 2]

        # NumPy 브로드캐스팅을 사용한 벡터화 연산
        # (K, 1, 2) - (1, M, 2) => (K, M, 2)
        diffs = samples_xy[:, np.newaxis, :] - centers[np.newaxis, :, :]
        
        # (K, M) - 각 샘플과 각 장애물 중심 간의 거리 (L2 norm)
        # np.linalg.norm(diffs, axis=2) 대신 수동 계산 (JAX/Numba에서 더 빠를 수 있음)
        mags = np.sqrt(np.sum(diffs**2, axis=2)) + 1e-8
        
        # (K, M) - 각 샘플과 각 장애물 표면 간의 SDF
        distances = mags - radii[np.newaxis, :]
        
        # (K,) - 각 샘플에 대해 *가장 가까운* 장애물과의 거리
        # (axis=1은 M개 장애물에 대한 축)
        min_distances = np.min(distances, axis=1)
        
        # 벡터화된 비용 계산

        costs = np.zeros_like(min_distances)
        
        # 1. Hard constraint: 충돌
        collision = min_distances < 0
        costs[collision] = 1e4

        # 2. Soft constraint: safe zone 내부 (강한 회피 신호)
        inside_safe = (min_distances >= 0) & (min_distances < safe_dist)
        # 기존 로직 유지
        costs[inside_safe] = ((safe_dist - min_distances[inside_safe]) / safe_dist)**2 * 100
        
        # 3. [추가] Long-tail constraint: safe zone 밖 (약한 회피 신호)
        # safe_dist보다 멀어도, 장애물에 가까운 것보다 먼 것이 '아주 조금' 더 좋다는 신호
        outside_safe = min_distances >= safe_dist
        # 거리가 멀수록 0에 수렴하는 작은 값 (예: 1/거리)
        # 100.0 은 weight 조절용, 1.0은 0나누기 방지
        costs[outside_safe] = 1.0 / (min_distances[outside_safe] + 1) 

        # print("Obstacle costs:", costs)
        
        return costs