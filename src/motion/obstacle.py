import numpy as np

class ObstacleMap:
    def __init__(self):
        self.circles = [] # (x, y, r)

    def add_circle(self, x, y, r):
        self.circles.append(np.array([x, y, r]))

    def get_obstacle_cost(self, samples: np.ndarray, safe_dist: float = 0.5, dt: float = 0.1) -> np.ndarray:
        """
        터널링 방지를 위한 Continuous Collision Detection (CCD)
        samples: (N, 4) [x, y, vx, vy]
        """
        if not self.circles:
            return np.zeros(samples.shape[0])

        # 1. 장애물 데이터 준비
        obs_array = np.array(self.circles)
        centers = obs_array[:, :2]  # (M, 2)
        radii = obs_array[:, 2]     # (M,)

        # 2. 선분의 시작점(P_start)과 끝점(P_end) 정의
        # P_end: 현재 샘플의 위치 (x, y)
        p_end = samples[:, :2] # (N, 2)
        
        # P_start: dt 시간 전의 위치 (x - vx*dt, y - vy*dt)
        # 만약 vx, vy가 없다면 단순 점 검사로 fallback 해야 함
        if samples.shape[1] >= 4:
            vel = samples[:, 2:4]
            p_start = p_end - vel * dt
        else:
            p_start = p_end # 속도 정보 없으면 점 검사

        # 3. 선분과 장애물 중심 간의 최소 거리 계산 (Vectorized)
        # 벡터 AB (선분): P_end - P_start
        vec_line = p_end - p_start # (N, 2)
        
        # 벡터 AC (시작점 -> 장애물): Center - P_start
        # Broadcasting: (N, 1, 2) - (1, M, 2) = (N, M, 2)
        vec_to_obs = centers[np.newaxis, :, :] - p_start[:, np.newaxis, :]
        
        # 선분 길이의 제곱
        len_sq = np.sum(vec_line**2, axis=1)[:, np.newaxis] # (N, 1)
        
        # 투영 (Projection) t 계산: dot(AC, AB) / dot(AB, AB)
        # vec_line은 (N, 2)이므로 (N, 1, 2)로 확장하여 연산
        dot_prod = np.sum(vec_to_obs * vec_line[:, np.newaxis, :], axis=2) # (N, M)
        
        # t값 클램핑 (0 ~ 1 사이로 제한하여 선분 내부만 검사)
        # 0보다 작으면 P_start랑 가깝고, 1보다 크면 P_end랑 가까움
        t = np.clip(dot_prod / (len_sq + 1e-8), 0.0, 1.0) # (N, M)
        
        # 선분 위의 가장 가까운 점 (Closest Point)
        # P_closest = P_start + t * vec_line
        closest_points = p_start[:, np.newaxis, :] + t[:, :, np.newaxis] * vec_line[:, np.newaxis, :]
        
        # 4. 최종 거리 계산 (Closest Point <-> Obstacle Center)
        diff = closest_points - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2) # (N, M)
        
        # 5. 비용 계산 (기존과 동일)
        sdf = dists - radii[np.newaxis, :] # Signed Distance Function
        
        total_cost = np.zeros(samples.shape[0])
        
        # 충돌 (선분이 장애물을 뚫고 지나감)
        collision_mask = sdf < 0.0
        total_cost += np.sum(collision_mask * 1000.0, axis=1)
        
        # 안전 거리 침범
        safe_mask = (sdf >= 0.0) & (sdf < safe_dist)
        soft_costs = np.exp(-2.0 * sdf)
        soft_costs[~safe_mask] = 0.0
        soft_costs[collision_mask] = 0.0
        
        total_cost += np.sum(soft_costs, axis=1) * 10.0
        
        return total_cost