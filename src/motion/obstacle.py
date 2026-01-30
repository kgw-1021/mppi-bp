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
        p_end = samples[:, :2] # (N, 2)
        
        # 속도 정보가 있으면 선분으로, 없으면 점으로 검사
        if samples.shape[1] >= 4:
            vel = samples[:, 2:4]
            p_start = p_end - vel * dt
        else:
            p_start = p_end 

        # 3. 선분과 장애물 중심 간의 최소 거리 계산 (Vectorized)
        # 벡터 AB (선분): P_end - P_start
        vec_line = p_end - p_start # (N, 2)
        
        # 벡터 AC (시작점 -> 장애물): Center - P_start
        # Broadcasting: (N, 1, 2) - (1, M, 2) = (N, M, 2)
        vec_to_obs = centers[np.newaxis, :, :] - p_start[:, np.newaxis, :]
        
        # 선분 길이의 제곱 (N, 1)
        len_sq = np.sum(vec_line**2, axis=1)[:, np.newaxis]
        
        # 투영 (Projection) t 계산: dot(AC, AB) / dot(AB, AB)
        # 1e-8은 0으로 나누기 방지 (로봇이 정지해 있을 때)
        dot_prod = np.sum(vec_to_obs * vec_line[:, np.newaxis, :], axis=2) # (N, M)
        t = np.clip(dot_prod / (len_sq + 1e-8), 0.0, 1.0) # (N, M)
        
        # 선분 위의 가장 가까운 점 (Closest Point)
        # P_closest = P_start + t * vec_line
        closest_points = p_start[:, np.newaxis, :] + t[:, :, np.newaxis] * vec_line[:, np.newaxis, :]
        
        # 4. 최종 거리 계산 (Closest Point <-> Obstacle Center)
        diff = closest_points - centers[np.newaxis, :, :]
        dists = np.linalg.norm(diff, axis=2) # (N, M)
        
        # 5. 비용 계산 (수정됨)
        sdf = dists - radii[np.newaxis, :] # (N, M)
        
        total_cost = np.zeros(samples.shape[0])
        
        # [수정 1] 충돌 (Hard Constraint + Gradient)
        # 충돌 시: 기본 벌점 1000 + 침투 깊이에 비례한 벌점
        # 깊게 박힐수록 비용이 커야 밖으로 밀어냄 (Penetration Penalty)
        collision_mask = sdf < 0.0
        # sdf가 음수이므로 -sdf는 양수(침투 깊이)
        penetration_cost = 1000.0 + (-sdf * 5000.0) 
        
        # 장애물 별로 비용 합산 (충돌한 장애물들에 대해)
        total_cost += np.sum(np.where(collision_mask, penetration_cost, 0.0), axis=1)
        
        # [수정 2] 안전 거리 (Soft Constraint)
        # 충돌하지 않았지만(sdf >= 0) 안전거리 내에 있는 경우
        safe_mask = (sdf >= 0.0) & (sdf < safe_dist)
        soft_costs = np.exp(-3.0 * sdf) # 지수함수로 가까울수록 급격히 증가
        
        # 충돌하지 않은 것들만 soft cost 적용
        total_cost += np.sum(np.where(safe_mask, soft_costs, 0.0), axis=1) * 100.0
        
        # print 제거 (속도 향상)
        
        return total_cost