# agent.py
import numpy as np
from fg.graph import Graph
from fg.factor_graph_mppi import SampleVNode
from .nodes import GoalSampleFNode, ObstacleSampleFNode, DistSampleFNode, KinematicsFNode

class SampleAgent:
    def __init__(self, name: str, start_pos, goal_pos, omap, horizon=10):
        self.name = name
        self.horizon = horizon
        self.goal_pos_ = goal_pos
        self.dims = [4]  # x, y, vx, vy
        
        self.graph = Graph()
        self.vars: list[SampleVNode] = []
        self.dist_factors: list[DistSampleFNode] = []

        
        
        # 4차원 목표 상태 생성 (위치는 goal_pos, 속도는 0)
        # goal_pos가 (2,)라고 가정
        full_goal_state = np.zeros(4)
        full_goal_state[:2] = self.goal_pos_
        
        # 1. 궤적 변수 생성 (x0 ... xH)
        for i in range(horizon):
            v = SampleVNode(f"{name}_v{i}", self.dims, num_particles=100)
            v.particles[:, :2] += start_pos  # 초기 위치 설정 (속도는 0으로 시작)
            
            self.vars.append(v)
            
            # 2. 팩터 연결
            
            # (A) Obstacle Factor (장애물)
            # Obstacle node 내부에서 samples[:, :2]만 쓰므로 그대로 둬도 됨
            of = ObstacleSampleFNode(f"{name}_obs{i}", self.dims, omap, strength=20.0)
            self.graph.connect(v, of)
            
            # (B) Goal Factor (목표)
            # 마지막 노드에 강하게, 나머지는 약하게
            str_val = 1.0 if i == horizon - 1 else 0.1
            
            # [수정 3] GoalFactor에 4차원 목표 전달
            gf = GoalSampleFNode(f"{name}_goal{i}", self.dims, full_goal_state, strength=str_val)
            self.graph.connect(v, gf)
            
            # (C) Distributed Factor (충돌 방지)
            df = DistSampleFNode(f"{name}_dist{i}", self.dims, min_dist=1.5, strength=5.0)
            self.graph.connect(v, df)
            self.dist_factors.append(df)
            
        # [수정 4] Dynamics Factor (Kinematics) 추가
        # x_t 와 x_{t+1} 을 연결
        for i in range(horizon - 1):
            curr_node = self.vars[i]
            next_node = self.vars[i+1]
            
            # Dynamics Factor 생성 (dt=0.1)
            dyn = KinematicsFNode(f"{name}_dyn_{i}", self.dims, dt=0.1, strength=5.0)
            
            # 순서대로 연결 (Graph Connect 순서가 중요할 수 있음)
            # KinematicsFNode 내부에서 edges[0]을 prev, edges[1]을 next로 가정했거나 이름순 정렬함
            self.graph.connect(curr_node, dyn)
            self.graph.connect(next_node, dyn)

    def set_neighbor_belief(self, neighbor_idx, timestep, mean, cov):
        if timestep < self.horizon:
            # DistFactor는 위치(2D)만 볼 수도 있고 4D를 볼 수도 있음.
            # DistSampleFNode 구현에 따라 다르지만, 보통 위치만 필요함.
            # 받은 mean이 4차원이면 그대로 넣어도 DistNode 내부에서 [:2]만 쓰면 됨.
            self.dist_factors[timestep].set_remote_belief(mean, cov)

    def step(self, iterations=5):
        """ 한 번의 제어 주기 동안의 최적화 """
        
        # EKI Iterations
        for _ in range(iterations):
            # 1. Update Factors (MPPI Exploration)
            for node in self.graph.nodes:
                if hasattr(node, 'update_factor'):
                    node.update_factor()
            
            # 2. Update Variables (EKI Transport)
            for v in self.vars:
                v.propagate(step_size=0.1)

        # Return next intended position (First step of trajectory)
        action_mean, _ = self.vars[0].get_belief_stats()
        
        # 다음 스텝을 위해 궤적 Shift (MPC)
        self.shift_trajectory()
        
        return action_mean[:2] # 위치만 반환

    def shift_trajectory(self):
        """ MPC 처럼 한 칸씩 당기기 """
        for i in range(self.horizon - 1):
            self.vars[i].particles = self.vars[i+1].particles.copy()
        
        # 마지막 노드는 이전 노드(이제 마지막이 된)에서 운동학적 전파 혹은 랜덤
        # 간단히 마지막 상태 유지 + 노이즈
        last_node = self.vars[-1]
        noise = np.random.randn(*last_node.particles.shape) * 0.1
        last_node.particles += noise
        # 속도 감쇠 (안정성을 위해)
        last_node.particles[:, 2:] *= 0.9

    def reached_goal(self, threshold=0.1):
        """ 목표 도달 여부 확인 """
        mean, _ = self.vars[-1].get_belief_stats()
        dist_to_goal = np.linalg.norm(self.goal_pos_ - mean[:2])
        return dist_to_goal < threshold