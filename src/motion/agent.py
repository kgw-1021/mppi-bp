# agent.py
import numpy as np
from fg.graph import Graph
from fg.factor_graph_mppi import SampleVNode
from .nodes import GoalSampleFNode, ObstacleSampleFNode, DistSampleFNode, PriorFactor

class SampleAgent:
    def __init__(self, name: str, start_pos, goal_pos, omap, horizon=5):
        self.name = name
        self.horizon = horizon
        self.dims = [2] # x, y
        
        self.graph = Graph()
        self.vars: list[SampleVNode] = []
        self.dist_factors: list[DistSampleFNode] = []
        
        # 1. 궤적 변수 생성 (x0 ... xH)
        for i in range(horizon):
            v = SampleVNode(f"{name}_v{i}", self.dims, num_particles=100)
            
            # 초기화: 시작에서 목표까지 선형 보간 + 노이즈
            alpha = i / (horizon - 1)
            init_pos = start_pos * (1-alpha) + goal_pos * alpha
            v.particles += init_pos # Shift particles
            
            self.vars.append(v)
            
            # 2. 팩터 연결
            
            # (A) Obstacle Factor
            of = ObstacleSampleFNode(f"{name}_obs{i}", self.dims, omap, strength=2.0)
            self.graph.connect(v, of)
            
            # (B) Goal Factor (마지막 노드에 강하게, 중간은 약하게 유도)
            str_val = 5.0 if i == horizon - 1 else 0.1
            gf = GoalSampleFNode(f"{name}_goal{i}", self.dims, goal_pos, strength=str_val)
            self.graph.connect(v, gf)
            
            # (C) Distributed Factor (다른 로봇 회피용)
            df = DistSampleFNode(f"{name}_dist{i}", self.dims, min_dist=1.5, strength=5.0)
            self.graph.connect(v, df)
            self.dist_factors.append(df)
            
            # (D) Smoothness / Prior Factor (t와 t-1 연결)
            # 여기서는 간단히 코드상에서 처리하거나, explicit factor로 추가 가능
            # 루프 내에서 처리하는 방식 사용 (아래 step 함수 참조)

    def set_neighbor_belief(self, neighbor_idx, timestep, mean, cov):
        if timestep < self.horizon:
            self.dist_factors[timestep].set_remote_belief(mean, cov)

    def step(self, iterations=20):
        """ 한 번의 제어 주기 동안의 최적화 """
        
        # EKI Iterations (Internal Loop)
        for _ in range(iterations):
            # 1. Update Factors (MPPI Exploration)
            for node in self.graph.nodes:
                if hasattr(node, 'update_factor'):
                    node.update_factor()
            
            # 2. Update Variables (EKI Transport)
            for i, v in enumerate(self.vars):
                # Smoothness Constraint (수동 주입)
                # 이전 노드의 평균 위치를 'Prior' 메시지처럼 흉내내서 보낼 수 있음
                # 여기서는 생략하고 자체 Factor Graph 로직에 맡김
                v.propagate(step_size=0.8)

        # Return next intended position (First step of trajectory)
        action_mean, _ = self.vars[0].get_belief_stats()
        
        # Receding Horizon Implementation
        # 실제 이동 후, 변수들을 shift 해주는 로직이 필요하지만
        # 시뮬레이션 단순화를 위해 첫 번째 변수 위치를 반환
        return action_mean

    def shift_trajectory(self, current_pos):
        """ MPC 처럼 한 칸씩 당기기 """
        for i in range(self.horizon - 1):
            self.vars[i].particles = self.vars[i+1].particles.copy()
        
        # 마지막은 그대로 유지하거나 목표 주변으로 재샘플링
        self.vars[-1].particles = self.vars[-1].particles + np.random.randn(*self.vars[-1].particles.shape) * 0.1