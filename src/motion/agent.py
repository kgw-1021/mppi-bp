# agent.py
import numpy as np
from typing import Dict, List
from fg.graph import Graph
from fg.factor_graph_mppi import SampleVNode, SampleFNode
from .nodes import GoalSampleFNode, ObstacleSampleFNode, DistSampleFNode, KinematicsFNode

class SampleAgent:
    def __init__(self, agent_id: int, graph: Graph, start_pos: np.ndarray, goal_pos: np.ndarray, omap, horizon=20):
        self.id = agent_id
        self.name = f"Agent{agent_id}"
        self.horizon = horizon
        self.goal_pos_ = goal_pos
        self.dims = [4]  # x, y, vx, vy
        
        # 외부 그래프 객체 참조 (연결용)
        self.graph = graph 

        # ---------------------------------------------------------
        # [관리 리스트] 자신의 노드들만 따로 저장
        # ---------------------------------------------------------
        self.vars: List[SampleVNode] = []       # 내 변수 노드들
        self.static_factors: List[SampleFNode] = [] # 정적 팩터 (Goal, Obs, Dyn)
        
        # 동적 팩터 (Collision) 관리 딕셔너리
        # Key: Other Agent ID, Value: List of DistSampleFNode
        self.collision_factors: Dict[int, List[DistSampleFNode]] = {}

        # 4차원 목표 상태 (위치=goal, 속도=0)
        full_goal_state = np.zeros(4)
        full_goal_state[:2] = self.goal_pos_
        
        # --- 그래프 구축 ---
        for i in range(horizon):
            # 1. Variable Node
            v = SampleVNode(f"{self.name}_v{i}", self.dims, num_particles=100)
            v.particles[:, :2] += start_pos 
            
            if v not in self.graph.nodes: self.graph.nodes.append(v)
            self.vars.append(v)
            
            # 2. Static Factors (Goal, Obstacle)
            
            # (A) Obstacle Factor
            of = ObstacleSampleFNode(f"{self.name}_obs{i}", self.dims, omap, strength=20.0)
            self.graph.connect(v, of)
            self.static_factors.append(of) # 리스트에 등록
            
            # (B) Goal Factor
            str_val = 2.0 if i == horizon - 1 else 0.1
            gf = GoalSampleFNode(f"{self.name}_goal{i}", self.dims, full_goal_state, strength=str_val)
            self.graph.connect(v, gf)
            self.static_factors.append(gf) # 리스트에 등록
            
        # 3. Dynamics Factor (Kinematics)
        for i in range(horizon - 1):
            dyn = KinematicsFNode(f"{self.name}_dyn_{i}", self.dims, dt=0.1, strength=20.0)
            self.graph.connect(self.vars[i], dyn)
            self.graph.connect(self.vars[i+1], dyn)
            self.static_factors.append(dyn) # 리스트에 등록

    def step(self, iterations=5):
        """
        [Distributed Optimization Loop]
        오직 '나'와 관련된 팩터와 변수만 업데이트합니다.
        """
        # EKI Inner Loop
        for _ in range(iterations):
            
            # 1. Update Static Factors (Goal, Obs, Kinematics)
            for factor in self.static_factors:
                if hasattr(factor, 'update_factor'):
                    factor.update_factor()

            # 2. Update Dynamic Factors (Collision)
            # 현재 연결된 모든 이웃과의 충돌 팩터 업데이트
            for neighbor_factors in self.collision_factors.values():
                for factor in neighbor_factors:
                    if hasattr(factor, 'update_factor'):
                        # 이때 상대방(Target Agent)의 위치 정보를 조회함
                        factor.update_factor()

            # 3. Propagate Variables (My Trajectory)
            # 내 파티클들만 이동
            for v in self.vars:
                v.propagate(step_size=0.1)

        # MPC: 다음 스텝 반환 및 궤적 밀기
        current_pos, _ = self.vars[0].get_belief_stats()
        self.shift_trajectory()
        
        return current_pos[:2]

    # ---------------------------------------------------------
    # 2. 동적 팩터 관리 메서드 (Main Loop에서 호출)
    # ---------------------------------------------------------

    def attach_collision_factor(self, other_agent: 'SampleAgent'):
        """
        [Dynamic] 상대방이 통신 범위 내에 들어오면 호출.
        나의 변수 노드와 상대방 객체를 연결하는 DistSampleFNode를 생성.
        """
        # 이미 연결되어 있다면 패스
        if other_agent.id in self.collision_factors:
            return 

        created_factors = []
        
        for t in range(self.horizon):
            # 팩터 이름 생성
            f_name = f"Coll_{self.name}_vs_{other_agent.name}_t{t}"
            
            # 팩터 생성 (상대방 Agent 객체 자체를 타겟으로 지정)
            factor = DistSampleFNode(
                name=f_name, 
                dims=[2], 
                target_vars=other_agent.vars, 
                time_step=t,
                min_dist=0.8,
                strength=20.0
            )
            
            self.graph.connect(self.vars[t], factor)
            
            created_factors.append(factor)
            
        # 관리 리스트에 추가
        self.collision_factors[other_agent.id] = created_factors

    def detach_collision_factor(self, other_agent_id: int):
        """
        [Dynamic] 상대방이 멀어지면 호출.
        관련된 팩터 노드들을 그래프에서 제거.
        """
        if other_agent_id not in self.collision_factors:
            return

        factors_to_remove = self.collision_factors[other_agent_id]
        
        for factor in factors_to_remove:
            self.graph.remove_node(factor)
            
        del self.collision_factors[other_agent_id]

    def shift_trajectory(self):
        """ MPC Shift Logic """
        for i in range(self.horizon - 1):
            self.vars[i].particles = self.vars[i+1].particles.copy()
        
        last_node = self.vars[-1]
        noise = np.random.randn(*last_node.particles.shape) * 0.1
        last_node.particles += noise
        last_node.particles[:, 2:] *= 0.9 # 속도 감쇠

    def reached_goal(self, threshold=0.1):
        """ 목표 도달 여부 확인 """
        mean, _ = self.vars[-1].get_belief_stats()
        dist_to_goal = np.linalg.norm(self.goal_pos_ - mean[:2])
        return dist_to_goal < threshold

    def is_connected_to(self, other_agent_id: int) -> bool:
        """현재 특정 에이전트와 충돌 팩터가 연결되어 있는지 확인"""
        return other_agent_id in self.collision_factors