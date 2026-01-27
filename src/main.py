# main.py
import numpy as np
import matplotlib.pyplot as plt
from motion.obstacle import ObstacleMap
from motion.agent import SampleAgent

def run_simulation():
    # 1. 맵 설정
    omap = ObstacleMap()
    omap.add_circle(0.0, 0.0, 2.0) # 중앙에 큰 장애물
    
    # 2. 로봇 생성 (서로 마주보는 위치)
    # Robot A: (-5, -2) -> (5, 2)
    agent_a = SampleAgent("A", np.array([-15.0, -15.0]), np.array([15.0, 15.0]), omap)
    
    # Robot B: (5, -2) -> (-5, 2)
    agent_b = SampleAgent("B", np.array([15.0, -15.0]), np.array([-15.0, 15.0]), omap)

    history_a = []
    history_b = []

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 8))

    # 3. 시뮬레이션 루프
    for t in range(50):
        print(f"Time Step: {t}")
        
        # (A) 통신: 서로의 계획 공유
        for h in range(agent_a.horizon):
            mu_a, cov_a = agent_a.vars[h].get_belief_stats()
            mu_b, cov_b = agent_b.vars[h].get_belief_stats()
            
            agent_b.set_neighbor_belief(0, h, mu_a, cov_a)
            agent_a.set_neighbor_belief(0, h, mu_b, cov_b)
        
        # (B) 계획 수립 (Optimize)
        next_pos_a = agent_a.step(iterations=10)
        next_pos_b = agent_b.step(iterations=10)
        
        # (C) 이동 (Shift Trajectory simulates movement)
        # 실제로는 물리 엔진이 있어야 하지만, 계획된 첫 위치로 이동했다고 가정
        history_a.append(next_pos_a)
        history_b.append(next_pos_b)
        
        agent_a.shift_trajectory()
        agent_b.shift_trajectory()

        # (D) 시각화
        ax.clear()
        
        # 장애물 그리기
        circle = plt.Circle((0, 0), 2.0, color='k', alpha=0.5)
        ax.add_patch(circle)
        
        # 궤적 그리기 (Particles)
        for v in agent_a.vars:
            ax.scatter(v.particles[:,0], v.particles[:,1], s=1, c='r', alpha=0.1)
        for v in agent_b.vars:
            ax.scatter(v.particles[:,0], v.particles[:,1], s=1, c='b', alpha=0.1)
            
        # 평균 경로
        path_a = np.array([v.get_belief_stats()[0] for v in agent_a.vars])
        path_b = np.array([v.get_belief_stats()[0] for v in agent_b.vars])
        ax.plot(path_a[:,0], path_a[:,1], 'r-', linewidth=2, label='Agent A')
        ax.plot(path_b[:,0], path_b[:,1], 'b-', linewidth=2, label='Agent B')
        
        # 과거 경로
        if history_a:
            ha = np.array(history_a)
            hb = np.array(history_b)
            ax.plot(ha[:,0], ha[:,1], 'r--', alpha=0.5)
            ax.plot(hb[:,0], hb[:,1], 'b--', alpha=0.5)

        ax.set_xlim(-20, 20)
        ax.set_ylim(-20, 20)
        ax.legend()
        ax.set_aspect('equal')
        plt.pause(0.1)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    run_simulation()