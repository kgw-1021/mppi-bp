import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter

from fg.graph import Graph
from motion.obstacle import ObstacleMap
from motion.agent import SampleAgent

MAX_SIM_TIME = 1000
COMM_RANGE = 4.0 

def run_simulation():
    graph = Graph()
    omap = ObstacleMap()
    obs1 = [0.0, 0.0, 4.0]
    omap.add_circle(obs1) 

    # 에이전트 생성
    agents = [
        SampleAgent(0, graph, np.array([-10.0, -10.0]), np.array([10.0, 10.0]), omap),
        SampleAgent(1, graph, np.array([10.0, -10.0]), np.array([-10.0, 10.0]), omap),
        SampleAgent(2, graph, np.array([10.0, 10.0]), np.array([-10.0, -10.0]), omap),
        SampleAgent(3, graph, np.array([-10.0, 10.0]), np.array([10.0, -10.0]), omap),
        SampleAgent(4, graph, np.array([-10.0, 0.0]), np.array([10.0, 0.0]), omap),
        SampleAgent(5, graph, np.array([10.0, 0.0]), np.array([-10.0, 0.0]), omap),
        SampleAgent(6, graph, np.array([0.0, -10.0]), np.array([0.0, 10.0]), omap),
        SampleAgent(7, graph, np.array([0.0, 10.0]), np.array([0.0, -10.0]), omap)
    ]

    histories = [[] for _ in agents]
    terminated = [False for _ in agents]
    
    # 시각화 설정
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 10))
    writer = PillowWriter(fps=10)
    
    with writer.saving(fig, "simulation_result_center_obs.gif", 100):
        for t in range(MAX_SIM_TIME):
            # -----------------------------------------------------
            # (A) Dynamic Topology (환경/센서 역할)
            # -----------------------------------------------------
            # 거리 기반으로 누가 누구를 인식하는지(Factor 연결)만 관리
            num_agents = len(agents)
            for i in range(num_agents):
                for j in range(i + 1, num_agents):
                    agent_a = agents[i]
                    agent_b = agents[j]

                    pos_a, _ = agent_a.vars[0].get_belief_stats()
                    pos_b, _ = agent_b.vars[0].get_belief_stats()
                    dist = np.linalg.norm(pos_a[:2] - pos_b[:2])

                    if dist < COMM_RANGE:
                        agent_a.attach_collision_factor(agent_b)
                        agent_b.attach_collision_factor(agent_a)
                    else:
                        agent_a.detach_collision_factor(agent_b.id)
                        agent_b.detach_collision_factor(agent_a.id)

            # -----------------------------------------------------
            # (B) Distributed Planning (개별 두뇌 역할)
            # -----------------------------------------------------
            # 각 에이전트가 독립적으로 최적화 수행
            for i, agent in enumerate(agents):
                # step() 내부에서 자기만의 그래프(정적+동적)를 최적화
                next_pos = agent.step(iterations=5)
                histories[i].append(next_pos)

            # -----------------------------------------------------
            # (C) Visualization
            # -----------------------------------------------------
            ax.clear()

            circle = plt.Circle((obs1[0], obs1[1]), obs1[2], color='gray', alpha=0.5)
            ax.add_patch(circle)

            colors = ['r', 'b', 'g', 'm', 'c', 'y', 'orange', 'purple']
            for i, agent in enumerate(agents):
                c = colors[i]
                # 현재 위치
                cur_pos = histories[i][-1]
                ax.plot(cur_pos[0], cur_pos[1], marker='o', color=c, markersize=10, label=f'{agent.name}')
                
                # 미래 궤적
                path = np.array([v.get_belief_stats()[0][:2] for v in agent.vars])
                ax.plot(path[:, 0], path[:, 1], linestyle='-', color=c, alpha=0.6)

                # 과거 궤적
                past = np.array(histories[i])
                ax.plot(past[:, 0], past[:, 1], linestyle='--', color=c, alpha=0.4)
                
                # 충돌 팩터 연결선 (디버깅)
                for neighbor_id in agent.collision_factors:
                    if neighbor_id > agent.id:
                        n_pos = histories[neighbor_id][-1]
                        ax.plot([cur_pos[0], n_pos[0]], [cur_pos[1], n_pos[1]], 'k--', alpha=0.3)

            ax.set_xlim(-15, 15)
            ax.set_ylim(-15, 15)
            ax.legend()
            plt.pause(0.01)
            writer.grab_frame()
            
            for agent in agents:
                if agent.reached_goal() and not terminated[agent.id]:
                    print(f"{agent.name} reached its goal!")
                    terminated[agent.id] = True
            if all(terminated):
                print("All agents have reached their goals.")
                break

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    run_simulation()