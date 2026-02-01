import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter

from fg.graph import Graph
from motion.obstacle import ObstacleMap
from motion.agent import SampleAgent

MAX_SIM_TIME = 1000
COMM_RANGE = 10.0 

def run_simulation():
    graph = Graph()
    omap = ObstacleMap()
    obs1 = [-2, 0.0, 2.0]
    obs2 = [4.0, 4.0, 1.0]
    obs3 = [-5.0, -5.0, 1.5]
    obs4 = [-6.0, 3.0, 1.0]
    obs5 = [5.0, 9.0, 1.2]
    obs6 = [7.0, -4.0, 1.5]
    obs7 = [1.0, -7.0, 2.0]

    omap.add_circle(obs1) 
    omap.add_circle(obs2)
    omap.add_circle(obs3)
    omap.add_circle(obs4)
    omap.add_circle(obs5)
    omap.add_circle(obs6)
    omap.add_circle(obs7)

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

    with writer.saving(fig, "simulation_result_multi_obs3.gif", 100):
        for t in range(MAX_SIM_TIME):
            # -----------------------------------------------------
            # (A) Dynamic Topology (환경/센서 역할)
            # -----------------------------------------------------
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
            for i, agent in enumerate(agents):
                if not terminated[i]: # 이미 도착한 에이전트는 계산 생략 가능 (선택사항)
                    next_pos = agent.step(iterations=5)
                    histories[i].append(next_pos)
                else:
                    # 멈춰 있어도 위치는 기록해야 그래프가 안 깨짐
                    histories[i].append(histories[i][-1])

            # -----------------------------------------------------
            # (C) Visualization
            # -----------------------------------------------------
            ax.clear()

            # 장애물 그리기
            circle1 = plt.Circle((obs1[0], obs1[1]), obs1[2], color='gray', alpha=0.5)
            circle2 = plt.Circle((obs2[0], obs2[1]), obs2[2], color='gray', alpha=0.5)
            circle3 = plt.Circle((obs3[0], obs3[1]), obs3[2], color='gray', alpha=0.5)
            circle4 = plt.Circle((obs4[0], obs4[1]), obs4[2], color='gray', alpha=0.5)
            circle5 = plt.Circle((obs5[0], obs5[1]), obs5[2], color='gray', alpha=0.5)
            circle6 = plt.Circle((obs6[0], obs6[1]), obs6[2], color='gray', alpha=0.5)
            circle7 = plt.Circle((obs7[0], obs7[1]), obs7[2], color='gray', alpha=0.5)

            ax.add_patch(circle1)
            ax.add_patch(circle2)
            ax.add_patch(circle3)
            ax.add_patch(circle4)
            ax.add_patch(circle5)
            ax.add_patch(circle6)
            ax.add_patch(circle7)

            colors = ['r', 'b', 'g', 'm', 'c', 'y', 'orange', 'purple']
            for i, agent in enumerate(agents):
                c = colors[i]
                if not histories[i]: continue

                cur_pos = histories[i][-1]
                ax.plot(cur_pos[0], cur_pos[1], marker='o', color=c, markersize=10, label=f'{agent.name}')
                
                path = np.array([v.get_belief_stats()[0][:2] for v in agent.vars])
                ax.plot(path[:, 0], path[:, 1], linestyle='-', color=c, alpha=0.6)

                past = np.array(histories[i])
                ax.plot(past[:, 0], past[:, 1], linestyle='--', linewidth=2, color=c, alpha=1.0)
                
                for neighbor_id in agent.collision_factors:
                    if neighbor_id > agent.id:
                        n_pos = histories[neighbor_id][-1]
                        ax.plot([cur_pos[0], n_pos[0]], [cur_pos[1], n_pos[1]], 'k--', alpha=0.3)

            ax.set_xlim(-15, 15)
            ax.set_ylim(-15, 15)
            ax.legend(loc='upper right')
            
            # [추가] 현재 프레임을 GIF에 저장
            writer.grab_frame()
            
            plt.pause(0.01)
            
            # 종료 조건 체크
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