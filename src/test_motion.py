import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import sys
import os

from fg.factor_graph_mppi import SampleMessage, SampleVNode, SampleFNode, SampleFactorGraph

from motion.obstacle import ObstacleMap
from motion.nodes import DynaSampleFNode, ObstacleSampleFNode, DistSampleFNode, RemoteSampleVNode
from motion.agent import SampleAgent, SampleEnv


def run_simulation_demo():
    """
    MPPI 팩터 그래프 기반 다중 에이전트 시뮬레이션 데모
    (스텝별 상세 로그 출력 기능 추가)
    """
    
    # 1. 시뮬레이션 환경 설정
    omap = ObstacleMap()
    omap.set_circle('obs1', 150, 150, 30)
    omap.set_circle('obs2', 300, 250, 40)
    omap.set_circle('obs3', 400, 100, 20)

    env = SampleEnv()

    # 2. 에이전트 설정
    agent_radius = 12
    global_plan_steps = 8  # 경로 계획 호라이즌 (전역 변수로 저장)
    
    state_a = np.array([50, 50, 0, 0])[:, None]
    target_a = np.array([450, 350, 0, 0])[:, None]
    
    state_b = np.array([450, 350, 0, 0])[:, None]
    target_b = np.array([50, 50, 0, 0])[:, None]

    common_weights = {
        'obstacle_weight': 1.0,
        'distance_weight':1.0,
        'target_position_weight': 10.0,
        'dynamic_position_weight': 10.0,
        'target_velocity_weight': 1.0,
        'dynamic_velocity_weight': 1.0,
    }

    agent_a = SampleAgent('AgentA', state_a, target_a, 
                          steps=global_plan_steps, radius=agent_radius, omap=omap,
                          **common_weights)
    
    agent_b = SampleAgent('AgentB', state_b, target_b, 
                          steps=global_plan_steps, radius=agent_radius, omap=omap,
                          **common_weights)

    env.add_agent(agent_a)
    env.add_agent(agent_b)
    
    agents = [agent_a, agent_b]
    colors = {
        agent_a.name: '#FF5733', 
        agent_b.name: '#3357FF',
    }
    
    # 3. Matplotlib 시각화 설정
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_aspect('equal')
    ax.set_xlim(-100, 600)
    ax.set_ylim(-100, 500)
    ax.set_title("Decentralized Agent Planning (MPPI Factor Graph)")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    fig.patch.set_facecolor('white')
    ax.set_facecolor('#F8F8F8')

    # --- 정적 요소 그리기 (장애물, 목표지점) ---
    static_artists = []
    
    for o in omap.objects.values():
        if o['type'] == 'circle':
            circle = patches.Circle((o['centerx'], o['centery']), o['radius'],
                                    facecolor='black', alpha=0.6, zorder=5)
            ax.add_patch(circle)
            static_artists.append(circle)

    for agent in agents:
        target = agent.get_target()
        star = ax.plot(target[0], target[1], '*', 
                       markersize=20, 
                       markerfacecolor=colors[agent.name], 
                       markeredgecolor='white', 
                       markeredgewidth=1.5,
                       zorder=10)
        static_artists.extend(star)

    # --- 동적 요소 (Artist) 초기화 ---
    artist_map = {}
    for agent in agents:
        name = agent.name
        color = colors[name]
        
        body = patches.Circle((agent.x, agent.y), agent.r, 
                              facecolor=color, alpha=0.9, zorder=15)
        ax.add_patch(body)
        
        safe_zone = patches.Circle((agent.x, agent.y), agent.r + agent.r,
                                   edgecolor=color, facecolor='none', 
                                   linestyle='--', linewidth=1.5, alpha=0.7, zorder=14)
        ax.add_patch(safe_zone)
        
        plan_line, = ax.plot([], [], 'o-', color=color, markersize=3, 
                             linewidth=2.0, alpha=0.7, zorder=20, label=f'Plan {name}')
                                
        history_line, = ax.plot([], [], '.-', color=color, 
                                linewidth=1.0, alpha=0.3, zorder=10)
        
        artist_map[name] = {
            'body': body,
            'safe_zone': safe_zone,
            'plan_line': plan_line,
            'history_line': history_line,
            'history_data': ([agent.x], [agent.y])
        }
    
    ax.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)

    # 4. 애니메이션 함수 정의

    def init_animation():
        all_artists = list(static_artists)
        for artists in artist_map.values():
            all_artists.extend(artists.values())
        all_artists = [a for a in all_artists if not isinstance(a, tuple)] 
        return all_artists

    # (!!!) 수정된 부분 (START) (!!!)
    def update_animation(frame):
        """애니메이션 매 프레임 업데이트 (로그 출력 포함)"""
        
        # 1. 시뮬레이션 스텝 실행
        # (step_plan: 경로 계산, step_move: 계산된 경로 1스텝 이동)
        env.step_plan(iters=20)  
        env.step_move()

        all_arrived = True
        for agent in env._agents:
            pos = agent._state[:2].flatten()
            goal = agent.get_target()[:2]
            if np.linalg.norm(pos - goal) > 5.0:
                all_arrived = False
                break

        if all_arrived:
            print("\nAll agents reached the destination. Simulation stopped.\n")
            ani.event_source.stop()  # 애니메이션 종료
            return []  # 빈 아티스트 반환

        # --- [요청사항] 스텝별 정보 출력 ---
        print("-" * 70)
        print(f" S T E P : {frame} ")
        print("-" * 70)

        updated_artists = []

        # 2. 각 에이전트의 시각적 요소 및 로그 업데이트
        for agent in env._agents:
            name = agent.name
            artists = artist_map[name]
            
            # (A) 현재 상태 (step_move가 방금 완료된 상태)
            # agent._state는 (4, 1) 배열이므로 보기 좋게 1D로 변환
            current_state_array = agent._state.flatten() 
            
            # (B) 계획된 경로 (step_plan이 방금 계산한 미래 경로)
            # get_state()는 [v0, v1, ..., vN-1]의 평균 상태 리스트를 반환
            # v0는 현재 상태(current_state_array)와 거의 동일
            # v1는 다음 스텝에 이동할 목표
            planned_states = agent.get_state() 
            
            # --- [요청사항] 에이전트 정보 출력 ---
            print(f" Agent: {name}")
            print(f"   Current State (t={frame}): "
                  f"x={current_state_array[0]:.2f}, y={current_state_array[1]:.2f}, "
                  f"vx={current_state_array[2]:.2f}, vy={current_state_array[3]:.2f}")

            print(f"   Planned Trajectory (t={frame} to t={frame + global_plan_steps - 1}):")
            
            # planned_states[0] (v0)는 현재 상태의 믿음(belief)
            if planned_states and planned_states[0] is not None:
                start_plan = planned_states[0]
                print(f"     -> Plan Start (v0):   x={start_plan[0]:.2f}, y={start_plan[1]:.2f}")
            
            # planned_states[1] (v1)는 다음 스텝의 목표
            if planned_states and len(planned_states) > 1 and planned_states[1] is not None:
                next_step_state = planned_states[1]
                print(f"     -> Next Step (v1):    x={next_step_state[0]:.2f}, y={next_step_state[1]:.2f}")
            
            # planned_states[-1] (vN-1)는 계획의 마지막 지점
            if planned_states and planned_states[-1] is not None:
                final_plan_state = planned_states[-1]
                print(f"     -> Horizon End (v{len(planned_states)-1}): x={final_plan_state[0]:.2f}, y={final_plan_state[1]:.2f}")
            print("") # 에이전트 간 공백

            # (C) 시각적 요소 업데이트 (기존 코드)
            x, y = current_state_array[0], current_state_array[1]

            artists['history_data'][0].append(x)
            artists['history_data'][1].append(y)
            artists['history_line'].set_data(artists['history_data'])
            
            artists['body'].set_center((x, y))
            artists['safe_zone'].set_center((x, y))
            
            if planned_states:
                plan_x = [s[0] for s in planned_states if s is not None]
                plan_y = [s[1] for s in planned_states if s is not None]
                artists['plan_line'].set_data(plan_x, plan_y)
            
            updated_artists.extend([
                artists['history_line'], artists['body'], 
                artists['safe_zone'], artists['plan_line']
            ])

        return updated_artists
    # (!!!) 수정된 부분 (END) (!!!)


    # 5. 애니메이션 실행
    ani = animation.FuncAnimation(
        fig, 
        update_animation, 
        frames=300, 
        init_func=init_animation,
        interval=50, 
        blit=False
    )
    
    plt.show()

# --- 메인 실행 ---
if __name__ == '__main__':
    run_simulation_demo()