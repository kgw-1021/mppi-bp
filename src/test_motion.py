import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
import sys
import os
from datetime import datetime
import imageio   # ← GIF 저장을 위한 라이브러리

from fg.factor_graph_mppi import SampleMessage, SampleVNode, SampleFNode, SampleFactorGraph

from motion.obstacle import ObstacleMap
from motion.nodes import DynaSampleFNode, ObstacleSampleFNode, DistSampleFNode, RemoteSampleVNode
from motion.agent import SampleAgent, SampleEnv


def run_simulation_demo(save_animation=True, output_dir='./animations'):
    
    # 출력 디렉토리 생성
    if save_animation and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. 시뮬레이션 환경 설정
    omap = ObstacleMap()
    omap.set_circle('obs1', 150, 150, 30)
    omap.set_circle('obs2', 300, 250, 40)
    omap.set_circle('obs3', 400, 100, 20)

    env = SampleEnv()

    # 2. 에이전트 설정
    agent_radius = 12
    global_plan_steps = 8
    
    state_a = np.array([50, 50, 0, 0])[:, None]
    target_a = np.array([450, 350, 0, 0])[:, None]

    common_weights = {
        'obstacle_weight': 0.1,       
        'distance_weight': 0.5,       
        'target_position_weight': 0.01,
        'dynamic_position_weight': 0.01,
        'target_velocity_weight': 0.01,
        'dynamic_velocity_weight': 0.01,
        'spd_limit_weight': 0.001
    }

    agent_a = SampleAgent('AgentA', state_a, target_a, 
                          steps=global_plan_steps, radius=agent_radius, omap=omap,
                          **common_weights)

    env.add_agent(agent_a)
    agents = [agent_a]

    colors = {agent_a.name: '#FF5733'}

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

    # 정적 요소 그리기
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

    # 동적 요소 초기화
    artist_map = {}
    for agent in agents:
        name = agent.name
        color = colors[name]
        
        body = patches.Circle((agent.x, agent.y), agent.r, 
                              facecolor=color, alpha=0.9, zorder=15)
        ax.add_patch(body)
        
        safe_zone = patches.Circle((agent.x, agent.y), agent.r * 2,
                                   edgecolor=color, facecolor='none', 
                                   linestyle='--', linewidth=1.5, alpha=0.7, zorder=14)
        ax.add_patch(safe_zone)
        
        plan_line, = ax.plot([], [], 'o-', color=color, markersize=3, 
                             linewidth=2.0, alpha=0.7, zorder=20)
        
        history_line, = ax.plot([], [], '.-', color=color, 
                                linewidth=1.0, alpha=0.3, zorder=10)
        
        artist_map[name] = {
            'body': body,
            'safe_zone': safe_zone,
            'plan_line': plan_line,
            'history_line': history_line,
            'history_data': ([agent.x], [agent.y])
        }
    
    plt.grid(True, linestyle=':', alpha=0.6)

    # GIF 프레임 버퍼
    frames_buffer = []

    # 4. 애니메이션 함수
    def init_animation():
        all_artists = list(static_artists)
        for artists in artist_map.values():
            all_artists.extend([artists['body'], artists['safe_zone'], 
                              artists['plan_line'], artists['history_line']])
        return all_artists

    def update_animation(frame):

        try:
            env.step_plan(iters=1)
            env.step_move()

            updated_artists = []

            all_arrived = True

            for agent in env._agents:
                name = agent.name
                artists = artist_map[name]

                current_state = agent._state.flatten()
                planned_states = agent.get_state()

                x, y = current_state[0], current_state[1]
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

                # 도착 여부 판단
                pos = agent._state[:2].flatten()
                goal = agent.get_target()[:2].flatten()
                if np.linalg.norm(pos - goal) > 10.0:
                    all_arrived = False

            # 프레임 저장 (RGB 버퍼)
            buf = fig.canvas.buffer_rgba()
            frame_img = np.asarray(buf)
            frames_buffer.append(frame_img.copy())

            if all_arrived:
                print("\nAll agents reached destination!\n")
                ani.event_source.stop()

            return updated_artists

        except Exception as e:
            print(f"ERROR at frame {frame}: {e}")
            ani.event_source.stop()
            return []

    # 5. 애니메이션 실행
    ani = animation.FuncAnimation(
        fig, 
        update_animation, 
        frames=300, 
        init_func=init_animation,
        interval=100,
        blit=False
    )
    
    # 실시간 시각화
    plt.show()

    # 6. show() 종료 후 GIF 저장
    if save_animation:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        gif_filename = os.path.join(output_dir, f'demo_{timestamp}.gif')

        print(f"\n Saving recorded frames as GIF: {gif_filename}")

        try:
            imageio.mimsave(gif_filename, frames_buffer, fps=5)
            print("GIF saved successfully!")
        except Exception as e:
            print(f"GIF save failed: {e}")


if __name__ == '__main__':
    run_simulation_demo(save_animation=True, output_dir='./animations')
