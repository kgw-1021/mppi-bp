
import sys
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches

# --- 사용자 정의 모듈 임포트 ---
# (motion/ 디렉토리에 obstacle.py, agent.py 등이 있다고 가정합니다.)
from motion.obstacle import ObstacleMap
from motion.agent import SampleAgent, SampleEnv
# ------------------------------


# --- 1. 환경 및 에이전트 설정 (기존 코드와 동일) ---
omap = ObstacleMap()
omap.set_circle('obs1', 145, 85, 20) # 원본 코드의 장애물 위치

# 원본 코드의 에이전트 설정
agent0 = SampleAgent('a0', [0, 0, 0, 0], [350, 100, 0, 0], steps=6, radius=15, omap=omap)
agent1 = SampleAgent('a1', [350, 100, 0, 0], [0, 50, 0, 0], steps=6, radius=15, omap=omap)

env = SampleEnv()
env.add_agent(agent0)
env.add_agent(agent1)

colors = {
    agent0: 'g',  # Matplotlib 색상 (green)
    agent1: 'b',  # (blue)
}

# --- 2. Matplotlib 시각화 설정 ---
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_aspect('equal')
# Pygame 윈도우 오프셋(500, 400)과 좌표계를 고려하여 범위 설정
ax.set_xlim(-100, 450)
ax.set_ylim(-100, 200)
ax.set_title("MPPI-PBP Multi-Agent Simulation")
ax.grid(True)

# 정적 장애물 그리기 (한 번만)
for o in omap.objects.values():
    if o['type'] == 'circle':
        ax.add_patch(patches.Circle((o['centerx'], o['centery']), o['radius'],
                                    edgecolor='r', facecolor='none', linewidth=2, label='Obstacle'))

# 애니메이션에서 업데이트될 'Artist' 객체 생성
agent_artists = {}
for agent, color in colors.items():
    # 1. 현재 위치 (마커)
    pos_marker, = ax.plot([], [], marker='o', color=color, markersize=10, label=f'Agent {agent.name}')
    # 2. 안전 반경 (채워진 원)
    radius_patch = patches.Circle((0, 0), agent.r,
                                  edgecolor=color, facecolor=color, alpha=0.2)
    ax.add_patch(radius_patch)
    # 3. 계획된 궤적 (점선)
    path_line, = ax.plot([], [], linestyle='--', color=color, linewidth=1.5, label=f'Plan {agent.name}')
    
    agent_artists[agent] = {
        'pos': pos_marker,
        'radius': radius_patch,
        'path': path_line
    }
ax.legend()

# --- 3. 애니메이션 함수 정의 ---

def init():
    """애니메이션 초기화 함수"""
    artists_list = []
    for agent in env._agents:
        artists_list.append(agent_artists[agent]['pos'])
        artists_list.append(agent_artists[agent]['radius'])
        artists_list.append(agent_artists[agent]['path'])
    return artists_list

def update(frame):
    """애니메이션 매 프레임 업데이트 함수"""
    
    # 1. 계획 (BP 실행)
    env.step_plan() # <- 성능 병목 지점
    
    updated_artists = []
    for agent in env._agents:
        # 계획된 궤적(belief) 가져오기
        state_trajectory = agent.get_state()
        if not state_trajectory or state_trajectory[0] is None:
            continue
        
        # 현재 상태 (t=0) 및 궤적 (t=0...T)
        current_state = state_trajectory[0]
        xs = [s[0] for s in state_trajectory if s is not None]
        ys = [s[1] for s in state_trajectory if s is not None]

        # 아티스트 업데이트
        artists_dict = agent_artists[agent]
        
        # 1. 현재 위치
        artists_dict['pos'].set_data([current_state[0]], [current_state[1]])
        updated_artists.append(artists_dict['pos'])
        
        # 2. 안전 반경
        artists_dict['radius'].set_center((current_state[0], current_state[1]))
        updated_artists.append(artists_dict['radius'])
        
        # 3. 계획된 궤적
        artists_dict['path'].set_data(xs, ys)
        updated_artists.append(artists_dict['path'])

    # 2. 이동 (시뮬레이션 1스텝 진행)
    env.step_move()
    
    return updated_artists

# --- 4. 애니메이션 실행 ---
if __name__ == '__main__':
    # frames: 총 시뮬레이션 스텝 수
    # interval: 각 스텝 사이의 시간 (ms) - env.step_plan() 시간에 따라 조절
    # blit=True: (효율적인) 부분 렌더링 사용
    ani = animation.FuncAnimation(fig, update, init_func=init,
                                  frames=200, interval=50, blit=True)
    
    plt.show()