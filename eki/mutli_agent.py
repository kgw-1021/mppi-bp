import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

from fg.agent import Agent
from fg.factor import DynamicsFactor, ObstacleFactor, InterRobotFactor, AgentGoalFactor
from fg.graph import FactorGraph
from fg.solver import MultiAgentEKISolver

# --- [2. 설정 클래스] ---
class Config:
    horizon = 50
    num_particles = 300
    state_dim = 4
    dt = 0.1
    r_diag = 1.0     # 관측 노이즈 (작을수록 제약조건 준수 강화)
    damping = 0.1    # 0.1 ~ 0.3 추천 (작을수록 부드럽게 수렴)

# --- [3. 메인 실행 함수] ---
def main():
    # 1. 에이전트 객체 생성 (이제 여기서 모든 속성을 관리!)
    agents = [
        Agent(id=0, 
              start_pose=[0.0, 0.0, 0.0, 0.0], 
              goal_pose=[10.0, 10.0, 0.0, 0.0], 
              priority=0.0,  # VIP
              radius=0.5, 
              color='red'),
              
        Agent(id=1, 
              start_pose=[10.0, 0.0, 0.0, 0.0], 
              goal_pose=[0.0, 10.0, 0.0, 0.0], 
              priority=0.0,  # 양보함
              radius=0.5,    # 덩치가 큼
              color='blue')
    ]
    
    # 2. 그래프 설정 (범용 팩터들만 추가)
    graph = FactorGraph(dt=0.1)
    graph.add_factor(DynamicsFactor(weight=100.0))
    graph.add_factor(ObstacleFactor(center=[5.0, 5.0], radius=1.0, weight=15.0))
    
    # Agent들의 정보를 참조할 범용 팩터들
    graph.add_factor(AgentGoalFactor(weight=10.0))
    graph.add_factor(InterRobotFactor(safety_margin=0.5, weight=10.0))

    # 3. 솔버 실행
    config = Config()
    solver = MultiAgentEKISolver(graph, config)
    
    key = jax.random.PRNGKey(42)
    
    print("Optimization Starting...")
    start_time = time.time()
    final_belief = solver.solve(key, agents, iterations=100)
    
    print("Done! Time: {:.4f} s".format(time.time() - start_time))
    
    # 결과 시각화 시에도 agents 객체의 정보(color, radius) 활용 가능
    plot_multi_agent(final_belief, agents, config)

def plot_multi_agent(belief, agents, config):
    """
    belief: (M, T, N, D) - 최적화된 결과
    agents: List[Agent] - 에이전트 객체 리스트 (색상, 반지름 등 정보 포함)
    config: 설정 객체 (dt 등)
    """
    # 평균 궤적 계산
    mean_traj = jnp.mean(belief, axis=2) # (M, T, D)
    M, T, _ = mean_traj.shape
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # --- 1. 정적 요소 설정 ---
    ax.set_xlim(-2, 12)
    ax.set_ylim(-2, 12)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_title("Multi-Agent EKI Planning (Object Oriented)")
    
    # 장애물 (예시로 고정된 위치, 실제로는 Graph에서 가져오거나 인자로 받아야 함)
    ax.add_patch(plt.Circle((5.0, 5.0), 1.0, color='gray', alpha=0.4, label='Obstacle'))

    # --- 2. 에이전트별 플롯 객체 초기화 ---
    lines = []        # 지나온 경로 (선)
    body_circles = [] # 로봇 몸체 (원) - 반지름 반영!
    clouds = []       # 파티클 구름 (점)
    
    for i, agent in enumerate(agents):
        # Agent 객체에서 속성 추출
        c = agent.color
        r = agent.radius
        
        # 시작점 (Start)
        ax.plot(agent.start_pose[0], agent.start_pose[1], 'o', color=c, alpha=0.5)
        # 목표점 (Goal)
        ax.plot(agent.goal_pose[0], agent.goal_pose[1], 'x', color=c, markersize=10, markeredgewidth=2)
        
        # 라벨 생성 (우선순위 표시)
        prio_text = f"Prio:{agent.priority}" if agent.priority > 0 else "Normal"
        label_text = f"Agent {agent.id} ({prio_text})"
        
        # 1) 궤적 선 (Line)
        line, = ax.plot([], [], '--', color=c, linewidth=1.5, alpha=0.7)
        lines.append(line)
        
        # 2) 로봇 몸체 (Circle Patch) -> 실제 크기 반영
        # 초기 위치는 (0,0)에 두고 update에서 이동시킴
        body = plt.Circle((0, 0), r, color=c, alpha=0.8, label=label_text)
        ax.add_patch(body)
        body_circles.append(body)
        
        # 3) 파티클 구름 (Scatter)
        cloud = ax.scatter([], [], s=1, color=c, alpha=0.15)
        clouds.append(cloud)

    ax.legend(loc='upper right')

    # --- 3. 애니메이션 업데이트 함수 ---
    def update(frame):
        ax.set_xlabel(f"Time: {frame * config.dt:.1f}s / {config.horizon * config.dt:.1f}s")
        
        artists = [] # 업데이트된 객체들을 리턴해야 blit이 잘 됨
        
        for i in range(M):
            # A. 파티클 구름 업데이트
            current_particles = belief[i, frame, :, :2]
            clouds[i].set_offsets(current_particles)
            artists.append(clouds[i])
            
            # B. 로봇 몸체(원) 이동
            # Circle 객체는 set_data가 아니라 center 속성을 바꿈
            pos = mean_traj[i, frame]
            body_circles[i].center = (pos[0], pos[1])
            artists.append(body_circles[i])
            
            # C. 지나온 궤적 선 그리기
            path = mean_traj[i, :frame+1]
            lines[i].set_data(path[:, 0], path[:, 1])
            artists.append(lines[i])
            
        return artists # 리스트 합치기

    # 애니메이션 실행
    ani = FuncAnimation(fig, update, frames=T, interval=100, blit=True)
    plt.show()

if __name__ == "__main__":
    main()