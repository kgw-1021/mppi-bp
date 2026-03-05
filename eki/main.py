import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time

from fg.agent import Agent
from fg.factor import ObstacleFactor, DynamicsFactor, SpeedLimitFactor, AgentGoalFactor
from fg.graph import FactorGraph
from fg.solver import MultiAgentEKISolver

class Config:
    horizon = 50
    num_particles = 500
    state_dim = 4
    dt = 0.1
    r_diag = 0.5
    damping = 0.5

def main():
    # 1. 에이전트 객체 생성 (단독)
    # 시작점, 목표점, 반지름, 색상 등을 여기서 정의
    single_agent = Agent(
        id=0,
        start_pose=[0.0, 0.0, 0.0, 0.0],
        goal_pose=[10.0, 10.0, 0.0, 0.0],
        radius=0.5,
        color='blue',
        priority=0.0
    )

    # 2. 그래프 및 팩터 설정
    graph = FactorGraph(dt=0.1)
    
    # 물리 법칙
    graph.add_factor(DynamicsFactor(pos_weight=30.0, vel_weight=10.0))
    
    # 장애물 (환경 요소)
    graph.add_factor(ObstacleFactor(center=[5.0, 5.0], radius=2.0, weight=15.0))
    graph.add_factor(ObstacleFactor(center=[10.0, 8.0], radius=1.5, weight=15.0))
    
    # 목표 도달
    graph.add_factor(AgentGoalFactor(weight=5.0))
    
    # 속도 제한
    graph.add_factor(SpeedLimitFactor(max_speed=5.0, weight=1.0))

    config = Config()
    
    # [변경 3] 솔버 교체 (MultiAgentEKISolver 재사용)
    solver = MultiAgentEKISolver(graph, config)
    
    # 3. 팩터 그래프 최적화 (Solving)
    key = jax.random.PRNGKey(42)
    print("Optimization Start...")
    
    start_time = time.time()
    
    # [변경 4] solve 함수에 '리스트' 형태로 전달
    # 결과 shape: (M, T, N, D) -> 여기선 M=1
    final_belief_all = solver.solve(key, [single_agent], iterations=100)
    
    end_time = time.time()
    print(f"Optimization Time: {end_time - start_time:.4f} s")
    
    # 단일 에이전트이므로 0번째 인덱스 추출
    # final_belief shape: (T, N, D)
    final_belief = final_belief_all[0] 

    # 4. 애니메이션 설정
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 정적 요소 (장애물)
    ax.add_patch(plt.Circle((5, 5), 2.0, color='gray', alpha=0.3, label='Obstacle'))
    ax.add_patch(plt.Circle((10, 8), 1.5, color='gray', alpha=0.3))
    
    # 시작/목표 (Agent 속성 활용)
    ax.plot(single_agent.start_pose[0], single_agent.start_pose[1], 'go', markersize=10, label='Start')
    ax.plot(single_agent.goal_pose[0], single_agent.goal_pose[1], 'rx', markersize=10, label='Goal')

    # 동적 요소 초기화
    particles_plot = ax.scatter([], [], s=1, c='red', alpha=0.3, label='Particle Belief')
    traj_line, = ax.plot([], [], 'b--', alpha=0.5, label='Planned Path')
    
    # [변경 5] 로봇을 점이 아닌 실제 크기(반지름)를 가진 원으로 표시
    robot_body = plt.Circle((0, 0), single_agent.radius, color=single_agent.color, alpha=0.8, label='Robot')
    ax.add_patch(robot_body)

    ax.set_xlim(-1, 12)
    ax.set_ylim(-1, 12)
    ax.legend(loc='upper left')
    ax.grid(True, linestyle='--')
    ax.set_title("Single Agent EKI (Object Oriented)")

    # 평균 궤적 계산
    mean_traj = jnp.mean(final_belief, axis=1) # (T, D)

    def update(frame):
        # 1. 파티클 업데이트
        current_particles = final_belief[frame] 
        particles_plot.set_offsets(current_particles[:, :2])
        
        # 2. 로봇 몸체 이동 (Circle center 업데이트)
        pos = mean_traj[frame]
        robot_body.center = (pos[0], pos[1])
        
        # 3. 궤적 선 업데이트
        traj_line.set_data(mean_traj[:frame+1, 0], mean_traj[:frame+1, 1])
        
        return particles_plot, robot_body, traj_line

    ani = FuncAnimation(fig, update, frames=config.horizon, interval=50, blit=True)
    plt.show()

if __name__ == "__main__":
    main()