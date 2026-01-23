import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from fg.factor import ObstacleFactor, GoalFactor, DynamicsFactor, SpeedLimitFactor
from fg.graph import FactorGraph
from fg.solver import FactorGraphEKISolver
import time 

class Config:
    horizon = 30
    num_particles = 500
    state_dim = 4
    dt = 0.1
    r_diag = 0.5
    damping = 0.2

def main():
    # 1. 그래프 및 팩터 설정
    graph = FactorGraph(dt=0.1)
    graph.add_factor(DynamicsFactor(weight=30.0)) # 물리 법칙 강하게
    graph.add_factor(ObstacleFactor(center=[5.0, 5.0], radius=2.0, weight=15.0))
    graph.add_factor(ObstacleFactor(center=[10.0, 8.0], radius=1.5, weight=15.0))
    
    goal_pos = jnp.array([10.0, 10.0, 0.0, 0.0])
    graph.add_factor(GoalFactor(goal_state=goal_pos, weight=2.0))
    
    # 속도 제한을 현실적으로 수정 (예: 5.0 m/s)
    graph.add_factor(SpeedLimitFactor(max_speed=5.0, weight=1.0))

    config = Config()
    solver = FactorGraphEKISolver(graph, config)
    
    # 2. 팩터 그래프 최적화 (Solving)
    start_pos = jnp.array([0.0, 0.0, 0.0, 0.0])
    key = jax.random.PRNGKey(42)
    
    print("Optimization Start...")
    # final_belief shape: (T, N, D)
    start_time = time.time()
    final_belief = solver.solve(key, start_pos, goal_pos, iterations=50)
    end_time = time.time()
    print(f"Optimization Time: {end_time - start_time:.4f} s")
    print("Optimization Finished.")


    # 3. 애니메이션 설정
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # 고정 요소 그리기 (목표, 장애물)
    ax.add_patch(plt.Circle((5, 5), 2.0, color='r', alpha=0.2))
    ax.add_patch(plt.Circle((10, 8), 1.5, color='r', alpha=0.2))
    ax.plot(start_pos[0], start_pos[1], 'go', markersize=10, label='Start')
    ax.plot(goal_pos[0], goal_pos[1], 'rx', markersize=10, label='Goal')

    # 동적 요소 초기화
    # - 파티클 구름 (연한 파란 점들)
    particles_plot = ax.scatter([], [], s=1, c='red', alpha=0.3, label='Particle Belief')
    # - 로봇 현재 위치 (평균값)
    robot_plot, = ax.plot([], [], 'ko', markersize=8, label='Robot (Mean)')
    # - 전체 궤적 선
    traj_line, = ax.plot([], [], 'b--', alpha=0.5, label='Planned Path')
    
    ax.set_xlim(-1, 12)
    ax.set_ylim(-1, 12)
    ax.legend(loc='upper left')
    ax.grid(True)

    # 평균 궤적 계산
    mean_traj = jnp.mean(final_belief, axis=1)

    print ("Final Mean Trajectory:")
    print (mean_traj)

    def update(frame):
        # 1. 현재 타임스텝의 파티클들 업데이트
        current_particles = final_belief[frame] # (N, D)
        particles_plot.set_offsets(current_particles[:, :2])
        
        # 2. 로봇 현재 위치 (평균) 업데이트
        robot_plot.set_data([mean_traj[frame, 0]], [mean_traj[frame, 1]])
        
        # 3. 지금까지 지나온 궤적 표시
        traj_line.set_data(mean_traj[:frame+1, 0], mean_traj[:frame+1, 1])
        
        ax.set_title(f"Time Step: {frame} (t={frame*config.dt:.1f}s)")
        return particles_plot, robot_plot, traj_line

    # 애니메이션 실행
    ani = FuncAnimation(fig, update, frames=config.horizon, interval=50, blit=True)
    
    plt.show()

if __name__ == "__main__":
    main()