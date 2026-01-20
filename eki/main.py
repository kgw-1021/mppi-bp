import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
from fg.factor import ObstacleFactor, GoalFactor, DynamicsFactor, SpeedLimitFactor
from fg.graph import FactorGraph
from fg.solver import FactorGraphEKISolver

class Config:
    horizon = 30
    num_particles = 500
    state_dim = 4
    dt = 0.1
    r_diag = 0.1
    damping = 0.2

def main():
    # 1. 그래프 생성
    graph = FactorGraph(dt=0.1)
    
    # 2. 팩터 조립 (원하는 대로 추가/삭제 가능!)
    #  - 동역학 추가
    graph.add_factor(DynamicsFactor(weight=2.0))
    
    #  - 장애물 1 추가
    graph.add_factor(ObstacleFactor(center=[5.0, 5.0], radius=2.0, weight=10.0))
    
    #  - 장애물 2 추가 (새로운 장애물도 쉽게 추가)
    graph.add_factor(ObstacleFactor(center=[8.0, 2.0], radius=1.5, weight=10.0))
    
    #  - 목표 지점 유도 추가
    goal_pos = jnp.array([10.0, 10.0, 0.0, 0.0])
    graph.add_factor(GoalFactor(goal_state=goal_pos, weight=1.0))

    #  - (옵션) 속도 제한 추가
    # graph.add_factor(SpeedLimitFactor(max_speed=2.0, weight=5.0))

    # 3. 솔버 설정
    config = Config()
    solver = FactorGraphEKISolver(graph, config)
    
    # 4. 실행
    start_pos = jnp.array([0.0, 0.0, 0.0, 0.0])
    key = jax.random.PRNGKey(42)
    
    print("Solving Factor Graph...")
    final_belief = solver.solve(key, start_pos, goal_pos, iterations=50)
    
    # 5. 결과 확인
    traj = jnp.mean(final_belief, axis=1)
    
    plt.figure(figsize=(8,8))
    plt.plot(traj[:,0], traj[:,1], 'b-o', label='Trajectory')
    
    # 장애물 시각화
    plt.gca().add_patch(plt.Circle((5,5), 2.0, color='r', alpha=0.3))
    plt.gca().add_patch(plt.Circle((8,2), 1.5, color='r', alpha=0.3))
    
    plt.plot(0,0,'go', label='Start')
    plt.plot(10,10,'rx', label='Goal')
    plt.legend()
    plt.grid()
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    main()