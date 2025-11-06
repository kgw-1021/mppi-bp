import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import time
from matplotlib.animation import PillowWriter

from motion.obstacle import ObstacleMap
from motion.agent import SampleAgent, SampleEnv
from visualize import Visualizer

if __name__ == '__main__':
    # 장애물 맵 생성
    omap = ObstacleMap()
    omap.set_circle('obstacle1', 100, 100, 20)
    
    # Sample-based 에이전트 생성 (MPPI 파라미터 포함)
    agent0 = SampleAgent('a0', [0, 0, 0, 0], [200, 200, 0, 0], 
                         steps=10, radius=5, omap=omap, num_particles=200,
                         target_position_weight=100, target_velocity_weight=10,
                         dynamic_position_weight=10, dynamic_velocity_weight=2,
                         obstacle_weight=50, distance_weight=15, dt=0.1)
    
    agent1 = SampleAgent('a1', [0, 200, 0, 0], [200, 0, 0, 0], 
                         steps=10, radius=5, omap=omap, num_particles=200,
                         target_position_weight=100, target_velocity_weight=10,
                         dynamic_position_weight=10, dynamic_velocity_weight=2,
                         obstacle_weight=50, distance_weight=15, dt=0.1)
    
    agent2 = SampleAgent('a2', [200, 200, 0, 0], [0, 0, 0, 0], 
                         steps=10, radius=5, omap=omap, num_particles=200,
                         target_position_weight=100, target_velocity_weight=10,
                         dynamic_position_weight=10, dynamic_velocity_weight=2,
                         obstacle_weight=50, distance_weight=15, dt=0.1)
    
    agent3 = SampleAgent('a3', [200, 0, 0, 0], [0, 200, 0, 0], 
                         steps=10, radius=5, omap=omap, num_particles=200,
                         target_position_weight=100, target_velocity_weight=10,
                         dynamic_position_weight=10, dynamic_velocity_weight=2,
                         obstacle_weight=50, distance_weight=15, dt=0.1)
    
    agent4 = SampleAgent('a4', [100, 0, 0, 0], [100, 200, 0, 0], 
                         steps=10, radius=5, omap=omap, num_particles=200,
                         target_position_weight=100, target_velocity_weight=10,
                         dynamic_position_weight=10, dynamic_velocity_weight=2,
                         obstacle_weight=50, distance_weight=15, dt=0.1)
    
    agent5 = SampleAgent('a5', [0, 100, 0, 0], [200, 100, 0, 0], 
                         steps=10, radius=5, omap=omap, num_particles=200,
                         target_position_weight=100, target_velocity_weight=10,
                         dynamic_position_weight=10, dynamic_velocity_weight=2,
                         obstacle_weight=50, distance_weight=15, dt=0.1)
    
    agent6 = SampleAgent('a6', [200, 100, 0, 0], [0, 100, 0, 0], 
                         steps=10, radius=5, omap=omap, num_particles=200,
                         target_position_weight=100, target_velocity_weight=10,
                         dynamic_position_weight=10, dynamic_velocity_weight=2,
                         obstacle_weight=50, distance_weight=15, dt=0.1)
    
    agent7 = SampleAgent('a7', [100, 200, 0, 0], [100, 0, 0, 0], 
                         steps=10, radius=5, omap=omap, num_particles=200,
                         target_position_weight=100, target_velocity_weight=10,
                         dynamic_position_weight=10, dynamic_velocity_weight=2,
                         obstacle_weight=50, distance_weight=15, dt=0.1)
    
    # 환경 설정 (SampleEnv 사용)
    env = SampleEnv()
    for a in [agent0, agent1, agent2, agent3, agent4, agent5, agent6, agent7]:
        env.add_agent(a)
    
    agents = [agent0, agent1, agent2, agent3, agent4, agent5, agent6, agent7]
    visualizer = Visualizer(omap, agents)

    print("종료: Ctrl+C 또는 창 닫기")
    print("MPPI 기반 Sample-based Belief Propagation 실행 중...")
    step_count = 0

    # --- 🎥 영상 저장 설정 ---
    metadata = dict(title='Sample-MPPI Simulation', artist='Geonwoo Kim', 
                   comment='MPPI-based particle trajectory simulation')
    writer = PillowWriter(fps=5)
    output_filename = "simulation_result_sample_mppi.gif"

    # fig 객체 가져오기 (Visualizer 내부의 figure 사용)
    fig = visualizer.fig if hasattr(visualizer, 'fig') else plt.gcf()

    try:
        with writer.saving(fig, output_filename, dpi=150):
            while step_count < 100:  # 최대 100 스텝
                # Sample-based BP + MPPI 기반 경로 계획
                # iters를 조금 줄여도 됨 (샘플링이므로 수렴이 더 느릴 수 있음)
                env.step_plan(iters=15)

                # 시각화 업데이트
                visualizer.update_visualization()

                # 프레임 저장
                writer.grab_frame()

                # 에이전트 이동
                env.step_move()

                if step_count % 10 == 0:
                    print(f"\nStep {step_count}")
                    all_reached_in_step = True
                    for agent in agents:
                        state = agent.get_state()
                        if state[0] is not None:
                            x, y, vx, vy = state[0]
                            target = agent.get_target()
                            if target is not None:
                                dist_to_goal = np.linalg.norm([x - target[0, 0], y - target[1, 0]])
                                print(f"  {agent.name}: pos=({x:.1f}, {y:.1f}), "
                                      f"vel=({vx:.2f}, {vy:.2f}), dist_to_goal={dist_to_goal:.1f}")
                                if dist_to_goal > 10:
                                    all_reached_in_step = False
                            else:
                                all_reached_in_step = False
                        else:
                            all_reached_in_step = False 
                    
                    if all_reached_in_step:
                        print("\n(출력) 모든 에이전트가 목표 근처 도달 시작")

                step_count += 1

                # 모든 에이전트가 목표에 도달했는지 확인
                all_reached = True
                for agent in agents:
                    state = agent.get_state()
                    if state[0] is not None: 
                        x, y, _, _ = state[0]
                        target = agent.get_target()
                        if target is not None:
                            dist = np.linalg.norm([x - target[0, 0], y - target[1, 0]])
                            if dist > 10:  
                                all_reached = False
                                break
                        else:
                            all_reached = False
                            break
                    else:
                        all_reached = False
                        break
                
                if all_reached:
                    print("\n모든 에이전트가 목표에 도달했습니다!")
                    print(f"총 {step_count} 스텝 소요")
                    print(f"\n시뮬레이션 결과가 '{output_filename}' 파일로 저장되었습니다.")
                    plt.pause(3)
                    break

    except KeyboardInterrupt:
        print("\n시뮬레이션 중단됨")

    finally:
        print(f"시뮬레이션 종료 (총 {step_count} 스텝)")
        print(f"MPPI 파라미터: K=200-400, lambda=0.1-1.0, particles={agents[0]._num_particles}")
        visualizer.close()