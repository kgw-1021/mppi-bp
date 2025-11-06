import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from motion.obstacle import ObstacleMap

class Visualizer:
    def __init__(self, omap: ObstacleMap, agents: list, xlim: tuple = (-100, 300), ylim: tuple = (-100, 300),
                 show_particles: bool = False, particle_alpha: float = 0.05) -> None:
        # Matplotlib 초기화 (실시간 플로팅 모드)
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.xlim = xlim
        self.ylim = ylim
        
        # 맵 범위 설정 (PyBullet 카메라 설정과 유사하게)
        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.ax.set_aspect('equal') # 1:1 비율 (원이 타원으로 보이지 않게)
        self.ax.grid(True)
        
        self.omap = omap
        self.agents = agents
        self.show_particles = show_particles
        self.particle_alpha = particle_alpha
        
        # 장애물 시각화 
        self._create_obstacles()
        
        # 에이전트 색상
        cmap = plt.colormaps.get_cmap('tab10')
        self.agent_colors = {agent: cmap(i / len(agents)) for i, agent in enumerate(agents)}

        # 에이전트 시각화 객체 (업데이트를 위해 저장)
        self.agent_patches = {}     # 에이전트 동체 (plt.Circle)
        self.target_markers = {}    # 에이전트 목표 (plt.plot)
        self.traj_lines = {}        # Sample-based 궤적 (plt.plot)
        self.particle_scatters = {} # 입자 시각화 (plt.scatter) - 선택적
        
        self._create_agents()
        
        self.ax.legend(loc='upper right')

    def _create_obstacles(self):
        """장애물을 Matplotlib Patch로 생성"""
        obstacle_added = False  # 범례 중복 방지용
        
        for name, obj in self.omap.objects.items():
            if obj['type'] == 'circle':
                circle = patches.Circle(
                    (obj['centerx'], obj['centery']),
                    obj['radius'],
                    color='black',
                    alpha=0.8,
                    label='Obstacle' if not obstacle_added else None
                )
                self.ax.add_patch(circle)
                obstacle_added = True
                
            elif obj['type'] == 'rectangle':
                # 직사각형 파라미터 추출
                cx, cy = obj['centerx'], obj['centery']
                width = obj['width']
                height = obj['height']
                theta = obj['theta']  # 회전 각도 (라디안)
                
                # Matplotlib의 Rectangle은 왼쪽 아래 모서리 좌표를 기준으로 함
                # 중심점 기준을 왼쪽 아래로 변환
                # 회전 변환을 위해 patches.Rectangle + set_transform 사용
                
                # 1. 중심 기준 직사각형 생성 (회전 전)
                rect = patches.Rectangle(
                    (-width/2, -height/2),  # 로컬 좌표계에서 중심이 (0,0)
                    width,
                    height,
                    color='black',
                    alpha=0.8,
                    label='Obstacle' if not obstacle_added else None
                )
                
                # 2. 변환 행렬 생성: 회전 + 이동
                # Matplotlib transform: Affine2D를 사용
                from matplotlib.transforms import Affine2D
                
                # 회전 각도를 degree로 변환 (theta는 라디안)
                angle_deg = np.degrees(theta)
                
                # 변환: 회전 후 이동
                t = Affine2D().rotate_around(0, 0, theta).translate(cx, cy) + self.ax.transData
                rect.set_transform(t)
                
                self.ax.add_patch(rect)
                obstacle_added = True
    
    def _create_agents(self):
        """에이전트 아티스트(Patch, Line)들을 생성"""
        for agent in self.agents:
            color = self.agent_colors[agent]
            
            # 1. 에이전트 동체 (Circle Patch)
            agent_patch = patches.Circle(
                (agent.x, agent.y),
                agent.r,
                color=color,
                alpha=0.9,
                label=f'Agent {agent.name}'
            )
            self.ax.add_patch(agent_patch)
            self.agent_patches[agent] = agent_patch
            
            # 2. 목표 지점 (X 마커)
            target = agent.get_target()
            if target is not None:
                target_marker, = self.ax.plot(
                    target[0, 0], 
                    target[1, 0], 
                    marker='x', 
                    markersize=10, 
                    color=color,
                    mew=2 # 마커 선 굵기
                )
                self.target_markers[agent] = target_marker
            
            # 3. Sample-based 궤적 (평균 경로)
            traj_line, = self.ax.plot([], [], color=color, linewidth=2, alpha=0.8)
            self.traj_lines[agent] = traj_line
            
            # 4. 입자 시각화 (선택적)
            if self.show_particles:
                scatter = self.ax.scatter([], [], s=5, color=color, alpha=self.particle_alpha)
                self.particle_scatters[agent] = scatter

    def update_visualization(self):
        """에이전트 위치 및 궤적 데이터 업데이트"""
        for agent in self.agents:
            patch = self.agent_patches[agent]
            traj_line = self.traj_lines[agent]
            
            state = agent.get_state()
            
            # 1. 에이전트 현재 위치 업데이트
            if state[0] is not None:
                x, y, _, _ = state[0]
                patch.center = (x, y)
            
            # 2. Sample-based 궤적 업데이트 (평균 경로)
            #   state 리스트에서 x, y 좌표만 추출
            traj_x = [s[0] for s in state if s is not None]
            traj_y = [s[1] for s in state if s is not None]
            traj_line.set_data(traj_x, traj_y)
            
            # 3. 입자 분포 시각화 (선택적)
            if self.show_particles and agent in self.particle_scatters:
                # 모든 시간 단계의 입자들을 표시
                all_particles_x = []
                all_particles_y = []
                
                for vnode in agent._vnodes[1:]:  # 첫 번째는 현재 위치이므로 제외
                    belief = vnode.belief
                    if belief is not None:
                        # 입자들의 x, y 좌표만 추출
                        particles_x = belief.samples[:, 0]  # x 좌표
                        particles_y = belief.samples[:, 1]  # y 좌표
                        all_particles_x.extend(particles_x)
                        all_particles_y.extend(particles_y)
                
                if len(all_particles_x) > 0:
                    self.particle_scatters[agent].set_offsets(
                        np.c_[all_particles_x, all_particles_y]
                    )
        
        # 캔버스 다시 그리기
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001) # GUI가 업데이트될 시간을 줌
    
    def close(self):
        plt.ioff() # 실시간 모드 끄기
        plt.close(self.fig)
        print("Matplotlib visualizer closed.")