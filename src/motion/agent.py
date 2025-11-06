from typing import Tuple, List, Dict
import numpy as np
from fg.factor_graph_mppi import SampleMessage, SampleVNode, SampleFNode, SampleFactorGraph

from .obstacle import ObstacleMap
from .nodes import DynaSampleFNode, ObstacleSampleFNode, DistSampleFNode, RemoteSampleVNode


class SampleAgent:
    def __init__(
            self, name: str, state, target=None, steps: int = 8, radius: int = 5, 
            omap: ObstacleMap = None, env: 'SampleEnv' = None,
            num_particles: int = 200,
            target_position_weight: float = 100,
            target_velocity_weight: float = 10,
            dynamic_position_weight: float = 10,
            dynamic_velocity_weight: float = 2,
            obstacle_weight: float = 50,
            distance_weight: float = 15,
            dt: float = 0.1
        ) -> None:
        assert steps > 1
        if np.shape(state) == ():
            state = np.array([[state]])
        elif len(np.shape(state)) == 1:
            state = np.array(state)[:, None]
        if target is not None:
            if np.shape(target) == ():
                target = np.array([[target]])
            elif len(np.shape(target)) == 1:
                target = np.array(target)[:, None]

        self._steps = steps
        self._name = name
        self._state = np.array(state)
        self._omap = omap
        self._radius = radius
        self._env = env
        self._dt = dt
        self._num_particles = num_particles

        # Weights for cost functions
        self._target_pos_weight = target_position_weight
        self._target_vel_weight = target_velocity_weight
        self._dyna_pos_weight = dynamic_position_weight
        self._dyna_vel_weight = dynamic_velocity_weight
        self._obstacle_weight = obstacle_weight
        self._dist_weight = distance_weight

        # Create VNodes with sample-based beliefs
        self._vnodes = []
        for i in range(steps):
            # Initialize particles around current state
            samples = self._initialize_samples(state, i)
            belief = SampleMessage([f'v{i}.x', f'v{i}.y', f'v{i}.vx', f'v{i}.vy'], 
                                   samples, np.ones(self._num_particles) / self._num_particles)
            vnode = SampleVNode(f'v{i}', [f'v{i}.x', f'v{i}.y', f'v{i}.vx', f'v{i}.vy'], belief)
            self._vnodes.append(vnode)

        # Create FNode for start (strong prior at current state)
        self._fnode_start = SampleFNode('fstart', [self._vnodes[0]], 
                                        mppi_params={'K': 200, 'lambda': 0.1, 'noise_std': 0.1})
        
        # Create FNode for target
        self._fnode_end = SampleFNode('fend', [self._vnodes[-1]],
                                      mppi_params={'K': 200, 'lambda': 0.5, 'noise_std': 0.5})
        
        self.set_state(state)
        self.set_target(target)

        # Create DynaFNode
        self._fnodes_dyna = [DynaSampleFNode(
            f'fd{i}{i+1}', [self._vnodes[i], self._vnodes[i+1]], 
            dt=self._dt, pos_weight=self._dyna_pos_weight, vel_weight=self._dyna_vel_weight,
            mppi_params={'K': 400, 'lambda': 1.0, 'noise_std': 0.5}
        ) for i in range(steps-1)]

        # Create ObstacleFNode
        self._fnodes_obst = [ObstacleSampleFNode(
            f'fo{i}', [self._vnodes[i]], omap=omap, safe_dist=self.r,
            obstacle_weight=self._obstacle_weight,
            mppi_params={'K': 300, 'lambda': 1.0, 'noise_std': 0.3}
        ) for i in range(1, steps)]

        # Build graph
        self._graph = SampleFactorGraph()
        self._graph.connect(self._vnodes[0], self._fnode_start)
        for v, f in zip(self._vnodes[:-1], self._fnodes_dyna):
            self._graph.connect(v, f)
        for v, f in zip(self._vnodes[1:], self._fnodes_dyna):
            self._graph.connect(v, f)
        for v, f in zip(self._vnodes[1:], self._fnodes_obst):
            self._graph.connect(v, f)
        self._graph.connect(self._vnodes[-1], self._fnode_end)

        self._others = {}

    def _initialize_samples(self, state, timestep):
        """시간 단계별로 샘플 초기화"""
        N = self._num_particles
        samples = np.zeros((N, 4))
        # 위치는 현재 + 예상 이동
        samples[:, :2] = state[:2, 0] + state[2:, 0] * self._dt * timestep
        samples[:, 2:] = state[2:, 0]
        # 약간의 노이즈 추가
        samples += np.random.randn(N, 4) * 0.1
        return samples

    def _start_cost(self, samples: np.ndarray) -> np.ndarray:
        """시작 위치 제약 비용 함수"""
        diff = samples - self._state[:, 0]
        costs = np.sum(diff[:, :2]**2, axis=1) * self._target_pos_weight
        costs += np.sum(diff[:, 2:]**2, axis=1) * self._target_vel_weight * 0.1
        return costs

    def _target_cost(self, samples: np.ndarray) -> np.ndarray:
        """목표 위치 제약 비용 함수"""
        if self._target is None:
            return np.zeros(samples.shape[0])
        diff = samples - self._target[:, 0]
        costs = np.sum(diff[:, :2]**2, axis=1) * self._target_pos_weight
        costs += np.sum(diff[:, 2:]**2, axis=1) * self._target_vel_weight
        return costs

    def __str__(self) -> str:
        return f'({self._name} s={self._state})'

    @property
    def name(self) -> str:
        return self._name
    
    @property
    def x(self) -> float:
        '''current x'''
        return self._state[0, 0]
    
    @property
    def y(self) -> float:
        '''current y'''
        return self._state[1, 0]
    
    @property
    def r(self) -> float:
        '''radius'''
        return self._radius

    def get_state(self) -> List[np.ndarray]:
        """각 시간 단계별 평균 상태 반환"""
        poss = []
        for v in self._vnodes:
            if v.belief is None:
                poss.append(None)
            else:
                # 샘플들의 가중 평균
                belief = v.belief
                mean_state = np.average(belief.samples, axis=0, weights=belief.weights)
                poss.append(mean_state)
        return poss

    def get_target(self) -> np.ndarray:
        return self._target

    def step_connect(self):
        # Search near agents
        others = self._env.find_near(self)
        for o in others:
            self.setup_com(o)
        for on in list(self._others.keys()):
            if self._others[on]['a'] not in others:
                self.end_com(on)

    def step_com(self):
        for o in self._others:
            self.send(o)

    def step_propagate(self):
        # MPPI로 각 팩터 업데이트
        factor_costs = {}
        
        # 시작 및 목표 팩터 비용
        if self._target is not None:
            factor_costs[self._fnode_start] = (self._start_cost, {'base': self._state[:, 0]})
            factor_costs[self._fnode_end] = (self._target_cost, {'base': self._target[:, 0]})
        else:
            factor_costs[self._fnode_start] = (self._start_cost, {'base': self._state[:, 0]})
        
        # 동역학 팩터는 cost_fn을 명시적으로 전달하지 않음 (자체 메서드 사용)
        # 빈 튜플로 전달하면 loopy_propagate에서 update_factor_with_mppi()를 인자 없이 호출
        for f in self._fnodes_dyna:
            factor_costs[f] = (None, {})
        
        # 장애물 팩터도 동일
        for f in self._fnodes_obst:
            factor_costs[f] = (None, {})
        
        # 거리 팩터도 동일
        for on in self._others:
            for f in self._others[on]['f']:
                factor_costs[f] = (None, {})
        
        # Loopy BP with MPPI
        self._graph.loopy_propagate(steps=1, factor_costs=factor_costs)

    def step_move(self):
        """실제 이동 시뮬레이션"""
        # v1의 평균으로 이동
        v1_belief = self._vnodes[1].belief
        next_state = np.average(v1_belief.samples, axis=0, weights=v1_belief.weights)
        next_state = next_state.reshape(-1, 1)
        
        # 약간의 노이즈 추가
        next_state += np.random.randn(4, 1) * 0.01
        next_state[:2] += next_state[2:] * self._dt
        
        # 모든 belief를 한 단계씩 이동
        for i in range(self._steps - 1):
            self._vnodes[i]._belief = self._vnodes[i+1]._belief.copy()
        
        # 마지막 노드는 extrapolation
        last_belief = self._vnodes[-1].belief
        samples = last_belief.samples.copy()
        samples[:, :2] += samples[:, 2:] * self._dt
        self._vnodes[-1]._belief = SampleMessage(self._vnodes[-1].dims, samples, last_belief.weights.copy())
        
        # 실제 상태 업데이트
        self._state = next_state
        self.set_state(self._state)

    def set_state(self, state):
        """현재 상태 설정"""
        self._state = np.array(state)
        # v0에 강한 prior 설정 (현재 위치 주변에 집중)
        samples = self._state[:, 0][None, :] + np.random.randn(self._num_particles, 4) * 0.05
        self._vnodes[0]._belief = SampleMessage(
            self._vnodes[0].dims, samples, 
            np.ones(self._num_particles) / self._num_particles
        )

    def set_target(self, target):
        """목표 위치 설정"""
        self._target = target

    def push_msg(self, msg):
        '''Called by other agent to simulate the other sending message to self agent.'''
        _type, aname, vname, msg_data = msg
        if msg_data is None:
            return
        if aname not in self._others:
            print(f'push msg: name {aname} not found')
            return
        vnodes: List[RemoteSampleVNode] = self._others[aname]['v']
        vnode: RemoteSampleVNode = None
        for v in vnodes:
            if v.name == vname:
                vnode = v
                break
        if vnode is None:
            print('vname not found')
            return

        # msg_data는 SampleMessage
        if _type == 'belief':
            vnode._belief = msg_data
        if _type == 'f2v':
            e = vnode.edges[0]
            e.set_message_from(e.get_other(vnode), msg_data)
        if _type == 'v2f':
            e = vnode.edges[0]
            vnode._msgs[e] = msg_data

    def setup_com(self, other: 'SampleAgent'):
        """다른 에이전트와 통신 설정"""
        on = other._name
        if on in self._others:
            return

        vnodes = [RemoteSampleVNode(
            f'{on}.v{i}', [f'{on}.v{i}.x', f'{on}.v{i}.y', f'{on}.v{i}.vx', f'{on}.v{i}.vy']
        ) for i in range(1, self._steps)]
        
        fnodes = [DistSampleFNode(
            f'{on}.f{i}', [vnodes[i-1], self._vnodes[i]], 
            safe_dist=self.r + other.r, distance_weight=self._dist_weight,
            mppi_params={'K': 300, 'lambda': 1.0, 'noise_std': 0.3}
        ) for i in range(1, self._steps)]

        for i in range(1, self._steps):
            self._graph.connect(self._vnodes[i], fnodes[i-1])
            self._graph.connect(vnodes[i-1], fnodes[i-1])
        
        self._others[on] = {'a': other, 'v': vnodes, 'f': fnodes}
        other.setup_com(self)
        self.send(on)

    def send(self, name: str):
        """다른 에이전트에게 메시지 전송"""
        if name not in self._others:
            return
        
        other = self._others[name]['a']
        for i in range(1, self._steps):
            vname = f'{self._name}.v{i}'
            v = self._vnodes[i]
            f: SampleFNode = self._others[name]['f'][i-1]

            belief = v.belief.copy()
            other.push_msg(('belief', self._name, vname, belief))

            f2v = f.edges[0].get_message_to(v)
            if f2v is not None:
                f2v = f2v.copy()
            other.push_msg(('f2v', self._name, vname, f2v))

            v2f = f.edges[0].get_message_to(f)
            if v2f is not None:
                v2f = v2f.copy()
            other.push_msg(('v2f', self._name, vname, v2f))

    def end_com(self, name: str):
        """다른 에이전트와 통신 종료"""
        if name not in self._others:
            return

        vnodes = self._others[name]['v']
        fnodes = self._others[name]['f']
        for v in vnodes:
            self._graph.remove_node(v)
        for f in fnodes:
            self._graph.remove_node(f)
        other_dict = self._others.pop(name)
        other_dict['a'].end_com(self._name)


class SampleEnv:
    def __init__(self) -> None:
        self._agents: List[SampleAgent] = []

    def add_agent(self, a: SampleAgent):
        if a not in self._agents:
            a._env = self
            self._agents.append(a)

    def find_near(self, this: SampleAgent, range: float = 1000, max_num: int = -1) -> List[SampleAgent]:
        agent_ds = []
        for a in self._agents:
            if a is this:
                continue
            d = np.sqrt((a.x - this.x)**2 + (a.y - this.y)**2)
            if d < range:
                agent_ds.append((a, d))
        agent_ds.sort(key=lambda ad: ad[1])
        if max_num > 0:
            agent_ds = agent_ds[:max_num]
        return [a for a, d in agent_ds]

    def step_plan(self, iters=12):
        """Planning step: 통신 설정 및 belief propagation"""
        for a in self._agents:
            a.step_connect()
        for i in range(iters):
            for a in self._agents:
                a.step_com()
            for a in self._agents:
                a.step_propagate()

    def step_move(self):
        """Move step: 실제 이동 시뮬레이션"""
        for a in self._agents:
            a.step_move()