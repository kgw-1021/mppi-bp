from typing import Tuple, List, Dict
import numpy as np
# factor_graph_mppi는 리팩토링된 버전을 사용한다고 가정
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
                                        mppi_params={'K': 400, 'lambda': 10.0, 'noise_std': 0.5})
        
        # Create FNode for target
        self._fnode_end = SampleFNode('fend', [self._vnodes[-1]],
                                      mppi_params={'K': 400, 'lambda': 10.0, 'noise_std': 1.0})
        
        self.set_state(state) 
        self.set_target(target)

        # Create DynaFNode
        self._fnodes_dyna = [DynaSampleFNode(
            f'fd{i}{i+1}', [self._vnodes[i], self._vnodes[i+1]], 
            dt=self._dt, pos_weight=self._dyna_pos_weight, vel_weight=self._dyna_vel_weight,
            mppi_params={'K': 400, 'lambda': 2.0, 'noise_std': 0.5}
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

    # -----------------------------------------------------------------
    # REFACTOR 2: (분산 BP) `send` 대신 `exchange_messages`를 호출하도록 수정
    # -----------------------------------------------------------------
    def step_com(self):
        """계산된 메시지를 다른 에이전트와 교환"""
        for o in self._others:
            self.exchange_messages(o)

    def step_propagate(self):
        """(수정됨) MPPI 팩터 업데이트 및 내부 메시지 전파"""
        factor_costs = {}

        factor_costs[self._fnode_start] = (self._start_cost, {'base': self._state[:, 0]})
        if self._target is not None:
            factor_costs[self._fnode_end] = (self._target_cost, {'base': self._target[:, 0]})

        mppi_nodes = self._fnodes_dyna + self._fnodes_obst
        for on in self._others:
            mppi_nodes.extend(self._others[on]['f'])

        # 딕셔너리 업데이트
        for f in mppi_nodes:
            if f not in factor_costs: # fstart/fend 중복 방지
                factor_costs[f] = (None, {})

        # Loopy BP with MPPI
        # (이 단계에서 모든 노드가 내부적으로 메시지를 계산하여 엣지에 저장함)
        self._graph.loopy_propagate(steps=1, factor_costs=factor_costs)

    # -----------------------------------------------------------------
    # REFACTOR 1: (웜 스타트) `set_state` 강제 호출을 제거하고 신념을 이동(shift)
    # -----------------------------------------------------------------
    def step_move(self):
        
        # 1. v1의 평균으로 다음 상태 결정
        v1_belief = self._vnodes[1].belief
        next_state = np.average(v1_belief.samples, axis=0, weights=v1_belief.weights)
        next_state = next_state.reshape(-1, 1)
        
        # 2. 모든 belief를 한 단계씩 앞으로 이동 (v[0] = v[1], v[1] = v[2], ...)
        for i in range(self._steps - 1): # i = 0 부터 (T-2) 까지
            self._vnodes[i]._belief = self._vnodes[i+1]._belief.copy()
        
        # 3. 마지막 노드(v[T-1])는 이전 노드(새 v[T-2], 즉 이전 v[T-1])로부터 외삽(extrapolate)
        last_belief = self._vnodes[-2].belief 
        samples = last_belief.samples.copy()
        samples[:, :2] += samples[:, 2:] * self._dt 
        self._vnodes[-1]._belief = SampleMessage(self._vnodes[-1].dims, samples, last_belief.weights.copy())

        # 4. 실제 에이전트 상태 업데이트
        self._state = next_state
        

    def set_state(self, state):
        """(초기화용) 현재 상태 설정. v0에 강한 prior 설정."""
        self._state = np.array(state)
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
        
        # --- 🚨 핵심 수정 로직 🚨 ---
        # 1. 수신된 VName('a0.v1')에서 노드 이름('v1') 부분만 추출
        try:
            v_suffix = vname.split('.', 1)[1] # 'a0.v1' -> 'v1'
        except IndexError:
            # 접두사 형식이 아닌 경우 (예: 'v1'만 들어온 경우)
            v_suffix = vname 

        # 2. 로컬 에이전트의 이름(self.name: 'a1')을 사용하여 로컬에서 등록된 이름 재구성
        # target_vname은 'a1.v1'이 됨
        target_vname = f'{self.name}.{v_suffix}'
        # -----------------------------
        
        # --- (디버깅용 출력) ---
        print(f"Target VName for search: {target_vname}")
        # ---------------------------

        for v in vnodes:
            # 재구성된 target_vname으로 검색
            print(f"Checking vnode name: {v.name}")
            if v.name == target_vname:
                vnode = v
                break
        if vnode is None:
            print('vname not found')
            return

        # msg_data는 SampleMessage
        if _type == 'belief':
            vnode._belief = msg_data
        if _type == 'f2v':
            # 팩터 -> 원격 vnode 엣지로 메시지 설정
            e = vnode.edges[0]
            e.set_message_from(e.get_other(vnode), msg_data)
        if _type == 'v2f':
            # 원격 vnode -> 팩터 엣지로 메시지 설정
            e = vnode.edges[0]
            # (수정) RemoteVNode는 calc_msg가 없으므로, 수신된 메시지를 엣지에 직접 저장
            e.set_message_from(vnode, msg_data)
            # (원본) vnode._msgs[e] = msg_data -> RemoteVNode에 _msgs가 정의되지 않음

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
        # (수정) 초기 메시지 전송은 step_com에서만 수행
        # self.send(on)

    # -----------------------------------------------------------------
    # REFACTOR 2: (분산 BP) `send`를 `exchange_messages`로 대체
    # -----------------------------------------------------------------
    def update_factors(self):
        """MPPI로 factor 업데이트만"""
        factor_costs = {
            self._fnode_start: (self._start_cost, {'base': self._state[:, 0]}),
        }
        if self._target is not None:
            factor_costs[self._fnode_end] = (self._target_cost, {'base': self._target[:, 0]})
        
        all_factors = (self._fnodes_dyna + self._fnodes_obst + 
                    [f for other in self._others.values() for f in other['f']])
        
        for f in all_factors:
            if f in factor_costs:
                cost_fn, kwargs = factor_costs[f]
                f.update_factor_with_mppi(cost_fn, kwargs.get('base'))
            else:
                f.update_factor_with_mppi()  # 자체 cost 사용

    def propagate_v_to_f(self):
        """Var → Factor 메시지만"""
        for v in self._vnodes:
            v.propagate()

    def propagate_f_to_v(self):
        """Factor → Var 메시지만"""
        all_factors = ([self._fnode_start, self._fnode_end] + 
                    self._fnodes_dyna + self._fnodes_obst + 
                    [f for other in self._others.values() for f in other['f']])
        for f in all_factors:
            f.propagate()

    def update_beliefs(self):
        """Belief 업데이트만"""
        for v in self._vnodes:
            v.update_belief()

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
        # (수정) 상대방의 end_com을 재귀 호출하지 않도록 방지
        if self._name in other_dict['a']._others:
            other_dict['a'].end_com(self._name)

    def exchange_messages(self, name: str):
        if name not in self._others:
            return

        other = self._others[name]['a']
        
        for i in range(1, self._steps):
            v_self = self._vnodes[i]
            v_remote: RemoteSampleVNode = self._others[name]['v'][i-1]
            f: SampleFNode = self._others[name]['f'][i-1] # DistSampleFNode

            edge_v_f = None
            for e in v_self.edges:
                if e.get_other(v_self) is f:
                    edge_v_f = e
                    break
            
            if edge_v_f:
                v2f_msg = edge_v_f.get_message_to(f) 
                if v2f_msg is not None:
                    other.push_msg(('v2f', self._name, v_remote.name, v2f_msg.copy()))

            edge_f_v = None
            for e in f.edges:
                if e.get_other(f) is v_remote:
                    edge_f_v = e
                    break
            
            if edge_f_v:

                f2v_msg = edge_f_v.get_message_to(v_remote)
                if f2v_msg is not None:
                    other.push_msg(('f2v', self._name, v_remote.name, f2v_msg.copy()))


class SampleEnv:
    def __init__(self) -> None:
        self._agents: List[SampleAgent] = []

    def add_agent(self, a: SampleAgent):
        if a not in self._agents:
            a._env = self
            self._agents.append(a)

    def find_near(self, this: SampleAgent, range: float = 500, max_num: int = -1) -> List[SampleAgent]:
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

    # -----------------------------------------------------------------
    # REFACTOR 3: (실행 순서) propagate (계산) -> com (교환) 순서로 변경
    # -----------------------------------------------------------------
    def step_plan(self, iters=12):
        for a in self._agents:
            a.step_connect()
        
        for i in range(iters):
            # 1. Factor 업데이트
            for a in self._agents:
                a.update_factors()
            
            # 2. V→F 메시지
            for a in self._agents:
                a.propagate_v_to_f()
            
            # 3. 메시지 교환 (v→f)
            for a in self._agents:
                a.step_com()  # v2f 메시지 전송
            
            # 4. F→V 메시지
            for a in self._agents:
                a.propagate_f_to_v()
            
            # 5. 메시지 교환 (f→v)
            for a in self._agents:
                a.step_com()  # f2v 메시지 전송
            
            # 6. Belief 업데이트
            for a in self._agents:
                a.update_beliefs()

    def step_move(self):
        """Move step: 실제 이동 시뮬레이션"""
        for a in self._agents:
            a.step_move()