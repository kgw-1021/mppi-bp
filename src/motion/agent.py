from typing import Tuple, List, Dict
import numpy as np
from fg.factor_graph_mppi import SampleMessage, SampleVNode, SampleFNode, SampleFactorGraph

from .obstacle import ObstacleMap
from .nodes import DynaSampleFNode, ObstacleSampleFNode, DistSampleFNode, RemoteSampleVNode, GoalSampleFNode


class SampleAgent:
    def __init__(
            self, name: str, state, target=None, steps: int = 8, radius: int = 5, 
            omap: ObstacleMap = None, env: 'SampleEnv' = None,
            num_particles: int = 200,
            target_position_weight: float = 0.01,
            target_velocity_weight: float = 0.01,
            dynamic_position_weight: float = 0.1,
            dynamic_velocity_weight: float =0.01,
            spd_limit_weight: float = 0.1,
            obstacle_weight: float = 0.1,
            distance_weight: float = 0.1,
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
        self.spd_limit_weight = spd_limit_weight

        self._target = target
        
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
                                        mppi_params={'K': 1000, 'lambda': 0.01, 'noise_std': 1.0, 'dt': self._dt})
        
        # 마지막 vnode
        v_last = self._vnodes[self._steps - 1]

        # Goal Factor 생성
        self._fnode_goal = GoalSampleFNode(
            name=f"f_goal",
            vnodes=[v_last],
            goal=self._target,   
            goal_pos_weight=self._target_pos_weight,
            goal_vel_weight=self._target_vel_weight,
            mppi_params={'K': 1000, 'lambda': 10.0, 'noise_std': 10.0},
            dt=self._dt,
            _message_exponent=1.0
)

        # Create DynaFNode
        self._fnodes_dyna = [DynaSampleFNode(
            f'fd{i}{i+1}', [self._vnodes[i], self._vnodes[i+1]], 
            dt=self._dt, pos_weight=self._dyna_pos_weight, vel_weight=self._dyna_vel_weight, limit_weight=self.spd_limit_weight,
            mppi_params={'K': 1000, 'lambda': 100.0, 'noise_std': 5.0}, _message_exponent=1.0
        ) for i in range(steps-1)]

        # Create ObstacleFNode
        self._fnodes_obst = [ObstacleSampleFNode(
            f'fo{i}', [self._vnodes[i]], omap=omap, safe_dist=self.r*3,
            obstacle_weight=self._obstacle_weight,
            mppi_params={'K': 1000, 'lambda': 100.0, 'noise_std': 20.0},
            dt=self._dt, _message_exponent=1.0
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
        self._graph.connect(v_last, self._fnode_goal)

        self._others = {}

    def _initialize_samples(self, state, timestep):
        N = self._num_particles
        samples = np.zeros((N, 4))
        
        start_pos = state[:2, 0]
        
        max_dist = 80.0 * self._dt * timestep * 2.0 # MAX_SPEED=5.0 가정
        
        target_pos = start_pos
        if self._target is not None:
            raw_target_pos = self._target[:2, 0]
            
            # 목표까지의 벡터
            diff = raw_target_pos - start_pos
            dist = np.linalg.norm(diff)
            
            if dist > 1e-6:
                direction = diff / dist
                
                # 단순히 선형 보간하되, 거리를 제한함
                ratio = timestep / max(1, self._steps - 1)
                # 너무 멀면 max_dist 제한에 걸림
                actual_dist = min(dist * ratio, max_dist)
                
                target_pos = start_pos + direction * actual_dist
                
                # 속도 초기화: 목표 방향으로 MAX_SPEED의 절반 정도?
                target_vel = direction * 2.0 # 초기 속도 2.0
            else:
                target_vel = np.zeros(2)
        else:
            target_vel = np.zeros(2)

        samples[:, :2] = target_pos
        samples[:, 2:] = target_vel
        
        # 노이즈 추가 (다양성 확보)
        samples[:, :2] += np.random.randn(N, 2) * 2.0 # 위치 노이즈
        samples[:, 2:] += np.random.randn(N, 2) * 0.5 # 속도 노이즈
        
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
            belief = v.belief
            if belief is None:
                poss.append(None)
            else:
                # 샘플들의 가중 평균
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
        """계산된 메시지를 다른 에이전트와 교환"""
        for o in self._others:
            self.exchange_messages(o)

    def step_propagate(self):
        """MPPI 팩터 업데이트 및 내부 메시지 전파"""
        factor_costs = {
            self._fnode_start: (self._start_cost, {'base': self._state[:, 0]}),
        }
        
        # Loopy BP with MPPI
        self._graph.loopy_propagate(steps=5, factor_costs=factor_costs)

    def step_move(self):
        """Move step: 실제 이동 시뮬레이션 (belief shift + v0 재설정)"""
        
        # 1. v1의 평균으로 다음 상태 결정
        v1_belief = self._vnodes[1].belief
        if v1_belief is None:
            print(f" {self._name}: v1 belief is None!")
            return
        
        next_state = np.average(v1_belief.samples, axis=0, weights=v1_belief.weights)
        next_state = next_state.reshape(-1, 1)
        
        # 2. 모든 belief를 한 단계씩 앞으로 이동
        for i in range(self._steps - 1):
            self._vnodes[i]._belief = self._vnodes[i+1]._belief.copy()
        
        # 3. 마지막 노드 외삽
        last_belief = self._vnodes[-2].belief
        if last_belief is not None:
            samples = last_belief.samples.copy()
            samples[:, :2] += samples[:, 2:] * self._dt 
            self._vnodes[-1]._belief = SampleMessage(
                self._vnodes[-1].dims, samples, last_belief.weights.copy()
            )

        # 4. 실제 에이전트 상태 업데이트
        self._state = next_state
        
        # 5. v0에 현재 상태에 대한 강한 prior 재설정
        samples = self._state[:, 0][None, :] + np.random.randn(self._num_particles, 4) * 0.05
        self._vnodes[0]._belief = SampleMessage(
            self._vnodes[0].dims, samples, 
            np.ones(self._num_particles) / self._num_particles
        )

        v_last = self._vnodes[self._steps - 1]
        self._fnode_goal._vnodes = [v_last]

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
        
        # aname 검증
        if aname not in self._others:
            # print(f"Unknown agent: {aname}")
            return
        
        vnodes: List[RemoteSampleVNode] = self._others[aname]['v']
        vnode: RemoteSampleVNode = None
        
        # [FIX] 이름 매칭 로직 개선
        # 메시지의 vname(예: 'a0.v1')에서 접미사('v1')만 추출
        try:
            if '.' in vname:
                suffix = vname.split('.', 1)[1]
            else:
                suffix = vname
        except:
            return

        # 현재 에이전트가 등록한 원격 노드 이름 형식으로 매칭
        # self._others[aname]['v']에 'a1.v1' 형태로 등록되어 있다면:
        target_name_local = f"{self.name}.{suffix}" 
        # 만약 'a0.v1' 형태로 등록되어 있다면:
        target_name_remote = f"{aname}.{suffix}"

        for v in vnodes:
            # 두 가지 경우 모두 확인
            if v.name == target_name_local or v.name == target_name_remote or v.name == vname:
                vnode = v
                break
        
        if vnode is None:
            # print(f"VNode not found for {vname} (tried {target_name_local})")
            return

        if _type == 'belief':
            vnode._belief = msg_data
            return
        
        if _type == 'f2v':
            e = vnode.edges[0] if vnode.edges else None
            if e:
                e.set_message_from(e.get_other(vnode), msg_data)
        
        if _type == 'v2f':
            e = vnode.edges[0] if vnode.edges else None
            if e:
                vnode.set_outgoing_msg(e, msg_data)

    def setup_com(self, other: 'SampleAgent', _recursive_call: bool = False):
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
            mppi_params={'K': 1000, 'lambda': 1000.0, 'noise_std': 10.0},
            dt=self._dt
        ) for i in range(1, self._steps)]

        for i in range(1, self._steps):
            self._graph.connect(self._vnodes[i], fnodes[i-1])
            self._graph.connect(vnodes[i-1], fnodes[i-1])
        
        self._others[on] = {'a': other, 'v': vnodes, 'f': fnodes}
        
        if not _recursive_call:
            other.setup_com(self, _recursive_call=True)

    def end_com(self, name: str, _recursive_call: bool = False):
        """다른 에이전트와 통신 종료"""
        if name not in self._others:
            return

        other_dict = self._others.pop(name)
        vnodes = other_dict['v']
        fnodes = other_dict['f']
        
        for v in vnodes:
            self._graph.remove_node(v)
        for f in fnodes:
            self._graph.remove_node(f)
        
        if not _recursive_call:
            other = other_dict['a']
            other.end_com(self._name, _recursive_call=True)

    def exchange_messages(self, name: str):
        """계산된 메시지를 다른 에이전트와 교환 + belief도 전송"""
        if name not in self._others:
            return

        other = self._others[name]['a']
        
        for i in range(1, self._steps):
            v_self = self._vnodes[i]
            v_remote: RemoteSampleVNode = self._others[name]['v'][i-1]
            f: SampleFNode = self._others[name]['f'][i-1]

            # 1. belief 전송
            belief = v_self.belief
            if belief is not None:
                other.push_msg(('belief', self._name, v_remote.name, belief.copy()))

            # 2. v->f 메시지 전송
            edge_v_f = None
            for e in v_self.edges:
                if e.get_other(v_self) is f:
                    edge_v_f = e
                    break
            
            if edge_v_f:
                v2f_msg = edge_v_f.get_message_to(f)
                if v2f_msg is not None:
                    other.push_msg(('v2f', self._name, v_remote.name, v2f_msg.copy()))

            # 3. f->v 메시지 전송
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

    def step_plan(self, iters=12):
        """계획 단계"""
        for a in self._agents:
            a.step_connect()
        
        for i in range(iters):
            for a in self._agents:
                a.step_propagate()
            for a in self._agents:
                a.step_com()

    def step_move(self):
        """Move step: 실제 이동 시뮬레이션"""
        for a in self._agents:
            a.step_move()