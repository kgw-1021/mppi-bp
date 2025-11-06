# example_usage.py
import numpy as np
from fg.factor_graph import SampleFactorGraph, SampleVNode, SampleFNode

# 1) 노드 생성 (각 변수는 이름과 dims 리스트)
x = SampleVNode('x', ['x'])
y = SampleVNode('y', ['y'])

# 2) 팩터 생성 (여기선 joint dims = ['x','y'])
f = SampleFNode('f_xy', [x, y], mppi_params={'K': 800, 'lambda': 1.0, 'noise_std': 1.0})

# 3) 그래프 연결
g = SampleFactorGraph()
g.connect(x, f)
g.connect(y, f)

# 4) cost 함수 정의 (샘플 행렬 (K,2) -> costs (K,))
def cost_fn(samples):
    # 예: x 근처(1.0), y 근처(2.0) 가 좋음
    x = samples[:, 0]
    y = samples[:, 1]
    return (x-1.0)**2 + 0.5*(y-2.0)**2

# 5) loopy propagate (팩터 업데이트에 cost_fn 제공)
beliefs = g.loopy_propagate(steps=5, factor_costs={f: (cost_fn, {})})

# 6) 결과 확인
bx = beliefs[x]  # SampleMessage
by = beliefs[y]
print("x samples mean:", (bx.samples * bx.weights[:,None]).sum(axis=0))
print("y samples mean:", (by.samples * by.weights[:,None]).sum(axis=0))
