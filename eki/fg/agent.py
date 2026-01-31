import jax.numpy as jnp
from dataclasses import dataclass

@dataclass
class Agent:
    id: int
    start_pose: jnp.ndarray  # [x, y, vx, vy]
    goal_pose: jnp.ndarray   # [x, y, vx, vy]
    priority: float = 0.0    # 높을수록 우선권 가짐
    radius: float = 0.5      # 로봇 크기
    color: str = 'blue'      # 시각화용 색상

    def __post_init__(self):
        # JAX 배열로 확실하게 변환
        self.start_pose = jnp.array(self.start_pose)
        self.goal_pose = jnp.array(self.goal_pose)