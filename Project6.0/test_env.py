import numpy as np
from environment_with_flags import GateEnvironmentWithFlags

env = GateEnvironmentWithFlags()

# Тест 1: 3 ворота + 2 флага (случайный action)
state = env.reset(n_gates=3, n_flags=2)
action = np.random.rand(30)
state, reward, done, info = env.step(action)
print(f"Random: reward={reward:.3f}, valid={env.is_valid()}, violations={len(info['violations'])}")

# Тест 2: почти идеальная трасса (треугольник)
ideal = np.zeros(30)
ideal[0:3] = [0.2, 0.2, 0.0]      # gate 0: (-3, -3)
ideal[3:6] = [0.8, 0.2, 0.25]     # gate 1: (3, -3)
ideal[6:9] = [0.5, 0.8, 0.5]      # gate 2: (0, 3)
ideal[9:11]  = [0.5, 0.2]         # flag 0: (0, -3)
ideal[11:13] = [0.65, 0.5]        # flag 1: (1.5, 0)

state, reward, done, info = env.step(ideal)
print(f"Ideal:  reward={reward:.3f}, valid={env.is_valid()}, bonuses={info['bonuses']}")