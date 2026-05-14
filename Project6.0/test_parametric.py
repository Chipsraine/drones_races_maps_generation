import numpy as np
from environment_with_flags import GateEnvironmentWithFlags

env = GateEnvironmentWithFlags()

# Тест: идеальный треугольник с флагами посередине (t=0.5, offset=0.5 → offset=0)
action = np.zeros(30)

# Ворота: треугольник
action[0:3] = [0.2, 0.2, 0.0]   # gate 0: (2, 2)
action[3:6] = [0.8, 0.2, 0.25]  # gate 1: (8, 2)
action[6:9] = [0.5, 0.8, 0.5]   # gate 2: (5, 8)

# Флаги: t=0.5 (середина), offset=0.5 (нулевое отклонение)
# t_raw = (0.5 - 0.1) / 0.8 = 0.5
# offset_raw = 0.5 (ноль метров)
action[9:11]  = [0.5, 0.5]   # flag 0: середина 0-1
action[11:13] = [0.5, 0.5]   # flag 1: середина 1-2
action[13:15] = [0.5, 0.5]   # flag 2: середина 2-0

env.reset(3, 3)
_, reward, _, info = env.step(action)
print(f"Valid: {env.is_valid()}")
print(f"Reward: {reward}")
print(f"Flags: {env.get_flags()}")