"""
Тест: ворота от model_3g0f + флаги от flag_network.
"""

import torch
import numpy as np
from pathlib import Path
from environment_with_flags import GateEnvironmentWithFlags, WORK_MIN, WORK_RANGE, MAX_GATES, MAX_FLAGS
from agent import PolicyNetwork
from flag_network import FlagNetwork

SAVE_DIR = Path("models")


def main():
    env = GateEnvironmentWithFlags()

    # Загружаем сеть ворот
    policy_gates = PolicyNetwork()
    policy_gates.load_state_dict(torch.load(SAVE_DIR / "model_3g0f.pt"))
    policy_gates.eval()

    # Загружаем сеть флагов
    policy_flags = FlagNetwork()
    policy_flags.load_state_dict(torch.load(SAVE_DIR / "flag_network.pt"))
    policy_flags.eval()

    print("=" * 60)
    print("ТЕСТ: ДВЕ СЕТИ (ворота + флаги)")
    print("=" * 60)

    for n_gates in [3, 4]:
        for n_flags in [0, 1, 2, 3]:
            if n_flags > n_gates:
                continue

            valid_count = 0
            total_reward = 0

            with torch.no_grad():
                for _ in range(200):
                    # Шаг 1: Генерируем ворота
                    state = env.reset(n_gates, n_flags=0)  # флаги пока 0
                    action_full, _, _, _, _ = policy_gates.select_action(state)

                    # Берём только ворота из action
                    gates_action = action_full[:n_gates * 3]

                    # Декодируем ворота
                    gates = []
                    for i in range(n_gates):
                        x = gates_action[i * 3] * 10.0
                        y = gates_action[i * 3 + 1] * 10.0
                        a = gates_action[i * 3 + 2] * 2 * np.pi
                        gates.append((x, y, a))

                    # Шаг 2: Генерируем флаги через вторую сеть
                    if n_flags > 0:
                        gates_np = np.array(gates, dtype=np.float32)
                        flags_pred = policy_flags.predict_flags(gates_np, n_flags)
                    else:
                        flags_pred = np.zeros((0, 2))

                    # Шаг 3: Собираем полный action
                    action = np.zeros(MAX_GATES * 3 + MAX_FLAGS * 2, dtype=np.float32)
                    for i, (x, y, a) in enumerate(gates):
                        action[i * 3] = (x - WORK_MIN) / WORK_RANGE
                        action[i * 3 + 1] = (y - WORK_MIN) / WORK_RANGE
                        action[i * 3 + 2] = a / (2 * np.pi)

                    flag_offset = n_gates * 3
                    for i, (fx, fy) in enumerate(flags_pred):
                        action[flag_offset + i * 2] = (fx - WORK_MIN) / WORK_RANGE
                        action[flag_offset + i * 2 + 1] = (fy - WORK_MIN) / WORK_RANGE

                    # Шаг 4: Проверяем
                    env.reset(n_gates, n_flags)
                    _, reward, _, _ = env.step(action)
                    total_reward += reward

                    if env.is_valid():
                        valid_count += 1

            print(f"{n_gates}в/{n_flags}ф: {valid_count}/200 валидных ({valid_count/2:.1f}%), reward={total_reward/200:.1f}")

    print("\nГотово!")


if __name__ == "__main__":
    main()