"""
Обучаем отдельную сеть для флагов на synthetic данных.
"""

import torch
import numpy as np
from pathlib import Path
from environment_with_flags import GateEnvironmentWithFlags, WORK_MIN, WORK_RANGE, MAX_GATES, MAX_FLAGS
from flag_network import FlagNetwork, FlagTrainer

SAVE_DIR = Path("models")
N_SAMPLES = 50000
EPOCHS = 100
BATCH_SIZE = 256
LR = 1e-3


def generate_flag_data():
    """Генерируем пары (ворота, флаги) где флаги посередине между воротами."""
    data = []

    for _ in range(N_SAMPLES):
        n_gates = np.random.randint(3, 7)
        n_flags = min(np.random.randint(1, n_gates + 1), n_gates)

        # Генерируем валидные ворота (примерно правильный треугольник/квадрат/пятиугольник)
        cx, cy = 5.0, 5.0
        radius = np.random.uniform(2.0, 4.0)

        gates = []
        for i in range(n_gates):
            angle = i * 2 * np.pi / n_gates + np.random.uniform(-0.3, 0.3)
            x = np.clip(cx + radius * np.cos(angle), 0.5, 9.5)
            y = np.clip(cy + radius * np.sin(angle), 0.5, 9.5)
            a = np.random.uniform(0, 2 * np.pi)
            gates.append((x, y, a))

        # Флаги — на отрезке между воротами + небольшой шум
        flags = []
        for i in range(n_flags):
            g1 = np.array(gates[i][:2])
            g2 = np.array(gates[(i + 1) % n_gates][:2])

            # Параметр t вдоль отрезка
            t = np.random.uniform(0.2, 0.8)
            mid = g1 + t * (g2 - g1)

            # Небольшое отклонение от линии (до 0.5м)
            perp = np.array([-(g2[1] - g1[1]), g2[0] - g1[0]])
            perp_norm = np.linalg.norm(perp)
            if perp_norm > 0:
                perp = perp / perp_norm
                offset = np.random.uniform(-0.3, 0.3)
                flag_pos = mid + perp * offset
            else:
                flag_pos = mid

            flags.append(np.clip(flag_pos, 0.5, 9.5))

        # Нормализуем ворота [0, 10] -> [0, 1]
        gates_norm = np.zeros(MAX_GATES * 3, dtype=np.float32)
        gates_flat = np.array(gates).flatten() / 10.0
        gates_norm[:len(gates_flat)] = gates_flat

        # Нормализуем флаги
        flags_norm = np.zeros(MAX_FLAGS * 2, dtype=np.float32)
        flags_flat = np.array(flags).flatten() / 10.0
        flags_norm[:len(flags_flat)] = flags_flat

        # Маска
        mask = np.zeros(MAX_FLAGS * 2, dtype=np.float32)
        mask[:n_flags * 2] = 1.0

        data.append((gates_norm, flags_norm, mask, n_gates, n_flags))

    return data


def validate(network, env, n_tests=100):
    """Проверяем: берём случайные ворота, предсказываем флаги, проверяем валидность."""
    network.eval()
    valid_count = 0

    with torch.no_grad():
        for _ in range(n_tests):
            n_gates = np.random.randint(3, 7)
            n_flags = min(np.random.randint(1, n_gates + 1), n_gates)

            # Генерируем ворота
            cx, cy = 5.0, 5.0
            radius = np.random.uniform(2.0, 4.0)
            gates = []
            for i in range(n_gates):
                angle = i * 2 * np.pi / n_gates + np.random.uniform(-0.3, 0.3)
                x = np.clip(cx + radius * np.cos(angle), 0.5, 9.5)
                y = np.clip(cy + radius * np.sin(angle), 0.5, 9.5)
                a = np.random.uniform(0, 2 * np.pi)
                gates.append((x, y, a))

            # Предсказываем флаги
            gates_np = np.array(gates, dtype=np.float32)
            flags_pred = network.predict_flags(gates_np, n_flags)

            # Собираем action для среды
            action = np.zeros(MAX_GATES * 3 + MAX_FLAGS * 2, dtype=np.float32)
            for i, (x, y, a) in enumerate(gates):
                action[i * 3] = (x - WORK_MIN) / WORK_RANGE
                action[i * 3 + 1] = (y - WORK_MIN) / WORK_RANGE
                action[i * 3 + 2] = a / (2 * np.pi)

            flag_offset = n_gates * 3
            for i, (fx, fy) in enumerate(flags_pred):
                action[flag_offset + i * 2] = (fx - WORK_MIN) / WORK_RANGE
                action[flag_offset + i * 2 + 1] = (fy - WORK_MIN) / WORK_RANGE

            # Проверяем валидность
            env.reset(n_gates, n_flags)
            env.step(action)
            if env.is_valid():
                valid_count += 1

    network.train()
    return valid_count / n_tests


def main():
    print("=" * 60)
    print("ОБУЧЕНИЕ СЕТИ ФЛАГОВ")
    print("=" * 60)

    env = GateEnvironmentWithFlags()
    network = FlagNetwork()
    trainer = FlagTrainer(network, lr=LR)

    print("Генерация данных...")
    data = generate_flag_data()
    print(f"Сгенерировано {len(data)} примеров")

    print("Обучение...")
    best_valid = 0.0

    for epoch in range(EPOCHS):
        np.random.shuffle(data)
        total_loss = 0
        n_batches = 0

        for i in range(0, len(data), BATCH_SIZE):
            batch = data[i:i + BATCH_SIZE]

            gates = torch.FloatTensor(np.array([x[0] for x in batch]))
            flags = torch.FloatTensor(np.array([x[1] for x in batch]))
            masks = torch.FloatTensor(np.array([x[2] for x in batch]))

            loss = trainer.update(gates, flags, masks)
            total_loss += loss
            n_batches += 1

        avg_loss = total_loss / n_batches

        # Валидация каждые 10 эпох
        if (epoch + 1) % 10 == 0:
            valid_rate = validate(network, env, n_tests=200)
            print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {avg_loss:.6f} | Valid: {valid_rate*100:.1f}%")

            if valid_rate > best_valid:
                best_valid = valid_rate
                torch.save(network.state_dict(), SAVE_DIR / "flag_network_best.pt")
                print(f"  -> Новый лучший результат!")

    # Сохраняем финальную
    torch.save(network.state_dict(), SAVE_DIR / "flag_network.pt")
    print(f"\nСохранено: {SAVE_DIR / 'flag_network.pt'}")
    print(f"Лучшая валидность: {best_valid*100:.1f}%")

    # Финальная проверка
    print("\nФинальная проверка:")
    for ng in [3, 4, 5, 6]:
        for nf in [1, 2, 3]:
            if nf > ng:
                continue
            valid_rate = validate_single_config(network, env, ng, nf)
            print(f"  {ng}в/{nf}ф: {valid_rate*100:.1f}%")

    print("\nГотово!")


def validate_single_config(network, env, n_gates, n_flags, n_tests=100):
    """Проверка на конкретной конфигурации."""
    network.eval()
    valid_count = 0

    with torch.no_grad():
        for _ in range(n_tests):
            # Генерируем ворота
            cx, cy = 5.0, 5.0
            radius = np.random.uniform(2.0, 4.0)
            gates = []
            for i in range(n_gates):
                angle = i * 2 * np.pi / n_gates + np.random.uniform(-0.3, 0.3)
                x = np.clip(cx + radius * np.cos(angle), 0.5, 9.5)
                y = np.clip(cy + radius * np.sin(angle), 0.5, 9.5)
                a = np.random.uniform(0, 2 * np.pi)
                gates.append((x, y, a))

            gates_np = np.array(gates, dtype=np.float32)
            flags_pred = network.predict_flags(gates_np, n_flags)

            action = np.zeros(MAX_GATES * 3 + MAX_FLAGS * 2, dtype=np.float32)
            for i, (x, y, a) in enumerate(gates):
                action[i * 3] = (x - WORK_MIN) / WORK_RANGE
                action[i * 3 + 1] = (y - WORK_MIN) / WORK_RANGE
                action[i * 3 + 2] = a / (2 * np.pi)

            flag_offset = n_gates * 3
            for i, (fx, fy) in enumerate(flags_pred):
                action[flag_offset + i * 2] = (fx - WORK_MIN) / WORK_RANGE
                action[flag_offset + i * 2 + 1] = (fy - WORK_MIN) / WORK_RANGE

            env.reset(n_gates, n_flags)
            env.step(action)
            if env.is_valid():
                valid_count += 1

    network.train()
    return valid_count / n_tests


if __name__ == "__main__":
    main()