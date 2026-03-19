"""
Генерация новых конфигураций ворот с помощью обученной GRU-модели.

Включает post-processing для соблюдения ограничений.
"""

import numpy as np
import torch
from pathlib import Path

from gate_model import GateGRU
from gate_dataset import denormalize_config, WORK_RANGE
from gate_generator import (
    WORK_MIN, WORK_MAX, MIN_DIST, MAX_DIST, MAX_ANGLE_DIFF,
    check_bounds, check_distance, check_angle_diff, check_min_distance_to_all,
    MIN_GATES, MAX_GATES,
)


def clamp_to_workzone(x: float, y: float) -> tuple[float, float]:
    """Ограничивает координаты рабочей зоной."""
    return np.clip(x, WORK_MIN, WORK_MAX), np.clip(y, WORK_MIN, WORK_MAX)


def fix_distance(x_prev, y_prev, x_new, y_new):
    """Корректирует позицию, чтобы расстояние было в [MIN_DIST, MAX_DIST]."""
    d = np.hypot(x_new - x_prev, y_new - y_prev)
    if d < 1e-6:
        # Слишком близко — сдвигаем случайно
        angle = np.random.uniform(0, 2 * np.pi)
        x_new = x_prev + MIN_DIST * np.cos(angle)
        y_new = y_prev + MIN_DIST * np.sin(angle)
        x_new, y_new = clamp_to_workzone(x_new, y_new)
        return x_new, y_new

    if d < MIN_DIST:
        # Растягиваем до мин. расстояния
        scale = MIN_DIST / d
        x_new = x_prev + (x_new - x_prev) * scale
        y_new = y_prev + (y_new - y_prev) * scale
    elif d > MAX_DIST:
        # Сжимаем до макс. расстояния
        scale = MAX_DIST / d
        x_new = x_prev + (x_new - x_prev) * scale
        y_new = y_prev + (y_new - y_prev) * scale

    x_new, y_new = clamp_to_workzone(x_new, y_new)
    return x_new, y_new


def fix_angle(prev_angle: float, new_angle: float) -> float:
    """Корректирует угол, чтобы разница была ≤ MAX_ANGLE_DIFF."""
    diff = (new_angle - prev_angle) % (2 * np.pi)
    if diff > np.pi:
        diff -= 2 * np.pi
    # Ограничиваем
    diff = np.clip(diff, -MAX_ANGLE_DIFF, MAX_ANGLE_DIFF)
    return (prev_angle + diff) % (2 * np.pi)


@torch.no_grad()
def generate_config(model: GateGRU, n_gates: int, device: torch.device,
                    temperature: float = 1.0) -> np.ndarray | None:
    """
    Генерирует одну конфигурацию ворот с помощью модели.

    Args:
        model: обученная модель
        n_gates: количество ворот
        device: устройство
        temperature: "температура" — шум к предсказаниям (0 = детерминистично)

    Returns:
        np.ndarray shape (n_gates, 3) или None при неудаче.
    """
    model.eval()
    hidden = model.init_hidden(1, device)

    # Первые ворота — случайные в рабочей зоне
    x = np.random.uniform(WORK_MIN, WORK_MAX)
    y = np.random.uniform(WORK_MIN, WORK_MAX)
    a = np.random.uniform(0, 2 * np.pi)

    gates = [(x, y, a)]

    # Нормализуем первые ворота
    normed = np.array([[(x - WORK_MIN) / WORK_RANGE,
                         (y - WORK_MIN) / WORK_RANGE,
                         a / (2 * np.pi)]], dtype=np.float32)
    current = torch.tensor(normed, device=device).unsqueeze(0)  # (1, 1, 3)

    for i in range(1, n_gates):
        pred, hidden = model.generate_step(current, hidden)
        pred = pred.cpu().numpy()[0]  # (3,)

        # Добавляем шум
        if temperature > 0:
            noise = np.random.normal(0, 0.02 * temperature, size=3)
            pred = np.clip(pred + noise, 0, 1)

        # Денормализация
        nx = pred[0] * WORK_RANGE + WORK_MIN
        ny = pred[1] * WORK_RANGE + WORK_MIN
        na = pred[2] * 2 * np.pi

        # Post-processing: корректируем ограничения
        nx, ny = clamp_to_workzone(nx, ny)
        nx, ny = fix_distance(gates[-1][0], gates[-1][1], nx, ny)
        na = fix_angle(gates[-1][2], na)

        gates.append((nx, ny, na))

        # Нормализуем для следующего шага
        normed = np.array([[(nx - WORK_MIN) / WORK_RANGE,
                             (ny - WORK_MIN) / WORK_RANGE,
                             na / (2 * np.pi)]], dtype=np.float32)
        current = torch.tensor(normed, device=device).unsqueeze(0)

    # Проверяем замкнутость
    gates_arr = np.array(gates, dtype=np.float32)
    if not check_distance(gates[-1][0], gates[-1][1], gates[0][0], gates[0][1]):
        return None
    if not check_angle_diff(gates[-1][2], gates[0][2]):
        return None

    return gates_arr


def generate_batch(model: GateGRU, n_configs: int, device: torch.device,
                   temperature: float = 1.0, max_attempts_per: int = 20) -> list[np.ndarray]:
    """Генерирует пачку валидных конфигураций."""
    results = []
    attempts = 0

    while len(results) < n_configs:
        n_gates = np.random.randint(MIN_GATES, MAX_GATES + 1)
        config = generate_config(model, n_gates, device, temperature)
        attempts += 1

        if config is not None:
            results.append(config)

        if attempts > n_configs * max_attempts_per:
            print(f"Слишком много попыток ({attempts}), "
                  f"сгенерировано {len(results)}/{n_configs}")
            break

    return results


def validate_config(config: np.ndarray) -> dict:
    """Проверяет все ограничения и возвращает отчёт."""
    n = len(config)
    report = {
        "n_gates": n,
        "bounds_ok": True,
        "distances_ok": True,
        "angles_ok": True,
        "closed_ok": True,
        "all_ok": True,
        "violations": [],
    }

    for i, (x, y, _) in enumerate(config):
        if not check_bounds(x, y):
            report["bounds_ok"] = False
            report["violations"].append(f"Gate {i}: out of bounds ({x:.2f}, {y:.2f})")

    for i in range(n):
        j = (i + 1) % n
        if not check_distance(config[i, 0], config[i, 1], config[j, 0], config[j, 1]):
            d = np.hypot(config[j, 0] - config[i, 0], config[j, 1] - config[i, 1])
            report["distances_ok"] = False
            report["violations"].append(f"Gates {i}-{j}: dist={d:.2f}m")

        if not check_angle_diff(config[i, 2], config[j, 2]):
            report["angles_ok"] = False
            report["violations"].append(f"Gates {i}-{j}: angle diff too large")

    last_first_dist = np.hypot(config[-1, 0] - config[0, 0], config[-1, 1] - config[0, 1])
    if not (MIN_DIST <= last_first_dist <= MAX_DIST):
        report["closed_ok"] = False

    report["all_ok"] = all([
        report["bounds_ok"], report["distances_ok"],
        report["angles_ok"], report["closed_ok"]
    ])

    return report


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path(__file__).parent / "models" / "best_model.pt"

    if not model_path.exists():
        print(f"Модель не найдена: {model_path}")
        print("Сначала запустите gate_train.py")
        return

    model = GateGRU().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

    print("Генерация конфигураций...")
    configs = generate_batch(model, n_configs=100, device=device, temperature=0.5)
    print(f"Сгенерировано: {len(configs)}")

    # Валидация
    valid = 0
    for config in configs:
        report = validate_config(config)
        if report["all_ok"]:
            valid += 1

    print(f"Validity rate: {valid}/{len(configs)} ({100 * valid / max(len(configs), 1):.1f}%)")

    # Визуализация лучших
    from gate_visualize import visualize_samples
    valid_configs = [c for c in configs if validate_config(c)["all_ok"]]
    if valid_configs:
        visualize_samples(
            valid_configs[:6],
            save_path=Path(__file__).parent / "data" / "viz_generated.png"
        )
    else:
        print("Нет полностью валидных конфигураций, показываем первые:")
        visualize_samples(
            configs[:6],
            save_path=Path(__file__).parent / "data" / "viz_generated.png"
        )


if __name__ == "__main__":
    main()
