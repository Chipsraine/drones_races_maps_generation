"""
Rule-based генератор валидных конфигураций ворот для дрон-рейсинга.

Стратегия: размещаем ворота по замкнутому контуру (деформированный круг),
затем добавляем шум для разнообразия. Это гарантирует замкнутость.
"""

import numpy as np
from pathlib import Path

# Ограничения
ARENA_SIZE = 10.0
MARGIN = 3.0
GATE_SIZE = 1.0
MIN_DIST = 3.0
MAX_DIST = 10.0
MAX_ANGLE_DIFF = np.pi
WORK_MIN = MARGIN          # 3.0
WORK_MAX = ARENA_SIZE - MARGIN  # 7.0
WORK_CENTER = (WORK_MIN + WORK_MAX) / 2  # 5.0
WORK_RANGE = WORK_MAX - WORK_MIN  # 4.0

MIN_GATES = 5
MAX_GATES = 10


def check_bounds(x: float, y: float) -> bool:
    return WORK_MIN <= x <= WORK_MAX and WORK_MIN <= y <= WORK_MAX


def check_distance(x1: float, y1: float, x2: float, y2: float) -> bool:
    d = np.hypot(x2 - x1, y2 - y1)
    return MIN_DIST <= d <= MAX_DIST


def check_angle_diff(a1: float, a2: float) -> bool:
    diff = abs(a1 - a2) % (2 * np.pi)
    if diff > np.pi:
        diff = 2 * np.pi - diff
    return diff <= MAX_ANGLE_DIFF


def check_min_distance_to_all(x: float, y: float, gates: list, min_dist: float = MIN_DIST) -> bool:
    for gx, gy, _ in gates:
        if np.hypot(x - gx, y - gy) < min_dist:
            return False
    return True


def generate_config(rng: np.random.Generator, max_attempts: int = 200) -> np.ndarray | None:
    """
    Генерирует одну валидную конфигурацию ворот.

    Стратегия:
    1. Выбираем N ворот, равномерно распределяем углы по кругу
    2. Сэмплируем радиус для каждых ворот (с деформацией)
    3. Добавляем шум к позициям
    4. Проверяем все ограничения
    """
    n_gates = rng.integers(MIN_GATES, MAX_GATES + 1)

    for _ in range(max_attempts):
        # Базовые углы по кругу (равномерно, с шумом)
        base_angles = np.linspace(0, 2 * np.pi, n_gates, endpoint=False)
        angle_noise = rng.uniform(-0.3, 0.3, size=n_gates)
        angles = base_angles + angle_noise
        # Сортируем, чтобы порядок был по часовой
        angles = np.sort(angles) % (2 * np.pi)

        # Радиусы (деформированный круг)
        base_radius = rng.uniform(1.2, 1.8)
        radii = base_radius + rng.uniform(-0.4, 0.4, size=n_gates)
        radii = np.clip(radii, 0.5, 2.0)

        # Центр с небольшим смещением
        cx = WORK_CENTER + rng.uniform(-0.5, 0.5)
        cy = WORK_CENTER + rng.uniform(-0.5, 0.5)

        # Координаты ворот
        xs = cx + radii * np.cos(angles)
        ys = cy + radii * np.sin(angles)

        # Проверка границ
        if not all(check_bounds(x, y) for x, y in zip(xs, ys)):
            continue

        # Углы наклона ворот — примерно касательные к контуру + шум
        gate_angles = np.zeros(n_gates)
        for i in range(n_gates):
            tangent = angles[i] + np.pi / 2  # касательная к окружности
            gate_angles[i] = (tangent + rng.uniform(-np.pi * 0.8, np.pi * 0.8)) % (2 * np.pi)

        # Собираем конфигурацию
        gates = list(zip(xs, ys, gate_angles))

        # Проверяем расстояния между соседними
        valid = True
        for i in range(n_gates):
            j = (i + 1) % n_gates
            if not check_distance(xs[i], ys[i], xs[j], ys[j]):
                valid = False
                break

        if not valid:
            continue

        # Проверяем разницу углов соседних
        for i in range(n_gates):
            j = (i + 1) % n_gates
            if not check_angle_diff(gate_angles[i], gate_angles[j]):
                valid = False
                break

        if not valid:
            continue

        # Проверяем мин. расстояние между НЕсоседними воротами
        for i in range(n_gates):
            for j in range(i + 2, n_gates):
                if i == 0 and j == n_gates - 1:
                    continue  # соседние (замкнутость)
                d = np.hypot(xs[i] - xs[j], ys[i] - ys[j])
                if d < MIN_DIST:
                    valid = False
                    break
            if not valid:
                break

        if not valid:
            continue

        return np.array(gates, dtype=np.float32)

    return None


def generate_dataset(n_samples: int = 10000, seed: int = 42) -> list[np.ndarray]:
    """Генерирует датасет валидных конфигураций."""
    rng = np.random.default_rng(seed)
    configs = []
    attempts = 0

    while len(configs) < n_samples:
        config = generate_config(rng)
        attempts += 1
        if config is not None:
            configs.append(config)

        if len(configs) % 1000 == 0 and len(configs) > 0:
            rate = len(configs) / attempts * 100
            print(f"  Сгенерировано {len(configs)}/{n_samples} "
                  f"(success rate: {rate:.0f}%)")

    rate = len(configs) / attempts * 100
    print(f"  Итого: {len(configs)} конфигураций за {attempts} попыток "
          f"(success rate: {rate:.0f}%)")
    return configs


def save_dataset(configs: list[np.ndarray], path: str | Path):
    """Сохраняет датасет как .npz."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, *configs)
    print(f"Сохранено {len(configs)} конфигураций в {path}")


def load_dataset(path: str | Path) -> list[np.ndarray]:
    """Загружает датасет из .npz."""
    data = np.load(path)
    return [data[k] for k in sorted(data.files, key=lambda x: int(x.split("_")[1]))]


if __name__ == "__main__":
    print("Генерация датасета...")
    configs = generate_dataset(n_samples=10000, seed=42)

    lengths = [len(c) for c in configs]
    print(f"\nСтатистика:")
    print(f"  Всего конфигураций: {len(configs)}")
    print(f"  Длины: min={min(lengths)}, max={max(lengths)}, "
          f"mean={np.mean(lengths):.1f}")

    save_path = Path(__file__).parent / "data" / "gate_configs.npz"
    save_dataset(configs, save_path)
