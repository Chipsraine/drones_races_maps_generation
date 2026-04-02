"""
Rule-based генератор валидных конфигураций ворот для дрон-рейсинга.

Стратегия:
1. Находим seed-конфигурации случайным перебором
2. Массово генерируем новые через деформацию seed'ов

Ограничение на расстояние — только между СОСЕДНИМИ по маршруту воротами.
"""

import numpy as np
from pathlib import Path
from collections import defaultdict

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


def validate_config(config: np.ndarray) -> bool:
    """
    Проверяет все ограничения конфигурации.
    Расстояние проверяется только между соседними по маршруту воротами.
    """
    n = len(config)
    if n < MIN_GATES or n > MAX_GATES:
        return False

    # Границы
    if np.any(config[:, 0] < WORK_MIN) or np.any(config[:, 0] > WORK_MAX):
        return False
    if np.any(config[:, 1] < WORK_MIN) or np.any(config[:, 1] > WORK_MAX):
        return False

    # Расстояния между соседними (включая замыкание last→first)
    for i in range(n):
        j = (i + 1) % n
        d = np.hypot(config[j, 0] - config[i, 0], config[j, 1] - config[i, 1])
        if d < MIN_DIST or d > MAX_DIST:
            return False

    # Углы между соседними
    for i in range(n):
        j = (i + 1) % n
        diff = abs(config[i, 2] - config[j, 2]) % (2 * np.pi)
        if diff > np.pi:
            diff = 2 * np.pi - diff
        if diff > MAX_ANGLE_DIFF:
            return False

    return True


def _find_seeds(rng: np.random.Generator, n_per_size: int = 20,
                max_attempts: int = 500_000) -> dict[int, list[np.ndarray]]:
    """Находит seed-конфигурации для каждого N случайным перебором."""
    seeds = defaultdict(list)
    needed = {n: n_per_size for n in range(MIN_GATES, MAX_GATES + 1)}

    for _ in range(max_attempts):
        if all(v <= 0 for v in needed.values()):
            break

        n = rng.integers(MIN_GATES, MAX_GATES + 1)
        if needed.get(n, 0) <= 0:
            continue

        xs = rng.uniform(WORK_MIN, WORK_MAX, size=n)
        ys = rng.uniform(WORK_MIN, WORK_MAX, size=n)

        # Генерируем углы с малыми приращениями
        a0 = rng.uniform(0, 2 * np.pi)
        deltas = rng.uniform(-0.8, 0.8, size=n - 1)
        angles = np.concatenate([[a0], a0 + np.cumsum(deltas)]) % (2 * np.pi)

        config = np.column_stack([xs, ys, angles]).astype(np.float32)

        if validate_config(config):
            seeds[n].append(config)
            needed[n] -= 1

    return dict(seeds)


def perturb_config(config: np.ndarray, rng: np.random.Generator,
                   noise_scale: float = 0.4) -> np.ndarray:
    """Деформирует конфигурацию случайным шумом."""
    new = config.copy()
    n = len(config)

    # Позиционный шум
    new[:, 0] += rng.normal(0, noise_scale, size=n)
    new[:, 1] += rng.normal(0, noise_scale, size=n)
    new[:, 2] += rng.normal(0, noise_scale * 0.5, size=n)

    # Клэмп
    new[:, 0] = np.clip(new[:, 0], WORK_MIN, WORK_MAX)
    new[:, 1] = np.clip(new[:, 1], WORK_MIN, WORK_MAX)
    new[:, 2] = new[:, 2] % (2 * np.pi)

    # Случайные трансформации для разнообразия
    r = rng.random()
    if r < 0.2:
        # Глобальный сдвиг
        dx, dy = rng.normal(0, 0.3, size=2)
        new[:, 0] = np.clip(new[:, 0] + dx, WORK_MIN, WORK_MAX)
        new[:, 1] = np.clip(new[:, 1] + dy, WORK_MIN, WORK_MAX)
    elif r < 0.4:
        # Отражение
        axis = rng.integers(0, 2)
        new[:, axis] = WORK_MIN + WORK_MAX - new[:, axis]
    elif r < 0.6:
        # Поворот вокруг центра масс
        theta = rng.uniform(0, 2 * np.pi)
        cx, cy = new[:, 0].mean(), new[:, 1].mean()
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        dx, dy = new[:, 0] - cx, new[:, 1] - cy
        new[:, 0] = np.clip(cx + dx * cos_t - dy * sin_t, WORK_MIN, WORK_MAX)
        new[:, 1] = np.clip(cy + dx * sin_t + dy * cos_t, WORK_MIN, WORK_MAX)
        new[:, 2] = (new[:, 2] + theta) % (2 * np.pi)
    elif r < 0.7:
        # Масштабирование
        scale = rng.uniform(0.8, 1.2)
        cx, cy = new[:, 0].mean(), new[:, 1].mean()
        new[:, 0] = np.clip(cx + (new[:, 0] - cx) * scale, WORK_MIN, WORK_MAX)
        new[:, 1] = np.clip(cy + (new[:, 1] - cy) * scale, WORK_MIN, WORK_MAX)

    return new


def generate_dataset(n_samples: int = 10000, seed: int = 42) -> list[np.ndarray]:
    """Генерирует датасет валидных конфигураций."""
    rng = np.random.default_rng(seed)

    # Шаг 1: находим seeds
    print("Поиск seed-конфигураций...")
    seeds = _find_seeds(rng, n_per_size=30)
    total_seeds = sum(len(v) for v in seeds.values())
    print(f"Найдено {total_seeds} seeds для n={sorted(seeds.keys())}")

    if total_seeds == 0:
        raise RuntimeError("Не удалось найти seed-конфигурации!")

    # Собираем все seeds в плоский список
    all_seeds = []
    for configs in seeds.values():
        all_seeds.extend(configs)

    # Шаг 2: генерируем через деформацию
    print("Генерация через деформацию...")
    configs = list(all_seeds)  # начинаем с seeds
    attempts = 0

    while len(configs) < n_samples:
        # Выбираем базу: seed или уже сгенерированную
        if len(configs) > 50 and rng.random() < 0.7:
            base = configs[rng.integers(0, len(configs))]
        else:
            base = all_seeds[rng.integers(0, len(all_seeds))]

        noise = rng.uniform(0.15, 0.6)
        new_config = perturb_config(base, rng, noise_scale=noise)
        attempts += 1

        if validate_config(new_config):
            configs.append(new_config)

        if len(configs) % 2000 == 0:
            rate = len(configs) / max(attempts, 1) * 100
            print(f"  {len(configs)}/{n_samples} (perturbation success: {rate:.0f}%)")

    rate = len(configs) / max(attempts, 1) * 100
    print(f"  Итого: {len(configs)} за {attempts} деформаций "
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

    # Распределение по длинам
    from collections import Counter
    cnt = Counter(lengths)
    for n in sorted(cnt):
        print(f"  n={n}: {cnt[n]} ({cnt[n]/len(configs)*100:.1f}%)")

    save_path = Path(__file__).parent / "data" / "gate_configs.npz"
    save_dataset(configs, save_path)
