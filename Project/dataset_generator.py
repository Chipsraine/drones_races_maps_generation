"""
Генератор синтетического датасета трасс для дронов.

Каждый пример — пара матриц:
  input:  карта с расставленными элементами (без пути)
  target: та же карта + путь дрона

Коды ячеек:
  0 = пусто
  1 = путь дрона
  2 = ворота  (полоса из 5 клеток, путь проходит СКВОЗЬ)
  3 = кольцо  (одна клетка, путь проходит СКВОЗЬ)
  4 = столбик (одна клетка, путь ОГИБАЕТ)
"""

import random
from collections import deque

import numpy as np


# ─── Константы ────────────────────────────────────────────────────────────────

EMPTY   = 0
PATH    = 1
GATE    = 2   # ворота — горизонтальная или вертикальная полоса 5 клеток
RING    = 3   # кольцо — одна клетка, насквозь
POLE    = 4   # столбик — одна клетка, вокруг

GRID_SCALE = 10          # 1 метр = 10 клеток
DEFAULT_SIZE_M = 10      # размер арены в метрах (10×10 м)
DEFAULT_GRID = DEFAULT_SIZE_M * GRID_SCALE   # 100×100 клеток


# ─── Размещение элементов ─────────────────────────────────────────────────────

def place_gate(grid, r, c, horizontal=True):
    """Размещает ворота (5 клеток) на карте. Возвращает True если успешно."""
    size = grid.shape[0]
    cells = []
    if horizontal:
        if c + 5 > size:
            return False
        cells = [(r, c + i) for i in range(5)]
    else:
        if r + 5 > size:
            return False
        cells = [(r + i, c) for i in range(5)]

    # Проверяем что место свободно + небольшой буфер вокруг
    for row, col in cells:
        for dr in range(-2, 3):
            for dc in range(-2, 3):
                nr, nc = row + dr, col + dc
                if 0 <= nr < size and 0 <= nc < size:
                    if grid[nr, nc] != EMPTY:
                        return False

    for row, col in cells:
        grid[row, col] = GATE
    return True


def place_element(grid, r, c, elem_type):
    """Размещает кольцо или столбик. Возвращает True если успешно."""
    size = grid.shape[0]
    if not (0 <= r < size and 0 <= c < size):
        return False
    for dr in range(-3, 4):
        for dc in range(-3, 4):
            nr, nc = r + dr, c + dc
            if 0 <= nr < size and 0 <= nc < size:
                if grid[nr, nc] != EMPTY:
                    return False
    grid[r, c] = elem_type
    return True


def generate_elements(grid_size=DEFAULT_GRID, n_gates=2, n_rings=2, n_poles=3, seed=None):
    """
    Создаёт матрицу с расставленными элементами.
    Возвращает grid (numpy array) и список «точек интереса» для построения пути.
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    grid = np.zeros((grid_size, grid_size), dtype=np.int8)
    margin = 10
    waypoints = []   # точки, которые путь должен пройти сквозь

    # Ворота
    placed = 0
    attempts = 0
    while placed < n_gates and attempts < 500:
        r = random.randint(margin, grid_size - margin - 5)
        c = random.randint(margin, grid_size - margin - 5)
        horizontal = random.random() > 0.5
        if place_gate(grid, r, c, horizontal):
            # Центр ворот — точка прохода
            if horizontal:
                waypoints.append((r, c + 2))
            else:
                waypoints.append((r + 2, c))
            placed += 1
        attempts += 1

    # Кольца
    placed = 0
    attempts = 0
    while placed < n_rings and attempts < 500:
        r = random.randint(margin, grid_size - margin)
        c = random.randint(margin, grid_size - margin)
        if place_element(grid, r, c, RING):
            waypoints.append((r, c))
            placed += 1
        attempts += 1

    # Столбики (путь огибает — не являются waypoints)
    placed = 0
    attempts = 0
    while placed < n_poles and attempts < 500:
        r = random.randint(margin, grid_size - margin)
        c = random.randint(margin, grid_size - margin)
        if place_element(grid, r, c, POLE):
            placed += 1
        attempts += 1

    return grid, waypoints


# ─── Построение пути (BFS) ────────────────────────────────────────────────────

def bfs_path(grid, start, end):
    """
    BFS на сетке от start до end.
    Клетки со столбиками (POLE) — непроходимы.
    Возвращает список координат пути или None.
    """
    size = grid.shape[0]
    queue = deque([(start, [start])])
    visited = {start}

    while queue:
        (r, c), path = queue.popleft()
        if (r, c) == end:
            return path
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r + dr, c + dc
            if (0 <= nr < size and 0 <= nc < size
                    and (nr, nc) not in visited
                    and grid[nr, nc] != POLE):
                visited.add((nr, nc))
                queue.append(((nr, nc), path + [(nr, nc)]))
    return None


def build_path(grid, waypoints, grid_size=DEFAULT_GRID):
    """
    Строит непрерывный путь через все waypoints.
    Начало и конец — случайные точки у краёв арены.
    Возвращает target-матрицу (копию grid + путь).
    """
    target = grid.copy()
    margin = 5

    # Точки входа и выхода
    start = (random.randint(margin, grid_size - margin), margin)
    end   = (random.randint(margin, grid_size - margin), grid_size - margin)

    # Перемешиваем waypoints и строим маршрут start → wp1 → wp2 → ... → end
    random.shuffle(waypoints)
    checkpoints = [start] + waypoints + [end]

    full_path = []
    for i in range(len(checkpoints) - 1):
        segment = bfs_path(target, checkpoints[i], checkpoints[i + 1])
        if segment is None:
            return None   # не удалось проложить путь — пример отбраковывается
        if full_path:
            segment = segment[1:]   # убираем дублирование точки стыка
        full_path.extend(segment)

    # Наносим путь на матрицу (не затираем элементы)
    for r, c in full_path:
        if target[r, c] == EMPTY:
            target[r, c] = PATH

    return target


# ─── Генерация датасета ───────────────────────────────────────────────────────

def generate_dataset(n_samples=500, grid_size=DEFAULT_GRID,
                     n_gates=2, n_rings=2, n_poles=3, seed=42):
    """
    Генерирует n_samples пар (input, target).

    Возвращает:
        inputs  — numpy array (N, grid_size, grid_size), dtype int8
        targets — numpy array (N, grid_size, grid_size), dtype int8
    """
    inputs  = []
    targets = []

    generated = 0
    attempts  = 0

    while generated < n_samples:
        attempts += 1
        grid, waypoints = generate_elements(
            grid_size=grid_size,
            n_gates=n_gates,
            n_rings=n_rings,
            n_poles=n_poles,
            seed=seed + attempts,
        )
        if not waypoints:
            continue

        target = build_path(grid, waypoints, grid_size=grid_size)
        if target is None:
            continue

        inputs.append(grid)
        targets.append(target)
        generated += 1

        if generated % 50 == 0:
            print(f"  Сгенерировано: {generated}/{n_samples} (попыток: {attempts})")

    return np.array(inputs), np.array(targets)


# ─── Запуск как скрипт ────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Генерация датасета...")
    inputs, targets = generate_dataset(n_samples=200, seed=42)

    np.save("data/inputs.npy", inputs)
    np.save("data/targets.npy", targets)

    print(f"Готово! inputs: {inputs.shape}, targets: {targets.shape}")
    print(f"Файлы сохранены в data/")

    # Визуализация одного примера
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(inputs[0], cmap="tab10", vmin=0, vmax=4)
        axes[0].set_title("Input (элементы без пути)")
        axes[1].imshow(targets[0], cmap="tab10", vmin=0, vmax=4)
        axes[1].set_title("Target (элементы + путь)")
        plt.tight_layout()
        plt.savefig("data/sample_visualization.png", dpi=100)
        print("Визуализация сохранена в data/sample_visualization.png")
    except Exception as e:
        print(f"Визуализация недоступна: {e}")
