"""
Генератор датасета трасс для дронов.

Архитектура (v2 — waypoint heatmap):
  - Вход/выход ВСЕГДА фиксированы: верхний левый → правый нижний угол
  - Модель предсказывает тепловую карту ключевых точек (waypoints),
    а не всю маску пути
  - BFS соединяет найденные точки → путь гарантированно шириной 1 клетка

Что попадает в heatmap (target):
  - Gate:   entry, center, exit  (3 точки, задают перпендикулярный подход)
  - Ring:   center               (1 точка)
  - Pole:   bypass point         (1 точка рядом со столбиком)
  - Фиксированные start/end в heatmap НЕ включаются — они всегда известны

Коды ячеек:
  0 = пусто     1 = путь
  2 = ворота    3 = кольцо    4 = столбик
"""

import random
from collections import deque

import numpy as np
from scipy.ndimage import gaussian_filter


# ─── Константы ────────────────────────────────────────────────────────────────

EMPTY, PATH, GATE, RING, POLE = 0, 1, 2, 3, 4

GRID_SCALE     = 10
DEFAULT_SIZE_M = 10
DEFAULT_GRID   = DEFAULT_SIZE_M * GRID_SCALE  # 100×100

MARGIN         = 12   # отступ от краёв арены
APPROACH_DIST  = 10   # расстояние точек подхода к воротам
POLE_BYPASS    = 8    # расстояние точки объезда от столбика
CLEARANCE      = 3    # буфер между элементами
HEATMAP_SIGMA  = 3    # размытие гауссовских блобов в пикселях


def get_fixed_points(grid_size=DEFAULT_GRID):
    """Фиксированные точки входа и выхода."""
    start = (MARGIN, MARGIN)
    end   = (grid_size - MARGIN - 1, grid_size - MARGIN - 1)
    return start, end


# ─── Вспомогательные функции ──────────────────────────────────────────────────

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def _is_clear(grid, cells, buf=CLEARANCE):
    size = grid.shape[0]
    for r, c in cells:
        for dr in range(-buf, buf + 1):
            for dc in range(-buf, buf + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < size and 0 <= nc < size:
                    if grid[nr, nc] != EMPTY:
                        return False
    return True


def make_heatmap(waypoints, grid_size, sigma=HEATMAP_SIGMA):
    """
    Создаёт тепловую карту: единичные импульсы в точках waypoints,
    размытые гауссовским фильтром.
    Значения в [0, 1].
    """
    heatmap = np.zeros((grid_size, grid_size), dtype=np.float32)
    for r, c in waypoints:
        r, c = int(r), int(c)
        if 0 <= r < grid_size and 0 <= c < grid_size:
            heatmap[r, c] = 1.0
    return gaussian_filter(heatmap, sigma=sigma)


# ─── Размещение элементов ─────────────────────────────────────────────────────

def place_gate(grid, r, c, horizontal=True):
    """
    Размещает ворота (5 клеток).
    Возвращает (ok, waypoints_group):
      waypoints_group = [entry, center, exit] — точки перпендикулярного прохода.
    """
    size = grid.shape[0]

    if horizontal:
        if c + 5 > size - MARGIN:
            return False, []
        cells  = [(r, c + i) for i in range(5)]
        center = (r, c + 2)
        entry  = (_clamp(r - APPROACH_DIST, MARGIN, size - MARGIN), c + 2)
        exit_  = (_clamp(r + APPROACH_DIST, MARGIN, size - MARGIN), c + 2)
    else:
        if r + 5 > size - MARGIN:
            return False, []
        cells  = [(r + i, c) for i in range(5)]
        center = (r + 2, c)
        entry  = (r + 2, _clamp(c - APPROACH_DIST, MARGIN, size - MARGIN))
        exit_  = (r + 2, _clamp(c + APPROACH_DIST, MARGIN, size - MARGIN))

    if not _is_clear(grid, cells):
        return False, []

    for row, col in cells:
        grid[row, col] = GATE

    # Случайно выбираем направление прохода
    group = [entry, center, exit_] if random.random() > 0.5 else [exit_, center, entry]
    return True, group


def place_ring(grid, r, c):
    """Размещает кольцо. Waypoint — центр."""
    size = grid.shape[0]
    if not (MARGIN <= r < size - MARGIN and MARGIN <= c < size - MARGIN):
        return False, []
    if not _is_clear(grid, [(r, c)]):
        return False, []
    grid[r, c] = RING
    return True, [(r, c)]


def place_pole(grid, r, c):
    """
    Размещает столбик. Waypoint — точка рядом со столбиком.
    Сам столбик непроходим → BFS огибает его.
    """
    size = grid.shape[0]
    if not (MARGIN <= r < size - MARGIN and MARGIN <= c < size - MARGIN):
        return False, []
    if not _is_clear(grid, [(r, c)]):
        return False, []
    grid[r, c] = POLE

    options = [
        (r, _clamp(c + POLE_BYPASS, MARGIN, size - MARGIN)),
        (r, _clamp(c - POLE_BYPASS, MARGIN, size - MARGIN)),
        (_clamp(r + POLE_BYPASS, MARGIN, size - MARGIN), c),
        (_clamp(r - POLE_BYPASS, MARGIN, size - MARGIN), c),
    ]
    bypass = random.choice(options)
    return True, [bypass]


# ─── Генерация карты ──────────────────────────────────────────────────────────

def generate_elements(grid_size=DEFAULT_GRID, n_gates=2, n_rings=2, n_poles=3, seed=None):
    """
    Расставляет элементы. Возвращает (grid, groups):
      groups — список групп waypoints, каждая группа = [wp1, wp2, ...]
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    grid   = np.zeros((grid_size, grid_size), dtype=np.int8)
    groups = []

    for _ in range(500):
        if sum(len(g) > 0 for g in groups if g) >= n_gates:
            break
        r = random.randint(MARGIN, grid_size - MARGIN - 5)
        c = random.randint(MARGIN, grid_size - MARGIN - 5)
        ok, wps = place_gate(grid, r, c, random.random() > 0.5)
        if ok:
            groups.append(wps)

    for _ in range(500):
        if sum(1 for g in groups if len(g) == 1 and grid[g[0][0], g[0][1]] == RING) >= n_rings:
            break
        r = random.randint(MARGIN, grid_size - MARGIN)
        c = random.randint(MARGIN, grid_size - MARGIN)
        ok, wps = place_ring(grid, r, c)
        if ok:
            groups.append(wps)

    for _ in range(500):
        pole_count = int(np.sum(grid == POLE))
        if pole_count >= n_poles:
            break
        r = random.randint(MARGIN, grid_size - MARGIN)
        c = random.randint(MARGIN, grid_size - MARGIN)
        ok, wps = place_pole(grid, r, c)
        if ok:
            groups.append(wps)

    return grid, groups


# ─── BFS ──────────────────────────────────────────────────────────────────────

def bfs_path(grid, start, end):
    """BFS: кратчайший путь от start до end. POLE непроходимы."""
    size = grid.shape[0]
    if not (0 <= start[0] < size and 0 <= start[1] < size):
        return None
    if not (0 <= end[0] < size and 0 <= end[1] < size):
        return None

    queue   = deque([(start, [start])])
    visited = {start}

    while queue:
        (r, c), path = queue.popleft()
        if (r, c) == end:
            return path
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if (0 <= nr < size and 0 <= nc < size
                    and (nr, nc) not in visited
                    and grid[nr, nc] != POLE):
                visited.add((nr, nc))
                queue.append(((nr, nc), path + [(nr, nc)]))
    return None


def build_path(grid, groups, grid_size=DEFAULT_GRID):
    """
    Строит путь: start → группа1 → группа2 → ... → end.
    Start и end фиксированы (верхний левый / правый нижний углы).
    Порядок групп случайный, внутри группы — фиксированный.
    Возвращает path-матрицу или None.
    """
    start, end = get_fixed_points(grid_size)
    target = grid.copy()

    shuffled = groups.copy()
    random.shuffle(shuffled)
    checkpoints = [start] + [wp for g in shuffled for wp in g] + [end]

    full_path = []
    for i in range(len(checkpoints) - 1):
        seg = bfs_path(target, checkpoints[i], checkpoints[i + 1])
        if seg is None:
            return None
        if full_path:
            seg = seg[1:]
        full_path.extend(seg)

    for r, c in full_path:
        if target[r, c] == EMPTY:
            target[r, c] = PATH

    return target


# ─── Генерация датасета ───────────────────────────────────────────────────────

def generate_dataset(n_samples=500, grid_size=DEFAULT_GRID,
                     n_gates=2, n_rings=2, n_poles=3, seed=42):
    """
    Генерирует датасет.

    Возвращает:
      inputs   (N, H, W) int8   — карты с элементами (без пути)
      heatmaps (N, H, W) float32 — тепловые карты waypoints (target для модели)
      paths    (N, H, W) int8   — полные BFS-пути (для визуализации)
    """
    inputs   = []
    heatmaps = []
    paths    = []

    generated = 0
    attempts  = 0

    while generated < n_samples:
        attempts += 1
        grid, groups = generate_elements(
            grid_size=grid_size,
            n_gates=n_gates, n_rings=n_rings, n_poles=n_poles,
            seed=seed + attempts,
        )
        if not groups:
            continue

        path_grid = build_path(grid, groups, grid_size=grid_size)
        if path_grid is None:
            continue

        # Собираем все промежуточные waypoints (без start/end — они фиксированы)
        all_waypoints = [wp for group in groups for wp in group]
        heatmap = make_heatmap(all_waypoints, grid_size)

        inputs.append(grid)
        heatmaps.append(heatmap)
        paths.append(path_grid)
        generated += 1

        if generated % 100 == 0:
            print(f"  Сгенерировано: {generated}/{n_samples} (попыток: {attempts})")

    return np.array(inputs), np.array(heatmaps), np.array(paths)


# ─── Запуск ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import ListedColormap

    os.makedirs("data", exist_ok=True)
    print("Генерация 5 примеров для проверки...")
    inputs, heatmaps, paths = generate_dataset(n_samples=5, seed=0)

    COLORS = ["#1a1a2e", "#00d4ff", "#ff6b35", "#7bc67e", "#e63946"]
    cmap   = ListedColormap(COLORS)
    start, end = get_fixed_points()

    fig, axes = plt.subplots(5, 3, figsize=(12, 18))
    fig.suptitle("Датасет v2: вход / тепловая карта / полный путь", fontsize=13)

    for i in range(5):
        axes[i, 0].imshow(inputs[i],   cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
        axes[i, 0].scatter([start[1], end[1]], [start[0], end[0]],
                           c=["lime", "yellow"], s=60, zorder=5)
        axes[i, 0].set_title(f"Вход #{i+1}")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(heatmaps[i], cmap="hot", vmin=0, vmax=1, interpolation="nearest")
        axes[i, 1].set_title("Heatmap (target)")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(paths[i],    cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
        axes[i, 2].set_title("Полный BFS-путь")
        axes[i, 2].axis("off")

    patches = [mpatches.Patch(color=COLORS[k], label=l)
               for k, l in enumerate(["Пусто","Путь","Ворота","Кольцо","Столбик"])]
    fig.legend(handles=patches, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.01))
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig("data/viz_v2_check.png", dpi=110, bbox_inches="tight")
    print("Сохранено: data/viz_v2_check.png")
    plt.show()
