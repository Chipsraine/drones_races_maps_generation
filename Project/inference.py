"""
Инференс: предсказанная тепловая карта → полный путь дрона шириной 1 клетка.

Pipeline:
  1. U-Net предсказывает тепловую карту waypoints
  2. scipy.ndimage находит локальные максимумы (пики)
  3. Жадный алгоритм упорядочивает пики (nearest-neighbour от start)
  4. BFS соединяет точки → путь гарантированно шириной 1 клетка

Запуск:
    uv run python Project/inference.py
"""

import numpy as np
import torch
from scipy.ndimage import maximum_filter, label

from model import TrackUNet, grid_to_tensor
from dataset_generator import bfs_path, get_fixed_points, POLE, EMPTY, PATH


# ─── Пиковый детектор ─────────────────────────────────────────────────────────

def find_peaks(heatmap: np.ndarray, min_distance: int = 8, threshold: float = 0.15):
    """
    Находит локальные максимумы на тепловой карте.

    Параметры:
        heatmap      — (H, W) float, значения [0, 1]
        min_distance — минимальное расстояние между пиками в пикселях
        threshold    — минимальное значение в пике (доля от максимума)

    Возвращает список (row, col) отсортированный по убыванию значения.
    """
    if heatmap.max() < 1e-6:
        return []

    # Локальные максимумы: пиксель = максимум в окне min_distance×min_distance
    footprint = np.ones((min_distance, min_distance), dtype=bool)
    local_max = maximum_filter(heatmap, footprint=footprint) == heatmap

    # Порог
    abs_threshold = threshold * heatmap.max()
    local_max &= (heatmap >= abs_threshold)

    # Координаты пиков
    rows, cols = np.where(local_max)
    if len(rows) == 0:
        return []

    peaks = list(zip(rows.tolist(), cols.tolist()))
    # Сортируем по убыванию значения
    peaks.sort(key=lambda p: heatmap[p[0], p[1]], reverse=True)
    return peaks


# ─── Упорядочивание пиков ─────────────────────────────────────────────────────

def order_peaks_greedy(peaks, start, end):
    """
    Упорядочивает пики жадным алгоритмом: всегда берём ближайший непосещённый.
    Начинаем от start, заканчиваем у end.
    """
    remaining = list(peaks)
    ordered   = []
    current   = start

    while remaining:
        dists = [abs(current[0] - p[0]) + abs(current[1] - p[1]) for p in remaining]
        nearest_idx = int(np.argmin(dists))
        ordered.append(remaining.pop(nearest_idx))
        current = ordered[-1]

    return ordered


# ─── Основной пайплайн ────────────────────────────────────────────────────────

def heatmap_to_path(grid: np.ndarray, heatmap: np.ndarray,
                    min_distance: int = 8, threshold: float = 0.15):
    """
    Полный пайплайн: тепловая карта → 1-клеточный путь.

    Параметры:
        grid     — (H, W) int8, карта с элементами (без пути)
        heatmap  — (H, W) float32, предсказанная тепловая карта

    Возвращает:
        path_grid — (H, W) int8, карта с проложенным путём (PATH=1)
        success   — bool, удалось ли проложить путь
    """
    grid_size = grid.shape[0]
    start, end = get_fixed_points(grid_size)

    peaks   = find_peaks(heatmap, min_distance=min_distance, threshold=threshold)
    ordered = order_peaks_greedy(peaks, start, end)

    checkpoints = [start] + ordered + [end]
    path_grid   = grid.copy()

    full_path = []
    for i in range(len(checkpoints) - 1):
        seg = bfs_path(path_grid, checkpoints[i], checkpoints[i + 1])
        if seg is None:
            # Путь недостижим — пробуем напрямую без waypoints
            seg = bfs_path(path_grid, checkpoints[i], checkpoints[-1])
            if seg is None:
                return path_grid, False
            full_path.extend(seg[1:] if full_path else seg)
            break
        if full_path:
            seg = seg[1:]
        full_path.extend(seg)

    for r, c in full_path:
        if path_grid[r, c] == EMPTY:
            path_grid[r, c] = PATH

    return path_grid, True


def predict_path(model: TrackUNet, grid: np.ndarray, device="cpu",
                 min_distance: int = 8, threshold: float = 0.15):
    """
    Полный пайплайн с нейросетью:
      grid → one-hot → U-Net → heatmap → peaks → BFS → path_grid

    Возвращает:
        pred_heatmap — (H, W) float32, предсказанная тепловая карта
        path_grid    — (H, W) int8, восстановленный путь
        success      — bool
    """
    model.eval()
    x = grid_to_tensor(grid).unsqueeze(0).to(device)   # (1, 5, H, W)
    with torch.no_grad():
        logits = model(x)                               # (1, 1, H, W)
    pred_heatmap = torch.sigmoid(logits).squeeze().cpu().numpy()   # (H, W)

    path_grid, success = heatmap_to_path(grid, pred_heatmap,
                                         min_distance=min_distance,
                                         threshold=threshold)
    return pred_heatmap, path_grid, success


# ─── Быстрая проверка ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import ListedColormap

    print("Загрузка данных и модели...")
    inputs   = np.load("data/inputs.npy")
    heatmaps = np.load("data/heatmaps.npy")
    paths    = np.load("data/paths.npy")

    model = TrackUNet(n_classes_in=5, base_features=32)
    model.load_state_dict(torch.load("models/best_model.pt", map_location="cpu", weights_only=True))

    COLORS = ["#1a1a2e", "#00d4ff", "#ff6b35", "#7bc67e", "#e63946"]
    cmap   = ListedColormap(COLORS)

    n = 4
    fig, axes = plt.subplots(n, 5, figsize=(20, n * 4))
    col_titles = ["Вход", "Heatmap (target)", "Predicted Heatmap", "Путь (BFS)", "Ground Truth"]
    for j, t in enumerate(col_titles):
        axes[0, j].set_title(t, fontsize=10)

    for i in range(n):
        pred_heatmap, path_grid, ok = predict_path(model, inputs[i])
        print(f"Пример {i+1}: путь {'проложен' if ok else 'НЕ ПРОЛОЖЕН'}")

        axes[i, 0].imshow(inputs[i],   cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(heatmaps[i], cmap="hot", vmin=0, interpolation="nearest")
        axes[i, 1].axis("off")

        axes[i, 2].imshow(pred_heatmap, cmap="hot", vmin=0, interpolation="nearest")
        axes[i, 2].axis("off")

        axes[i, 3].imshow(path_grid,   cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
        axes[i, 3].axis("off")

        axes[i, 4].imshow(paths[i],    cmap=cmap, vmin=0, vmax=4, interpolation="nearest")
        axes[i, 4].axis("off")

    patches = [mpatches.Patch(color=COLORS[k], label=l)
               for k, l in enumerate(["Пусто", "Путь", "Ворота", "Кольцо", "Столбик"])]
    fig.legend(handles=patches, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.01))
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig("data/viz_inference.png", dpi=110, bbox_inches="tight")
    print("Сохранено: data/viz_inference.png")
    plt.show()
