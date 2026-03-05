"""
Визуализация датасета и предсказаний модели.

Запуск:
    uv run python Project/visualize.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from matplotlib.colors import ListedColormap

from model import TrackUNet, grid_to_tensor
from inference import predict_path
from dataset_generator import get_fixed_points


# ─── Цвета и легенда ──────────────────────────────────────────────────────────

COLORS = ["#1a1a2e", "#00d4ff", "#ff6b35", "#7bc67e", "#e63946"]
LABELS = ["Пусто", "Путь дрона", "Ворота", "Кольцо", "Столбик"]
CMAP   = ListedColormap(COLORS)


def _add_legend(fig):
    patches = [mpatches.Patch(color=COLORS[k], label=LABELS[k]) for k in range(5)]
    fig.legend(handles=patches, loc="lower center", ncol=5,
               fontsize=9, framealpha=0.9, bbox_to_anchor=(0.5, 0.01))


def _plot_grid(ax, grid, title=""):
    ax.imshow(grid, cmap=CMAP, vmin=0, vmax=4, interpolation="nearest")
    if title:
        ax.set_title(title, fontsize=10, pad=6)
    ax.axis("off")


def _plot_heatmap(ax, hmap, title=""):
    ax.imshow(hmap, cmap="hot", vmin=0, interpolation="nearest")
    if title:
        ax.set_title(title, fontsize=10, pad=6)
    ax.axis("off")


# ─── 1. Примеры из датасета ───────────────────────────────────────────────────

def show_dataset_samples(n=4):
    """Показывает n примеров: вход / тепловая карта waypoints / полный BFS-путь."""
    inputs   = np.load("data/inputs.npy")
    heatmaps = np.load("data/heatmaps.npy")
    paths    = np.load("data/paths.npy")

    start, end = get_fixed_points()

    fig, axes = plt.subplots(n, 3, figsize=(12, n * 3.5))
    fig.suptitle("Примеры из датасета", fontsize=14, y=1.01)

    col_titles = ["Вход (элементы)", "Heatmap waypoints (target)", "BFS-путь (визуализация)"]
    for j, t in enumerate(col_titles):
        axes[0, j].set_title(t, fontsize=10, pad=6)

    for i in range(n):
        _plot_grid(axes[i, 0], inputs[i])
        axes[i, 0].scatter([start[1], end[1]], [start[0], end[0]],
                           c=["lime", "yellow"], s=60, zorder=5)
        axes[i, 0].set_ylabel(f"#{i+1}", fontsize=10, rotation=0, labelpad=20, va="center")

        _plot_heatmap(axes[i, 1], heatmaps[i])
        _plot_grid(axes[i, 2], paths[i])

    _add_legend(fig)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig("data/viz_dataset.png", dpi=120, bbox_inches="tight")
    print("Сохранено: data/viz_dataset.png")
    plt.show()


# ─── 2. Предсказания модели ───────────────────────────────────────────────────

def show_predictions(n=4):
    """
    5-панельная визуализация:
    Вход | Heatmap target | Predicted Heatmap | Восстановленный путь | GT путь
    """
    inputs   = np.load("data/inputs.npy")
    heatmaps = np.load("data/heatmaps.npy")
    paths    = np.load("data/paths.npy")

    model = TrackUNet(n_classes_in=5, base_features=32)
    model.load_state_dict(torch.load("models/best_model.pt", map_location="cpu", weights_only=True))

    fig, axes = plt.subplots(n, 5, figsize=(20, n * 3.5))
    fig.suptitle("Предсказания модели", fontsize=14, y=1.01)

    col_titles = ["Вход", "Heatmap (target)", "Predicted Heatmap", "Путь (BFS)", "GT путь"]
    for j, t in enumerate(col_titles):
        axes[0, j].set_title(t, fontsize=10, pad=6)

    for i in range(n):
        pred_heatmap, path_grid, ok = predict_path(model, inputs[i])

        # Cosine similarity тепловых карт
        ph_flat = pred_heatmap.flatten()
        th_flat = heatmaps[i].flatten()
        norm = np.linalg.norm(ph_flat) * np.linalg.norm(th_flat)
        sim  = float(np.dot(ph_flat, th_flat) / norm) if norm > 1e-6 else 0.0

        _plot_grid(axes[i, 0], inputs[i], f"Пример {i+1}")
        _plot_heatmap(axes[i, 1], heatmaps[i])
        _plot_heatmap(axes[i, 2], pred_heatmap, f"sim={sim:.2f}")
        _plot_grid(axes[i, 3], path_grid, "OK" if ok else "FAIL")
        _plot_grid(axes[i, 4], paths[i])

    _add_legend(fig)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig("data/viz_predictions.png", dpi=120, bbox_inches="tight")
    print("Сохранено: data/viz_predictions.png")
    plt.show()


# ─── 3. Детальный анализ одного примера ──────────────────────────────────────

def show_single(idx=0):
    """5-панельный детальный анализ одного примера."""
    inputs   = np.load("data/inputs.npy")
    heatmaps = np.load("data/heatmaps.npy")
    paths    = np.load("data/paths.npy")

    model = TrackUNet(n_classes_in=5, base_features=32)
    model.load_state_dict(torch.load("models/best_model.pt", map_location="cpu", weights_only=True))

    pred_heatmap, path_grid, ok = predict_path(model, inputs[idx])

    # Сравнение путей: GT путь vs восстановленный
    gt_path   = (paths[idx] == 1)
    pred_path = (path_grid == 1)
    intersection = (gt_path & pred_path).sum()
    union        = (gt_path | pred_path).sum()
    iou = intersection / (union + 1e-6)

    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    fig.suptitle(f"Пример #{idx} — детальный анализ", fontsize=14)

    _plot_grid(axes[0],     inputs[idx],    "Вход (элементы)")
    _plot_heatmap(axes[1],  heatmaps[idx],  "Heatmap target")
    _plot_heatmap(axes[2],  pred_heatmap,   "Predicted Heatmap")
    _plot_grid(axes[3],     path_grid,      f"Восст. путь (IoU={iou:.3f})")
    _plot_grid(axes[4],     paths[idx],     "GT путь")

    _add_legend(fig)
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(f"data/viz_single_{idx}.png", dpi=120, bbox_inches="tight")
    print(f"Сохранено: data/viz_single_{idx}.png  |  IoU={iou:.3f}  |  path={'OK' if ok else 'FAIL'}")
    plt.show()


# ─── Запуск ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== 1. Примеры из датасета ===")
    show_dataset_samples(n=4)

    print("\n=== 2. Предсказания модели ===")
    show_predictions(n=4)

    print("\n=== 3. Детальный анализ ===")
    show_single(idx=0)
    show_single(idx=1)
