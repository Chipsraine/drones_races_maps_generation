"""
Визуализация конфигураций ворот для дрон-рейсинга.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

from gate_generator import ARENA_SIZE, MARGIN, GATE_SIZE


def draw_gate(ax, x: float, y: float, angle: float, color: str = "blue", lw: float = 2.5):
    """Рисует ворота как отрезок длиной GATE_SIZE с центром в (x, y) под углом angle."""
    dx = (GATE_SIZE / 2) * np.cos(angle)
    dy = (GATE_SIZE / 2) * np.sin(angle)
    ax.plot([x - dx, x + dx], [y - dy, y + dy], color=color, linewidth=lw, solid_capstyle="round")
    # Стрелка направления (перпендикуляр)
    nx = 0.3 * np.cos(angle + np.pi / 2)
    ny = 0.3 * np.sin(angle + np.pi / 2)
    ax.annotate("", xy=(x + nx, y + ny), xytext=(x, y),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5))


def draw_config(ax, config: np.ndarray, title: str = "", show_distances: bool = True):
    """
    Отрисовка одной конфигурации ворот на арене.

    Args:
        config: shape (N, 3) — [x, y, angle]
    """
    # Арена
    ax.set_xlim(0, ARENA_SIZE)
    ax.set_ylim(0, ARENA_SIZE)
    ax.set_aspect("equal")
    ax.set_xlabel("X (м)")
    ax.set_ylabel("Y (м)")

    # Рабочая зона
    work_rect = patches.Rectangle(
        (MARGIN, MARGIN),
        ARENA_SIZE - 2 * MARGIN,
        ARENA_SIZE - 2 * MARGIN,
        linewidth=1.5, edgecolor="green", facecolor="lightgreen", alpha=0.15,
        linestyle="--", label="Рабочая зона"
    )
    ax.add_patch(work_rect)

    # Граница арены
    arena_rect = patches.Rectangle(
        (0, 0), ARENA_SIZE, ARENA_SIZE,
        linewidth=2, edgecolor="black", facecolor="none"
    )
    ax.add_patch(arena_rect)

    n = len(config)
    cmap = plt.cm.tab10

    # Линии маршрута (замкнутый)
    for i in range(n):
        j = (i + 1) % n
        ax.plot(
            [config[i, 0], config[j, 0]],
            [config[i, 1], config[j, 1]],
            color="gray", linewidth=1, linestyle="--", alpha=0.5, zorder=1
        )

        if show_distances:
            d = np.hypot(config[j, 0] - config[i, 0], config[j, 1] - config[i, 1])
            mx = (config[i, 0] + config[j, 0]) / 2
            my = (config[i, 1] + config[j, 1]) / 2
            ax.text(mx, my, f"{d:.1f}м", fontsize=7, ha="center", va="bottom",
                    color="gray", alpha=0.8)

    # Ворота
    for i, (x, y, a) in enumerate(config):
        color = cmap(i % 10)
        draw_gate(ax, x, y, a, color=color)
        ax.text(x, y + 0.4, str(i + 1), fontsize=9, ha="center", va="bottom",
                fontweight="bold", color=color)

    if title:
        ax.set_title(title, fontsize=11)


def visualize_samples(configs: list[np.ndarray], n_show: int = 6,
                      save_path: str | Path | None = None):
    """Показывает несколько примеров конфигураций."""
    n_show = min(n_show, len(configs))
    cols = 3
    rows = (n_show + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
    axes = np.atleast_2d(axes)

    for idx in range(n_show):
        ax = axes[idx // cols, idx % cols]
        draw_config(ax, configs[idx], title=f"Конфигурация #{idx + 1} ({len(configs[idx])} ворот)")

    # Скрыть пустые
    for idx in range(n_show, rows * cols):
        axes[idx // cols, idx % cols].set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Сохранено: {save_path}")
    plt.show()


def visualize_comparison(real: np.ndarray, generated: np.ndarray,
                         save_path: str | Path | None = None):
    """Сравнивает реальную и сгенерированную конфигурации."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    draw_config(ax1, real, title=f"Ground Truth ({len(real)} ворот)")
    draw_config(ax2, generated, title=f"Сгенерированная ({len(generated)} ворот)")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Сохранено: {save_path}")
    plt.show()


if __name__ == "__main__":
    from gate_generator import load_dataset

    data_path = Path(__file__).parent / "data" / "gate_configs.npz"
    if data_path.exists():
        configs = load_dataset(data_path)
        visualize_samples(
            configs[:6],
            save_path=Path(__file__).parent / "data" / "viz_samples.png"
        )
    else:
        print(f"Датасет не найден: {data_path}")
        print("Сначала запустите gate_generator.py")
