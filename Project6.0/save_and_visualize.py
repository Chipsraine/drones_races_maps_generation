"""
Сохранение весов и визуализация для конкретных конфигураций.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

from environment_with_flags import GateEnvironmentWithFlags, ARENA_SIZE, MAX_GATES, MAX_FLAGS, GATE_SIZE
from agent import PolicyNetwork

SAVE_DIR = Path(__file__).parent / "models"
VIZ_DIR = Path(__file__).parent / "data"


def draw_gate(ax, x, y, angle, color="blue", lw=2.5):
    dx = (GATE_SIZE / 2) * np.cos(angle)
    dy = (GATE_SIZE / 2) * np.sin(angle)
    ax.plot([x - dx, x + dx], [y - dy, y + dy], color=color, linewidth=lw, solid_capstyle="round")
    nx = 0.3 * np.cos(angle + np.pi / 2)
    ny = 0.3 * np.sin(angle + np.pi / 2)
    ax.annotate("", xy=(x + nx, y + ny), xytext=(x, y),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5))


def visualize_config(config, flags, valid, reward, title, save_path):
    """Рисует одну трассу."""
    n = len(config)
    nf = len(flags) if flags is not None else 0
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    cmap = plt.cm.tab10
    
    ax.set_xlim(0, ARENA_SIZE)
    ax.set_ylim(0, ARENA_SIZE)
    ax.set_aspect("equal")
    ax.set_xlabel("X (м)")
    ax.set_ylabel("Y (м)")
    
    # Арена
    ax.add_patch(patches.Rectangle((0, 0), ARENA_SIZE, ARENA_SIZE,
                                    linewidth=2, edgecolor="black",
                                    facecolor="lightyellow", alpha=0.15))
    
    # Маршрут
    for i in range(n):
        j = (i + 1) % n
        ax.plot([config[i, 0], config[j, 0]], [config[i, 1], config[j, 1]],
                color="gray", linewidth=1.5, linestyle="--", alpha=0.5)
        d = np.hypot(config[j, 0] - config[i, 0], config[j, 1] - config[i, 1])
        mx = (config[i, 0] + config[j, 0]) / 2
        my = (config[i, 1] + config[j, 1]) / 2
        ax.text(mx, my, f"{d:.1f}м", fontsize=8, ha="center", color="gray")
    
    # Ворота
    for i, (x, y, a) in enumerate(config):
        color = cmap(i % 10)
        draw_gate(ax, x, y, a, color=color)
        ax.text(x, y + 0.5, str(i + 1), fontsize=10, ha="center",
                fontweight="bold", color=color)
    
    # Флаги
    if nf > 0:
        for fi, (fx, fy) in enumerate(flags):
            ax.plot(fx, fy, 'ro', markersize=10, markeredgecolor='darkred', markeredgewidth=1.5)
            ax.text(fx, fy + 0.4, f"F{fi+1}", fontsize=9, ha="center",
                    fontweight="bold", color="red")
    
    status = "VALID" if valid else "INVALID"
    color_title = "green" if valid else "red"
    ax.set_title(f"{title}\n[{status}] Reward={reward:.1f}", fontsize=12, color=color_title)
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Сохранено: {save_path}")


def generate_and_visualize(policy, env, n_gates, n_flags, n_samples=10):
    """Генерирует n_samples трасс и визуализирует валидные."""
    policy.eval()
    valid_count = 0
    
    with torch.no_grad():
        for idx in range(n_samples):
            state = env.reset(n_gates, n_flags)
            action, _, _, _, _ = policy.select_action(state)
            env.step(action)
            
            config = env.get_config()
            flags = env.get_flags()
            valid = env.is_valid()
            reward, _ = env._evaluate_config()
            
            # Добавляем бонус за валидность как в step()
            n_violations = len([v for v in [reward] if False])  # заглушка, reward уже с бонусом
            # Пересчитаем чистый reward
            reward, info = env._evaluate_config()
            n_violations = len(info["violations"])
            if n_violations == 0:
                reward += 200.0
            
            if valid:
                valid_count += 1
                title = f"{n_gates}в/{n_flags}ф — Валидная #{valid_count}"
                save_path = VIZ_DIR / f"valid_{n_gates}g{n_flags}f_{valid_count}.png"
                visualize_config(config, flags, valid, reward, title, save_path)
    
    print(f"\n{n_gates} ворот, {n_flags} флагов: {valid_count}/{n_samples} валидных ({valid_count*10}%)")
    return valid_count


def main():
    VIZ_DIR.mkdir(parents=True, exist_ok=True)
    
    env = GateEnvironmentWithFlags()
    policy = PolicyNetwork()
    
    # === ВАРИАНТ 1: 3 ворота, 0 флагов (итерация 1000) ===
    print("=" * 50)
    print("ВАРИАНТ 1: 3 ворота, 0 флагов")
    print("=" * 50)
    
    # Загружаем веса с 1000-й итерации (если есть best_model.pt)
    # Или используем текущие обученные веса
    checkpoint_3g = SAVE_DIR / "best_model.pt"
    if checkpoint_3g.exists():
        policy.load_state_dict(torch.load(checkpoint_3g))
        print(f"Загружены веса: {checkpoint_3g}")
    else:
        print("Веса не найдены, используем случайные (нужно обучить!)")
        return
    
    # Визуализация 3 ворот без флагов
    generate_and_visualize(policy, env, n_gates=3, n_flags=0, n_samples=20)
    
    # Визуализация 3 ворот с 1 флагом
    generate_and_visualize(policy, env, n_gates=3, n_flags=1, n_samples=20)
    
    # === ВАРИАНТ 2: 4 ворота, 4 флага (итерация 4000) ===
    print("\n" + "=" * 50)
    print("ВАРИАНТ 2: 4 ворота, 4 флага")
    print("=" * 50)
    
    # Тут нужно будет загрузить веса с 4000-й итерации
    # Пока используем те же — для демонстрации
    generate_and_visualize(policy, env, n_gates=4, n_flags=2, n_samples=20)
    generate_and_visualize(policy, env, n_gates=4, n_flags=4, n_samples=20)
    
    print("\nГотово! Все изображения в папке:", VIZ_DIR)


if __name__ == "__main__":
    main()