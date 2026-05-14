"""
Красивая визуализация трасс для отчёта.
3 ворота + 0-3 флага, обученная модель.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from matplotlib.patches import FancyBboxPatch

from environment_with_flags import GateEnvironmentWithFlags, ARENA_SIZE, WORK_MIN, WORK_RANGE, MAX_GATES, MAX_FLAGS, GATE_SIZE
from agent import PolicyNetwork

SAVE_DIR = Path("models")
VIZ_DIR = Path("data")
VIZ_DIR.mkdir(exist_ok=True)


def generate_flags(gates, n_flags):
    """Жадная генерация флагов — середина отрезка."""
    flags = []
    for i in range(n_flags):
        g1 = np.array(gates[i][:2])
        g2 = np.array(gates[(i + 1) % len(gates)][:2])
        mid = (g1 + g2) / 2
        noise = np.random.uniform(-0.2, 0.2, 2)
        flag = mid + noise
        
        dx, dy = g2[0] - g1[0], g2[1] - g1[1]
        length = np.hypot(dx, dy)
        if length > 0:
            t = ((flag[0] - g1[0]) * dx + (flag[1] - g1[1]) * dy) / (length ** 2)
            t = np.clip(t, 0.15, 0.85)
            proj_x = g1[0] + t * dx
            proj_y = g1[1] + t * dy
            perp_x = -dy / length
            perp_y = dx / length
            offset = np.random.uniform(-0.2, 0.2)
            flag = np.array([proj_x + offset * perp_x, proj_y + offset * perp_y])
        
        flags.append(flag)
    return np.array(flags)


def draw_gate(ax, x, y, angle, color, label, lw=3):
    """Рисует ворота со стрелкой направления."""
    dx = (GATE_SIZE / 2) * np.cos(angle)
    dy = (GATE_SIZE / 2) * np.sin(angle)
    
    # Ворота
    ax.plot([x - dx, x + dx], [y - dy, y + dy], 
            color=color, linewidth=lw, solid_capstyle="round", zorder=5)
    
    # Стрелка направления
    nx = 0.4 * np.cos(angle + np.pi / 2)
    ny = 0.4 * np.sin(angle + np.pi / 2)
    ax.annotate("", xy=(x + nx, y + ny), xytext=(x, y),
                arrowprops=dict(arrowstyle="->", color=color, lw=2), zorder=5)
    
    # Номер
    ax.text(x, y + 0.6, label, fontsize=12, ha="center",
            fontweight="bold", color=color, zorder=6,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor=color, alpha=0.9))


def draw_flag(ax, x, y, label, color="red"):
    """Рисует флаг как круг с подписью."""
    circle = plt.Circle((x, y), 0.25, color=color, ec="darkred", linewidth=2, zorder=5)
    ax.add_patch(circle)
    ax.text(x, y + 0.45, label, fontsize=10, ha="center",
            fontweight="bold", color="darkred", zorder=6)


def visualize_track(config, flags, valid, reward, title, save_path):
    """Рисует одну трассу на красивом фоне."""
    n = len(config)
    nf = len(flags) if flags is not None else 0
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Фон арены
    ax.set_facecolor("#f5f5f0")
    ax.add_patch(patches.Rectangle((0, 0), ARENA_SIZE, ARENA_SIZE,
                                    linewidth=3, edgecolor="#333333",
                                    facecolor="#fafaf5", alpha=0.8, zorder=1))
    
    # Сетка
    for i in range(11):
        ax.axhline(i, color="#dddddd", linewidth=0.5, zorder=1)
        ax.axvline(i, color="#dddddd", linewidth=0.5, zorder=1)
    
    # Траектория (сплошная линия между воротами)
    for i in range(n):
        j = (i + 1) % n
        ax.plot([config[i, 0], config[j, 0]], [config[i, 1], config[j, 1]],
                color="#4488cc", linewidth=2.5, linestyle="-", alpha=0.7, zorder=2)
    
    # Траектория через флаги (пунктир)
    if nf > 0:
        for i in range(nf):
            g1 = config[i]
            g2 = config[(i + 1) % n]
            flag = flags[i]
            
            # Отрезок gate_i → flag → gate_{i+1}
            ax.plot([g1[0], flag[0]], [g1[1], flag[1]],
                    color="#66aadd", linewidth=1.5, linestyle="--", alpha=0.5, zorder=2)
            ax.plot([flag[0], g2[0]], [flag[1], g2[1]],
                    color="#66aadd", linewidth=1.5, linestyle="--", alpha=0.5, zorder=2)
    
    # Ворота
    colors = ["#e74c3c", "#2ecc71", "#3498db", "#9b59b6", "#f39c12", "#1abc9c"]
    for i, (x, y, a) in enumerate(config):
        draw_gate(ax, x, y, a, colors[i % len(colors)], f"G{i+1}")
    
    # Флаги
    for i, (fx, fy) in enumerate(flags):
        draw_flag(ax, fx, fy, f"F{i+1}")
    
    # Информация
    status = "✓ ВАЛИДНАЯ" if valid else "✗ НЕВАЛИДНАЯ"
    status_color = "#27ae60" if valid else "#e74c3c"
    
    info_text = f"{title}\n{status} | Reward: {reward:.1f}"
    ax.text(0.5, 9.5, info_text, fontsize=14, fontweight="bold",
            verticalalignment="top", horizontalalignment="left",
            bbox=dict(boxstyle="round,pad=0.5", facecolor=status_color, 
                     edgecolor="black", alpha=0.15, linewidth=2))
    
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 10.5)
    ax.set_aspect("equal")
    ax.set_xlabel("X, м", fontsize=12)
    ax.set_ylabel("Y, м", fontsize=12)
    ax.set_title("DroneTrack — Генерация трассы FPV-дрона", fontsize=16, fontweight="bold", pad=20)
    
    # Легенда
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="#4488cc", lw=2.5, label="Маршрут"),
        Line2D([0], [0], color="#66aadd", lw=1.5, linestyle="--", label="С флагом"),
        plt.Circle((0, 0), 0.1, color="red", label="Флаг"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=10)
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Сохранено: {save_path}")


def main():
    print("=" * 60)
    print("ВИЗУАЛИЗАЦИЯ: 3 ворота + флаги")
    print("=" * 60)
    
    env = GateEnvironmentWithFlags()
    policy = PolicyNetwork()
    
    # Загружаем обученные веса
    checkpoint = SAVE_DIR / "model_gates_greedy_flags.pt"
    if not checkpoint.exists():
        print(f"❌ Нет весов: {checkpoint}")
        return
    
    policy.load_state_dict(torch.load(checkpoint))
    policy.eval()
    print(f"✅ Загружены веса: {checkpoint}")
    
    # Генерируем по 3 примера для каждой конфигурации
    configs_to_test = [
        (3, 0, "3 ворота, без флагов"),
        (3, 1, "3 ворота, 1 флаг"),
        (3, 2, "3 ворота, 2 флага"),
        (3, 3, "3 ворота, 3 флага"),
    ]
    
    for n_gates, n_flags, desc in configs_to_test:
        print(f"\n{desc}:")
        
        generated = 0
        attempts = 0
        
        with torch.no_grad():
            while generated < 3 and attempts < 100:
                attempts += 1
                
                # Генерируем ворота
                state = env.reset(n_gates, n_flags)
                action_full, _, _, _, _ = policy.select_action(state)
                gates_action = action_full[:n_gates * 3]
                
                gates = []
                for i in range(n_gates):
                    x = gates_action[i * 3] * 10.0
                    y = gates_action[i * 3 + 1] * 10.0
                    a = gates_action[i * 3 + 2] * 2 * np.pi
                    gates.append((x, y, a))
                
                # Генерируем флаги
                if n_flags > 0:
                    flags = generate_flags(gates, n_flags)
                else:
                    flags = np.zeros((0, 2))
                
                # Собираем action
                action = np.zeros(MAX_GATES * 3 + MAX_FLAGS * 2, dtype=np.float32)
                for i, (x, y, a) in enumerate(gates):
                    action[i * 3] = (x - WORK_MIN) / WORK_RANGE
                    action[i * 3 + 1] = (y - WORK_MIN) / WORK_RANGE
                    action[i * 3 + 2] = a / (2 * np.pi)
                
                flag_offset = n_gates * 3
                for i, (fx, fy) in enumerate(flags):
                    action[flag_offset + i * 2] = (fx - WORK_MIN) / WORK_RANGE
                    action[flag_offset + i * 2 + 1] = (fy - WORK_MIN) / WORK_RANGE
                
                # Проверяем
                env.reset(n_gates, n_flags)
                _, reward, _, _ = env.step(action)
                valid = env.is_valid()
                
                if valid:
                    generated += 1
                    config = env.get_config()
                    title = f"{desc} — Пример {generated}"
                    save_path = VIZ_DIR / f"track_3g{n_flags}f_{generated}.png"
                    visualize_track(config, flags, valid, reward, title, save_path)
        
        print(f"  Сгенерировано {generated}/3 валидных (попыток: {attempts})")
    
    print(f"\n✅ Все изображения в папке: {VIZ_DIR}")
    print("Файлы:")
    for f in sorted(VIZ_DIR.glob("track_*.png")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()