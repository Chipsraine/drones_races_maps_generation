"""
Обучение RL-агента — подход "всё за один шаг".

Цикл обучения (значительно проще v2):
1. Агент получает число ворот → выдаёт ВСЕ позиции сразу
2. Среда оценивает конфигурацию → возвращает reward
3. PPO обновляет веса
4. Повторяем

Ключевое отличие: каждый эпизод = 1 шаг. Нет sequential decision-making.
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

from environment_simple import GateEnvironment, WORK_MIN, WORK_MAX, ARENA_SIZE, GATE_SIZE, MAX_GATES
from agent import PolicyNetwork, PPOTrainer

# === НАСТРОЙКИ ОБУЧЕНИЯ ===
N_ITERATIONS = 5000
EPISODES_PER_ITER = 64       # больше эпизодов — стабильнее обновления
EVAL_EVERY = 50
N_EVAL = 100
SAVE_DIR = Path(__file__).parent / "models"
VIZ_DIR = Path(__file__).parent / "data"

MIN_GATES = 3


def collect_episode(env: GateEnvironment, policy: PolicyNetwork,
                    n_gates: int) -> dict:
    """Один эпизод = один шаг: state → action → reward."""
    state = env.reset(n_gates)
    action, log_prob, value = policy.select_action(state)
    next_state, reward, done, info = env.step(action)

    return {
        "state": state,
        "action": action,
        "reward": reward,
        "log_prob": log_prob,
        "value": value,
        "config": env.get_config(),
        "valid": env.is_valid(),
        "info": info,
    }


def evaluate(env, policy, n_episodes=100):
    """Оценка без обучения."""
    policy.eval()
    valid_count = 0
    total_rewards = []
    valid_by_n = {}  # статистика по числу ворот

    with torch.no_grad():
        for _ in range(n_episodes):
            n_gates = np.random.randint(MIN_GATES, MAX_GATES + 1)
            ep = collect_episode(env, policy, n_gates)
            total_rewards.append(ep["reward"])

            if n_gates not in valid_by_n:
                valid_by_n[n_gates] = {"total": 0, "valid": 0}
            valid_by_n[n_gates]["total"] += 1

            if ep["valid"]:
                valid_count += 1
                valid_by_n[n_gates]["valid"] += 1

    policy.train()
    return {
        "validity_rate": valid_count / n_episodes,
        "mean_reward": np.mean(total_rewards),
        "valid_count": valid_count,
        "valid_by_n": valid_by_n,
    }


def draw_gate(ax, x, y, angle, color="blue", lw=2.5):
    """Рисует ворота как отрезок."""
    dx = (GATE_SIZE / 2) * np.cos(angle)
    dy = (GATE_SIZE / 2) * np.sin(angle)
    ax.plot([x - dx, x + dx], [y - dy, y + dy], color=color, linewidth=lw,
            solid_capstyle="round")
    nx = 0.3 * np.cos(angle + np.pi / 2)
    ny = 0.3 * np.sin(angle + np.pi / 2)
    ax.annotate("", xy=(x + nx, y + ny), xytext=(x, y),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5))


def visualize_results(policy, env, save_path, n_show=12):
    """Генерирует и визуализирует конфигурации."""
    policy.eval()
    configs = []

    with torch.no_grad():
        for _ in range(200):
            n = np.random.randint(MIN_GATES, MAX_GATES + 1)
            ep = collect_episode(env, policy, n)
            configs.append((ep["config"], ep["valid"], ep["reward"]))

    configs.sort(key=lambda x: (-int(x[1]), -x[2]))

    n_show = min(n_show, len(configs))
    cols = 3
    rows = (n_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
    axes = np.atleast_2d(axes)
    cmap = plt.cm.tab10

    for idx in range(n_show):
        ax = axes[idx // cols, idx % cols]
        config, valid, reward = configs[idx]
        n = len(config)

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
                    color="gray", linewidth=1, linestyle="--", alpha=0.5)
            d = np.hypot(config[j, 0] - config[i, 0], config[j, 1] - config[i, 1])
            mx = (config[i, 0] + config[j, 0]) / 2
            my = (config[i, 1] + config[j, 1]) / 2
            ax.text(mx, my, f"{d:.1f}м", fontsize=7, ha="center", color="gray")

        # Ворота
        for i, (x, y, a) in enumerate(config):
            color = cmap(i % 10)
            draw_gate(ax, x, y, a, color=color)
            ax.text(x, y + 0.4, str(i + 1), fontsize=9, ha="center",
                    fontweight="bold", color=color)

        status = "VALID" if valid else "INVALID"
        color_title = "green" if valid else "red"
        ax.set_title(f"#{idx+1} ({n} ворот) [{status}] R={reward:.1f}",
                     fontsize=11, color=color_title)

    for idx in range(n_show, rows * cols):
        axes[idx // cols, idx % cols].set_visible(False)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Визуализация: {save_path}")


def main():
    print("=" * 60)
    print("ОБУЧЕНИЕ RL-АГЕНТА (ALL-AT-ONCE) ДЛЯ РАССТАНОВКИ ВОРОТ")
    print("=" * 60)

    # === ClearML ===
    from clearml import Task
    env_name = "Colab" if "google.colab" in sys.modules else "Local"
    task = Task.init(project_name="DroneTrack", task_name=f"RL v4 Exact Copy - {env_name}")
    task.connect({
    "n_iterations": N_ITERATIONS,
    "episodes_per_iter": EPISODES_PER_ITER,
    "min_gates": MIN_GATES,
    "max_gates": MAX_GATES,
    "lr": 3e-4,
    "clip_eps": 0.2,
    "ppo_epochs": 10,
    "entropy_coef": 0.02,
    "hidden_dim": 256,
    "approach": "gates_only",  # без столбиков
})
    logger = task.get_logger()

    env = GateEnvironment()
    policy = PolicyNetwork()
    trainer = PPOTrainer(policy, lr=3e-4)

    n_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"Параметров в модели: {n_params:,}")
    print(f"Итераций: {N_ITERATIONS}, эпизодов/итерацию: {EPISODES_PER_ITER}")
    print(f"Подход: ALL-AT-ONCE (1 шаг = все ворота)")
    print()

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    history = {"iter": [], "reward": [], "validity": [], "loss": []}
    best_validity = 0.0

    for iteration in range(1, N_ITERATIONS + 1):
        # === 1. Собираем эпизоды ===
        episodes = []
        ep_rewards = []

        for _ in range(EPISODES_PER_ITER):
            n_gates = np.random.randint(MIN_GATES, MAX_GATES + 1)
            ep = collect_episode(env, policy, n_gates)
            episodes.append(ep)
            ep_rewards.append(ep["reward"])

        # === 2. Обновляем веса PPO ===
        loss_info = trainer.update(episodes)

        # === 3. Логируем ===
        mean_reward = np.mean(ep_rewards)

        if iteration % EVAL_EVERY == 0 or iteration == 1:
            eval_result = evaluate(env, policy, N_EVAL)
            validity = eval_result["validity_rate"]

            history["iter"].append(iteration)
            history["reward"].append(mean_reward)
            history["validity"].append(validity)
            history["loss"].append(loss_info["loss"])

            # ClearML
            logger.report_scalar("Reward", "mean", mean_reward, iteration)
            logger.report_scalar("Validity", "rate", validity * 100, iteration)
            logger.report_scalar("Loss", "total", loss_info["loss"], iteration)
            logger.report_scalar("Loss", "policy", loss_info["policy_loss"], iteration)
            logger.report_scalar("Loss", "value", loss_info["value_loss"], iteration)

            # Статистика по числу ворот
            detail = ""
            for ng in sorted(eval_result["valid_by_n"]):
                info = eval_result["valid_by_n"][ng]
                pct = info["valid"] / info["total"] * 100 if info["total"] > 0 else 0
                detail += f" {ng}g:{pct:.0f}%"
                logger.report_scalar("Validity by N", f"{ng} gates", pct, iteration)

            print(f"Iter {iteration:4d}/{N_ITERATIONS} | "
                  f"reward={mean_reward:6.2f} | "
                  f"validity={validity*100:5.1f}% |{detail} | "
                  f"loss={loss_info['loss']:.4f}")

            if validity > best_validity:
                best_validity = validity
                torch.save(policy.state_dict(), SAVE_DIR / "best_model.pt")
                print(f"  -> Лучшая модель! (validity={validity*100:.1f}%)")

    # === ФИНАЛЬНАЯ ОЦЕНКА ===
    print("\n" + "=" * 60)
    print("ФИНАЛЬНАЯ ОЦЕНКА")

    policy.load_state_dict(torch.load(SAVE_DIR / "best_model.pt", weights_only=True))
    final = evaluate(env, policy, 500)
    print(f"Validity rate: {final['validity_rate']*100:.1f}% "
          f"({final['valid_count']}/500)")
    print(f"Mean reward: {final['mean_reward']:.2f}")

    print("\nПо числу ворот:")
    for ng in sorted(final["valid_by_n"]):
        info = final["valid_by_n"][ng]
        pct = info["valid"] / info["total"] * 100 if info["total"] > 0 else 0
        print(f"  {ng} ворот: {info['valid']}/{info['total']} ({pct:.1f}%)")

    # === ГРАФИКИ ===
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history["iter"], history["reward"], "b-", linewidth=2)
    ax1.set_xlabel("Итерация")
    ax1.set_ylabel("Средняя награда")
    ax1.set_title("Награда за эпизод")
    ax1.grid(True, alpha=0.3)

    ax2.plot(history["iter"], [v * 100 for v in history["validity"]],
             "g-", linewidth=2)
    ax2.set_xlabel("Итерация")
    ax2.set_ylabel("Validity rate (%)")
    ax2.set_title("Доля валидных конфигураций")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(VIZ_DIR / "training_curves.png", dpi=150)
    plt.close()
    print(f"\nГрафики: {VIZ_DIR / 'training_curves.png'}")

    # === ВИЗУАЛИЗАЦИЯ ===
    visualize_results(policy, env, VIZ_DIR / "viz_rl_results.png", n_show=12)

    # ClearML images
    logger.report_image("Results", "Training Curves",
                        local_path=str(VIZ_DIR / "training_curves.png"))
    logger.report_image("Results", "Generated Configs",
                        local_path=str(VIZ_DIR / "viz_rl_results.png"))

    print("\nГотово!")


if __name__ == "__main__":
    main()
