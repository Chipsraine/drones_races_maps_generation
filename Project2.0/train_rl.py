"""
Обучение RL-агента для расстановки ворот.

Цикл обучения:
1. Агент играет эпизоды (расставляет ворота)
2. Среда выдаёт награды за правильные/неправильные действия
3. PPO обновляет веса нейросети
4. Повторяем, пока агент не научится

Ключевые метрики:
- reward: средняя награда за эпизод (растёт = агент учится)
- validity_rate: % полностью валидных конфигураций (главная метрика)
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

from environment import GateEnvironment, WORK_MIN, WORK_MAX, ARENA_SIZE, MARGIN, GATE_SIZE
from agent import PolicyNetwork, PPOTrainer

# === НАСТРОЙКИ ОБУЧЕНИЯ ===
N_ITERATIONS = 1500      # сколько итераций обучения
EPISODES_PER_ITER = 40   # сколько эпизодов за итерацию
EVAL_EVERY = 25          # как часто оценивать качество
N_EVAL = 50              # сколько эпизодов для оценки
SAVE_DIR = Path(__file__).parent / "models"
VIZ_DIR = Path(__file__).parent / "data"

MIN_GATES = 3
MAX_GATES = 6  # max 6: в зоне 4x4 с новыми правилами (без пересечений) 7-8 нереально


def get_n_gates_curriculum(iteration: int) -> tuple[int, str]:
    """
    Плавный curriculum learning: постепенно расширяем диапазон ворот.

    Вместо резкого переключения фаз (что вызывало обвал reward),
    используем плавное подмешивание: вероятность сложных эпизодов
    растёт постепенно через переходные зоны.

    Фазы:
    - Итерации 1–600       (40%): 3–4 ворот  — базовые маршруты
    - Итерации 601–675    (5%):  переход: 80% из [3,4], 20% из [5]
    - Итерации 676–975    (20%): 3–5 ворот  — средние конфигурации
    - Итерации 976–1050   (5%):  переход: 80% из [3,5], 20% из [6]
    - Итерации 1051+      (30%): 3–6 ворот  — полный диапазон
    """
    t = iteration / N_ITERATIONS

    if t <= 0.40:
        return np.random.randint(3, 5), "phase1(3-4)"
    elif t <= 0.45:
        # Переход 1→2: подмешиваем 5 ворот с вероятностью 20%
        if np.random.random() < 0.2:
            return 5, "trans1→2"
        return np.random.randint(3, 5), "trans1→2"
    elif t <= 0.65:
        return np.random.randint(3, 6), "phase2(3-5)"
    elif t <= 0.70:
        # Переход 2→3: подмешиваем 6 ворот с вероятностью 20%
        if np.random.random() < 0.2:
            return 6, "trans2→3"
        return np.random.randint(3, 6), "trans2→3"
    else:
        return np.random.randint(MIN_GATES, MAX_GATES + 1), "phase3(3-6)"


def collect_episode(env: GateEnvironment, policy: PolicyNetwork,
                    n_gates: int) -> dict:
    """
    Один эпизод: агент расставляет n_gates ворот.
    Возвращает собранный опыт.
    """
    state = env.reset(n_gates)
    hidden = policy.init_hidden()

    states, actions, rewards, log_probs, values = [], [], [], [], []
    done = False

    for _ in range(n_gates):
        action, log_prob, value, hidden = policy.select_action(state, hidden)
        next_state, reward, done, info = env.step(action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        log_probs.append(log_prob)
        values.append(value.squeeze())

        state = next_state

    return {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "log_probs": log_probs,
        "values": values,
        "done": done,
        "config": env.get_config(),
        "valid": env.is_valid(),
        "total_reward": sum(rewards),
    }


def evaluate(env, policy, n_episodes=50, max_gates_eval: int = MAX_GATES):
    """
    Оценка: запускаем эпизоды без обучения, считаем метрики.

    max_gates_eval позволяет оценивать на диапазоне текущей фазы curriculum,
    а не на полном диапазоне — это даёт честную картину прогресса агента.
    """
    policy.eval()
    valid_count = 0
    total_rewards = []

    with torch.no_grad():
        for _ in range(n_episodes):
            n_gates = np.random.randint(MIN_GATES, max_gates_eval + 1)
            ep = collect_episode(env, policy, n_gates)
            total_rewards.append(ep["total_reward"])
            if ep["valid"]:
                valid_count += 1

    policy.train()
    return {
        "validity_rate": valid_count / n_episodes,
        "mean_reward": np.mean(total_rewards),
        "valid_count": valid_count,
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


def visualize_results(policy, env, save_path, n_show=6):
    """Генерирует и визуализирует конфигурации."""
    policy.eval()
    configs = []

    with torch.no_grad():
        for _ in range(100):
            n = np.random.randint(MIN_GATES, MAX_GATES + 1)
            ep = collect_episode(env, policy, n)
            configs.append((ep["config"], ep["valid"], ep["total_reward"]))

    # Сначала валидные, потом по reward
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

        # Рабочая зона
        rect = patches.Rectangle((MARGIN, MARGIN), ARENA_SIZE - 2 * MARGIN,
                                  ARENA_SIZE - 2 * MARGIN, linewidth=1.5,
                                  edgecolor="green", facecolor="lightgreen",
                                  alpha=0.15, linestyle="--")
        ax.add_patch(rect)
        ax.add_patch(patches.Rectangle((0, 0), ARENA_SIZE, ARENA_SIZE,
                                        linewidth=2, edgecolor="black",
                                        facecolor="none"))

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
    print("ОБУЧЕНИЕ RL-АГЕНТА ДЛЯ РАССТАНОВКИ ВОРОТ")
    print("=" * 60)

    # === ClearML ===
    from clearml import Task
    env_name = "Colab" if "google.colab" in sys.modules else "Local"
    task = Task.init(project_name="DroneTrack", task_name=f"RL Gate Placement v4 - {env_name}")
    task.connect({
        "n_iterations": N_ITERATIONS,
        "episodes_per_iter": EPISODES_PER_ITER,
        "min_gates": MIN_GATES,
        "max_gates": MAX_GATES,
        "lr": 1e-4,
        "gamma": 0.99,
        "clip_eps": 0.2,
        "ppo_epochs": 8,
        "state_dim": 25,
        "hidden_dim": 128,
        "curriculum": "smooth",
    })
    logger = task.get_logger()

    env = GateEnvironment()
    policy = PolicyNetwork()
    trainer = PPOTrainer(policy, lr=1e-4, ppo_epochs=8)

    n_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"Параметров в модели: {n_params:,}")
    print(f"Итераций: {N_ITERATIONS}, эпизодов/итерацию: {EPISODES_PER_ITER}")
    print()

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    # История для графиков
    history = {"iter": [], "reward": [], "validity": [], "loss": []}

    best_validity = 0.0

    for iteration in range(1, N_ITERATIONS + 1):
        # === 1. Собираем эпизоды ===
        episodes = []
        ep_rewards = []

        for _ in range(EPISODES_PER_ITER):
            n_gates, curriculum_phase = get_n_gates_curriculum(iteration)
            ep = collect_episode(env, policy, n_gates)
            episodes.append(ep)
            ep_rewards.append(ep["total_reward"])

        # === 2. Обновляем веса PPO ===
        loss_info = trainer.update(episodes)

        # === 3. Логируем ===
        mean_reward = np.mean(ep_rewards)

        if iteration % EVAL_EVERY == 0 or iteration == 1:
            # Оцениваем на диапазоне текущей фазы (не на полном MAX_GATES)
            _, curriculum_phase = get_n_gates_curriculum(iteration)
            phase_max_map = {
                "phase1(3-4)": 4, "trans1→2": 4,
                "phase2(3-5)": 5, "trans2→3": 5,
                "phase3(3-6)": 6,
            }
            phase_max = phase_max_map[curriculum_phase]
            eval_result = evaluate(env, policy, N_EVAL, max_gates_eval=phase_max)
            validity = eval_result["validity_rate"]

            history["iter"].append(iteration)
            history["reward"].append(mean_reward)
            history["validity"].append(validity)
            history["loss"].append(loss_info["loss"])

            # ClearML логирование
            logger.report_scalar("Reward", "mean", mean_reward, iteration)
            logger.report_scalar("Validity", "rate", validity * 100, iteration)
            logger.report_scalar("Loss", "total", loss_info["loss"], iteration)
            logger.report_scalar("Loss", "policy", loss_info["policy_loss"], iteration)
            logger.report_scalar("Loss", "value", loss_info["value_loss"], iteration)

            print(f"Iter {iteration:4d}/{N_ITERATIONS} | "
                  f"{curriculum_phase} | "
                  f"reward={mean_reward:6.2f} | "
                  f"validity={validity*100:5.1f}% | "
                  f"loss={loss_info['loss']:.4f}")

            # Сохраняем лучшую модель
            if validity > best_validity:
                best_validity = validity
                torch.save(policy.state_dict(), SAVE_DIR / "best_model.pt")
                print(f"  -> Лучшая модель! (validity={validity*100:.1f}%)")

    # === ФИНАЛЬНАЯ ОЦЕНКА ===
    print("\n" + "=" * 60)
    print("ФИНАЛЬНАЯ ОЦЕНКА")

    # Загружаем лучшую модель
    policy.load_state_dict(torch.load(SAVE_DIR / "best_model.pt", weights_only=True))
    final = evaluate(env, policy, 200)
    print(f"Validity rate: {final['validity_rate']*100:.1f}% "
          f"({final['valid_count']}/200)")
    print(f"Mean reward: {final['mean_reward']:.2f}")

    # === ГРАФИКИ ОБУЧЕНИЯ ===
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
    print(f"Графики: {VIZ_DIR / 'training_curves.png'}")

    # === ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ ===
    visualize_results(policy, env, VIZ_DIR / "viz_rl_results.png", n_show=12)

    # Загружаем картинки в ClearML
    logger.report_image("Results", "Training Curves",
                        local_path=str(VIZ_DIR / "training_curves.png"))
    logger.report_image("Results", "Generated Configs",
                        local_path=str(VIZ_DIR / "viz_rl_results.png"))

    print("\nГотово!")


if __name__ == "__main__":
    main()
