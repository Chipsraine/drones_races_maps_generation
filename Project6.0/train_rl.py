"""
Обучение RL-агента — подход "всё за один шаг" с Curriculum Learning.
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

from environment_with_flags import GateEnvironmentWithFlags, WORK_MIN, WORK_MAX, ARENA_SIZE, MAX_GATES, MAX_FLAGS, GATE_SIZE, ACTION_DIM
from agent import PolicyNetwork, PPOTrainer

# === НАСТРОЙКИ ОБУЧЕНИЯ ===
N_ITERATIONS = 10000
EPISODES_PER_ITER = 128
lr = 1e-3                     # было 3e-4 — в 3 раза быстрее
entropy_coef = 0.15           # было 0.05 — больше исследования
clip_eps = 0.3
EVAL_EVERY = 50
N_EVAL = 100
SAVE_DIR = Path(__file__).parent / "models"
VIZ_DIR = Path(__file__).parent / "data"

MIN_GATES = 3


def collect_episode(env, policy, n_gates, n_flags=None):
    state = env.reset(n_gates, n_flags)
    action, log_prob, value, log_prob_per_dim, entropy_per_dim = policy.select_action(state)
    next_state, reward, done, info = env.step(action)
    
    # Маска: 1 на активных выходах, 0 на неиспользуемых
    nf = n_flags if n_flags is not None else n_gates
    active_dim = n_gates * 3 + nf * 2
    mask = np.zeros(ACTION_DIM, dtype=np.float32)
    mask[:active_dim] = 1.0

    return {
        "state": state,
        "action": action,
        "reward": reward,
        "log_prob": log_prob,
        "log_prob_per_dim": log_prob_per_dim,
        "entropy_per_dim": entropy_per_dim,
        "mask": mask,
        "value": value,
        "config": env.get_config(),
        "flags": env.get_flags(),
        "valid": env.is_valid(),
        "info": info,
    }


def evaluate(env, policy, n_gates_fixed=None, n_flags_fixed=None, n_episodes=100):
    policy.eval()
    valid_count = 0
    total_rewards = []
    valid_by_n = {}
    with torch.no_grad():
        for _ in range(n_episodes):
            if n_gates_fixed is not None:
                n_gates = n_gates_fixed
                n_flags = n_flags_fixed if n_flags_fixed is not None else 0
            else:
                n_gates = np.random.randint(3, MAX_GATES + 1)
                n_flags = np.random.randint(0, min(n_gates + 1, MAX_FLAGS + 1))
            ep = collect_episode(env, policy, n_gates, n_flags)
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
    dx = (GATE_SIZE / 2) * np.cos(angle)
    dy = (GATE_SIZE / 2) * np.sin(angle)
    ax.plot([x - dx, x + dx], [y - dy, y + dy], color=color, linewidth=lw, solid_capstyle="round")
    nx = 0.3 * np.cos(angle + np.pi / 2)
    ny = 0.3 * np.sin(angle + np.pi / 2)
    ax.annotate("", xy=(x + nx, y + ny), xytext=(x, y),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5))


def visualize_results(policy, env, save_path, n_show=12, n_flags=None):
    policy.eval()
    configs = []
    with torch.no_grad():
        for _ in range(200):
            n = np.random.randint(MIN_GATES, MAX_GATES + 1)
            nf = n_flags if n_flags is not None else np.random.randint(0, min(n + 1, MAX_FLAGS + 1))
            ep = collect_episode(env, policy, n, nf)
            configs.append((ep["config"], ep["flags"], ep["valid"], ep["reward"]))
    configs.sort(key=lambda x: (-int(x[2]), -x[3]))
    n_show = min(n_show, len(configs))
    cols = 3
    rows = (n_show + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
    axes = np.atleast_2d(axes)
    cmap = plt.cm.tab10
    for idx in range(n_show):
        ax = axes[idx // cols, idx % cols]
        config, flags, valid, reward = configs[idx]
        n = len(config)
        nf = len(flags) if flags is not None else 0
        ax.set_xlim(0, ARENA_SIZE)
        ax.set_ylim(0, ARENA_SIZE)
        ax.set_aspect("equal")
        ax.set_xlabel("X (м)")
        ax.set_ylabel("Y (м)")
        ax.add_patch(patches.Rectangle((0, 0), ARENA_SIZE, ARENA_SIZE,
                                        linewidth=2, edgecolor="black",
                                        facecolor="lightyellow", alpha=0.15))
        for i in range(n):
            j = (i + 1) % n
            ax.plot([config[i, 0], config[j, 0]], [config[i, 1], config[j, 1]],
                    color="gray", linewidth=1, linestyle="--", alpha=0.5)
            d = np.hypot(config[j, 0] - config[i, 0], config[j, 1] - config[i, 1])
            mx = (config[i, 0] + config[j, 0]) / 2
            my = (config[i, 1] + config[j, 1]) / 2
            ax.text(mx, my, f"{d:.1f}м", fontsize=7, ha="center", color="gray")
        for i, (x, y, a) in enumerate(config):
            color = cmap(i % 10)
            draw_gate(ax, x, y, a, color=color)
            ax.text(x, y + 0.4, str(i + 1), fontsize=9, ha="center",
                    fontweight="bold", color=color)
        if nf > 0:
            for fi, (fx, fy) in enumerate(flags):
                ax.plot(fx, fy, 'ro', markersize=8, markeredgecolor='darkred', markeredgewidth=1)
                ax.text(fx, fy + 0.35, f"F{fi+1}", fontsize=8, ha="center",
                        fontweight="bold", color="red")
        status = "VALID" if valid else "INVALID"
        color_title = "green" if valid else "red"
        ax.set_title(f"#{idx+1} ({n}в/{nf}ф) [{status}] R={reward:.1f}",
                     fontsize=11, color=color_title)
    for idx in range(n_show, rows * cols):
        axes[idx // cols, idx % cols].set_visible(False)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Визуализация: {save_path}")


def main():
    print("=" * 60)
    print("ОБУЧЕНИЕ RL v6.1 — Project5.0 + флаги")
    print("=" * 60)

    from clearml import Task
    env_name = "Colab" if "google.colab" in sys.modules else "Local"
    task = Task.init(project_name="DroneTrack", task_name=f"RL v6.1 FixedConsts - {env_name}")
    task.connect({
        "n_iterations": N_ITERATIONS,
        "episodes_per_iter": EPISODES_PER_ITER,
        "min_gates": MIN_GATES,
        "max_gates": MAX_GATES,
        "lr": 5e-4,
        "clip_eps": 0.2,
        "ppo_epochs": 20,
        "entropy_coef": 0.1,
        "hidden_dim": 512,
        "approach": "gates_with_flags_masked",
        "max_flags": MAX_FLAGS,
    })
    logger = task.get_logger()

    env = GateEnvironmentWithFlags()
    policy = PolicyNetwork()
    trainer = PPOTrainer(policy, lr=5e-4, entropy_coef=0.1, ppo_epochs=20)

    n_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"Параметров в модели: {n_params:,}")
    print(f"Итераций: {N_ITERATIONS}, эпизодов/итерацию: {EPISODES_PER_ITER}")
    print()

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    history = {"iter": [], "reward": [], "validity": [], "loss": []}
    best_validity = 0.0

    for iteration in range(1, N_ITERATIONS + 1):
        episodes = []
        ep_rewards = []

        for _ in range(EPISODES_PER_ITER):
            # === CURRICULUM: мягкий переход ===
            if iteration <= 1000:
                # Фаза 1: 3 ворота, 0 флагов (уже освоено)
                n_gates = 3
                n_flags = 0
            elif iteration <= 2000:
                # Фаза 2: 3 ворота + 1 флаг
                n_gates = 3
                n_flags = 1
            elif iteration <= 3000:
                # Фаза 3: 3 ворота + 3 флага
                n_gates = 3
                n_flags = 3
            elif iteration <= 4000:
                # Фаза 4: 4 ворота + 2 флага
                n_gates = 4
                n_flags = 2
            elif iteration <= 5000:
                # Фаза 5: 4 ворота + 4 флага
                n_gates = 4
                n_flags = 4
            elif iteration <= 7000:
                # Фаза 6: 5 ворот + 3-5 флагов
                n_gates = 5
                n_flags = np.random.randint(3, 6)
            elif iteration <= 10000:
                # Фаза 7: 6 ворот + 4-6 флагов
                n_gates = 6
                n_flags = np.random.randint(4, 7)
            else:
                # Фаза 8: полный random
                n_gates = np.random.randint(3, MAX_GATES + 1)
                n_flags = np.random.randint(0, min(n_gates + 1, MAX_FLAGS + 1))
            
            ep = collect_episode(env, policy, n_gates, n_flags)
            episodes.append(ep)
            ep_rewards.append(ep["reward"])

        if len(episodes) == 0:
            continue

        loss_info = trainer.update(episodes)
        mean_reward = np.mean(ep_rewards)

        if iteration % EVAL_EVERY == 0 or iteration == 1:
            # Оцениваем ТЕКУЩУЮ фазу + random для сравнения
            if iteration <= 1000:
                eval_result = evaluate(env, policy, n_gates_fixed=3, n_flags_fixed=0, n_episodes=N_EVAL)
                phase = "[3g0f]"
            elif iteration <= 2000:
                eval_result = evaluate(env, policy, n_gates_fixed=3, n_flags_fixed=1, n_episodes=N_EVAL)
                phase = "[3g1f]"
            elif iteration <= 3000:
                eval_result = evaluate(env, policy, n_gates_fixed=3, n_flags_fixed=3, n_episodes=N_EVAL)
                phase = "[3g3f]"
            elif iteration <= 4000:
                eval_result = evaluate(env, policy, n_gates_fixed=4, n_flags_fixed=2, n_episodes=N_EVAL)
                phase = "[4g2f]"
            elif iteration <= 5000:
                eval_result = evaluate(env, policy, n_gates_fixed=4, n_flags_fixed=4, n_episodes=N_EVAL)
                phase = "[4g4f]"
            elif iteration <= 7000:
                eval_result = evaluate(env, policy, n_gates_fixed=5, n_flags_fixed=5, n_episodes=N_EVAL)
                phase = "[5g5f]"
            elif iteration <= 10000:
                eval_result = evaluate(env, policy, n_gates_fixed=6, n_flags_fixed=6, n_episodes=N_EVAL)
                phase = "[6g6f]"
            else:
                eval_result = evaluate(env, policy, n_episodes=N_EVAL)
                phase = "[random]"
            
            validity = eval_result["validity_rate"]

            history["iter"].append(iteration)
            history["reward"].append(mean_reward)
            history["validity"].append(validity)
            history["loss"].append(loss_info["loss"])

            logger.report_scalar("Reward", "mean", mean_reward, iteration)
            logger.report_scalar("Validity", "rate", validity * 100, iteration)
            logger.report_scalar("Loss", "total", loss_info["loss"], iteration)

            detail = ""
            for ng in sorted(eval_result["valid_by_n"]):
                info = eval_result["valid_by_n"][ng]
                pct = info["valid"] / info["total"] * 100 if info["total"] > 0 else 0
                detail += f" {ng}g:{pct:.0f}%"
                logger.report_scalar("Validity by N", f"{ng} gates", pct, iteration)

            phase = "[3g0f]" if iteration <= 1000 else "[mixed]"
            print(f"Iter {iteration:4d}/{N_ITERATIONS} {phase} | "
                  f"reward={mean_reward:7.2f} | "
                  f"validity={validity*100:5.1f}% |{detail} | "
                  f"loss={loss_info['loss']:.4f}")

            if validity > best_validity:
                best_validity = validity
                torch.save(policy.state_dict(), SAVE_DIR / "best_model.pt")
                print(f"  -> Лучшая модель! (validity={validity*100:.1f}%)")

                        # === СОХРАНЕНИЕ ПРОМЕЖУТОЧНЫХ ВЕСОВ ===
            if iteration == 1000:
                torch.save(policy.state_dict(), SAVE_DIR / "model_3g0f.pt")
                print(f"  -> [CHECKPOINT] Сохранены веса: 3 ворота, 0 флагов")
            elif iteration == 2000:
                torch.save(policy.state_dict(), SAVE_DIR / "model_3g3f.pt")
                print(f"  -> [CHECKPOINT] Сохранены веса: 3 ворота, 3 флага")
            elif iteration == 3000:
                torch.save(policy.state_dict(), SAVE_DIR / "model_4g2f.pt")
                print(f"  -> [CHECKPOINT] Сохранены веса: 4 ворота, 2 флага")
            elif iteration == 4000:
                torch.save(policy.state_dict(), SAVE_DIR / "model_4g4f.pt")
                print(f"  -> [CHECKPOINT] Сохранены веса: 4 ворота, 4 флага")

    # Финал
    print("\n" + "=" * 60)
    print("ФИНАЛЬНАЯ ОЦЕНКА")
    final = evaluate(env, policy, n_episodes=500)
    print(f"Validity rate: {final['validity_rate']*100:.1f}% ({final['valid_count']}/500)")
    print(f"Mean reward: {final['mean_reward']:.2f}")
    for ng in sorted(final["valid_by_n"]):
        info = final["valid_by_n"][ng]
        pct = info["valid"] / info["total"] * 100 if info["total"] > 0 else 0
        print(f"  {ng} ворот: {info['valid']}/{info['total']} ({pct:.1f}%)")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(history["iter"], history["reward"], "b-", linewidth=2)
    ax1.set_xlabel("Итерация"); ax1.set_ylabel("Средняя награда"); ax1.grid(True, alpha=0.3)
    ax2.plot(history["iter"], [v * 100 for v in history["validity"]], "g-", linewidth=2)
    ax2.set_xlabel("Итерация"); ax2.set_ylabel("Validity (%)"); ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(VIZ_DIR / "training_curves.png", dpi=150); plt.close()
    visualize_results(policy, env, VIZ_DIR / "viz_rl_results.png", n_show=12)
    logger.report_image("Results", "Training Curves", local_path=str(VIZ_DIR / "training_curves.png"))
    logger.report_image("Results", "Generated Configs", local_path=str(VIZ_DIR / "viz_rl_results.png"))
    print("\nГотово!")


if __name__ == "__main__":
    main()