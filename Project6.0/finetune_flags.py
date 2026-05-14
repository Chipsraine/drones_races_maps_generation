"""
Fine-tuning: берём веса 3g0f и учим сеть ставить флаги.
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from clearml import Task

from environment_with_flags import GateEnvironmentWithFlags, ARENA_SIZE, MAX_GATES, MAX_FLAGS, GATE_SIZE, ACTION_DIM
from agent import PolicyNetwork, PPOTrainer

SAVE_DIR = Path("models")
VIZ_DIR = Path("data")
VIZ_DIR.mkdir(exist_ok=True)

EPISODES_PER_ITER = 128
N_ITERATIONS = 3000
EVAL_EVERY = 50
N_EVAL = 100


def collect_episode(env, policy, n_gates, n_flags):
    state = env.reset(n_gates, n_flags)
    action, log_prob, value, log_prob_per_dim, entropy_per_dim = policy.select_action(state)
    env.step(action)
    
    nf = n_flags
    active_dim = n_gates * 3 + nf * 2
    mask = np.zeros(ACTION_DIM, dtype=np.float32)
    mask[:active_dim] = 1.0
    
    # Получаем reward из step()
    _, reward, _, info = env.step(action)
    
    return {
        "state": state, "action": action, "reward": reward,
        "log_prob": log_prob, "log_prob_per_dim": log_prob_per_dim,
        "entropy_per_dim": entropy_per_dim, "mask": mask,
        "value": value, "config": env.get_config(), "flags": env.get_flags(),
        "valid": env.is_valid(), "info": info,
    }


def evaluate(env, policy, n_gates, n_flags, n_episodes=100):
    policy.eval()
    valid_count = 0
    total_rewards = []
    
    with torch.no_grad():
        for _ in range(n_episodes):
            ep = collect_episode(env, policy, n_gates, n_flags)
            total_rewards.append(ep["reward"])
            if ep["valid"]:
                valid_count += 1
    
    policy.train()
    return {
        "validity_rate": valid_count / n_episodes,
        "mean_reward": np.mean(total_rewards),
    }


def draw_gate(ax, x, y, angle, color="blue", lw=2.5):
    dx = (GATE_SIZE / 2) * np.cos(angle)
    dy = (GATE_SIZE / 2) * np.sin(angle)
    ax.plot([x - dx, x + dx], [y - dy, y + dy], color=color, linewidth=lw, solid_capstyle="round")
    nx = 0.3 * np.cos(angle + np.pi / 2)
    ny = 0.3 * np.sin(angle + np.pi / 2)
    ax.annotate("", xy=(x + nx, y + ny), xytext=(x, y),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.5))


def visualize(policy, env, n_gates, n_flags, save_path, n_show=12):
    policy.eval()
    configs = []
    
    with torch.no_grad():
        for _ in range(200):
            state = env.reset(n_gates, n_flags)
            action, _, _, _, _ = policy.select_action(state)
            env.step(action)
            configs.append((env.get_config(), env.get_flags(), env.is_valid(), env._evaluate_config()[0]))
    
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
        ax.add_patch(patches.Rectangle((0, 0), ARENA_SIZE, ARENA_SIZE,
                    linewidth=2, edgecolor="black", facecolor="lightyellow", alpha=0.15))
        
        for i in range(n):
            j = (i + 1) % n
            ax.plot([config[i, 0], config[j, 0]], [config[i, 1], config[j, 1]],
                    color="gray", linewidth=1.5, linestyle="--", alpha=0.5)
        
        for i, (x, y, a) in enumerate(config):
            color = cmap(i % 10)
            draw_gate(ax, x, y, a, color=color)
            ax.text(x, y + 0.4, str(i + 1), fontsize=9, ha="center", fontweight="bold", color=color)
        
        if nf > 0:
            for fi, (fx, fy) in enumerate(flags):
                ax.plot(fx, fy, 'ro', markersize=8, markeredgecolor='darkred', markeredgewidth=1)
                ax.text(fx, fy + 0.35, f"F{fi+1}", fontsize=8, ha="center", fontweight="bold", color="red")
        
        status = "VALID" if valid else "INVALID"
        color_title = "green" if valid else "red"
        ax.set_title(f"#{idx+1} ({n}в/{nf}ф) [{status}] R={reward:.1f}", fontsize=11, color=color_title)
    
    for idx in range(n_show, rows * cols):
        axes[idx // cols, idx % cols].set_visible(False)
    
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Визуализация: {save_path}")


def main():
    print("=" * 60)
    print("FINE-TUNING: 3 ворота + флаги")
    print("=" * 60)
    
    # ClearML
    env_name = "Colab" if "google.colab" in sys.modules else "Local"
    task = Task.init(project_name="DroneTrack", task_name=f"RL Fine-tune Flags - {env_name}")
    logger = task.get_logger()
    
    env = GateEnvironmentWithFlags()
    policy = PolicyNetwork()
    
    # Загружаем веса 3g0f
    checkpoint = SAVE_DIR / "model_3g0f.pt"
    if not checkpoint.exists():
        print(f"❌ Нет весов {checkpoint}! Сначала обучи 3g0f.")
        return
    
    policy.load_state_dict(torch.load(checkpoint))
    print(f"✅ Загружены веса: {checkpoint}")
    
    # Fine-tuning: маленький lr, чтобы не забыть ворота
    trainer = PPOTrainer(policy, lr=1e-4, entropy_coef=0.05, ppo_epochs=20)
    
    print(f"Параметров: {sum(p.numel() for p in policy.parameters() if p.requires_grad):,}")
    print(f"Итераций: {N_ITERATIONS}, эпизодов/итерацию: {EPISODES_PER_ITER}")
    print(f"Learning rate: 1e-4 (fine-tuning)")
    print()
    
    best_validity = 0.0
    history = {"iter": [], "reward": [], "validity": []}
    
    for iteration in range(1, N_ITERATIONS + 1):
        episodes = []
        ep_rewards = []
        
        # Curriculum для флагов
        if iteration <= 500:
            n_flags = 1
        elif iteration <= 1000:
            n_flags = 2
        elif iteration <= 1500:
            n_flags = 3
        else:
            n_flags = np.random.randint(1, 4)  # 1-3 флага random
        
        for _ in range(EPISODES_PER_ITER):
            ep = collect_episode(env, policy, n_gates=3, n_flags=n_flags)
            episodes.append(ep)
            ep_rewards.append(ep["reward"])
        
        loss_info = trainer.update(episodes)
        mean_reward = np.mean(ep_rewards)
        
        if iteration % EVAL_EVERY == 0 or iteration == 1:
            # Оцениваем на 3 воротах + 3 флагах (самая сложная конфигурация)
            eval_result = evaluate(env, policy, n_gates=3, n_flags=3, n_episodes=N_EVAL)
            validity = eval_result["validity_rate"]
            
            history["iter"].append(iteration)
            history["reward"].append(mean_reward)
            history["validity"].append(validity)
            
            logger.report_scalar("Reward", "mean", mean_reward, iteration)
            logger.report_scalar("Validity", "3g3f", validity * 100, iteration)
            logger.report_scalar("Loss", "total", loss_info["loss"], iteration)
            
            phase = "[3g1f]" if iteration <= 500 else "[3g2f]" if iteration <= 1000 else "[3g3f]" if iteration <= 1500 else "[3g_random]"
            print(f"Iter {iteration:4d}/{N_ITERATIONS} {phase} | "
                  f"reward={mean_reward:7.2f} | "
                  f"validity(3g3f)={validity*100:5.1f}% | "
                  f"loss={loss_info['loss']:.4f}")
            
            if validity > best_validity:
                best_validity = validity
                torch.save(policy.state_dict(), SAVE_DIR / "model_3g_flags_best.pt")
                print(f"  -> Лучшая модель! (validity={validity*100:.1f}%)")
    
    # Финальная оценка
    print("\n" + "=" * 60)
    print("ФИНАЛЬНАЯ ОЦЕНКА")
    
    for nf in [0, 1, 2, 3]:
        result = evaluate(env, policy, n_gates=3, n_flags=nf, n_episodes=200)
        print(f"3 ворота, {nf} флагов: {result['validity_rate']*100:.1f}% valid, reward={result['mean_reward']:.1f}")
    
    # Визуализация
    visualize(policy, env, n_gates=3, n_flags=3, save_path=VIZ_DIR / "viz_3g3f.png")
    
    print("\nГотово!")


if __name__ == "__main__":
    main()