"""
Простой подход: сеть учит только ворота, флаги = середина отрезка.
Это гарантирует, что флаги всегда на линии между воротами.
"""

import sys
import torch
import numpy as np
from pathlib import Path
from clearml import Task

from environment_with_flags import GateEnvironmentWithFlags, WORK_MIN, WORK_RANGE, MAX_GATES, MAX_FLAGS
from agent import PolicyNetwork, PPOTrainer

SAVE_DIR = Path("models")
N_ITERATIONS = 6000
EPISODES_PER_ITER = 128
EVAL_EVERY = 50
N_EVAL = 100


def generate_flags(gates, n_flags):
    """Флаги = середина отрезка + небольшой шум, проекция на отрезок."""
    flags = []
    for i in range(n_flags):
        g1 = np.array(gates[i][:2])
        g2 = np.array(gates[(i + 1) % len(gates)][:2])
        
        # Середина
        mid = (g1 + g2) / 2
        
        # Небольшой шум
        noise = np.random.uniform(-0.3, 0.3, 2)
        flag = mid + noise
        
        # Проекция на отрезок (чтобы гарантированно на линии)
        dx, dy = g2[0] - g1[0], g2[1] - g1[1]
        length = np.hypot(dx, dy)
        if length > 0:
            # Параметр t
            t = ((flag[0] - g1[0]) * dx + (flag[1] - g1[1]) * dy) / (length ** 2)
            t = np.clip(t, 0.1, 0.9)
            
            # Проекция
            proj_x = g1[0] + t * dx
            proj_y = g1[1] + t * dy
            
            # Отклонение не более 0.5м
            perp_x = -dy / length
            perp_y = dx / length
            offset = np.random.uniform(-0.3, 0.3)
            
            flag = np.array([proj_x + offset * perp_x, proj_y + offset * perp_y])
        
        flags.append(flag)
    
    return np.array(flags)


def collect_episode(env, policy, n_gates, n_flags):
    state = env.reset(n_gates, n_flags)
    action_full, log_prob, value, log_prob_per_dim, entropy_per_dim = policy.select_action(state)
    
    # Берём только ворота из action сети
    gates_action = action_full[:n_gates * 3]
    
    # Декодируем ворота
    gates = []
    for i in range(n_gates):
        x = gates_action[i * 3] * 10.0
        y = gates_action[i * 3 + 1] * 10.0
        a = gates_action[i * 3 + 2] * 2 * np.pi
        gates.append((x, y, a))
    
    # Флаги — жадно, не от сети
    if n_flags > 0:
        flags = generate_flags(gates, n_flags)
    else:
        flags = np.zeros((0, 2))
    
    # Собираем полный action для среды
    action = np.zeros(MAX_GATES * 3 + MAX_FLAGS * 2, dtype=np.float32)
    for i, (x, y, a) in enumerate(gates):
        action[i * 3] = (x - WORK_MIN) / WORK_RANGE
        action[i * 3 + 1] = (y - WORK_MIN) / WORK_RANGE
        action[i * 3 + 2] = a / (2 * np.pi)
    
    flag_offset = n_gates * 3
    for i, (fx, fy) in enumerate(flags):
        action[flag_offset + i * 2] = (fx - WORK_MIN) / WORK_RANGE
        action[flag_offset + i * 2 + 1] = (fy - WORK_MIN) / WORK_RANGE
    
    # Маска: обучаем только ворота (флаги не от сети)
    mask = np.zeros(MAX_GATES * 3 + MAX_FLAGS * 2, dtype=np.float32)
    mask[:n_gates * 3] = 1.0
    
    # Оцениваем
    _, reward, _, info = env.step(action)
    
    return {
        "state": state,
        "action": action,
        "reward": reward,
        "log_prob": log_prob,
        "log_prob_per_dim": log_prob_per_dim,
        "entropy_per_dim": entropy_per_dim,
        "mask": mask,
        "value": value,
        "valid": env.is_valid(),
        "info": info,
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


def main():
    print("=" * 60)
    print("ОБУЧЕНИЕ ВОРОТ + ЖАДНЫЕ ФЛАГИ")
    print("=" * 60)
    
    env_name = "Colab" if "google.colab" in sys.modules else "Local"
    task = Task.init(project_name="DroneTrack", task_name=f"RL Gates+GreedyFlags - {env_name}")
    logger = task.get_logger()
    
    env = GateEnvironmentWithFlags()
    policy = PolicyNetwork()
    
    # Загружаем веса 3g0f
    checkpoint = SAVE_DIR / "model_3g0f.pt"
    if checkpoint.exists():
        policy.load_state_dict(torch.load(checkpoint))
        print(f"Загружены веса: {checkpoint}")
    else:
        print("Стартуем с нуля")
    
    trainer = PPOTrainer(policy, lr=5e-4, entropy_coef=0.1, ppo_epochs=20)
    
    print(f"Параметров: {sum(p.numel() for p in policy.parameters() if p.requires_grad):,}")
    print(f"Итераций: {N_ITERATIONS}")
    print()
    
    best_validity = 0.0
    
    for iteration in range(1, N_ITERATIONS + 1):
        episodes = []
        ep_rewards = []
        
        # Curriculum: сначала 1 флаг, потом 2, потом 3
        if iteration <= 1000:
            n_flags = 1
        elif iteration <= 2000:
            n_flags = 2
        else:
            n_flags = 3
        
        for _ in range(EPISODES_PER_ITER):
            n_gates = np.random.randint(3, 5)  # 3 или 4 ворота
            ep = collect_episode(env, policy, n_gates, n_flags)
            episodes.append(ep)
            ep_rewards.append(ep["reward"])
        
        loss_info = trainer.update(episodes)
        mean_reward = np.mean(ep_rewards)
        
        if iteration % EVAL_EVERY == 0 or iteration == 1:
            # Оцениваем на текущей конфигурации
            eval_n_flags = n_flags
            eval_result = evaluate(env, policy, n_gates=3, n_flags=eval_n_flags, n_episodes=N_EVAL)
            validity = eval_result["validity_rate"]
            
            # Дополнительно: 3g3f
            eval_3g3f = evaluate(env, policy, n_gates=3, n_flags=3, n_episodes=50)
            
            logger.report_scalar("Reward", "mean", mean_reward, iteration)
            logger.report_scalar("Validity", f"3g{eval_n_flags}f", validity * 100, iteration)
            logger.report_scalar("Validity", "3g3f", eval_3g3f["validity_rate"] * 100, iteration)
            
            phase = f"[3g{eval_n_flags}f]"
            print(f"Iter {iteration:4d}/{N_ITERATIONS} {phase} | "
                  f"reward={mean_reward:7.2f} | "
                  f"validity(3g{eval_n_flags}f)={validity*100:5.1f}% | "
                  f"validity(3g3f)={eval_3g3f['validity_rate']*100:5.1f}% | "
                  f"loss={loss_info['loss']:.4f}")
            
            if validity > best_validity:
                best_validity = validity
                torch.save(policy.state_dict(), SAVE_DIR / "model_gates_greedy_flags.pt")
                print(f"  -> Лучшая модель! (validity={validity*100:.1f}%)")
    
    # Финал
    print("\n" + "=" * 60)
    print("ФИНАЛЬНАЯ ОЦЕНКА")
    
    for ng in [3, 4]:
        for nf in [0, 1, 2, 3]:
            if nf > ng:
                continue
            result = evaluate(env, policy, ng, nf, n_episodes=200)
            print(f"{ng}в/{nf}ф: {result['validity_rate']*100:.1f}% valid, reward={result['mean_reward']:.1f}")
    
    print("\nГотово!")


if __name__ == "__main__":
    main()