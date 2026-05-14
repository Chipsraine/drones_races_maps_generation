"""
Обучаем сеть ворот с учётом флагов.
Сеть флагов уже обучена и заморожена.
"""

import sys
import torch
import numpy as np
from pathlib import Path
from clearml import Task

from environment_with_flags import GateEnvironmentWithFlags, WORK_MIN, WORK_RANGE, MAX_GATES, MAX_FLAGS
from agent import PolicyNetwork, PPOTrainer
from flag_network import FlagNetwork

SAVE_DIR = Path("models")
N_ITERATIONS = 5000
EPISODES_PER_ITER = 128
EVAL_EVERY = 50
N_EVAL = 100


def collect_episode(env, policy_gates, policy_flags, n_gates, n_flags):
    """
    Генерируем ворота → добавляем флаги через frozen сеть → считаем награду.
    """
    state = env.reset(n_gates, n_flags)
    
    # Шаг 1: Генерируем ворота
    action_full, log_prob, value, log_prob_per_dim, entropy_per_dim = policy_gates.select_action(state)
    gates_action = action_full[:n_gates * 3]
    
    # Декодируем ворота
    gates = []
    for i in range(n_gates):
        x = gates_action[i * 3] * 10.0
        y = gates_action[i * 3 + 1] * 10.0
        a = gates_action[i * 3 + 2] * 2 * np.pi
        gates.append((x, y, a))
    
    # Шаг 2: Генерируем флаги через frozen сеть
    if n_flags > 0:
        gates_np = np.array(gates, dtype=np.float32)
        with torch.no_grad():
            flags_pred = policy_flags.predict_flags(gates_np, n_flags)
    else:
        flags_pred = np.zeros((0, 2))
    
    # Шаг 3: Собираем полный action
    action = np.zeros(MAX_GATES * 3 + MAX_FLAGS * 2, dtype=np.float32)
    for i, (x, y, a) in enumerate(gates):
        action[i * 3] = (x - WORK_MIN) / WORK_RANGE
        action[i * 3 + 1] = (y - WORK_MIN) / WORK_RANGE
        action[i * 3 + 2] = a / (2 * np.pi)
    
    flag_offset = n_gates * 3
    for i, (fx, fy) in enumerate(flags_pred):
        action[flag_offset + i * 2] = (fx - WORK_MIN) / WORK_RANGE
        action[flag_offset + i * 2 + 1] = (fy - WORK_MIN) / WORK_RANGE
    
    # Шаг 4: Оцениваем в среде
    _, reward, _, info = env.step(action)
    
    # Маска только для выходов ворот (флаги генерируются отдельно)
    mask = np.zeros(MAX_GATES * 3 + MAX_FLAGS * 2, dtype=np.float32)
    mask[:n_gates * 3] = 1.0  # Обучаем только ворота!
    
    return {
        "state": state,
        "action": action,  # Полный action для совместимости с PPO
        "reward": reward,
        "log_prob": log_prob,
        "log_prob_per_dim": log_prob_per_dim,
        "entropy_per_dim": entropy_per_dim,
        "mask": mask,
        "value": value,
        "valid": env.is_valid(),
        "info": info,
    }


def evaluate(env, policy_gates, policy_flags, n_gates, n_flags, n_episodes=100):
    policy_gates.eval()
    valid_count = 0
    total_rewards = []
    
    with torch.no_grad():
        for _ in range(n_episodes):
            ep = collect_episode(env, policy_gates, policy_flags, n_gates, n_flags)
            total_rewards.append(ep["reward"])
            if ep["valid"]:
                valid_count += 1
    
    policy_gates.train()
    return {
        "validity_rate": valid_count / n_episodes,
        "mean_reward": np.mean(total_rewards),
    }


def main():
    print("=" * 60)
    print("ОБУЧЕНИЕ ВОРОТ С УЧЁТОМ ФЛАГОВ")
    print("=" * 60)
    
    env_name = "Colab" if "google.colab" in sys.modules else "Local"
    task = Task.init(project_name="DroneTrack", task_name=f"RL Gates+Flags Joint - {env_name}")
    logger = task.get_logger()
    
    env = GateEnvironmentWithFlags()
    
    # Сеть ворот — обучаемая
    policy_gates = PolicyNetwork()
    
    # Загружаем веса 3g0f как стартовую точку (опционально)
    checkpoint_gates = SAVE_DIR / "model_3g0f.pt"
    if checkpoint_gates.exists():
        policy_gates.load_state_dict(torch.load(checkpoint_gates))
        print(f"Загружены стартовые веса: {checkpoint_gates}")
    else:
        print("Стартуем с нуля")
    
    # Сеть флагов — замороженная
    policy_flags = FlagNetwork()
    policy_flags.load_state_dict(torch.load(SAVE_DIR / "flag_network.pt"))
    for param in policy_flags.parameters():
        param.requires_grad = False
    policy_flags.eval()
    print("Сеть флагов загружена и заморожена")
    
    # Обучаем только ворота, но награда включает флаги
    trainer = PPOTrainer(policy_gates, lr=5e-4, entropy_coef=0.1, ppo_epochs=20)
    
    print(f"Параметров ворот: {sum(p.numel() for p in policy_gates.parameters() if p.requires_grad):,}")
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
            ep = collect_episode(env, policy_gates, policy_flags, n_gates, n_flags)
            episodes.append(ep)
            ep_rewards.append(ep["reward"])
        
        loss_info = trainer.update(episodes)
        mean_reward = np.mean(ep_rewards)
        
        if iteration % EVAL_EVERY == 0 or iteration == 1:
            # Оцениваем на ТОЙ ЖЕ конфигурации, что и обучаем
            if iteration <= 1000:
                eval_n_flags = 1
            elif iteration <= 2000:
                eval_n_flags = 2
            else:
                eval_n_flags = 3
            
            eval_result = evaluate(env, policy_gates, policy_flags, n_gates=3, n_flags=eval_n_flags, n_episodes=N_EVAL)
            validity = eval_result["validity_rate"]
            
            # Дополнительно: оценим на 3g3f для информации
            eval_3g3f = evaluate(env, policy_gates, policy_flags, n_gates=3, n_flags=3, n_episodes=50)
            
            logger.report_scalar("Reward", "mean", mean_reward, iteration)
            logger.report_scalar("Validity", f"3g{eval_n_flags}f", validity * 100, iteration)
            logger.report_scalar("Validity", "3g3f_extra", eval_3g3f["validity_rate"] * 100, iteration)
            
            phase = f"[3g{eval_n_flags}f]"
            print(f"Iter {iteration:4d}/{N_ITERATIONS} {phase} | "
                  f"reward={mean_reward:7.2f} | "
                  f"validity(3g{eval_n_flags}f)={validity*100:5.1f}% | "
                  f"validity(3g3f)={eval_3g3f['validity_rate']*100:5.1f}% | "
                  f"loss={loss_info['loss']:.4f}")
    
    # Финальная оценка
    print("\n" + "=" * 60)
    print("ФИНАЛЬНАЯ ОЦЕНКА")
    
    for ng in [3, 4]:
        for nf in [0, 1, 2, 3]:
            if nf > ng:
                continue
            result = evaluate(env, policy_gates, policy_flags, ng, nf, n_episodes=200)
            print(f"{ng}в/{nf}ф: {result['validity_rate']*100:.1f}% valid, reward={result['mean_reward']:.1f}")
    
    print("\nГотово!")


if __name__ == "__main__":
    main()