"""
Supervised warm-up для флагов: учим сеть ставить флаги посередине между воротами.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from environment_with_flags import GateEnvironmentWithFlags, WORK_MIN, WORK_RANGE, ACTION_DIM
from agent import PolicyNetwork

SAVE_DIR = Path("models")
N_SAMPLES = 10000
EPOCHS = 50
BATCH_SIZE = 128
LR = 1e-3


def generate_supervised_data(n_gates=3, n_flags=3):
    """
    Генерирует идеальные трассы: ворота в треугольнике/квадрате, флаги посередине.
    """
    data = []
    
    for _ in range(N_SAMPLES):
        # Случайный треугольник внутри арены [0, 10]
        # Центр + случайные отклонения
        cx, cy = 5.0, 5.0
        radius = np.random.uniform(2.0, 3.5)
        
        gates = []
        for i in range(n_gates):
            angle = i * 2 * np.pi / n_gates + np.random.uniform(-0.3, 0.3)
            x = cx + radius * np.cos(angle)
            y = cy + radius * np.sin(angle)
            a = np.random.uniform(0, 2 * np.pi)
            gates.append((x, y, a))
        
        # Флаги — посередине между воротами
        flags = []
        for i in range(n_flags):
            g1 = gates[i]
            g2 = gates[(i + 1) % n_gates]
            fx = (g1[0] + g2[0]) / 2 + np.random.uniform(-0.3, 0.3)
            fy = (g1[1] + g2[1]) / 2 + np.random.uniform(-0.3, 0.3)
            flags.append((fx, fy))
        
        # Кодируем в action [0, 1]
        action = np.zeros(ACTION_DIM)
        for i, (x, y, a) in enumerate(gates):
            action[i * 3] = (x - WORK_MIN) / WORK_RANGE
            action[i * 3 + 1] = (y - WORK_MIN) / WORK_RANGE
            action[i * 3 + 2] = a / (2 * np.pi)
        
        flag_offset = n_gates * 3
        for i, (fx, fy) in enumerate(flags):
            action[flag_offset + i * 2] = (fx - WORK_MIN) / WORK_RANGE
            action[flag_offset + i * 2 + 1] = (fy - WORK_MIN) / WORK_RANGE
        
        state = np.array([n_gates / 6, n_flags / 6], dtype=np.float32)
        data.append((state, action))
    
    return data


def main():
    print("=" * 60)
    print("SUPERVISED WARM-UP для флагов")
    print("=" * 60)
    
    env = GateEnvironmentWithFlags()
    policy = PolicyNetwork()
    
    # Загружаем веса 3g0f
    checkpoint = SAVE_DIR / "model_3g0f.pt"
    if checkpoint.exists():
        policy.load_state_dict(torch.load(checkpoint))
        print(f"Загружены веса: {checkpoint}")
    else:
        print("Веса 3g0f не найдены, начинаем с нуля")
    
    # Генерируем данные
    print("Генерация supervised данных...")
    data = generate_supervised_data(n_gates=3, n_flags=3)
    print(f"Сгенерировано {len(data)} примеров")
    
    # Оптимизатор
    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)
    
    # Обучение: MSE между предсказанным action и идеальным
    policy.train()
    
    for epoch in range(EPOCHS):
        np.random.shuffle(data)
        total_loss = 0
        n_batches = 0
        
        for i in range(0, len(data), BATCH_SIZE):
            batch = data[i:i + BATCH_SIZE]
            states = torch.FloatTensor(np.array([x[0] for x in batch]))
            targets = torch.FloatTensor(np.array([x[1] for x in batch]))
            
            # Маска: активные выходы для 3 ворот + 3 флага = 15
            mask = torch.zeros(ACTION_DIM)
            mask[:15] = 1.0
            
            mean, std, _ = policy.forward(states)
            
            # MSE только по активным выходам
            loss = ((mean - targets) ** 2 * mask).sum() / mask.sum()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {total_loss / n_batches:.6f}")
    
    # Сохраняем
    torch.save(policy.state_dict(), SAVE_DIR / "model_3g3f_supervised.pt")
    print(f"\nСохранено: {SAVE_DIR / 'model_3g3f_supervised.pt'}")
    
    # Проверка: генерируем несколько трасс
    print("\nПроверка:")
    policy.eval()
    valid_count = 0
    with torch.no_grad():
        for _ in range(100):
            state = env.reset(3, 3)
            action, _, _, _, _ = policy.select_action(state)
            env.step(action)
            if env.is_valid():
                valid_count += 1
    
    print(f"Валидность на 3g3f: {valid_count}/100 ({valid_count}%)")
    
    # Возвращаем std к нормальному значению (после supervised std может схлопнуться)
    with torch.no_grad():
        policy.actor_log_std.data[:] = -2.0  # std = 0.135
    
    torch.save(policy.state_dict(), SAVE_DIR / "model_3g3f_warmup.pt")
    print(f"Сохранено с восстановленным std: {SAVE_DIR / 'model_3g3f_warmup.pt'}")


if __name__ == "__main__":
    main()