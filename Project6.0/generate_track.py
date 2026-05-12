import torch
import numpy as np
import matplotlib.pyplot as plt
from agent import PolicyNetwork
from environment_simple import GateEnvironment, WORK_MIN, WORK_MAX, ARENA_SIZE

def generate_track(n_gates=5, model_path="models/best_model_5gates_96percent.pt"):
    """Генерация и визуализация трассы."""
    
    # Загрузка модели
    policy = PolicyNetwork()
    policy.load_state_dict(torch.load(model_path, weights_only=True))
    policy.eval()
    
    # Создание среды
    env = GateEnvironment()
    
    # Генерация трассы
    with torch.no_grad():
        state = env.reset(n_gates)
        action, _, _ = policy.select_action(state)
        next_state, reward, done, info = env.step(action)
    
    # Получение конфигурации ворот
    config = env.config
    
    # Проверка валидности
    is_valid = env.is_valid()
    
    # Визуализация
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Рисуем арену (0,0 до ARENA_SIZE, ARENA_SIZE)
    arena = plt.Rectangle((0, 0), ARENA_SIZE, ARENA_SIZE, 
                          fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(arena)
    
    # Рабочая зона (где могут быть ворота)
    work_size = WORK_MAX - WORK_MIN
    work_offset = -WORK_MIN
    
    # Рисуем ворота
    for i, (x, y, angle) in enumerate(config):
        # Преобразуем координаты из рабочей зоны в арену
        # x,y в диапазоне [WORK_MIN, WORK_MAX] -> [0, ARENA_SIZE]
        gate_x = (x + work_offset) / work_size * ARENA_SIZE
        gate_y = (y + work_offset) / work_size * ARENA_SIZE
        
        # Рисуем ворота как линию перпендикулярную направлению
        gate_length = 0.8  # длина ворот в метрах
        dx = np.cos(angle + np.pi/2) * gate_length / 2
        dy = np.sin(angle + np.pi/2) * gate_length / 2
        
        # Масштабируем длину ворот
        dx_scaled = dx / work_size * ARENA_SIZE
        dy_scaled = dy / work_size * ARENA_SIZE
        
        ax.plot([gate_x - dx_scaled, gate_x + dx_scaled], 
                [gate_y - dy_scaled, gate_y + dy_scaled], 
                'b-', linewidth=4, label=f'Gate {i+1}' if i == 0 else "")
        
        # Номер ворот
        ax.text(gate_x, gate_y, str(i+1), fontsize=12, ha='center', va='center',
                bbox=dict(boxstyle='circle', facecolor='white', edgecolor='blue', alpha=0.8))
    
    # Рисуем путь (соединяем ворота)
    for i in range(len(config)):
        j = (i + 1) % len(config)
        x1 = (config[i, 0] + work_offset) / work_size * ARENA_SIZE
        y1 = (config[i, 1] + work_offset) / work_size * ARENA_SIZE
        x2 = (config[j, 0] + work_offset) / work_size * ARENA_SIZE
        y2 = (config[j, 1] + work_offset) / work_size * ARENA_SIZE
        
        ax.plot([x1, x2], [y1, y2], 'g--', alpha=0.5, linewidth=1)
    
    # Настройки графика
    ax.set_xlim(-0.5, ARENA_SIZE + 0.5)
    ax.set_ylim(-0.5, ARENA_SIZE + 0.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'FPV Track - {n_gates} gates\nValid: {is_valid}, Reward: {reward:.2f}')
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f'track_{n_gates}gates.png', dpi=150, bbox_inches='tight')
    print(f"✅ Трасса сохранена: track_{n_gates}gates.png")
    print(f"   Valid: {is_valid}, Reward: {reward:.2f}")
    print(f"   Gates positions:")
    for i, (x, y, a) in enumerate(config):
        print(f"     Gate {i+1}: x={x:.2f}, y={y:.2f}, angle={np.degrees(a):.1f}°")
    
    return fig, ax

if __name__ == "__main__":
    # Генерация трасс с разным числом ворот
    for n in [3, 4, 5, 6]:
        try:
            generate_track(n_gates=n)
        except Exception as e:
            print(f"❌ Ошибка при генерации {n} ворот: {e}")
            import traceback
            traceback.print_exc()
