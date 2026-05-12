import numpy as np
from typing import Tuple

# === КОНСТАНТЫ ===
ARENA_SIZE = 10.0
WORK_MIN = -5.0
WORK_MAX = 5.0
WORK_RANGE = WORK_MAX - WORK_MIN  # 10.0

MIN_DIST = 3.0
MAX_DIST = 10.0
GLOBAL_MIN_DIST = 2.0
MAX_ANGLE_DIFF = np.pi / 6  # 30 градусов
MAX_TRAVEL_ANGLE = np.pi / 2  # 90 градусов

MAX_GATES = 6
MAX_FLAGS = 6  # Флагов не больше ворот

# ACTION_DIM: ворота (x,y,angle) + флаги (x,y)
# Для 6 ворот и 6 флагов: 6*3 + 6*2 = 30
ACTION_DIM = MAX_GATES * 3 + MAX_FLAGS * 2

PILLAR_RADIUS = 0.3  # Радиус флага


def _segments_intersect(p1, p2, q1, q2):
    """Проверка пересечения двух отрезков."""
    # ... (тот же код, что в environment_simple.py)
    def ccw(A, B, C):
        return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])
    return ccw(p1, q1, q2) != ccw(p2, q1, q2) and ccw(p1, p2, q1) != ccw(p1, p2, q2)


class GateEnvironmentWithFlags:
    """
    Среда для генерации трасс с воротами и флагами.
    
    Флаги располагаются между воротами на траектории движения.
    """
    
    def __init__(self):
        self.n_gates = 5
        self.n_flags = 5  # По умолчанию флагов = воротам
        self.config = None      # (n_gates, 3) - ворота
        self.flags = None       # (n_flags, 2) - флаги
    
    def reset(self, n_gates: int, n_flags: int | None = None) -> np.ndarray:
        """Сброс среды."""
        self.n_gates = n_gates
        
        if n_flags is None:
            self.n_flags = n_gates  # По умолчанию флагов = воротам
        else:
            self.n_flags = min(n_flags, n_gates)  # Флагов не больше ворот
        
        self.config = None
        self.flags = None
        
        return self._get_state()
    
    def _get_state(self) -> np.ndarray:
        """Состояние = [n_gates, n_flags]."""
        return np.array([self.n_gates / MAX_GATES, self.n_flags / MAX_FLAGS], dtype=np.float32)
    
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        """
        Декодирование action:
        - Первые n_gates * 3: ворота (x, y, angle)
        - Следующие n_flags * 2: флаги (x, y)
        """
        n = self.n_gates
        nf = self.n_flags
        
        # === ДЕКОДИРОВАНИЕ ВОРОТ ===
        gates = []
        for i in range(n):
            x = action[i * 3]     * WORK_RANGE + WORK_MIN
            y = action[i * 3 + 1] * WORK_RANGE + WORK_MIN
            a = action[i * 3 + 2] * 2 * np.pi
            gates.append((x, y, a))
        self.config = np.array(gates, dtype=np.float32)
        
        # === ДЕКОДИРОВАНИЕ ФЛАГОВ ===
        flag_offset = MAX_GATES * 3  # Начало флагов в action
        flags = []
        for i in range(nf):
            fx = action[flag_offset + i * 2]     * WORK_RANGE + WORK_MIN
            fy = action[flag_offset + i * 2 + 1] * WORK_RANGE + WORK_MIN
            flags.append((fx, fy))
        self.flags = np.array(flags, dtype=np.float32)
        
        # === ПРОВЕРКА РАСПОЛОЖЕНИЯ ФЛАГОВ ===
        # Флаг должен быть на отрезке между воротами
        # Проверим и скорректируем при необходимости
        
        reward, info = self._evaluate_config()
        return self._get_state(), reward, True, info
    
    def _evaluate_config(self) -> tuple[float, dict]:
        """Оценка конфигурации с флагами."""
        config = self.config
        flags = self.flags
        n = len(config)
        nf = len(flags) if flags is not None else 0
        
        reward = 0.0
        info = {"violations": [], "bonuses": []}
        
        # === 1. БАЗОВЫЙ БОНУС ===
        reward += 2.0 * n
        
        # === 2. ПРОВЕРКА ВОРОТ (как в environment_simple.py) ===
        # ... (границы, дистанции, углы, пересечения)
        # Копируем из environment_simple.py
        
        # === 3. ПРОВЕРКА ФЛАГОВ ===
        if nf > 0:
            # Флаг должен быть между двумя воротами
            for i in range(nf):
                flag = flags[i]
                gate1 = config[i]
                gate2 = config[(i + 1) % n]
                
                # Проверка: флаг на отрезке gate1-gate2?
                # Используем параметр t вдоль отрезка
                dx = gate2[0] - gate1[0]
                dy = gate2[1] - gate1[1]
                length = np.sqrt(dx**2 + dy**2)
                
                if length > 0:
                    # Проекция флага на отрезок
                    t = ((flag[0] - gate1[0]) * dx + (flag[1] - gate1[1]) * dy) / (length**2)
                    t = np.clip(t, 0.1, 0.9)  # Флаг не должен быть слишком близко к воротам
                    
                    # Расстояние от флага до линии gate1-gate2
                    dist_to_line = abs((flag[1] - gate1[1]) * dx - (flag[0] - gate1[0]) * dy) / length
                    
                    # Бонус если флаг близко к линии (на траектории)
                    if dist_to_line < 0.5:  # Допуск 0.5м
                        reward += 2.0
                        info["bonuses"].append(f"flag_on_path_{i}")
                    else:
                        reward -= 1.0 * dist_to_line
                        info["violations"].append(f"flag_off_path_{i}")
                    
                    # Флаг не должен быть слишком близко к воротам
                    dist_to_g1 = np.hypot(flag[0] - gate1[0], flag[1] - gate1[1])
                    dist_to_g2 = np.hypot(flag[0] - gate2[0], flag[1] - gate2[1])
                    
                    if dist_to_g1 < 1.0 or dist_to_g2 < 1.0:
                        reward -= 2.0
                        info["violations"].append(f"flag_too_close_to_gate_{i}")
        
        # === 4. ФИНАЛЬНЫЙ БОНУС ===
        n_violations = len(info["violations"])
        if n_violations == 0:
            reward += 20.0
        
        info["n_violations"] = n_violations
        return reward, info
    
    def is_valid(self) -> bool:
        """Проверка валидности с флагами."""
        # ... (копируем из environment_simple.py + добавляем проверки флагов)
        pass
    
    def get_config(self):
        return self.config
    
    def get_flags(self):
        return self.flags