"""
Среда с параметрическими флагами: флаг задаётся как (t, offset) на отрезке между воротами.
Это гораздо проще для сети, чем свободные (x, y).
"""

import numpy as np

# ... (все константы как в environment_with_flags.py) ...

# ACTION_DIM: ворота (x,y,angle) + флаги (t, offset)
# t ∈ [0, 1] → [0.1, 0.9] (позиция на отрезке)
# offset ∈ [0, 1] → [-0.5, 0.5] (отклонение от линии, в метрах)
# Для 6 ворот и 6 флагов: 6*3 + 6*2 = 30 (тот же размер!)

class GateEnvironmentParametricFlags:
    """
    Флаг = точка на отрезке gate_i → gate_{i+1} с параметрами (t, offset).
    """
    
    def __init__(self):
        self.n_gates = 5
        self.n_flags = 5
        self.config = None
        self.flags = None
    
    def reset(self, n_gates, n_flags=None):
        self.n_gates = n_gates
        if n_flags is None:
            self.n_flags = n_gates
        else:
            self.n_flags = min(n_flags, n_gates)
        self.config = None
        self.flags = None
        return self._get_state()
    
    def _get_state(self):
        return np.array([self.n_gates / 6, self.n_flags / 6], dtype=np.float32)
    
    def step(self, action):
        n = self.n_gates
        nf = self.n_flags
        
        # Декодируем ворота (как раньше)
        gates = []
        for i in range(n):
            x = action[i * 3] * 10.0
            y = action[i * 3 + 1] * 10.0
            a = action[i * 3 + 2] * 2 * np.pi
            gates.append((x, y, a))
        self.config = np.array(gates, dtype=np.float32)
        
        # Декодируем флаги: (t, offset) → (x, y)
        flag_offset = n * 3
        flags = []
        for i in range(nf):
            t_raw = action[flag_offset + i * 2]
            offset_raw = action[flag_offset + i * 2 + 1]
            
            # t ∈ [0.1, 0.9]
            t = 0.1 + t_raw * 0.8
            
            # offset ∈ [-0.5, 0.5] метров
            offset = (offset_raw - 0.5) * 1.0
            
            gate1 = self.config[i]
            gate2 = self.config[(i + 1) % n]
            
            # Точка на отрезке
            px = gate1[0] + t * (gate2[0] - gate1[0])
            py = gate1[1] + t * (gate2[1] - gate1[1])
            
            # Отклонение перпендикулярно отрезку
            dx = gate2[0] - gate1[0]
            dy = gate2[1] - gate1[1]
            length = np.hypot(dx, dy)
            
            if length > 0:
                # Перпендикулярный вектор (нормализованный)
                perp_x = -dy / length
                perp_y = dx / length
                px += offset * perp_x
                py += offset * perp_y
            
            flags.append((px, py))
        
        self.flags = np.array(flags, dtype=np.float32)
        
        reward, info = self._evaluate_config()
        return self._get_state(), reward, True, info
    
    def _evaluate_config(self):
        # ... (точно такая же награда, как в environment_with_flags.py) ...
        pass
    
    def is_valid(self):
        # ... (точно такая же проверка, как в environment_with_flags.py) ...
        pass