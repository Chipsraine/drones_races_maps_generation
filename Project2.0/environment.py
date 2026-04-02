"""
Среда (Environment) для RL-агента, расставляющего ворота.

Агент последовательно ставит ворота на арену.
Среда проверяет правила и выдаёт награду (reward).

Терминология RL:
- State (состояние): текущие размещённые ворота
- Action (действие): координаты и угол следующих ворот (x, y, angle)
- Reward (награда): +/- в зависимости от соблюдения правил
- Episode (эпизод): одна полная расстановка ворот от начала до конца
"""

import numpy as np


# === ПРАВИЛА АРЕНЫ ===
ARENA_SIZE = 10.0            # размер арены (м)
MARGIN = 3.0                 # мин. расстояние от края
GATE_SIZE = 1.0              # размер ворот (м)
MIN_DIST = 3.0               # мин. расстояние между соседними воротами
MAX_DIST = 10.0              # макс. расстояние между соседними воротами
MAX_ANGLE_DIFF = np.pi       # макс. разница углов ворот (180°)

WORK_MIN = MARGIN            # 3.0
WORK_MAX = ARENA_SIZE - MARGIN  # 7.0
WORK_RANGE = WORK_MAX - WORK_MIN  # 4.0

# === ДОПОЛНИТЕЛЬНЫЕ ПРАВИЛА ===
GLOBAL_MIN_DIST = 2.0        # мин. расстояние между ЛЮБЫМИ воротами (не только соседними)
MAX_TRAVEL_ANGLE = np.radians(150)  # макс. разворот направления движения

# === РАЗМЕР СОСТОЯНИЯ ===
MAX_GATES_STATE = 8          # максимум ворот в state (= MAX_GATES в train_rl)
STATE_DIM = 1 + MAX_GATES_STATE * 3  # 25: прогресс + 8 ворот по 3 числа


def _segments_intersect(p1, p2, p3, p4):
    """
    Проверяет, пересекаются ли два отрезка p1-p2 и p3-p4.
    Использует знаки векторных произведений (cross product).
    """
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)

    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False


class GateEnvironment:
    """
    Среда для размещения ворот.

    На каждом шаге агент выдаёт action = (x, y, angle) ∈ [0, 1]³
    Среда переводит в реальные координаты, проверяет правила, даёт reward.

    Правила (проверяются в step):
    1. Границы рабочей зоны [3, 7] × [3, 7]
    2. Дистанция до предыдущих ворот: 3–10 м
    3. Угол ворот: разница с предыдущими ≤ 180°
    4. Глобальная мин. дистанция 2 м до ЛЮБЫХ ворот
    5. Пересечение сегментов пути
    6. Резкий разворот направления движения (> 150°)
    7. Прогрессивная подсказка к замыканию (последние 2 шага)
    8. Замыкание маршрута + coverage-бонус
    """

    def __init__(self, n_gates: int = 5):
        self.n_gates = n_gates
        self.gates = []       # [(x, y, angle), ...]
        self.step_num = 0
        self.done = False

    def reset(self, n_gates: int | None = None) -> np.ndarray:
        """Начинает новый эпизод."""
        if n_gates is not None:
            self.n_gates = n_gates
        self.gates = []
        self.step_num = 0
        self.done = False
        return self._get_state()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        """
        Агент делает действие — ставит ворота.

        Args:
            action: (3,) массив [x, y, angle] в диапазоне [0, 1]

        Returns:
            state, reward, done, info
        """
        x = action[0] * WORK_RANGE + WORK_MIN
        y = action[1] * WORK_RANGE + WORK_MIN
        angle = action[2] * 2 * np.pi

        reward = 0.0
        info = {"violations": []}

        # --- Правило 1: Границы ---
        if not (WORK_MIN <= x <= WORK_MAX and WORK_MIN <= y <= WORK_MAX):
            reward -= 1.0
            info["violations"].append("bounds")

        if len(self.gates) > 0:
            prev = self.gates[-1]

            # --- Правило 2: Дистанция до предыдущих ворот ---
            dist = np.hypot(x - prev[0], y - prev[1])
            if dist < MIN_DIST:
                reward -= 2.0
                info["violations"].append(f"too_close({dist:.1f}m)")
            elif dist > MAX_DIST:
                reward -= 2.0
                info["violations"].append(f"too_far({dist:.1f}m)")
            else:
                reward += 1.0

            # --- Правило 3: Угол ворот ---
            angle_diff = abs(angle - prev[2]) % (2 * np.pi)
            if angle_diff > np.pi:
                angle_diff = 2 * np.pi - angle_diff
            if angle_diff > MAX_ANGLE_DIFF:
                reward -= 1.0
                info["violations"].append("gate_angle")
            else:
                reward += 0.5

            # --- Правило 4: Глобальная мин. дистанция (до всех, кроме предыдущих) ---
            for i, g in enumerate(self.gates[:-1]):
                d_global = np.hypot(x - g[0], y - g[1])
                if d_global < GLOBAL_MIN_DIST:
                    reward -= 2.0
                    info["violations"].append(f"global_too_close(g{i},{d_global:.1f}m)")
                    break

            # --- Правило 5: Пересечение сегментов ---
            # Новый сегмент: prev → (x, y)
            # Сравниваем со всеми существующими, кроме последнего (смежного)
            if len(self.gates) >= 2:
                new_p1 = (prev[0], prev[1])
                new_p2 = (x, y)
                for i in range(len(self.gates) - 2):
                    seg_a = (self.gates[i][0], self.gates[i][1])
                    seg_b = (self.gates[i + 1][0], self.gates[i + 1][1])
                    if _segments_intersect(new_p1, new_p2, seg_a, seg_b):
                        reward -= 2.0
                        info["violations"].append(f"crossing(seg{i})")
                        break

            # --- Правило 6: Разворот направления движения ---
            if len(self.gates) >= 2:
                prev2 = self.gates[-2]
                dir_prev = np.arctan2(prev[1] - prev2[1], prev[0] - prev2[0])
                dir_curr = np.arctan2(y - prev[1], x - prev[0])
                travel_diff = abs(dir_curr - dir_prev) % (2 * np.pi)
                if travel_diff > np.pi:
                    travel_diff = 2 * np.pi - travel_diff
                if travel_diff > MAX_TRAVEL_ANGLE:
                    reward -= 1.5
                    info["violations"].append(f"sharp_turn({np.degrees(travel_diff):.0f}°)")
                else:
                    reward += 0.3

            # --- Правило 7: Прогрессивная подсказка к замыканию ---
            # На последних 2 шагах поощряем нахождение в правильной дистанции от первых ворот
            steps_left = self.n_gates - self.step_num - 1
            if steps_left <= 1:
                first = self.gates[0]
                dist_to_first = np.hypot(x - first[0], y - first[1])
                if MIN_DIST <= dist_to_first <= MAX_DIST:
                    reward += 1.5
                elif dist_to_first < MIN_DIST * 1.5 or dist_to_first < MAX_DIST * 1.2:
                    reward += 0.5

        # Размещаем ворота
        self.gates.append((x, y, angle))
        self.step_num += 1

        # --- Правило 8: Замыкание (последний шаг) ---
        if self.step_num >= self.n_gates:
            self.done = True
            first = self.gates[0]
            last = self.gates[-1]
            close_dist = np.hypot(last[0] - first[0], last[1] - first[1])

            if MIN_DIST <= close_dist <= MAX_DIST:
                reward += 5.0
                info["closed"] = True
            else:
                reward -= 3.0
                info["closed"] = False
                info["violations"].append(f"closure({close_dist:.1f}m)")

            # Угол замыкания
            angle_diff_close = abs(last[2] - first[2]) % (2 * np.pi)
            if angle_diff_close > np.pi:
                angle_diff_close = 2 * np.pi - angle_diff_close
            if angle_diff_close <= MAX_ANGLE_DIFF:
                reward += 1.0
            else:
                reward -= 1.0
                info["violations"].append("closure_angle")

            # Пересечение замыкающего сегмента
            close_p1 = (last[0], last[1])
            close_p2 = (first[0], first[1])
            # Проверяем со всеми сегментами кроме (0,1) и (n-2, n-1) — смежных с замыканием
            for i in range(1, len(self.gates) - 2):
                seg_a = (self.gates[i][0], self.gates[i][1])
                seg_b = (self.gates[i + 1][0], self.gates[i + 1][1])
                if _segments_intersect(close_p1, close_p2, seg_a, seg_b):
                    reward -= 2.0
                    info["violations"].append("closure_crossing")
                    break

            # Coverage-бонус: вознаграждаем за разброс ворот по арене
            coords = np.array([(g[0], g[1]) for g in self.gates])
            spread = coords.std(axis=0).mean()  # среднее стандартное отклонение по x и y
            reward += 0.5 * spread
            info["spread"] = round(float(spread), 2)

        return self._get_state(), reward, self.done, info

    def _get_state(self) -> np.ndarray:
        """
        Возвращает текущее состояние как вектор длиной STATE_DIM=25.

        Формат: [прогресс, x1,y1,a1, x2,y2,a2, ..., x8,y8,a8]
        Ворота заполняются в порядке размещения.
        Незаполненные слоты = нули.

        В отличие от старого state (7 чисел с только первыми/последними),
        агент теперь видит ВСЕ размещённые ворота — это критично для
        избежания пересечений и кластеризации.
        """
        state = np.zeros(STATE_DIM, dtype=np.float32)
        state[0] = self.step_num / self.n_gates

        for i, (gx, gy, ga) in enumerate(self.gates):
            if i >= MAX_GATES_STATE:
                break
            base = 1 + i * 3
            state[base]     = (gx - WORK_MIN) / WORK_RANGE  # x нормализованный
            state[base + 1] = (gy - WORK_MIN) / WORK_RANGE  # y нормализованный
            state[base + 2] = ga / (2 * np.pi)              # angle нормализованный

        return state

    def get_config(self) -> np.ndarray:
        """Возвращает текущую конфигурацию как массив (N, 3)."""
        return np.array(self.gates, dtype=np.float32)

    def is_valid(self) -> bool:
        """
        Проверяет валидность конфигурации по ВСЕМ правилам:
        - границы
        - дистанции между соседними воротами (3–10 м)
        - углы ворот (≤ 180°)
        - отсутствие пересечений сегментов
        - глобальная мин. дистанция 2 м между любыми воротами
        """
        if len(self.gates) < self.n_gates:
            return False

        config = self.get_config()
        n = len(config)

        # Границы
        for i in range(n):
            if not (WORK_MIN <= config[i, 0] <= WORK_MAX):
                return False
            if not (WORK_MIN <= config[i, 1] <= WORK_MAX):
                return False

        # Дистанции между соседними (включая замыкание)
        for i in range(n):
            j = (i + 1) % n
            d = np.hypot(config[j, 0] - config[i, 0], config[j, 1] - config[i, 1])
            if d < MIN_DIST or d > MAX_DIST:
                return False

        # Углы ворот между соседними
        for i in range(n):
            j = (i + 1) % n
            diff = abs(config[i, 2] - config[j, 2]) % (2 * np.pi)
            if diff > np.pi:
                diff = 2 * np.pi - diff
            if diff > MAX_ANGLE_DIFF:
                return False

        # Пересечения сегментов (для замкнутого маршрута)
        # Сегменты: (0,1), (1,2), ..., (n-2,n-1), (n-1,0)
        for i in range(n):
            p1 = (config[i, 0], config[i, 1])
            p2 = (config[(i + 1) % n, 0], config[(i + 1) % n, 1])
            for j in range(i + 2, n):
                # Пропускаем смежные сегменты через замыкание
                if i == 0 and j == n - 1:
                    continue
                q1 = (config[j, 0], config[j, 1])
                q2 = (config[(j + 1) % n, 0], config[(j + 1) % n, 1])
                if _segments_intersect(p1, p2, q1, q2):
                    return False

        # Глобальная мин. дистанция между любыми воротами (не только соседними)
        for i in range(n):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue  # соседние через замыкание
                d = np.hypot(config[j, 0] - config[i, 0], config[j, 1] - config[i, 1])
                if d < GLOBAL_MIN_DIST:
                    return False

        return True
