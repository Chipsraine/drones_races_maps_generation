"""
Среда (Environment) для RL-агента — подход "всё за один шаг".

Агент выдаёт ВСЕ ворота и столбики сразу, среда оценивает конфигурацию целиком.

Объекты на арене:
- Ворота (gates): дрон пролетает сквозь них по порядку (замкнутый маршрут)
- Столбики (pillars): дрон должен их облетать (путь не проходит через столбик)
"""

import numpy as np


# === ПРАВИЛА АРЕНЫ ===
ARENA_SIZE = 10.0
GATE_SIZE = 1.0
MIN_DIST = 3.0
MAX_DIST = 10.0
MAX_ANGLE_DIFF = np.pi       # 180°

WORK_MIN = 0.0
WORK_MAX = ARENA_SIZE         # 10.0
WORK_RANGE = WORK_MAX - WORK_MIN  # 10.0

# === ДОПОЛНИТЕЛЬНЫЕ ПРАВИЛА ===
GLOBAL_MIN_DIST = 2.0        # мин. расстояние между ЛЮБЫМИ воротами
MAX_TRAVEL_ANGLE = np.radians(150)

# === СТОЛБИКИ ===
MAX_PILLARS = 2               # кол-во столбиков
PILLAR_RADIUS = 0.5           # радиус «no-fly» зоны вокруг столбика
PILLAR_MIN_DIST_GATE = 1.5    # мин. расстояние от столбика до ворот
PILLAR_MIN_DIST_PILLAR = 2.0  # мин. расстояние между столбиками

MAX_GATES = 6

# Размерность action: ворота (x,y,angle) + столбики (x,y)
ACTION_DIM = MAX_GATES * 3  # 18 + 4 = 22


def _segments_intersect(p1, p2, p3, p4):
    """Проверяет, пересекаются ли два отрезка p1-p2 и p3-p4."""
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


def _point_to_segment_dist(px, py, x1, y1, x2, y2):
    """Минимальное расстояние от точки (px,py) до отрезка (x1,y1)-(x2,y2)."""
    dx, dy = x2 - x1, y2 - y1
    len_sq = dx * dx + dy * dy
    if len_sq == 0:
        return np.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / len_sq))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return np.hypot(px - proj_x, py - proj_y)


class GateEnvironment:
    """
    Одношаговая среда: агент выдаёт ворота + столбики за один шаг.
    """

    def __init__(self):
        self.n_gates = 5
        self.config = None      # (n_gates, 3) — ворота
        self.pillars = None     # (MAX_PILLARS, 2) — столбики

    def reset(self, n_gates: int | None = None) -> np.ndarray:
        if n_gates is not None:
            self.n_gates = n_gates
        self.config = None
        self.pillars = None
        return self._get_state()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        """
        Args:
            action: (ACTION_DIM,) — [gate1_x,y,a, ..., gateN_x,y,a, pillar1_x,y, pillar2_x,y]
        """
        n = self.n_gates

        # Декодируем ворота
        gates = []
        for i in range(n):
            x = action[i * 3]     * WORK_RANGE + WORK_MIN
            y = action[i * 3 + 1] * WORK_RANGE + WORK_MIN
            a = action[i * 3 + 2] * 2 * np.pi
            gates.append((x, y, a))
        self.config = np.array(gates, dtype=np.float32)

        # Столбики убраны для упрощения
        self.pillars = np.array([], dtype=np.float32).reshape(0, 2)

        reward, info = self._evaluate_config()
        return self._get_state(), reward, True, info

    def _evaluate_config(self) -> tuple[float, dict]:
        config = self.config
        pillars = self.pillars
        n = len(config)
        reward = 0.0
        info = {"violations": [], "bonuses": []}

        # === 1. Границы ворот ===
        for i in range(n):
            if WORK_MIN <= config[i, 0] <= WORK_MAX and WORK_MIN <= config[i, 1] <= WORK_MAX:
                reward += 0.3
            else:
                reward -= 1.0
                info["violations"].append(f"bounds(g{i})")

        # === 2. Дистанции между соседними ===
        dist_ok = 0
        for i in range(n):
            j = (i + 1) % n
            d = np.hypot(config[j, 0] - config[i, 0], config[j, 1] - config[i, 1])
            if MIN_DIST <= d <= MAX_DIST:
                reward += 1.5
                dist_ok += 1
            elif d < MIN_DIST:
                reward -= 2.0
                info["violations"].append(f"too_close({i}→{j},{d:.1f}m)")
            else:
                reward -= 2.0
                info["violations"].append(f"too_far({i}→{j},{d:.1f}m)")

        if dist_ok == n:
            reward += 5.0
            info["bonuses"].append("all_dist_ok")

        # === 3. Углы ворот ===
        for i in range(n):
            j = (i + 1) % n
            diff = abs(config[i, 2] - config[j, 2]) % (2 * np.pi)
            if diff > np.pi:
                diff = 2 * np.pi - diff
            if diff <= MAX_ANGLE_DIFF:
                reward += 0.5
            else:
                reward -= 1.0
                info["violations"].append(f"gate_angle({i}→{j})")

        # === 4. Глобальная мин. дистанция ===
        for i in range(n):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                d = np.hypot(config[j, 0] - config[i, 0], config[j, 1] - config[i, 1])
                if d < GLOBAL_MIN_DIST:
                    reward -= 2.0
                    info["violations"].append(f"global_close({i},{j},{d:.1f}m)")

        # === 5. Пересечения сегментов ===
        crossing_count = 0
        for i in range(n):
            p1 = (config[i, 0], config[i, 1])
            p2 = (config[(i + 1) % n, 0], config[(i + 1) % n, 1])
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                q1 = (config[j, 0], config[j, 1])
                q2 = (config[(j + 1) % n, 0], config[(j + 1) % n, 1])
                if _segments_intersect(p1, p2, q1, q2):
                    crossing_count += 1

        if crossing_count > 0:
            reward -= 2.0 * crossing_count
            info["violations"].append(f"crossings({crossing_count})")
        else:
            reward += 3.0
            info["bonuses"].append("no_crossings")

        # === 6. Направление движения ===
        sharp_turns = 0
        for i in range(n):
            prev = (i - 1) % n
            nxt = (i + 1) % n
            dir_in = np.arctan2(config[i, 1] - config[prev, 1],
                                config[i, 0] - config[prev, 0])
            dir_out = np.arctan2(config[nxt, 1] - config[i, 1],
                                 config[nxt, 0] - config[i, 0])
            travel_diff = abs(dir_out - dir_in) % (2 * np.pi)
            if travel_diff > np.pi:
                travel_diff = 2 * np.pi - travel_diff
            if travel_diff > MAX_TRAVEL_ANGLE:
                sharp_turns += 1

        if sharp_turns > 0:
            reward -= 1.0 * sharp_turns
            info["violations"].append(f"sharp_turns({sharp_turns})")
        else:
            reward += 1.0
            info["bonuses"].append("smooth_path")

        # === 7. Coverage-бонус ===
        coords = config[:, :2]
        spread = coords.std(axis=0).mean()
        reward += spread
        info["spread"] = round(float(spread), 2)

        # === 8. Столбики ===
        info["n_violations"] = len(info["violations"])
        info["closed"] = dist_ok == n

        return reward, info
    def _get_state(self) -> np.ndarray:
        return np.array([self.n_gates / MAX_GATES], dtype=np.float32)

    def get_config(self) -> np.ndarray:
        return self.config

    def get_pillars(self) -> np.ndarray:
        return self.pillars

    def is_valid(self) -> bool:
        """Проверка валидности без столбиков."""
        config = self.config
        n = len(config)
        if n == 0:
            return False

        # 1. Границы ворот
        for i in range(n):
            if not (WORK_MIN <= config[i, 0] <= WORK_MAX and 
                    WORK_MIN <= config[i, 1] <= WORK_MAX):
                return False

        # 2. Дистанции между соседними
        for i in range(n):
            j = (i + 1) % n
            d = np.hypot(config[j, 0] - config[i, 0], config[j, 1] - config[i, 1])
            if not (MIN_DIST <= d <= MAX_DIST):
                return False

        # 3. Углы ворот
        for i in range(n):
            j = (i + 1) % n
            diff = abs(config[i, 2] - config[j, 2]) % (2 * np.pi)
            if diff > np.pi:
                diff = 2 * np.pi - diff
            if diff > MAX_ANGLE_DIFF:
                return False

        # 4. Глобальная мин. дистанция
        for i in range(n):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                d = np.hypot(config[j, 0] - config[i, 0], config[j, 1] - config[i, 1])
                if d < GLOBAL_MIN_DIST:
                    return False

        # 5. Пересечения сегментов
        for i in range(n):
            p1 = (config[i, 0], config[i, 1])
            p2 = (config[(i + 1) % n, 0], config[(i + 1) % n, 1])
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                q1 = (config[j, 0], config[j, 1])
                q2 = (config[(j + 1) % n, 0], config[(j + 1) % n, 1])
                if _segments_intersect(p1, p2, q1, q2):
                    return False

        # 6. Направление движения
        for i in range(n):
            prev = (i - 1) % n
            nxt = (i + 1) % n
            dir_in = np.arctan2(config[i, 1] - config[prev, 1],
                                config[i, 0] - config[prev, 0])
            dir_out = np.arctan2(config[nxt, 1] - config[i, 1],
                                 config[nxt, 0] - config[i, 0])
            travel_diff = abs(dir_out - dir_in) % (2 * np.pi)
            if travel_diff > np.pi:
                travel_diff = 2 * np.pi - travel_diff
            if travel_diff > MAX_TRAVEL_ANGLE:
                return False

        return True
