"""
Среда для генерации трасс с воротами и флагами.
База — рабочий environment_simple.py (Project5.0) + флаги.
"""

import numpy as np

# === ПРАВИЛА АРЕНЫ (как в Project5.0) ===
ARENA_SIZE = 10.0
GATE_SIZE = 1.0
MIN_DIST = 3.0
MAX_DIST = 10.0
MAX_ANGLE_DIFF = np.pi       # 180°

WORK_MIN = 0.0
WORK_MAX = ARENA_SIZE
WORK_RANGE = WORK_MAX - WORK_MIN

GLOBAL_MIN_DIST = 2.0
MAX_TRAVEL_ANGLE = np.radians(150)

MAX_GATES = 6
MAX_FLAGS = 6
PILLAR_RADIUS = 0.3

ACTION_DIM = MAX_GATES * 3 + MAX_FLAGS * 2


def _segments_intersect(p1, p2, p3, p4):
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
    dx, dy = x2 - x1, y2 - y1
    len_sq = dx * dx + dy * dy
    if len_sq == 0:
        return np.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / len_sq))
    proj_x = x1 + t * dx
    proj_y = y1 + t * dy
    return np.hypot(px - proj_x, py - proj_y)


class GateEnvironmentWithFlags:
    
    def __init__(self):
        self.n_gates = 5
        self.n_flags = 5
        self.config = None
        self.flags = None

    def reset(self, n_gates: int, n_flags: int | None = None) -> np.ndarray:
        self.n_gates = n_gates
        if n_flags is None:
            self.n_flags = n_gates
        else:
            self.n_flags = min(n_flags, n_gates)
        self.config = None
        self.flags = None
        return self._get_state()

    def _get_state(self) -> np.ndarray:
        return np.array([self.n_gates / MAX_GATES, self.n_flags / MAX_FLAGS], dtype=np.float32)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        n = self.n_gates
        nf = self.n_flags

        gates = []
        for i in range(n):
            x = action[i * 3]     * WORK_RANGE + WORK_MIN
            y = action[i * 3 + 1] * WORK_RANGE + WORK_MIN
            a = action[i * 3 + 2] * 2 * np.pi
            gates.append((x, y, a))
        self.config = np.array(gates, dtype=np.float32)

        flag_offset = n * 3
        flags = []
        for i in range(nf):
            fx = action[flag_offset + i * 2]     * WORK_RANGE + WORK_MIN
            fy = action[flag_offset + i * 2 + 1] * WORK_RANGE + WORK_MIN
            flags.append((fx, fy))
        self.flags = np.array(flags, dtype=np.float32)

        reward, info = self._evaluate_config()
        
                # === БОНУС ЗА ВАЛИДНОСТЬ ===
        n_violations = len(info["violations"])
        
        if n_violations == 0:
            reward += 200.0  # КОСМИЧЕСКИЙ бонус за идеал
            info["bonuses"].append("PERFECT")
        elif n_violations == 1:
            reward += 80.0
            info["bonuses"].append("ALMOST_1")
        elif n_violations == 2:
            reward += 30.0
            info["bonuses"].append("ALMOST_2")
        elif n_violations == 3:
            reward += 10.0
            info["bonuses"].append("ALMOST_3")

        return self._get_state(), reward, True, info

    def _evaluate_config(self) -> tuple[float, dict]:
        config = self.config
        flags = self.flags
        n = len(config)
        nf = len(flags) if flags is not None else 0
        reward = 0.0
        info = {"violations": [], "bonuses": []}

        # 1. Границы
        for i in range(n):
            if WORK_MIN <= config[i, 0] <= WORK_MAX and WORK_MIN <= config[i, 1] <= WORK_MAX:
                reward += 0.3
            else:
                reward -= 1.0
                info["violations"].append(f"bounds(g{i})")

        # 2. Дистанции соседних
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

        # 3. Углы ворот (orientation) — с 180° всегда ок
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

        # 4. Глобальная мин. дистанция
        for i in range(n):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                d = np.hypot(config[j, 0] - config[i, 0], config[j, 1] - config[i, 1])
                if d < GLOBAL_MIN_DIST:
                    reward -= 2.0
                    info["violations"].append(f"global_close({i},{j},{d:.1f}m)")

        # 5. Пересечения
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

        # 6. Направление движения
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

        # 7. Coverage
        coords = config[:, :2]
        spread = coords.std(axis=0).mean()
        reward += spread
        info["spread"] = round(float(spread), 2)

        # 8. Флаги
        if nf > 0:
            for i in range(nf):
                flag = flags[i]
                gate1 = config[i]
                gate2 = config[(i + 1) % n]
                
                dx = gate2[0] - gate1[0]
                dy = gate2[1] - gate1[1]
                length = np.hypot(dx, dy)
                
                if length > 0:
                    dist_to_line = abs((flag[1] - gate1[1]) * dx - (flag[0] - gate1[0]) * dy) / length
                    if dist_to_line < 0.5:
                        reward += 2.0
                        info["bonuses"].append(f"flag_on_line_{i}")
                    else:
                        reward -= dist_to_line
                        info["violations"].append(f"flag_off_line_{i}")
                    
                    t = ((flag[0] - gate1[0]) * dx + (flag[1] - gate1[1]) * dy) / (length ** 2)
                    if 0.1 <= t <= 0.9:
                        reward += 1.0
                    else:
                        reward -= 1.0
                        info["violations"].append(f"flag_not_between_{i}")
                    
                    dist_to_g1 = np.hypot(flag[0] - gate1[0], flag[1] - gate1[1])
                    dist_to_g2 = np.hypot(flag[0] - gate2[0], flag[1] - gate2[1])
                    if dist_to_g1 < 1.0 or dist_to_g2 < 1.0:
                        reward -= 2.0
                        info["violations"].append(f"flag_too_close_{i}")

        info["n_violations"] = len(info["violations"])
        info["closed"] = dist_ok == n
        return reward, info

    def is_valid(self) -> bool:
        if self.config is None:
            return False
        
        config = self.config
        flags = self.flags
        n = len(config)
        nf = len(flags) if flags is not None else 0

        for i in range(n):
            if not (WORK_MIN <= config[i, 0] <= WORK_MAX and 
                    WORK_MIN <= config[i, 1] <= WORK_MAX):
                return False

        for i in range(n):
            j = (i + 1) % n
            d = np.hypot(config[j, 0] - config[i, 0], config[j, 1] - config[i, 1])
            if not (MIN_DIST <= d <= MAX_DIST):
                return False

        for i in range(n):
            j = (i + 1) % n
            diff = abs(config[i, 2] - config[j, 2]) % (2 * np.pi)
            if diff > np.pi:
                diff = 2 * np.pi - diff
            if diff > MAX_ANGLE_DIFF:
                return False

        for i in range(n):
            for j in range(i + 2, n):
                if i == 0 and j == n - 1:
                    continue
                d = np.hypot(config[j, 0] - config[i, 0], config[j, 1] - config[i, 1])
                if d < GLOBAL_MIN_DIST:
                    return False

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

        if nf > 0:
            for i in range(nf):
                flag = flags[i]
                gate1 = config[i]
                gate2 = config[(i + 1) % n]
                
                dx = gate2[0] - gate1[0]
                dy = gate2[1] - gate1[1]
                length = np.hypot(dx, dy)
                if length == 0:
                    return False
                
                dist_to_line = abs((flag[1] - gate1[1]) * dx - (flag[0] - gate1[0]) * dy) / length
                if dist_to_line > 0.5:
                    return False
                
                t = ((flag[0] - gate1[0]) * dx + (flag[1] - gate1[1]) * dy) / (length ** 2)
                if not (0.1 <= t <= 0.9):
                    return False
                
                dist_to_g1 = np.hypot(flag[0] - gate1[0], flag[1] - gate1[1])
                dist_to_g2 = np.hypot(flag[0] - gate2[0], flag[1] - gate2[1])
                if dist_to_g1 < 1.0 or dist_to_g2 < 1.0:
                    return False

        return True

    def get_config(self):
        return self.config

    def get_flags(self):
        return self.flags