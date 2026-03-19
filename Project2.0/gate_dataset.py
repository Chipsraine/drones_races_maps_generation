"""
PyTorch Dataset для конфигураций ворот.

Нормализация, паддинг, подготовка для авторегрессивного обучения.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

from gate_generator import WORK_MIN, WORK_MAX, MAX_GATES

WORK_RANGE = WORK_MAX - WORK_MIN  # 4.0


def normalize_config(config: np.ndarray) -> np.ndarray:
    """
    Нормализует конфигурацию в [0, 1].
    x, y: (val - WORK_MIN) / WORK_RANGE
    angle: val / (2π)
    """
    normed = config.copy()
    normed[:, 0] = (config[:, 0] - WORK_MIN) / WORK_RANGE
    normed[:, 1] = (config[:, 1] - WORK_MIN) / WORK_RANGE
    normed[:, 2] = config[:, 2] / (2 * np.pi)
    return normed


def denormalize_config(normed: np.ndarray) -> np.ndarray:
    """Обратная нормализация."""
    config = normed.copy()
    config[:, 0] = normed[:, 0] * WORK_RANGE + WORK_MIN
    config[:, 1] = normed[:, 1] * WORK_RANGE + WORK_MIN
    config[:, 2] = normed[:, 2] * (2 * np.pi)
    return config


class GateDataset(Dataset):
    """
    Датасет для авторегрессивного обучения.

    Каждый элемент:
    - input_seq: (max_len, 3) — нормализованная последовательность ворот (вход)
    - target_seq: (max_len, 3) — сдвинутая на 1 последовательность (таргет)
    - mask: (max_len,) — маска реальных элементов (1=реальный, 0=padding)
    - length: скаляр — реальная длина последовательности
    """

    def __init__(self, configs: list[np.ndarray], max_len: int = MAX_GATES + 1):
        """
        Args:
            configs: список конфигураций, каждая shape (N_i, 3)
            max_len: макс. длина с учётом замыкания (+1 для повтора первых ворот)
        """
        self.max_len = max_len
        self.samples = []

        for config in configs:
            # Добавляем первые ворота в конец (замкнутость)
            closed = np.vstack([config, config[0:1]])
            normed = normalize_config(closed)

            seq_len = len(normed)

            # Input: все кроме последнего (ворота 0..N-1)
            # Target: все кроме первого (ворота 1..N, где N = повтор 0-х)
            input_seq = np.zeros((max_len, 3), dtype=np.float32)
            target_seq = np.zeros((max_len, 3), dtype=np.float32)
            mask = np.zeros(max_len, dtype=np.float32)

            real_len = seq_len - 1  # кол-во пар input→target
            input_seq[:real_len] = normed[:-1]
            target_seq[:real_len] = normed[1:]
            mask[:real_len] = 1.0

            self.samples.append({
                "input_seq": torch.tensor(input_seq),
                "target_seq": torch.tensor(target_seq),
                "mask": torch.tensor(mask),
                "length": torch.tensor(real_len, dtype=torch.long),
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def create_datasets(configs: list[np.ndarray], val_ratio: float = 0.2, seed: int = 42):
    """Разбивает конфигурации на train/val и создаёт датасеты."""
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(configs))
    split = int(len(configs) * (1 - val_ratio))

    train_configs = [configs[i] for i in indices[:split]]
    val_configs = [configs[i] for i in indices[split:]]

    train_ds = GateDataset(train_configs)
    val_ds = GateDataset(val_configs)

    print(f"Train: {len(train_ds)}, Val: {len(val_ds)}")
    return train_ds, val_ds
