"""
Отдельная сеть для флагов: получает ворота → выдаёт флаги.
"""

import torch
import torch.nn as nn
import numpy as np


class FlagNetwork(nn.Module):
    """
    Вход: позиции ворот (max_gates * 3) — нормализованные [0,1]
    Выход: позиции флагов (max_flags * 2) — нормализованные [0,1]
    """

    def __init__(self, max_gates=6, max_flags=6, hidden_dim=256):
        super().__init__()
        self.max_gates = max_gates
        self.max_flags = max_flags
        input_dim = max_gates * 3
        output_dim = max_flags * 2

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid(),  # [0, 1]
        )

    def forward(self, gates):
        """
        Args:
            gates: (batch, max_gates * 3) — нормализованные ворота
        Returns:
            flags: (batch, max_flags * 2) — нормализованные флаги
        """
        return self.net(gates)

    def predict_flags(self, gates_np, n_flags):
        """
        gates_np: (n_gates, 3) — реальные координаты ворот [0, 10]
        n_flags: int

        Returns: (n_flags, 2) — реальные координаты флагов [0, 10]
        """
        # Нормализуем ворота
        gates_flat = gates_np.flatten() / 10.0
        # Дополняем нулями до max_gates * 3
        gates_padded = np.zeros(self.max_gates * 3, dtype=np.float32)
        gates_padded[:len(gates_flat)] = gates_flat

        gates_t = torch.FloatTensor(gates_padded).unsqueeze(0)

        with torch.no_grad():
            flags_norm = self.forward(gates_t).squeeze(0).numpy()

        # Денормализуем флаги
        flags = flags_norm[:n_flags * 2] * 10.0
        return flags.reshape(n_flags, 2)


class FlagTrainer:
    def __init__(self, network, lr=1e-3):
        self.network = network
        self.optimizer = torch.optim.Adam(network.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def update(self, batch_gates, batch_flags, batch_masks):
        """
        batch_gates: (batch, max_gates * 3)
        batch_flags: (batch, max_flags * 2)
        batch_masks: (batch, max_flags * 2) — 1 на активных флагах
        """
        pred = self.network(batch_gates)
        loss = ((pred - batch_flags) ** 2 * batch_masks).sum() / batch_masks.sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()