"""
Авторегрессивная GRU-модель для генерации конфигураций ворот.

Вход на каждом шаге: (x, y, angle) предыдущих ворот
Выход: (x, y, angle) следующих ворот
"""

import torch
import torch.nn as nn


class GateGRU(nn.Module):
    """
    GRU-модель для последовательной генерации ворот.

    Архитектура:
        input (3) → Linear → GRU (2 layers) → Linear → output (3)
        x, y: sigmoid (→ [0,1])
        angle: sigmoid (→ [0,1])
    """

    def __init__(self, input_dim: int = 3, hidden_dim: int = 128,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Encoder: проецирует (x, y, angle) в пространство эмбеддингов
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # GRU
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Decoder: предсказывает (x, y, angle)
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),  # все выходы в [0, 1]
        )

    def forward(self, x: torch.Tensor, hidden: torch.Tensor | None = None):
        """
        Args:
            x: (batch, seq_len, 3) — нормализованные ворота
            hidden: (num_layers, batch, hidden_dim) или None

        Returns:
            output: (batch, seq_len, 3) — предсказания следующих ворот
            hidden: (num_layers, batch, hidden_dim) — финальное скрытое состояние
        """
        # (batch, seq_len, 3) → (batch, seq_len, hidden_dim)
        embedded = self.input_proj(x)

        # GRU
        gru_out, hidden = self.gru(embedded, hidden)

        # (batch, seq_len, hidden_dim) → (batch, seq_len, 3)
        output = self.output_proj(gru_out)

        return output, hidden

    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Инициализация скрытого состояния нулями."""
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)

    def generate_step(self, x: torch.Tensor, hidden: torch.Tensor):
        """
        Один шаг генерации (для инференса).

        Args:
            x: (batch, 1, 3) — текущие ворота
            hidden: (num_layers, batch, hidden_dim)

        Returns:
            pred: (batch, 3) — предсказание следующих ворот
            hidden: новое скрытое состояние
        """
        output, hidden = self.forward(x, hidden)
        return output[:, 0, :], hidden


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = GateGRU()
    print(f"Параметров: {count_parameters(model):,}")

    # Тест forward
    batch = torch.randn(4, 8, 3).sigmoid()
    out, h = model(batch)
    print(f"Input:  {batch.shape}")
    print(f"Output: {out.shape}")
    print(f"Hidden: {h.shape}")
    print(f"Output range: [{out.min().item():.3f}, {out.max().item():.3f}]")
