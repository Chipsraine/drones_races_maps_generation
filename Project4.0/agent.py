"""
RL-агент на основе PPO — подход "всё за один шаг".

Ключевое отличие от v2:
- Нет GRU (не нужен — нет последовательности шагов)
- Модель получает число ворот → выдаёт ВСЕ позиции сразу
- Один эпизод = один шаг → PPO работает стабильнее

Архитектура:
  n_gates (1) → [Linear 256 → ReLU] × 3 → два выхода:
    - Actor: mean для всех ворот (MAX_GATES * 3)
    - Critic: value (1 число)
"""

import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal

from environment import MAX_GATES


ACTION_DIM = MAX_GATES * 3


class PolicyNetwork(nn.Module):
    """
    Нейросеть: число ворот → позиции всех ворот.

    Вход: (batch, 1) — нормализованное число ворот
    Выход:
      - action_mean: (batch, 18) — средние для Gaussian policy
      - action_std: (batch, 18) — стандартные отклонения
      - value: (batch,) — оценка качества состояния
    """

    def __init__(self, state_dim: int = 1, hidden_dim: int = 256):
        super().__init__()

        # Общий энкодер
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Actor: предсказывает позиции всех ворот
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, ACTION_DIM),
            nn.Sigmoid(),  # [0, 1] — далее среда переводит в реальные координаты
        )
        self.actor_log_std = nn.Parameter(torch.zeros(ACTION_DIM))

        # Critic: оценивает качество текущего состояния
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, state):
        """
        Args:
            state: (batch, STATE_DIM)
        Returns:
            action_mean: (batch, ACTION_DIM)
            action_std: (batch, ACTION_DIM)
            value: (batch,)
        """
        x = self.encoder(state)
        action_mean = self.actor_mean(x)
        action_std = self.actor_log_std.exp().expand_as(action_mean)
        value = self.critic(x).squeeze(-1)
        return action_mean, action_std, value

    def select_action(self, state: np.ndarray):
        """
        Выбирает действие для одного состояния.

        Returns:
            action, log_prob, value
        """
        state_t = torch.FloatTensor(state).unsqueeze(0)
        mean, std, value = self.forward(state_t)
        dist = Normal(mean, std)
        action = dist.sample()
        action = action.clamp(0, 1)
        log_prob = dist.log_prob(action).sum(-1)

        return (action.squeeze(0).detach().numpy(),
                log_prob.squeeze(0).detach(),
                value.squeeze(0).detach())

    def evaluate(self, states, actions):
        """Оценивает действия (при обновлении PPO)."""
        mean, std, values = self.forward(states)
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_probs, values, entropy


class PPOTrainer:
    """
    PPO для одношаговых эпизодов.

    Упрощение: каждый эпизод = 1 шаг, поэтому returns = reward (без дисконта).
    """

    def __init__(self, policy: PolicyNetwork, lr: float = 3e-4,
                 clip_eps: float = 0.2, value_coef: float = 0.5,
                 entropy_coef: float = 0.02, ppo_epochs: int = 10):
        self.policy = policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.ppo_epochs = ppo_epochs

    def update(self, episodes_data: list[dict]) -> dict:
        """
        Обновляет веса по собранным эпизодам.

        Для одношаговых эпизодов: return = reward (без дисконта).
        """
        states = torch.FloatTensor(np.array([ep["state"] for ep in episodes_data]))
        actions = torch.FloatTensor(np.array([ep["action"] for ep in episodes_data]))
        rewards = torch.FloatTensor([ep["reward"] for ep in episodes_data])
        old_log_probs = torch.stack([ep["log_prob"] for ep in episodes_data])
        old_values = torch.stack([ep["value"] for ep in episodes_data])

        # Returns = rewards (1-step episodes, no discount)
        returns = rewards

        # Advantages
        advantages = returns - old_values
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        total_loss_val = 0
        for _ in range(self.ppo_epochs):
            new_log_probs, new_values, entropy = self.policy.evaluate(states, actions)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantages
            surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = (new_values - returns).pow(2).mean()
            entropy_loss = -entropy.mean()

            loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
            self.optimizer.step()

            total_loss_val += loss.item()

        return {
            "loss": total_loss_val / self.ppo_epochs,
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
        }
