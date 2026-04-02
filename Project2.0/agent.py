"""
RL-агент на основе PPO (Proximal Policy Optimization).

Что такое PPO (для объяснения преподавателю):
- Агент (нейросеть) выбирает действия (ставит ворота)
- Среда даёт reward за правильные действия и penalty за неправильные
- PPO обновляет веса сети так, чтобы вероятность "хороших" действий
  увеличивалась, а "плохих" — уменьшалась
- "Proximal" = обновления ограничены, чтобы обучение было стабильным

Архитектура нейросети:
  State (7) → [Linear 64 → ReLU] → [GRU 64] → два выхода:
    - Policy head (actor): предсказывает mean и std для действий
    - Value head (critic): оценивает "качество" текущего состояния
"""

import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal


class PolicyNetwork(nn.Module):
    """
    Нейросеть агента с двумя головами:

    1. Actor (политика): решает, КУДА ставить ворота
       - Выдаёт среднее (mean) и разброс (std) для нормального распределения
       - Из этого распределения сэмплируется действие (x, y, angle)

    2. Critic (оценщик): оценивает, НАСКОЛЬКО ХОРОШО текущее состояние
       - Помогает понять: "мы на пути к хорошей конфигурации или нет?"
    """

    def __init__(self, state_dim: int = 25, action_dim: int = 3, hidden_dim: int = 128):
        super().__init__()

        # Общая часть: обрабатывает состояние
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
        )

        # GRU — запоминает историю предыдущих шагов
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        # Actor: предсказывает параметры распределения действий
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Sigmoid(),  # действия в [0, 1]
        )
        # Логарифм стандартного отклонения (learnable)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim) - 0.5)

        # Critic: оценивает ценность состояния (одно число)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self.hidden_dim = hidden_dim

    def forward(self, state, hidden=None):
        """
        Args:
            state: (batch, state_dim) или (batch, 1, state_dim)
            hidden: скрытое состояние GRU

        Returns:
            action_mean, action_std, value, hidden
        """
        if state.dim() == 2:
            state = state.unsqueeze(1)  # (batch, 1, state_dim)

        x = self.state_encoder(state)  # (batch, 1, hidden)
        gru_out, hidden = self.gru(x, hidden)  # (batch, 1, hidden)
        gru_out = gru_out.squeeze(1)  # (batch, hidden)

        action_mean = self.actor_mean(gru_out)   # (batch, 3)
        action_std = self.actor_log_std.exp().expand_as(action_mean)  # (batch, 3)
        value = self.critic(gru_out).squeeze(-1)  # (batch,)

        return action_mean, action_std, value, hidden

    def init_hidden(self, batch_size=1):
        return torch.zeros(1, batch_size, self.hidden_dim)

    def select_action(self, state, hidden=None):
        """
        Выбирает действие для одного состояния (при сборе опыта).

        Returns:
            action, log_prob, value, hidden
        """
        state_t = torch.FloatTensor(state).unsqueeze(0)  # (1, state_dim)
        if hidden is None:
            hidden = self.init_hidden()

        mean, std, value, hidden = self.forward(state_t, hidden)
        dist = Normal(mean, std)
        action = dist.sample()
        action = action.clamp(0, 1)  # ограничиваем [0, 1]
        log_prob = dist.log_prob(action).sum(-1)  # сумма по (x, y, angle)

        return (action.squeeze(0).detach().numpy(),
                log_prob.squeeze(0).detach(),
                value.detach(),
                hidden.detach())

    def evaluate(self, states, actions, hiddens=None):
        """
        Оценивает действия (при обновлении PPO).
        """
        mean, std, values, _ = self.forward(states, hiddens)
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_probs, values, entropy


class PPOTrainer:
    """
    PPO алгоритм обучения.

    Принцип работы:
    1. Агент играет N эпизодов, собирает опыт (состояния, действия, награды)
    2. Вычисляет "advantage" — насколько действие было лучше среднего
    3. Обновляет веса сети, чтобы хорошие действия стали вероятнее
    4. Ограничивает размер обновления (clip) для стабильности
    """

    def __init__(self, policy: PolicyNetwork, lr: float = 3e-4,
                 gamma: float = 0.99, clip_eps: float = 0.2,
                 value_coef: float = 0.5, entropy_coef: float = 0.01,
                 ppo_epochs: int = 4):
        self.policy = policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        self.gamma = gamma           # дисконт будущих наград
        self.clip_eps = clip_eps     # ограничение обновления PPO
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.ppo_epochs = ppo_epochs

    def compute_returns(self, rewards: list[float], values: list[torch.Tensor],
                        done: bool) -> torch.Tensor:
        """
        Вычисляет дисконтированные возвраты (returns).
        Return_t = reward_t + gamma * Return_{t+1}
        """
        returns = []
        R = 0 if done else values[-1].item()
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        return torch.FloatTensor(returns)

    def update(self, episodes_data: list[dict]) -> dict:
        """
        Обновляет веса по собранным эпизодам.

        Args:
            episodes_data: список эпизодов, каждый содержит
                states, actions, rewards, log_probs, values, dones
        """
        # Собираем все данные в тензоры
        all_states = []
        all_actions = []
        all_returns = []
        all_old_log_probs = []
        all_advantages = []

        for ep in episodes_data:
            returns = self.compute_returns(ep["rewards"], ep["values"], ep["done"])
            values_t = torch.stack(ep["values"])
            advantages = returns - values_t

            all_states.append(torch.FloatTensor(np.array(ep["states"])))
            all_actions.append(torch.FloatTensor(np.array(ep["actions"])))
            all_returns.append(returns)
            all_old_log_probs.append(torch.stack(ep["log_probs"]))
            all_advantages.append(advantages)

        states = torch.cat(all_states)
        actions = torch.cat(all_actions)
        returns = torch.cat(all_returns)
        old_log_probs = torch.cat(all_old_log_probs)
        advantages = torch.cat(all_advantages)

        # Нормализация advantages
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO обновление (несколько эпох по одним данным)
        total_loss_val = 0
        for _ in range(self.ppo_epochs):
            new_log_probs, new_values, entropy = self.policy.evaluate(states, actions)

            # Ratio: насколько новая политика отличается от старой
            ratio = (new_log_probs - old_log_probs).exp()

            # Clipped surrogate loss (суть PPO)
            surr1 = ratio * advantages
            surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = (new_values - returns).pow(2).mean()

            # Entropy bonus (поощряем разнообразие действий)
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
