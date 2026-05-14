"""
RL-агент на основе PPO с маскированием неиспользуемых выходов.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Normal

from environment_with_flags import MAX_GATES, MAX_FLAGS, ACTION_DIM

class PolicyNetwork(nn.Module):
    """
    [n_gates, n_flags] (2 числа) → все ворота и флаги.
    Маскирование: градиенты идут только на активные выходы.
    """

    def __init__(self, state_dim: int = 2, hidden_dim: int = 512):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, ACTION_DIM),
            nn.Sigmoid(),
        )
        # Начальный std = 0.135 (вместо 1.0) — точнее, меньше шума
        self.actor_log_std = nn.Parameter(torch.ones(ACTION_DIM) * (-2.0))

        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, state):
        x = self.encoder(state)
        action_mean = self.actor_mean(x)
        action_std = self.actor_log_std.exp().expand_as(action_mean)
        value = self.critic(x).squeeze(-1)
        return action_mean, action_std, value

    def select_action(self, state: np.ndarray):
        """
        Возвращает action, log_prob (скаляр), value, log_prob_per_dim (вектор), entropy_per_dim (вектор)
        """
        state_t = torch.FloatTensor(state).unsqueeze(0)
        mean, std, value = self.forward(state_t)
        dist = Normal(mean, std)
        action = dist.sample()
        action = action.clamp(0, 1)
        
        log_prob_per_dim = dist.log_prob(action).squeeze(0)      # (ACTION_DIM,)
        entropy_per_dim = dist.entropy().squeeze(0)             # (ACTION_DIM,)
        log_prob = log_prob_per_dim.sum()                       # скаляр

        return (action.squeeze(0).detach().numpy(),
                log_prob.detach(),
                value.squeeze(0).detach(),
                log_prob_per_dim.detach(),
                entropy_per_dim.detach())

    def evaluate(self, states, actions):
        """Возвращает log_prob и entropy ПО КАЖДОМУ ИЗМЕРЕНИЮ (для маскирования)."""
        mean, std, values = self.forward(states)
        dist = Normal(mean, std)
        log_probs = dist.log_prob(actions)          # (batch, ACTION_DIM)
        entropy = dist.entropy()                    # (batch, ACTION_DIM) — БЕЗ аргументов!
        return log_probs, values, entropy


class PPOTrainer:
    def __init__(self, policy: PolicyNetwork, lr: float = 5e-4,
                 clip_eps: float = 0.2, value_coef: float = 0.5,
                 entropy_coef: float = 0.1, ppo_epochs: int = 20):
        self.policy = policy
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.ppo_epochs = ppo_epochs

    def update(self, episodes_data: list[dict]) -> dict:
        states = torch.FloatTensor(np.array([ep["state"] for ep in episodes_data]))
        actions = torch.FloatTensor(np.array([ep["action"] for ep in episodes_data]))
        rewards = torch.FloatTensor([ep["reward"] for ep in episodes_data])
        old_log_probs_per_dim = torch.stack([ep["log_prob_per_dim"] for ep in episodes_data])
        old_values = torch.stack([ep["value"] for ep in episodes_data])
        masks = torch.FloatTensor(np.array([ep["mask"] for ep in episodes_data]))  # (batch, ACTION_DIM)

        # Суммируем log_prob только по активным выходам
        old_log_probs = (old_log_probs_per_dim * masks).sum(-1)
        returns = rewards

        advantages = returns - old_values
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        total_loss_val = 0
        for _ in range(self.ppo_epochs):
            new_log_probs_per_dim, new_values, entropy_per_dim = self.policy.evaluate(states, actions)
            
            new_log_probs = (new_log_probs_per_dim * masks).sum(-1)
            entropy = (entropy_per_dim * masks).sum(-1)

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