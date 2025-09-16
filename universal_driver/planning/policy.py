"""Policy network for vehicle control."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Normal

from ..config import PolicyConfig


@dataclass
class PolicyOutput:
    action: torch.Tensor
    log_prob: torch.Tensor
    value: torch.Tensor
    entropy: torch.Tensor
    recurrent_state: Optional[torch.Tensor]


class ActorHead(nn.Module):
    def __init__(self, hidden_size: int, action_dim: int) -> None:
        super().__init__()
        self.mean = nn.Linear(hidden_size, action_dim)
        self.log_std = nn.Linear(hidden_size, action_dim)

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = torch.tanh(self.mean(features))
        log_std = torch.clamp(self.log_std(features), min=-5.0, max=2.0)
        return mean, log_std


class CriticHead(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.value = nn.Linear(hidden_size, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.value(features)


class UniversalPolicy(nn.Module):
    """Actor-critic policy that supports optional recurrence."""

    def __init__(self, input_dim: int, config: PolicyConfig) -> None:
        super().__init__()
        hidden_layers = []
        last_dim = input_dim
        activation = nn.ReLU if config.activation == "relu" else nn.ELU
        for hidden in config.hidden_sizes:
            hidden_layers.append(nn.Linear(last_dim, hidden))
            hidden_layers.append(activation())
            last_dim = hidden
        self.backbone = nn.Sequential(*hidden_layers)
        self.policy_config = config
        self.actor = ActorHead(last_dim, config.action_dim)
        self.critic = CriticHead(last_dim)
        if config.use_lstm:
            self.rnn = nn.LSTM(last_dim, last_dim, batch_first=True)
        else:
            self.rnn = None

    def forward(
        self, features: torch.Tensor, recurrent_state: Optional[torch.Tensor] = None
    ) -> PolicyOutput:
        x = self.backbone(features)
        rnn_state = recurrent_state
        if self.rnn is not None:
            if rnn_state is None:
                h0 = torch.zeros(1, x.size(0), x.size(1), device=x.device)
                c0 = torch.zeros(1, x.size(0), x.size(1), device=x.device)
            else:
                h0, c0 = rnn_state
            x, (h1, c1) = self.rnn(x.unsqueeze(1), (h0, c0))
            x = x.squeeze(1)
            rnn_state = (h1, c1)
        mean, log_std = self.actor(x)
        std = log_std.exp()
        dist = Normal(mean, std)
        if self.policy_config.stochastic:
            action = dist.rsample()
        else:
            action = mean
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        value = self.critic(x)
        return PolicyOutput(
            action=action,
            log_prob=log_prob,
            value=value,
            entropy=entropy,
            recurrent_state=rnn_state,
        )

    def act(
        self, features: torch.Tensor, recurrent_state: Optional[torch.Tensor] = None
    ) -> PolicyOutput:
        return self.forward(features, recurrent_state=recurrent_state)

