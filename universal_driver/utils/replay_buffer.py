"""Experience replay buffer for off-policy training."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterator, Union

import numpy as np
import torch

TensorDict = Dict[str, Union[np.ndarray, torch.Tensor]]


@dataclass
class Transition:
    observation: TensorDict
    action: Union[np.ndarray, torch.Tensor]
    reward: Union[float, torch.Tensor]
    next_observation: TensorDict
    done: Union[bool, torch.Tensor]
    log_prob: Union[float, torch.Tensor]
    value: Union[float, torch.Tensor]


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.memory: Deque[Transition] = deque(maxlen=capacity)

    def add(self, transition: Transition) -> None:
        self.memory.append(transition)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.memory)

    def sample(self, batch_size: int) -> Transition:
        indices = np.random.choice(len(self.memory), size=batch_size, replace=False)
        batch = [self.memory[idx] for idx in indices]
        return stack_transitions(batch)


def stack_transitions(transitions: Iterator[Transition]) -> Transition:
    obs: Dict[str, list[np.ndarray]] = {}
    next_obs: Dict[str, list[np.ndarray]] = {}
    actions = []
    rewards = []
    dones = []
    log_probs = []
    values = []
    for transition in transitions:
        actions.append(np.asarray(transition.action))
        rewards.append(float(np.asarray(transition.reward)))
        dones.append(float(np.asarray(transition.done)))
        log_probs.append(float(np.asarray(transition.log_prob)))
        values.append(float(np.asarray(transition.value)))
        for key, value in transition.observation.items():
            obs.setdefault(key, []).append(np.asarray(value))
        for key, value in transition.next_observation.items():
            next_obs.setdefault(key, []).append(np.asarray(value))
    obs_tensor = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in obs.items()}
    next_obs_tensor = {
        k: torch.as_tensor(v, dtype=torch.float32) for k, v in next_obs.items()
    }
    action_tensor = torch.as_tensor(actions, dtype=torch.float32)
    reward_tensor = torch.as_tensor(rewards, dtype=torch.float32).unsqueeze(-1)
    done_tensor = torch.as_tensor(dones, dtype=torch.float32).unsqueeze(-1)
    log_prob_tensor = torch.as_tensor(log_probs, dtype=torch.float32).unsqueeze(-1)
    value_tensor = torch.as_tensor(values, dtype=torch.float32).unsqueeze(-1)
    return Transition(
        observation=obs_tensor,
        action=action_tensor,
        reward=reward_tensor,
        next_observation=next_obs_tensor,
        done=done_tensor,
        log_prob=log_prob_tensor,
        value=value_tensor,
    )

