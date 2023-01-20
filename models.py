from typing import Tuple
from torch import nn
import torch


class PolicyNetwork(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super(PolicyNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ValueNetwork(nn.Module):
    def __init__(self, input_size: int) -> None:
        super(ValueNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class BatchNormValueNetwork(nn.Module):
    def __init__(self, input_size: int) -> None:
        super(BatchNormValueNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ActorCriticNetwork(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super(ActorCriticNetwork, self).__init__()
        self.shared_layers = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh())
        self.policy_layers = nn.Sequential(
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, output_size),
            nn.Softmax(dim=-1))
        self.value_layers = nn.Sequential(
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.shared_layers(x)
        return self.policy_layers(z), self.value_layers(z)
    
    def get_action(self, x: torch.Tensor) -> torch.Tensor:
        return self.policy_layers(self.shared_layers(x))

    def get_value(self, x: torch.Tensor) -> torch.Tensor:
        return self.value_layers(self.shared_layers(x))
