from torch.distributions import Categorical
from models import ActorCriticNetwork
from typing import List, Tuple
from torch import optim
from torch import nn
import numpy as np
import torch


class Agent:
    def __init__(
            self,
            input_size: int,
            output_size: int,
            policy_lr: float,
            value_lr: float,
            gamma: float,
            lambda_: float,
            epsilon: float,
            target_kl_div: float,
            entropy_weight: float,
            max_policy_train_iters: int,
            value_train_iters: int) -> None:
        self.network = ActorCriticNetwork(input_size, output_size)
        policy_params = list(self.network.shared_layers.parameters()) + \
            list(self.network.policy_layers.parameters())
        self.policy_optimizer = optim.RMSprop(policy_params, policy_lr)
        value_params = list(self.network.shared_layers.parameters()) + \
            list(self.network.value_layers.parameters())
        self.value_optimizer = optim.RMSprop(value_params, value_lr)
        self.gamma = gamma
        self.lambda_ = lambda_
        self.lower_eps = 1 - epsilon
        self.upper_eps = 1 + epsilon
        self.target_kl_div = target_kl_div
        self.entropy_weight = entropy_weight
        self.max_policy_train_iters = max_policy_train_iters
        self.value_train_iters = value_train_iters
        self.log_cache = 1 / torch.log(torch.tensor(output_size))

        self.state_memory = []
        self.action_memory = []
        self.reward_memory = []
        self.state_value_memory = []
        self.action_log_prob_memory = []

    def choose_action(self, np_state: np.ndarray, is_train: bool, env,
                      player: int) -> int:
        state = torch.from_numpy(np_state)
        action_probs, state_value = self.network(state)

        if is_train:
            action_dist = Categorical(action_probs)
            action = action_dist.sample().item()
            if not env.is_valid_action(action, player):
                _, indices = torch.sort(action_probs.detach(), descending=True)
                for index in indices:
                    if env.is_valid_action(index, player):
                        action = index.item()
                        break

            action_log_prob = action_dist.log_prob(action).item()
            self.state_memory.append(state.unsqueeze(dim=0))
            self.action_memory.append(action)
            self.state_value_memory.append(state_value.item())
            self.action_log_prob_memory.append(action_log_prob)
            return int(action)
        else:
            _, indices = torch.sort(action_probs.detach(), descending=True)
            for index in indices:
                if env.is_valid_action(index, player):
                    return index
            return env.sample_valid_action(player)

    def store_reward(self, reward: float) -> None:
        self.reward_memory.append(reward)

    def update(self) -> Tuple[float, float]:
        states = torch.cat(self.state_memory)
        actions = torch.from_numpy(np.array(self.action_memory, dtype=np.int8))
        returns = torch.from_numpy(self.get_returns(self.reward_memory))
        action_log_probs = torch.from_numpy(np.array(
            self.action_log_prob_memory, dtype=np.float32))
        gaes = torch.from_numpy(self.get_gaes(self.reward_memory,
                                              self.state_value_memory))

        policy_loss = self.train_policy(states, actions, action_log_probs, gaes)
        value_loss = self.train_value(states, returns)
        self.clear_memory()
        return policy_loss, value_loss
    
    def train_policy(self, states: torch.Tensor, actions: torch.Tensor,
                     old_log_probs: torch.Tensor, gaes: torch.Tensor) -> float:
        loss = 0.0
        for _ in range(self.max_policy_train_iters):
            self.policy_optimizer.zero_grad()
            new_action_probs = self.network.get_action(states)
            new_action_dists = Categorical(new_action_probs)
            new_log_probs = new_action_dists.log_prob(actions)
            entropies = new_action_dists.entropy() * self.log_cache

            policy_ratio = torch.exp(new_log_probs - old_log_probs)
            clipped_ratio = policy_ratio.clamp(self.lower_eps, self.upper_eps)
            policy_loss = -(torch.min(policy_ratio, clipped_ratio) * gaes +
                            self.entropy_weight * entropies).mean()
            loss += policy_loss.item()
            policy_loss.backward()
            self.policy_optimizer.step()

            if (old_log_probs - new_log_probs).mean() >= self.target_kl_div:
                break
        return loss

    def train_value(self, states: torch.Tensor, returns: torch.Tensor) -> float:
        loss = 0.0
        for _ in range(self.value_train_iters):
            self.value_optimizer.zero_grad()
            state_values = self.network.get_value(states)
            value_loss = nn.functional.mse_loss(returns.view(-1, 1),
                                                state_values)
            loss += value_loss.item()
            value_loss.backward()
            self.value_optimizer.step()
        return loss

    def get_returns(self, reward_memory: List[float]) -> np.ndarray:
        T = len(reward_memory)
        returns = np.zeros(T, dtype=np.float32)
        returns_sum = 0.0
        for i in range(T - 1, -1, -1):
            returns_sum = self.reward_memory[i] + self.gamma * returns_sum
            returns[i] = returns_sum
        return returns

    def get_gaes(self, reward_memory: List[float],
                 state_value_memory: List[float]) -> np.ndarray:
        next_state_values = np.concatenate([state_value_memory[1:], [0]])

        T = len(reward_memory)
        deltas = np.zeros(T, dtype=np.float32)
        for i in range(T):
            deltas[i] = reward_memory[i] + self.gamma * next_state_values[i] - \
                state_value_memory[i]

        gaes = np.zeros(T, dtype=np.float32)
        gaes[-1] = deltas[-1]
        for i in range(T - 2, -1, -1):
            gaes[i] = deltas[i] + self.lambda_ * self.gamma * gaes[i + 1]
        return gaes

    def clear_memory(self) -> None:
        self.state_memory.clear()
        self.action_memory.clear()
        self.reward_memory.clear()
        self.state_value_memory.clear()
        self.action_log_prob_memory.clear()
