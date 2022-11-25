from models import PolicyNetwork, ValueNetwork
from torch.distributions import Categorical
from typing import Tuple
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
        entropy_weight: float,
        batch: int
    ) -> None:
        assert 0 <= policy_lr <= 1
        assert 0 <= value_lr <= 1
        assert 0 <= gamma <= 1
        assert 0 <= entropy_weight <= 1

        self.policy_network = PolicyNetwork(input_size, output_size)
        self.policy_optimizer = optim.RMSprop(self.policy_network.parameters(),
                                              policy_lr)
        self.value_network = ValueNetwork(input_size)
        self.value_optimizer = optim.RMSprop(self.value_network.parameters(),
                                             value_lr)
        self.gamma = gamma
        self.entropy_weight = entropy_weight
        self.batch = batch

        self.state_memory = []
        self.state_memory_batch = np.empty(self.batch, dtype=np.object0)
        self.log_prob_memory = []
        self.log_prob_memory_batch = np.empty(self.batch, dtype=np.object0)
        self.reward_memory = []
        self.reward_memory_batch = np.empty(self.batch, dtype=np.object0)
        self.entropy_memory = []
        self.entropy_memory_batch = np.empty(self.batch, dtype=np.object0)
        self.return_memory_batch = np.empty(self.batch, dtype=np.object0)
        self.memory_ptr = 0
        self.log_cache = 1 / torch.log(torch.tensor(
            output_size, dtype=torch.float32))

    def choose_action(self, np_state: np.ndarray, is_train: bool, env,
                      player: int) -> int:
        state = torch.from_numpy(np_state)
        action_probs = self.policy_network(state)
        if is_train:
            action_dist = Categorical(action_probs)
            action = int(action_dist.sample().item())
            if not env.is_valid_action(action, player):
                _, indices = torch.sort(action_probs.detach(), descending=True)
                for index in indices:
                    if env.is_valid_action(index, player):
                        action = index.item()
                        break
            self.state_memory.append(state.unsqueeze(dim=0))
            self.log_prob_memory.append(action_dist.log_prob(torch.tensor(
                action)))
            self.entropy_memory.append(action_dist.entropy() * self.log_cache)
            return action
        else:
            _, indices = torch.sort(action_probs.detach(), descending=True)
            for index in indices:
                if env.is_valid_action(index, player):
                    return index
            return env.sample_valid_action(player)

    def store_reward(self, reward: float) -> None:
        self.reward_memory.append(reward)

    def update(self, iteration: int) -> Tuple[float, float]:
        self.save_data()
        self.value_optimizer.zero_grad()
        T = len(self.reward_memory)
        returns = torch.zeros(T, dtype=torch.float32)
        returns_sum = 0.0
        for i in range(T - 1, -1, -1):
            returns_sum = self.reward_memory[i] + self.gamma * returns_sum
            returns[i] = returns_sum
        returns = (returns - returns.mean()) / returns.std()
        self.return_memory_batch[self.memory_ptr] = returns
        self.memory_ptr += 1
        state_values = self.value_network(torch.concat(self.state_memory)
                                          ).view(T)
        value_loss = nn.functional.mse_loss(state_values, returns)
        value_loss.backward()
        self.value_optimizer.step()

        policy_loss = 0.
        if (iteration + 1) % self.batch == 0:
            policy_loss += self.update_policy()
        self.clear_memory()
        return policy_loss, value_loss.item()

    def update_policy(self) -> float:
        self.policy_optimizer.zero_grad()
        policy_loss = torch.tensor(0.0, dtype=torch.float32)
        for state_memory, log_prob_memory, return_memory, entropy_memory in zip(
            self.state_memory_batch, self.log_prob_memory_batch,
            self.return_memory_batch, self.entropy_memory_batch):
            T = len(state_memory)
            state_values = self.value_network.forward(torch.concat(state_memory)
                                                      ).view(T)
            for return_, state_value, log_prob, entropy in zip(
                return_memory, state_values, log_prob_memory, entropy_memory):
                policy_loss -= (log_prob * (return_ - state_value).detach() +
                                entropy * self.entropy_weight)

        policy_loss /= self.batch
        policy_loss.backward()
        self.policy_optimizer.step()
        self.memory_ptr = 0
        return policy_loss.item()

    def save_data(self) -> None:
        self.state_memory_batch[self.memory_ptr] = self.state_memory.copy()
        self.log_prob_memory_batch[self.memory_ptr] = \
            self.log_prob_memory.copy()
        self.reward_memory_batch[self.memory_ptr] = self.reward_memory.copy()
        self.entropy_memory_batch[self.memory_ptr] = self.entropy_memory.copy()

    def clear_memory(self) -> None:
        self.state_memory.clear()
        self.log_prob_memory.clear()
        self.reward_memory.clear()
        self.entropy_memory.clear()
