from Congkak import Congkak
from PPO import Agent
import torch


if __name__ == "__main__":
    env = Congkak(True)
    agent1 = Agent(
        input_size=16,
        output_size=7,
        policy_lr=3e-4,
        value_lr=1e-3,
        gamma=0.99,
        lambda_=0.95,
        epsilon=0.2,
        target_kl_div=0.05,
        entropy_weight=0.1,
        max_policy_train_iters=128,
        value_train_iters=128)
    agent1.network.load_state_dict(torch.load("Agent1_Congkak_PPO.pt"))
    agent1.network.eval()
    # agent2 = Agent(
    #     input_size=16,
    #     output_size=7,
    #     policy_lr=3e-4,
    #     value_lr=1e-3,
    #     gamma=0.99,
    #     lambda_=0.95,
    #     epsilon=0.2,
    #     target_kl_div=0.05,
    #     entropy_weight=0.1,
    #     max_policy_train_iters=128,
    #     value_train_iters=128)
    # agent2.network.load_state_dict(torch.load("Agent2_Congkak_PPO.pt"))
    # agent2.network.eval()

    state = env.reset(True)
    print(env)
    done = False
    info = 0
    while not done:
        while not done and info == 0:
            action_1 = agent1.choose_action(state, False, env, 1)
            action_2 = int(input("Player 2 Entry: "))
            state, _, _, done, info = env.two_agent_step(action_1, action_2)
        while not done and info == 1:
            action_1 = agent1.choose_action(state, False, env, 1)
            print(f"Player 1 play entry: {action_1}")
            print("Press Enter to continue")
            input()
            state, _, done, info = env.step(action_1, 1)
        while not done and info == 2:
            action_2 = int(input("Player 2 Entry: "))
            state, _, done, info = env.step(action_2, 2)
