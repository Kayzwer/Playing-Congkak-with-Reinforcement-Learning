from Congkak import Congkak
from PPO import Agent
import torch


if __name__ == "__main__":
    env = Congkak(True)
    agent2 = Agent(
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
    agent2.network.load_state_dict(torch.load("Agent2_Congkak_PPO.pt"))
    agent2.network.eval()

    state = env.reset(True)
    print(env)
    done = False
    info = 0
    while not done:
        while not done and info == 0:
            action_1 = int(input("Player 1 Entry: "))
            action_2 = agent2.choose_action(state, False, env, 2)
            state, _, _, done, info = env.two_agent_step(action_1, action_2)
        while not done and info == 1:
            action_1 = int(input("Player 1 Entry: "))
            state, _, done, info = env.step(action_1, 1)
        while not done and info == 2:
            action_2 = agent2.choose_action(state, False, env, 2)
            print(f"Player 2 play entry: {action_2}")
            print("Press Enter to continue")
            input()
            state, _, done, info = env.step(action_2, 2)
