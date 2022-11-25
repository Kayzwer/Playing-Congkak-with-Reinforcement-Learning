import torch
from Congkak import Congkak
from PolicyGradient import Agent


if __name__ == "__main__":
    env = Congkak()
    agent1 = Agent(16, 7, 0.0001, 0.0001, 0.99, 0.1, 5)
    agent2 = Agent(16, 7, 0.0001, 0.0001, 0.99, 0.1, 5)
    agent1.policy_network.load_state_dict(torch.load(
        "Agent1_Congkak_PolicyGradient.pt"))
    agent2.policy_network.load_state_dict(torch.load(
        "Agent2_Congkak_PolicyGradient.pt"))
    agent1.policy_network.eval()
    agent2.policy_network.eval()
    state = env.reset()
    done = False
    info = 0
    while not done:
        while info == 0 and not done:
            action1 = agent1.choose_action(state, False, env, 1)
            action2 = agent2.choose_action(state, False, env, 2)
            state, reward1, reward2, done, info = env.two_agent_step(action1,
                                                                     action2)
            print(env)
        while info == 1 and not done:
            action1 = agent1.choose_action(state, False, env, 1)
            state, reward, done, info = env.step(action1, 1)
            print(env)
        while info == 2 and not done:
            action2 = agent2.choose_action(state, False, env, 2)
            state, reward, done, info = env.step(action2, 2)
            print(env)
