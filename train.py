from Congkak import Congkak
from PPO import Agent
import torch


if __name__ == "__main__":
    env = Congkak()
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
    agent1.network.train()
    agent2.network.train()

    for e in range(1000000000):
        state = env.reset()
        done = False
        info = 0
        score_1 = 0.
        score_2 = 0.
        while not done:
            while info == 0 and not done:
                action_1 = agent1.choose_action(state, True, env, 1)
                action_2 = agent2.choose_action(state, True, env, 2)
                state, reward_1, reward_2, done, info = env.two_agent_step(
                    action_1, action_2)
                agent1.store_reward(reward_1)
                agent2.store_reward(reward_2)
                score_1 += reward_1
                score_2 += reward_2
            while info == 1 and not done:
                action = agent1.choose_action(state, True, env, 1)
                state, reward, done, info = env.step(action, 1)
                agent1.store_reward(reward)
                score_1 += reward
            while info == 2 and not done:
                action = agent2.choose_action(state, True, env, 2)
                state, reward, done, info = env.step(action, 2)
                agent2.store_reward(reward)
                score_2 += reward
        policy_loss_1, value_loss_1 = agent1.update()
        policy_loss_2, value_loss_2 = agent2.update()
        if (e + 1) % 100000 == 0:
            print(f"Episode: {e + 1}, Player 1 score: {score_1}, Player 2 scor"
                  f"e: {score_2}")
            print(f"Player 1, Policy Loss: {policy_loss_1:.6f}, Value Loss: "
                  f"{value_loss_1:.6f} | Player 2, Policy Loss: "
                  f"{policy_loss_2:.6f}, Value Loss: {value_loss_2:.6f}")
        if (e + 1) % 500 == 0:
            torch.save(agent1.network.state_dict(),
                       "Agent1_Congkak_PolicyGradient.pt")
            torch.save(agent2.network.state_dict(),
                       "Agent2_Congkak_PolicyGradient.pt")
