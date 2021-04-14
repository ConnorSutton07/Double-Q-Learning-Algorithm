"""
Entrance to program
Serves as driver

"""
from agent import AgentQ
from grid import Grid
import matplotlib.pyplot as plt
import argparse

def train_agent(num_episodes = 1000) -> tuple:
    agent = AgentQ()
    env = Grid()
    rewards = []

    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        terminal = False
        while not terminal:
            action_id, action = agent.act(state)
            next_state, reward, terminal = env.step(action)
            episode_reward += reward

            agent.q_update(state, action_id, reward, next_state, terminal)
            state = next_state
        rewards.append(episode_reward)
    k = 3
    reward_plot = rolling_average(rewards, k)
    return reward_plot, agent.best_policy()
    

def normalize(X: list) -> list:
    """ 
    Centers list of data around origin

    """ 
    max_value = max(X)
    min_value = min(X)
    return [(x - min_value) / (max_value - min_value) for x in X]

def rolling_average(buffer: list, k: int) -> list:
    avg = []
    for i in range(0, len(buffer) - k, k):
        cur_avg = sum(buffer[i:i+k]) / k
        for j in range(k):
            avg.append(cur_avg)
    return normalize(avg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deep-Q Learning Example")
    parser.add_argument("--episodes", type=int, nargs='?',
                        const=True, default=1000, 
                        help="Number of training episodes")
    args = parser.parse_args()
    reward_plot, best_policy = train_agent(args.episodes)
    print("Best policy: ", best_policy)
    plt.plot(reward_plot)
    plt.show()
    