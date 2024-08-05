# https://gymnasium.farama.org/environments/toy_text/frozen_lake/
import gymnasium as gym; print(f"gym.__version__: {gym.__version__}")
import time
from warnings import filterwarnings
filterwarnings(action='ignore', category=DeprecationWarning)


ACTION_STRING_LIST = [" LEFT", " DOWN", "RIGHT", "   UP"]


def frozen_lake_1():
    env = gym.make('FrozenLake-v1', map_name="4x4", render_mode="human")
    return env


def frozen_lake_2():
    env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=True, render_mode="human")
    return env


def frozen_lake_3():
    env = gym.make('FrozenLake-v1', map_name="8x8", render_mode="human")
    return env


def frozen_lake_4():
    env = gym.make('FrozenLake-v1', map_name="8x8", is_slippery=False, render_mode="human")
    return env


def frozen_lake_5():
    desc = ["SFFF", "HFHH", "FFFH", "HFFF", "FFFG"]
    env = gym.make('FrozenLake-v1', desc=desc, render_mode="human")
    return env


def frozen_lake_6():
    from gymnasium.envs.toy_text.frozen_lake import generate_random_map
    env = gym.make('FrozenLake-v1', desc=generate_random_map(size=8), render_mode="human")
    return env


def run(env):
    # This sets the initial state at S, our starting point
    # We can render the environment to see where we are on the 4x4 frozenlake gridworld
    observation, info = env.reset()

    action = 2  # RIGHT
    next_observation, reward, terminated, truncated, info = env.step(action)

    # Prob = 1: deterministic policy, if we choose to go right, we'll go right
    print("Obs.: {0}, Action: {1}({2}), Next Obs.: {3}, Reward: {4}, Terminated: {5}, Truncated: {6}, Info: {7}".format(
        observation, action, ACTION_STRING_LIST[action], next_observation, reward, terminated, truncated, info
    ))

    observation = next_observation

    time.sleep(3)

    action = 1  # DOWN
    next_observation, reward, terminated, truncated, info = env.step(action)

    print("Obs.: {0}, Action: {1}({2}), Next Obs.: {3}, Reward: {4}, Terminated: {5}, Truncated: {6}, Info: {7}".format(
        observation, action, ACTION_STRING_LIST[action], next_observation, reward, terminated, truncated, info
    ))

    print("*" * 80)
    time.sleep(3)


if __name__ == "__main__":
    #env = frozen_lake_1()
    #env = frozen_lake_2()
    #env = frozen_lake_3()
    #env = frozen_lake_4()
    #env = frozen_lake_5()
    env = frozen_lake_6()

    run(env)
