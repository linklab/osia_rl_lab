# https://gymnasium.farama.org/environments/toy_text/frozen_lake/
# pip install gymnasium[toy-text]
import time

import gymnasium as gym; print(f"gym.__version__: {gym.__version__}")

env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=False, render_mode="human")

ACTION_STRING_LIST = [" LEFT", " DOWN", "RIGHT", "   UP"]


def env_info_details():
    #####################
    # observation space #
    #####################
    print("*" * 80)
    print("[observation_space]")
    print(env.observation_space)
    print(env.observation_space.n)
    # We should expect to see 15 possible grids from 0 to 15 when
    # we uniformly randomly sample from our observation space
    for i in range(10):
        print(env.observation_space.sample(), end=" ")
    print()

    print("*" * 80)
    ################
    # action space #
    ################
    print("[action_space]")
    print(env.action_space)
    print(env.action_space.n)
    # We should expect to see 4 actions when
    # we uniformly randomly sample:
    #     1. LEFT: 0
    #     2. DOWN: 1
    #     3. RIGHT: 2
    #     4. UP: 3
    for i in range(10):
        print(env.action_space.sample(), end=" ")
    print()

    print("*" * 80)
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
    env_info_details()
