# https://gymnasium.farama.org/environments/classic_control/pendulum
import time

import gymnasium as gym
import numpy as np

print("gym.__version__:", gym.__version__)
np.set_printoptions(formatter={'float': '{:5.2f}'.format})

env = gym.make('Pendulum-v1', render_mode="human")


def env_info_details():
    #####################
    # observation space #
    #####################
    print("*" * 80)
    print("[observation_space]")
    print(env.observation_space)
    for i in range(10):
        print(env.observation_space.sample())
    print()

    print("*" * 80)
    ################
    # action space #
    ################
    print("[action_space]")
    print(env.action_space)
    for i in range(10):
        print(env.action_space.sample())
    print()

    print("*" * 80)
    observation, info = env.reset()

    action = env.action_space.sample()
    next_observation, reward, terminated, truncated, info = env.step(action)

    print("Obs.: {0}, Action: {1}, Next Obs.: {2}, Reward: {3:6.2f}, "
          "Terminated: {4}, Truncated: {5}, Info: {6}".format(
        observation, action, next_observation, reward, terminated, truncated, info
    ))

    observation = next_observation

    time.sleep(2)

    action = env.action_space.sample()
    next_observation, reward, terminated, truncated, info = env.step(action)

    print("Obs.: {0}, Action: {1}, Next Obs.: {2}, Reward: {3:6.2f}, "
          "Terminated: {4}, Truncated: {5}, Info: {6}".format(
        observation, action, next_observation, reward, terminated, truncated, info
    ))

    print("*" * 80)
    time.sleep(2)


if __name__ == "__main__":
    env_info_details()