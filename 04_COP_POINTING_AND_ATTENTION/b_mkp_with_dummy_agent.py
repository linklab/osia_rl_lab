import random

import gymnasium as gym
import numpy as np

from a_mkp_env import MultiDimKnapsack

gym.register_envs(MultiDimKnapsack)


class Dummy_Agent:
    def get_action(self, observation: np.ndarray, available_actions: np.ndarray) -> int:
        action = random.choice(available_actions)
        return action


def main() -> None:
    env: MultiDimKnapsack = gym.make("MultiDimKnapsack-v0", n_items=5, n_resources=3)
    obs, info = env.reset()

    print("Initial state:")
    print(env.get_wrapper_attr("_inintial_state").get_observation())
    print()
    print("Current obs:")
    print(obs)
    print()

    available_actions = info["available_actions"]
    action = available_actions[0]
    obs, reward, truncated, terminated, info = env.step(action)

    print("After taking action:")
    print(obs)
    print(f"Reward: {reward}")
    print(f"Truncated: {truncated}")
    print(f"Terminated: {terminated}")
    print(f"Info: {info}")


if __name__ == "__main__":
    main()


def run_env() -> None:
    print("START RUN!!!")
    env: MultiDimKnapsack = gym.make("MultiDimKnapsack-v0", n_items=5, n_resources=3)
    agent = Dummy_Agent()
    observation, info = env.reset()

    done = False
    episode_step = 1
    while not done:
        action = agent.get_action(observation, info["available_actions"])
        next_observation, reward, terminated, truncated, info = env.step(action)

        print(
            "[Step: {0:3}] Obs.: {1:>2}, Action: {2}({3}), Next Obs.: {4}, "
            "Reward: {5}, terminated: {6}, Truncated: {7}, Info: {8}".format(
                episode_step,
                str(observation),
                action,
                info["available_actions"],
                str(next_observation),
                reward,
                terminated,
                truncated,
                info,
            )
        )
        observation = next_observation
        done = terminated or truncated
        episode_step += 1


if __name__ == "__main__":
    run_env()
