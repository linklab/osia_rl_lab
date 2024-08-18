import random
import time

import gymnasium as gym
import numpy as np

print(f"gym.__version__: {gym.__version__}")


env = gym.make("FrozenLake-v1", desc=None, map_name="4x4", is_slippery=False, render_mode="human")
ACTION_STRING_LIST = [" LEFT", " DOWN", "RIGHT", "   UP"]


# Random Action
class Dummy_Agent:
    def get_action(self, observation: np.ndarray) -> int:
        # observation is not used
        available_action_ids = [0, 1, 2, 3]
        action_id = random.choice(available_action_ids)
        return action_id


def main() -> None:
    print("START RUN!!!")
    agent = Dummy_Agent()
    observation, info = env.reset()

    episode_step = 0
    done = False

    while not done:
        action = agent.get_action(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)

        episode_step += 1
        print(
            "[Step: {0:3}] Obs.: {1:>2}, Action: {2}({3}), Next Obs.: {4:>2}, "
            "Reward: {5}, Terminated: {6}, Truncated: {7}, Info: {8}".format(
                episode_step,
                observation,
                action,
                ACTION_STRING_LIST[action],
                next_observation,
                reward,
                terminated,
                truncated,
                info,
            )
        )
        observation = next_observation
        done = terminated or truncated
        time.sleep(0.5)


if __name__ == "__main__":
    main()
