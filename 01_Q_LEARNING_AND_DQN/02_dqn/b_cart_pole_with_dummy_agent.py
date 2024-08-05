# https://gymnasium.farama.org/environments/classic_control/cart_pole/
import gymnasium as gym; print("gym.__version__:", gym.__version__)
import random
import time

env = gym.make('CartPole-v1', render_mode="human")
ACTION_STRING_LIST = [" LEFT", "RIGHT"]


class Dummy_Agent:
    def get_action(self, observation):
        available_action_ids = [0, 1]
        action_id = random.choice(available_action_ids)
        return action_id


def run_env():
    print("START RUN!!!")
    agent = Dummy_Agent()
    observation, info = env.reset()

    done = False
    episode_step = 1
    while not done:
        action = agent.get_action(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)

        print("[Step: {0:3}] Obs.: {1:>2}, Action: {2}({3}), Next Obs.: {4}, "
              "Reward: {5}, terminated: {6}, Truncated: {7}, Info: {8}".format(
            episode_step, str(observation), action, ACTION_STRING_LIST[action], str(next_observation),
            reward, terminated, truncated, info
        ))
        observation = next_observation
        done = terminated or truncated
        episode_step += 1
        time.sleep(0.1)


if __name__ == "__main__":
    run_env()

