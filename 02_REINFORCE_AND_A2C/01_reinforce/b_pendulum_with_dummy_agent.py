# https://gymnasium.farama.org/environments/classic_control/pendulum
import gymnasium as gym
import numpy as np

np.set_printoptions(formatter={'float': '{:5.2f}'.format})
env = gym.make('Pendulum-v1', render_mode="human")


class Dummy_Agent:
    def get_action(self, observation):
        action = np.expand_dims(np.random.uniform(-2.0, 2.0), axis=0)
        return action


def run_env():
    print("START RUN!!!")
    agent = Dummy_Agent()
    observation, info = env.reset()

    done = False
    episode_step = 1
    while not done:
        action = agent.get_action(observation)
        next_observation, reward, terminated, truncated, info = env.step(action)

        print("[Step: {0:3}] Obs.: {1:>2}, Action: {2}, Next Obs.: {3}, "
              "Reward: {4:6.2f}, terminated: {5}, Truncated: {6}, Info: {7}".format(
            episode_step, str(observation), action, str(next_observation),
            reward, terminated, truncated, info
        ))
        observation = next_observation
        done = terminated or truncated
        episode_step += 1


if __name__ == "__main__":
    run_env()

