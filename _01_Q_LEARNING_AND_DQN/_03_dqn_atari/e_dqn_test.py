# https://gymnasium.farama.org/environments/classic_control/cart_pole/
import os

import gymnasium as gym
import torch
from gymnasium.wrappers import FrameStackObservation, AtariPreprocessing

from c_qnet import MODEL_DIR, QNetCNN


def test(env: gym.Env, q, num_episodes):
    for i in range(num_episodes):
        episode_reward = 0  # cumulative_reward

        # Environment 초기화와 변수 초기화
        observation, _ = env.reset()

        episode_steps = 0

        done = False

        while not done:
            episode_steps += 1
            action = q.get_action(observation, epsilon=0.0)

            next_observation, reward, terminated, truncated, _ = env.step(action)

            episode_reward += reward
            observation = next_observation
            done = terminated or truncated

        print("[EPISODE: {0}] EPISODE_STEPS: {1:3d}, EPISODE REWARD: {2:4.1f}".format(i, episode_steps, episode_reward))


def main_play(num_episodes: int, env_name: str) -> None:
    env = gym.make(env_name, render_mode="rgb_array")
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=(84, 84),
        grayscale_obs=True,
        grayscale_newaxis=False,
        scale_obs=True
    )
    env = FrameStackObservation(env, stack_size=4)

    q = QNetCNN(n_actions=6)
    model_params = torch.load(os.path.join(MODEL_DIR, "dqn_{0}_latest.pth".format(env_name)), weights_only=True)
    q.load_state_dict(model_params)
    q.eval()

    test(env, q, num_episodes=num_episodes)

    env.close()


if __name__ == "__main__":
    NUM_EPISODES = 3
    ENV_NAME = "PongNoFrameskip-v4"

    main_play(num_episodes=NUM_EPISODES, env_name=ENV_NAME)
