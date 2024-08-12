import os

import gymnasium as gym
import torch

from a_mkp_env import MultiDimKnapsack
from c_qnet import MODEL_DIR, QNet
from g_google_or_tools import solve

gym.register_envs(MultiDimKnapsack)


def test(env: gym.Env | MultiDimKnapsack, q: QNet, num_episodes: int) -> None:
    for i in range(num_episodes):
        episode_reward = 0  # cumulative_reward

        # Environment 초기화와 변수 초기화
        observation, info = env.reset()

        or_tools_log = solve(
            n_items=env.n_items,
            n_resources=env.n_resources,
            demands=env.get_initial_state().demands,
            values=env.get_initial_state().values,
            capacities=env.get_initial_state().capacities,
        )

        episode_steps = 0

        done = False

        while not done:
            episode_steps += 1
            action = q.get_action(
                obs=observation,
                available_actions=info["available_actions"],
                epsilon=0.0
            )

            next_observation, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            observation = next_observation
            done = terminated or truncated

        dqn_log = env.get_log()
        print("[OR-TOOLS] ", or_tools_log)
        print("[DQN]      ", dqn_log)

        print("[EPISODE: {0}] EPISODE_STEPS: {1:3d}, EPISODE REWARD: {2:4.1f}".format(i, episode_steps, episode_reward))


def main_play() -> None:
    NUM_EPISODES = 3
    ENV_NAME = "MultiDimKnapsack-v0"
    N_ITEMS = 20
    N_RESOURCES = 1

    def make_env() -> gym.Env:
        env = gym.make(ENV_NAME, n_items=N_ITEMS, n_resources=N_RESOURCES)
        env = gym.wrappers.FlattenObservation(env)
        return env

    env = make_env()
    n_features = env.observation_space.shape[0]
    n_actions = env.action_space.n
    q = QNet(n_features=n_features, n_actions=n_actions)
    model_params = torch.load(
        os.path.join(MODEL_DIR, f"dqn_{ENV_NAME}_{N_ITEMS}x{N_RESOURCES}_latest.pth"),
        weights_only=True
    )
    q.load_state_dict(model_params)
    q.eval()

    test(env, q, num_episodes=NUM_EPISODES)

    env.close()


if __name__ == "__main__":
    main_play()
