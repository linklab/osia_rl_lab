import random
import time

import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym; print(f"gym.__version__: {gym.__version__}")


np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)


ACTION_STRING_LIST = [" LEFT", " DOWN", "RIGHT", "   UP"]
IS_SLIPPERY = False
MAP_NAME = "4x4"
DESC = None


class QTableAgent:
    def __init__(self, env, num_episodes, validation_num_episodes, alpha, gamma, epsilon):
        self.env = env
        self.num_episodes = num_episodes
        self.validation_num_episodes = validation_num_episodes
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # Q-Table 초기화
        self.q_table = np.zeros(
            [env.observation_space.n, env.action_space.n]
        )

    def greedy_action(self, observation):
        action_values = self.q_table[observation, :]
        max_value = np.max(action_values)
        action = np.random.choice(
            [action_ for action_, value_ in enumerate(action_values) if value_ == max_value]
        )
        return action

    def epsilon_greedy_action(self, observation):
        action_values = self.q_table[observation, :]
        if np.random.rand() < self.epsilon:
            action = random.choice(range(len(action_values)))
        else:
            max_value = np.max(action_values)
            action = np.random.choice(
                [action_ for action_, value_ in enumerate(action_values) if value_ == max_value]
            )
        return action

    def train(self):
        episode_reward_list = []
        episode_td_error_list = []

        training_time_steps = 0
        is_train_success = False

        for episode in range(self.num_episodes):
            episode_reward = 0.0
            episode_td_error = 0.0

            observation, _ = self.env.reset()
            visited_states = [observation]

            episode_step = 0
            done = False

            while not done:
                action = self.epsilon_greedy_action(observation)
                next_observation, reward, terminated, truncated, _ = self.env.step(action)
                episode_step += 1
                episode_reward += reward

                # Q-Learning
                td_error = reward + self.gamma * np.max(self.q_table[next_observation, :]) \
                           - self.q_table[observation, action]

                self.q_table[observation, action] = self.q_table[observation, action] + self.alpha * td_error
                episode_td_error += td_error

                training_time_steps += 1  # Q-table 업데이트 횟수

                visited_states.append(next_observation)
                observation = next_observation

                done = terminated or truncated

            print(
                "[EPISODE: {0:>2}]".format(episode + 1, observation),
                "Episode Steps: {0:>2}, Visited States Length: {1:>2}, Episode Reward: {2}".format(
                    episode_step, len(visited_states), episode_reward
                ),
                "GOAL" if done and observation == 15 else ""
            )
            episode_reward_list.append(episode_reward)
            episode_td_error_list.append(episode_td_error / episode_step)

            if (episode + 1) % 10 == 0:
                episode_reward_list_test, avg_episode_reward_test = self.validate()
                print("[VALIDATION RESULTS: {0} Episodes, Episode Reward List: {1}] Episode Reward Mean: {2:.3f}".format(
                    self.validation_num_episodes, episode_reward_list_test, avg_episode_reward_test
                ))
                if avg_episode_reward_test == 1.0:
                    print("***** TRAINING DONE!!! *****")
                    is_train_success = True
                    break

        return episode_reward_list, episode_td_error_list, is_train_success

    def validate(self):
        episode_reward_lst = np.zeros(shape=(self.validation_num_episodes,), dtype=float)

        test_env = gym.make('FrozenLake-v1', desc=DESC, map_name=MAP_NAME, is_slippery=IS_SLIPPERY)

        for episode in range(self.validation_num_episodes):
            episode_reward = 0  # cumulative_reward
            episode_step = 1

            observation, _ = test_env.reset()

            done = truncated = False
            while not done and not truncated:
                action = self.greedy_action(observation)
                next_observation, reward, done, truncated, _ = test_env.step(action)
                episode_reward += reward
                observation = next_observation
                episode_step += 1

            episode_reward_lst[episode] = episode_reward

        return episode_reward_lst, np.mean(episode_reward_lst)


def main():
    NUM_EPISODES = 200
    VALIDATION_NUM_EPISODES = 10
    ALPHA = 0.1
    GAMMA = 0.95
    EPSILON = 0.1

    env = gym.make('FrozenLake-v1', desc=DESC, map_name=MAP_NAME, is_slippery=IS_SLIPPERY)
    q_table_agent = QTableAgent(
        env=env, num_episodes=NUM_EPISODES, validation_num_episodes=VALIDATION_NUM_EPISODES,
        alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON
    )

    episode_reward_list, episode_td_error_list, is_train_success = q_table_agent.train()
    print("\nFinal Q-Table Values")
    print("    LEFT   DOWN  RIGHT     UP")
    for idx, observation in enumerate(q_table_agent.q_table):
        print("{0:2d}".format(idx), end=":")
        for action_state in observation:
            print("{0:5.3f} ".format(action_state), end=" ")
        print()

    plt.plot(range(len(episode_reward_list)), episode_reward_list, color="blue")
    plt.xlabel("episodes")
    plt.ylabel("training episode reward (blue)")
    plt.show()

    plt.plot(range(len(episode_td_error_list)), episode_td_error_list, color="red")
    plt.xlabel("episodes")
    plt.ylabel("td_error (red)")
    plt.show()

    if is_train_success:
        q_learning_test(q_table_agent=q_table_agent)
    else:
        print("NO PLAYING!!!")


def q_learning_test(q_table_agent):
    play_env = gym.make('FrozenLake-v1', desc=DESC, map_name=MAP_NAME, is_slippery=IS_SLIPPERY, render_mode="human")
    observation, _ = play_env.reset()
    time.sleep(1)

    done = False
    episode_reward = 0.0
    episode_step = 1

    while not done:
        action = q_table_agent.greedy_action(observation)
        next_observation, reward, terminated, truncated, _ = play_env.step(action)
        episode_reward += reward
        observation = next_observation
        done = terminated or truncated
        episode_step += 1
        time.sleep(1)

    if episode_reward >= 1.0:
        print("PLAY EPISODE SUCCESS!!! (TOTAL STEPS: {0})".format(episode_step))
    else:
        print("PLAY EPISODE FAILED!!! (TOTAL STEPS: {0})".format(episode_step))


if __name__ == "__main__":
    main()