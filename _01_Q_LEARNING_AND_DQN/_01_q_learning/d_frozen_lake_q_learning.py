import random
import time
from datetime import datetime
import gymnasium as gym
import numpy as np

import wandb

print(f"gym.__version__: {gym.__version__}")

ACTION_STRING_LIST = [" LEFT", " DOWN", "RIGHT", "   UP"]
IS_SLIPPERY = False
MAP_NAME = "4x4"
DESC = None


class QTableAgent:
    def __init__(
            self, env, config, use_wandb
    ):
        self.env = env
        self.num_episodes = config["num_episodes"]
        self.train_num_episodes_before_next_validation = config["train_num_episodes_before_next_validation"]
        self.validation_num_episodes = config["validation_num_episodes"]
        self.alpha = config["alpha"]
        self.gamma = config["gamma"]
        self.epsilon = config["epsilon"]
        self.use_wandb = use_wandb

        # Q-Table 초기화
        self.q_table = np.zeros([env.observation_space.n, env.action_space.n])

        self.current_time = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")

        if self.use_wandb:
            wandb.init(project="Q_LEARNING_FROZEN_LAKE_2", config=config)

    def greedy_action(self, observation: np.ndarray) -> int:
        action_values = self.q_table[observation, :]
        max_value = np.max(action_values)
        action = np.random.choice([action_ for action_, value_ in enumerate(action_values) if value_ == max_value])
        return action

    def epsilon_greedy_action(self, observation: np.ndarray) -> int:
        action_values = self.q_table[observation, :]
        if np.random.rand() < self.epsilon:
            action = random.choice(range(len(action_values)))
        else:
            max_value = np.max(action_values)
            action = np.random.choice([action_ for action_, value_ in enumerate(action_values) if value_ == max_value])
        return action

    def train(self) -> tuple[list[float], list[float], bool]:
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
            avg_episode_reward_validation = 0.0
            done = False

            while not done:
                action = self.epsilon_greedy_action(observation)
                next_observation, reward, terminated, truncated, _ = self.env.step(action)
                episode_step += 1
                episode_reward += reward

                # Q-Learning
                td_error = reward + self.gamma * np.max(self.q_table[next_observation, :]) - self.q_table[observation, action]

                self.q_table[observation, action] = self.q_table[observation, action] + self.alpha * td_error
                episode_td_error += td_error

                training_time_steps += 1  # Q-table 업데이트 횟수

                visited_states.append(next_observation)
                observation = next_observation

                done = terminated or truncated

            print(
                "[EPISODE: {0:>2}]".format(
                    episode + 1,
                ),
                "Episode Steps: {0:>2}, Visited States Length: {1:>2}, Episode Reward: {2}".format(
                    episode_step, len(visited_states), episode_reward
                ),
                "GOAL" if done and observation == 15 else "",
            )

            if (episode + 1) % self.train_num_episodes_before_next_validation == 0:
                episode_reward_list_validation, avg_episode_reward_validation = self.validate()
                print(
                    "[VALIDATION RESULTS: {0} Episodes, Episode Reward List: {1}] Episode Reward Mean: {2:.3f}".format(
                        self.validation_num_episodes,
                        episode_reward_list_validation,
                        avg_episode_reward_validation
                    )
                )
                if avg_episode_reward_validation == 1.0:
                    print("***** TRAINING DONE!!! *****")
                    is_train_success = True
                    break

            if self.use_wandb:
                wandb.log({
                    "[TRAIN] Length of Visited States": len(visited_states),
                    "[TRAIN] Episode Reward": episode_reward,
                    "[TRAIN] Average Episode TD Error": episode_td_error / episode_step,
                    "[VALIDATION] Average Episode Reward": avg_episode_reward_validation
                })

        if self.use_wandb:
            wandb.finish()

        return episode_reward_list, episode_td_error_list, is_train_success

    def validate(self):
        episode_reward_lst = np.zeros(shape=(self.validation_num_episodes,), dtype=float)

        test_env = gym.make("FrozenLake-v1", desc=DESC, map_name=MAP_NAME, is_slippery=IS_SLIPPERY)

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
    config = {
        "num_episodes": 200,
        "train_num_episodes_before_next_validation": 10,
        "validation_num_episodes": 10,
        "alpha": 0.1,
        "gamma": 0.95,
        "epsilon": 0.1
    }

    env = gym.make("FrozenLake-v1", desc=DESC, map_name=MAP_NAME, is_slippery=IS_SLIPPERY)

    use_wandb = True

    q_table_agent = QTableAgent(
        env=env,
        config=config,
        use_wandb=use_wandb,
    )

    episode_reward_list, episode_td_error_list, is_train_success = q_table_agent.train()
    print("\nFinal Q-Table Values")
    print("    LEFT   DOWN  RIGHT     UP")
    for idx, observation in enumerate(q_table_agent.q_table):
        print("{0:2d}".format(idx), end=":")
        for action_state in observation:
            print("{0:5.3f} ".format(action_state), end=" ")
        print()

    if is_train_success:
        q_learning_test(q_table_agent=q_table_agent)
    else:
        print("NO PLAYING!!!")


def q_learning_test(q_table_agent: QTableAgent) -> None:
    play_env = gym.make("FrozenLake-v1", desc=DESC, map_name=MAP_NAME, is_slippery=IS_SLIPPERY, render_mode="human")
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
