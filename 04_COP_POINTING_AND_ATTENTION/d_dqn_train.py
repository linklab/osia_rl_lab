import os
import time
from datetime import datetime
from shutil import copyfile

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb

from a_mkp_env import MultiDimKnapsack
from c_qnet import MODEL_DIR, QNet, ReplayBuffer, Transition

gym.register_envs(MultiDimKnapsack)


class DQN:
    def __init__(self, env: gym.Env | MultiDimKnapsack, test_env: gym.Env, config: dict, use_wandb: bool):
        self.env = env
        self.test_env = test_env
        self.use_wandb = use_wandb

        self.env_name = config["env_name"]

        self.current_time = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")

        if self.use_wandb:
            self.wandb = wandb.init(project="DQN_{0}".format(self.env_name), name=self.current_time, config=config)

        self.max_num_episodes = config["max_num_episodes"]
        self.batch_size = config["batch_size"]
        self.learning_rate = config["learning_rate"]
        self.gamma = config["gamma"]
        self.steps_between_train = config["steps_between_train"]
        self.target_sync_step_interval = config["target_sync_step_interval"]
        self.replay_buffer_size = config["replay_buffer_size"]
        self.epsilon_start = config["epsilon_start"]
        self.epsilon_end = config["epsilon_end"]
        self.epsilon_final_scheduled_percent = config["epsilon_final_scheduled_percent"]
        self.print_episode_interval = config["print_episode_interval"]
        self.train_num_episodes_before_next_validation = config["train_num_episodes_before_next_validation"]
        self.validation_num_episodes = config["validation_num_episodes"]
        self.episode_reward_avg_solved = config["episode_reward_avg_solved"]

        self.epsilon_scheduled_last_episode = self.max_num_episodes * self.epsilon_final_scheduled_percent

        # network
        n_features = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n
        self.q = QNet(n_features=n_features, n_actions=n_actions)
        self.target_q = QNet(n_features=n_features, n_actions=n_actions)
        self.target_q.load_state_dict(self.q.state_dict())

        self.optimizer = optim.Adam(self.q.parameters(), lr=self.learning_rate)

        # agent
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)

        self.time_steps = 0
        self.total_time_steps = 0
        self.training_time_steps = 0

    def epsilon_scheduled(self, current_episode: int) -> float:
        fraction = min(current_episode / self.epsilon_scheduled_last_episode, 1.0)

        epsilon = min(self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start), self.epsilon_start)
        return epsilon

    def train_loop(self) -> None:
        loss = 0.0

        total_train_start_time = time.time()

        validation_episode_reward_avg = 0.0

        is_terminated = False

        for n_episode in range(1, self.max_num_episodes + 1):
            epsilon = self.epsilon_scheduled(n_episode)

            episode_reward = 0

            observation, info = self.env.reset()

            done = False
            terminated = False

            while not done:
                self.time_steps += 1
                self.total_time_steps += 1

                action = self.q.get_action(
                    obs=observation,
                    available_actions=info["available_actions"],
                    epsilon=epsilon,
                )

                next_observation, reward, terminated, truncated, info = self.env.step(action)

                transition = Transition(observation, action, next_observation, reward, terminated)

                self.replay_buffer.append(transition)

                episode_reward += reward
                observation = next_observation
                done = terminated or truncated

                if self.total_time_steps % self.steps_between_train == 0 and self.time_steps > self.batch_size:
                    loss = self.train()

            total_training_time = time.time() - total_train_start_time
            total_training_time = time.strftime("%H:%M:%S", time.gmtime(total_training_time))

            if n_episode % self.print_episode_interval == 0:
                print(
                    "[Episode {:3,}, Time Steps {:6,}]".format(n_episode, self.time_steps),
                    "Episode Reward: {:>5},".format(episode_reward),
                    "Replay buffer: {:>6,},".format(self.replay_buffer.size()),
                    "Loss: {:6.3f},".format(loss),
                    "Epsilon: {:4.2f},".format(epsilon),
                    "Training Steps: {:5,},".format(self.training_time_steps),
                    "Elapsed Time: {}".format(total_training_time),
                )

            if n_episode % self.train_num_episodes_before_next_validation == 0:
                validation_episode_reward_lst, validation_episode_reward_avg = self.validate()

                print(
                    "[Validation Episode Reward: {0}] Average: {1:.3f}".format(
                        validation_episode_reward_lst, validation_episode_reward_avg
                    )
                )

                self.model_save(validation_episode_reward_avg)

            if self.use_wandb:
                self.wandb.log(
                    {
                        "[VALIDATION] Mean Episode Reward ({0} Episodes)".format(
                            self.validation_num_episodes
                        ): validation_episode_reward_avg,
                        "[TRAIN] Episode Reward": episode_reward,
                        "[TRAIN] Loss": loss if loss != 0.0 else 0.0,
                        "[TRAIN] Epsilon": epsilon,
                        "[TRAIN] Replay buffer": self.replay_buffer.size(),
                        "Training Episode": n_episode,
                        "Training Steps": self.training_time_steps,
                    }
                )

            if is_terminated:
                break

        total_training_time = time.time() - total_train_start_time
        total_training_time = time.strftime("%H:%M:%S", time.gmtime(total_training_time))
        print("Total Training End : {}".format(total_training_time))

        if self.use_wandb:
            self.wandb.finish()

    def train(self) -> float:
        self.training_time_steps += 1

        batch = self.replay_buffer.sample(self.batch_size)

        # observations.shape: torch.Size([32, 4]),
        # actions.shape: torch.Size([32, 1]),
        # next_observations.shape: torch.Size([32, 4]),
        # rewards.shape: torch.Size([32, 1]),
        # dones.shape: torch.Size([32])
        observations, actions, next_observations, rewards, dones = batch

        # state_action_values.shape: torch.Size([32, 1])
        q_out = self.q(observations)
        q_values = q_out.gather(dim=-1, index=actions)

        with torch.no_grad():
            q_prime_out = self.target_q(next_observations)
            # next_state_values.shape: torch.Size([32, 1])
            max_q_prime = q_prime_out.max(dim=1, keepdim=True).values
            max_q_prime[dones] = 0.0

            # target_state_action_values.shape: torch.Size([32, 1])
            targets = rewards + self.gamma * max_q_prime

        # loss is just scalar torch value
        loss = F.mse_loss(targets.detach(), q_values)

        # print("observations.shape: {0}, actions.shape: {1}, "
        #       "next_observations.shape: {2}, rewards.shape: {3}, dones.shape: {4}".format(
        #     observations.shape, actions.shape,
        #     next_observations.shape, rewards.shape, dones.shape
        # ))
        # print("state_action_values.shape: {0}".format(state_action_values.shape))
        # print("next_state_values.shape: {0}".format(next_state_values.shape))
        # print("target_state_action_values.shape: {0}".format(
        #     target_state_action_values.shape
        # ))
        # print("loss.shape: {0}".format(loss.shape))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # sync
        if self.time_steps % self.target_sync_step_interval == 0:
            self.target_q.load_state_dict(self.q.state_dict())

        return loss.item()

    def model_save(self, validation_episode_reward_avg: float) -> None:
        n_items = self.env.get_wrapper_attr("n_items")
        n_resources = self.env.get_wrapper_attr("n_resources")
        filename = (
            f"dqn_{self.env_name}_{n_items}x{n_resources}_"
            f"{validation_episode_reward_avg:4.1f}_{self.current_time}.pth"
        )
        torch.save(self.q.state_dict(), os.path.join(MODEL_DIR, filename))

        copyfile(
            src=os.path.join(MODEL_DIR, filename),
            dst=os.path.join(MODEL_DIR, f"dqn_{self.env_name}_{n_items}x{n_resources}_latest.pth")
        )

    def validate(self) -> tuple[np.ndarray, float]:
        episode_reward_lst = np.zeros(shape=(self.validation_num_episodes,), dtype=float)

        for i in range(self.validation_num_episodes):
            episode_reward = 0

            observation, info = self.test_env.reset()

            done = False

            while not done:
                action = self.q.get_action(
                    obs=observation,
                    available_actions=info["available_actions"],
                    epsilon=0.0,
                )

                next_observation, reward, terminated, truncated, info = self.test_env.step(action)

                episode_reward += reward
                observation = next_observation
                done = terminated or truncated

            episode_reward_lst[i] = episode_reward

        return episode_reward_lst, np.average(episode_reward_lst)


def main() -> None:
    print("TORCH VERSION:", torch.__version__)
    ENV_NAME = "MultiDimKnapsack-v0"
    N_ITEMS = 10
    N_RESOURCES = 2

    def make_env() -> gym.Env:
        env = gym.make(ENV_NAME, n_items=N_ITEMS, n_resources=N_RESOURCES)
        env = gym.wrappers.FlattenObservation(env)
        return env

    env = make_env()
    test_env = make_env()

    config = {
        "env_name": ENV_NAME,                              # 환경의 이름
        "max_num_episodes": 20_000,                        # 훈련을 위한 최대 에피소드 횟수
        "batch_size": 256,                                 # 훈련시 배치에서 한번에 가져오는 랜덤 배치 사이즈
        "learning_rate": 0.0001,                           # 학습율
        "gamma": 1.00,                                     # 감가율
        "steps_between_train": 64,                         # 훈련 사이의 환경 스텝 수
        "target_sync_step_interval": 1000,                 # 기존 Q 모델을 타깃 Q 모델로 동기화시키는 step 간격
        "replay_buffer_size": 50_000,                      # 리플레이 버퍼 사이즈
        "epsilon_start": 0.95,                             # Epsilon 초기 값
        "epsilon_end": 0.01,                               # Epsilon 최종 값
        "epsilon_final_scheduled_percent": 0.75,           # Epsilon 최종 값으로 스케줄되는 마지막 에피소드 비율
        "print_episode_interval": 100,                     # Episode 통계 출력에 관한 에피소드 간격
        "train_num_episodes_before_next_validation": 500,  # 검증 사이 마다 각 훈련 episode 간격
        "validation_num_episodes": 50,                     # 검증에 수행하는 에피소드 횟수
        "episode_reward_avg_solved": np.inf,               # 훈련 종료를 위한 검증 에피소드 리워드의 Average
    }

    use_wandb = True
    dqn = DQN(env=env, test_env=test_env, config=config, use_wandb=use_wandb)
    dqn.train_loop()


if __name__ == "__main__":
    main()
