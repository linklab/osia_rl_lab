# https://gymnasium.farama.org/environments/classic_control/cart_pole/
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
import ale_py
gym.register_envs(ale_py)

from gymnasium.wrappers import FrameStackObservation, AtariPreprocessing

from c_qnet import MODEL_DIR, ReplayBuffer, Transition, QNetCNN


class DQN:
    def __init__(self, env: gym.Env, valid_env: gym.Env, config: dict, use_wandb: bool):
        self.env = env
        self.valid_env = valid_env
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
        self.q = QNetCNN(n_actions=6)
        self.target_q = QNetCNN(n_actions=6)
        self.target_q.load_state_dict(self.q.state_dict())

        self.optimizer = optim.Adam(self.q.parameters(), lr=self.learning_rate)

        # agent
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)

        self.time_steps = 0
        self.training_time_steps = 0

    def epsilon_scheduled(self, current_episode: int) -> float:
        fraction = min(current_episode / self.epsilon_scheduled_last_episode, 1.0)

        epsilon = min(self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start), self.epsilon_start)
        return epsilon

    def train_loop(self) -> None:
        loss = 0.0

        total_train_start_time = time.time()

        validation_episode_reward_avg = -21.0

        is_terminated = False

        for n_episode in range(1, self.max_num_episodes + 1):
            epsilon = self.epsilon_scheduled(n_episode)

            episode_reward = 0

            observation, _ = self.env.reset()

            done = False

            while not done:
                self.time_steps += 1

                action = self.q.get_action(observation, epsilon)

                next_observation, reward, terminated, truncated, _ = self.env.step(action)

                transition = Transition(observation, action, next_observation, reward, terminated)

                self.replay_buffer.append(transition)

                episode_reward += reward
                observation = next_observation
                done = terminated or truncated

                if self.time_steps % self.steps_between_train == 0 and self.time_steps > self.batch_size:
                    loss = self.train()

            if n_episode % self.print_episode_interval == 0:
                print(
                    "[Episode {:3,}, Time Steps {:6,}]".format(n_episode, self.time_steps),
                    "Episode Reward: {:>5},".format(episode_reward),
                    "Replay buffer: {:>6,},".format(self.replay_buffer.size()),
                    "Loss: {:7.5f},".format(loss),
                    "Epsilon: {:4.2f},".format(epsilon),
                    "Training Steps: {:5,},".format(self.training_time_steps),
                )

            if n_episode % self.train_num_episodes_before_next_validation == 0:
                validation_episode_reward_lst, validation_episode_reward_avg = self.validate()

                total_training_time = time.time() - total_train_start_time
                total_training_time = time.strftime("%H:%M:%S", time.gmtime(total_training_time))

                print(
                    "[Validation Episode Reward: {0}] Average: {1:.3f}, Elapsed Time: {2}".format(
                        validation_episode_reward_lst, validation_episode_reward_avg, total_training_time
                    )
                )

                if validation_episode_reward_avg > self.episode_reward_avg_solved:
                    print("Solved in {0:,} time steps ({1:,} training steps)!".format(self.time_steps, self.training_time_steps))
                    self.model_save(validation_episode_reward_avg)
                    is_terminated = True

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
        filename = "dqn_{0}_{1:4.1f}_{2}.pth".format(self.env_name, validation_episode_reward_avg, self.current_time)
        torch.save(self.q.state_dict(), os.path.join(MODEL_DIR, filename))

        copyfile(src=os.path.join(MODEL_DIR, filename), dst=os.path.join(MODEL_DIR, "dqn_{0}_latest.pth".format(self.env_name)))

    def validate(self) -> tuple[np.ndarray, float]:
        episode_reward_lst = np.zeros(shape=(self.validation_num_episodes,), dtype=float)

        for i in range(self.validation_num_episodes):
            episode_reward = 0

            observation, _ = self.valid_env.reset()

            done = False

            while not done:
                action = self.q.get_action(observation, epsilon=0.0)

                next_observation, reward, terminated, truncated, _ = self.valid_env.step(action)

                episode_reward += reward
                observation = next_observation
                done = terminated or truncated

            episode_reward_lst[i] = episode_reward

        return episode_reward_lst, np.average(episode_reward_lst)


def main() -> None:
    print("TORCH VERSION:", torch.__version__)
    ENV_NAME = "PongNoFrameskip-v4"

    env = gym.make(ENV_NAME)
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

    valid_env = gym.make(ENV_NAME)
    valid_env = AtariPreprocessing(
        valid_env,
        frame_skip=4,
        screen_size=(84, 84),
        grayscale_obs=True,
        grayscale_newaxis=False,
        scale_obs=True
    )
    valid_env = FrameStackObservation(valid_env, stack_size=4)

    config = {
        "env_name": ENV_NAME,                             # 환경의 이름
        "max_num_episodes": 2_000,                      # 훈련을 위한 최대 에피소드 횟수
        "batch_size": 64,                                # 훈련시 배치에서 한번에 가져오는 랜덤 배치 사이즈
        "learning_rate": 0.0001,                          # 학습율
        "gamma": 0.99,                                    # 감가율
        "steps_between_train": 2,                         # 훈련 사이의 환경 스텝 수
        "target_sync_step_interval": 500,               # 기존 Q 모델을 타깃 Q 모델로 동기화시키는 step 간격
        "replay_buffer_size": 1_000_000,                  # 리플레이 버퍼 사이즈
        "epsilon_start": 0.99,                            # Epsilon 초기 값
        "epsilon_end": 0.01,                              # Epsilon 최종 값
        "epsilon_final_scheduled_percent": 0.75,          # Epsilon 최종 값으로 스케줄되는 마지막 에피소드 비율
        "print_episode_interval": 5,                      # Episode 통계 출력에 관한 에피소드 간격
        "train_num_episodes_before_next_validation": 50,  # 검증 사이 마다 각 훈련 episode 간격
        "validation_num_episodes": 3,                     # 검증에 수행하는 에피소드 횟수
        "episode_reward_avg_solved": 20,                  # 훈련 종료를 위한 검증 에피소드 리워드의 Average
    }

    use_wandb = True
    dqn = DQN(env=env, valid_env=valid_env, config=config, use_wandb=use_wandb)
    dqn.train_loop()


if __name__ == "__main__":
    main()
