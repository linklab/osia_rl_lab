# https://gymnasium.farama.org/environments/classic_control/cart_pole/
import os
import time
from datetime import datetime
from shutil import copyfile

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from a_sac_models import MODEL_DIR, GaussianPolicy, QNetwork, ReplayBuffer, Transition, DEVICE

import wandb


class SAC:
    def __init__(self, env: gym.Env, test_env: gym.Env, config: dict, use_wandb: bool):
        self.env = env
        self.test_env = test_env
        self.use_wandb = use_wandb

        self.env_name = config["env_name"]

        self.current_time = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")

        if use_wandb:
            self.wandb = wandb.init(project="sac_{0}".format(self.env_name), name=self.current_time, config=config)
        else:
            self.wandb = None

        self.max_num_episodes = config["max_num_episodes"]
        self.batch_size = config["batch_size"]
        self.learning_rate = config["learning_rate"]
        self.gamma = config["gamma"]
        self.print_episode_interval = config["print_episode_interval"]
        self.train_num_episodes_before_next_validation = config["train_num_episodes_before_next_validation"]
        self.validation_num_episodes = config["validation_num_episodes"]
        self.episode_reward_avg_solved = config["episode_reward_avg_solved"]
        self.steps_between_train = config["steps_between_train"]
        self.soft_update_tau = config["soft_update_tau"]
        self.replay_buffer_size = config["replay_buffer_size"]
        self.learning_starts = config["learning_starts"]
        self.automatic_entropy_tuning = config["automatic_entropy_tuning"]

        n_features = env.observation_space.shape[0]
        n_actions = env.action_space.shape[0]

        self.policy = GaussianPolicy(n_features=n_features, n_actions=n_actions, action_space=env.action_space)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)

        self.q_network_1 = QNetwork(n_features=n_features, n_actions=n_actions)
        self.q_network_2 = QNetwork(n_features=n_features, n_actions=n_actions)
        self.target_q_network_1 = QNetwork(n_features=n_features, n_actions=n_actions)
        self.target_q_network_2 = QNetwork(n_features=n_features, n_actions=n_actions)

        self.target_q_network_1.load_state_dict(self.q_network_1.state_dict())
        self.target_q_network_2.load_state_dict(self.q_network_2.state_dict())

        self.q_network_1_optimizer = optim.Adam(self.q_network_1.parameters(), lr=self.learning_rate)
        self.q_network_2_optimizer = optim.Adam(self.q_network_2.parameters(), lr=self.learning_rate)

        self.replay_buffer = ReplayBuffer(capacity=self.replay_buffer_size)

        if self.automatic_entropy_tuning:
            self.target_entropy = -torch.prod(torch.Tensor(env.action_space.shape).to(DEVICE)).item()
            print("TARGET ENTROPY: {0}".format(self.target_entropy))
            self.log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.learning_rate)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = 0.2

        self.time_steps = 0
        self.training_time_steps = 0

        self.max_alpha = 5.0

    def train_loop(self) -> None:
        total_train_start_time = time.time()

        validation_episode_reward_avg = -1500
        policy_loss = q_1_td_loss = q_2_td_loss = alpha_loss = mu = entropy = 0.0

        is_terminated = False

        for n_episode in range(1, self.max_num_episodes + 1):
            episode_reward = 0

            observation, _ = self.env.reset()

            done = False

            while not done:
                self.time_steps += 1

                if self.time_steps < self.learning_starts:
                    action = self.env.action_space.sample()
                else:
                    action = self.policy.get_action(observation)

                next_observation, reward, terminated, truncated, _ = self.env.step(action)

                episode_reward += reward

                transition = Transition(observation, action, next_observation, reward, terminated)

                self.replay_buffer.append(transition)

                observation = next_observation
                done = terminated or truncated

                if self.time_steps % self.steps_between_train == 0 and self.time_steps > self.batch_size:
                    policy_loss, q_1_td_loss, q_2_td_loss, alpha_loss, mu, entropy = self.train()

            if n_episode % self.print_episode_interval == 0:
                print(
                    "[Epi. {:3,}, Time Steps {:6,}]".format(n_episode, self.time_steps),
                    "Epi. Reward: {:>9.3f},".format(episode_reward),
                    "Policy L.: {:>7.3f},".format(policy_loss),
                    "Critic L.: {:>7.3f}, {:>7.3f}".format(q_1_td_loss, q_2_td_loss),
                    "Alpha L.: {:>7.3f},".format(alpha_loss),
                    "Alpha: {:>7.3f},".format(self.alpha),
                    "Entropy: {:>7.3f},".format(entropy),
                    "Train Steps: {:5,}, ".format(self.training_time_steps),
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

            if self.use_wandb and n_episode > self.train_num_episodes_before_next_validation:
                self.log_wandb(
                    validation_episode_reward_avg,
                    episode_reward,
                    policy_loss,
                    q_1_td_loss, q_2_td_loss,
                    alpha_loss,
                    mu,
                    entropy,
                    n_episode,
                )

            if is_terminated:
                if self.wandb:
                    for _ in range(5):
                        self.log_wandb(
                            validation_episode_reward_avg,
                            episode_reward,
                            policy_loss,
                            q_1_td_loss, q_2_td_loss,
                            alpha_loss,
                            mu,
                            entropy,
                            n_episode,
                        )
                break

        total_training_time = time.time() - total_train_start_time
        total_training_time = time.strftime("%H:%M:%S", time.gmtime(total_training_time))
        print("Total Training End : {}".format(total_training_time))
        if self.use_wandb:
            self.wandb.finish()

    def log_wandb(
        self,
        validation_episode_reward_avg: float,
        episode_reward: float,
        policy_loss: float,
        q_1_td_loss: float, q_2_td_loss: float,
        alpha_loss: float,
        mu: float,
        entropy: float,
        n_episode: float,
    ) -> None:
        self.wandb.log(
            {
                "[VALIDATION] Mean Episode Reward ({0} Episodes)".format(
                    self.validation_num_episodes
                ): validation_episode_reward_avg,
                "[TRAIN] episode reward": episode_reward,
                "[TRAIN] policy loss": policy_loss,
                "[TRAIN] critic 1 loss": q_1_td_loss,
                "[TRAIN] critic 2 loss": q_2_td_loss,
                "[TRAIN] alpha loss": alpha_loss,
                "[TRAIN] alpha": self.alpha,
                "[TRAIN] mu": mu,
                "[TRAIN] entropy": entropy,
                "[TRAIN] Replay buffer": self.replay_buffer.size(),
                "training episode": n_episode,
                "training steps": self.training_time_steps,
            }
        )

    def train(self):
        self.training_time_steps += 1

        observations, actions, next_observations, rewards, dones = self.replay_buffer.sample(self.batch_size)

        ####################
        # Q NETWORK UPDATE #
        ####################
        with torch.no_grad():
            next_state_action, next_state_log_pi, _, _ = self.policy.sample(next_observations)
            qf1_next_target = self.target_q_network_1(next_observations, next_state_action)
            qf2_next_target = self.target_q_network_2(next_observations, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            min_qf_next_target[dones] = 0.0
            target_values = rewards + self.gamma * min_qf_next_target
            # target_values = (target_values - torch.mean(target_values)) / (torch.std(target_values) + 1e-7)

        # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1 = self.q_network_1(observations, actions)
        qf2 = self.q_network_2(observations, actions)
        qf1_loss = F.mse_loss(qf1, target_values)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, target_values)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]

        self.q_network_1_optimizer.zero_grad()
        qf1_loss.backward()
        nn.utils.clip_grad_norm_(self.q_network_1.parameters(), 3.0)
        self.q_network_1_optimizer.step()

        self.q_network_2_optimizer.zero_grad()
        qf2_loss.backward()
        nn.utils.clip_grad_norm_(self.q_network_2.parameters(), 3.0)
        self.q_network_2_optimizer.step()

        #################
        # Policy UPDATE #
        #################
        sample_actions, log_pi, mu, entropy = self.policy.sample(observations, reparameterization_trick=True)

        qf1_pi = self.q_network_1(observations, sample_actions)
        qf2_pi = self.q_network_2(observations, sample_actions)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = -1.0 * (min_qf_pi - self.alpha * log_pi).mean()  # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 3.0)
        self.policy_optimizer.step()

        #################
        # Alpha UPDATE #
        #################
        if self.automatic_entropy_tuning:
            alpha_loss = -1.0 * (self.log_alpha.exp() * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            nn.utils.clip_grad_norm_([self.log_alpha], 3.0)
            self.alpha_optimizer.step()

            # log_alpha를 기반으로 알파 값 계산 (상한선을 설정)
            with torch.no_grad():
                self.alpha = self.log_alpha.exp().item()

                # 클램핑 기법을 사용해 알파 값이 상한선을 넘지 않도록 제한
                if self.alpha > self.max_alpha:
                    self.log_alpha.data = torch.log(torch.tensor([self.max_alpha]))
        else:
            alpha_loss = torch.tensor(0.).to(DEVICE)

        # sync, TAU: 0.995
        self.soft_synchronize_models(
            source_model=self.q_network_1, target_model=self.target_q_network_1, tau=self.soft_update_tau
        )
        self.soft_synchronize_models(
            source_model=self.q_network_2, target_model=self.target_q_network_2, tau=self.soft_update_tau
        )

        return policy_loss.item(), qf1_loss.item(), qf2_loss.item(), alpha_loss.item(), mu.mean().item(), entropy.item()

    def soft_synchronize_models(self, source_model, target_model, tau):
        source_model_state = source_model.state_dict()
        target_model_state = target_model.state_dict()
        for k, v in source_model_state.items():
            target_model_state[k] = tau * target_model_state[k] + (1.0 - tau) * v
        target_model.load_state_dict(target_model_state)

    def model_save(self, validation_episode_reward_avg: float) -> None:
        filename = "sac_{0}_{1:4.1f}_{2}.pth".format(self.env_name, validation_episode_reward_avg, self.current_time)
        torch.save(self.policy.state_dict(), os.path.join(MODEL_DIR, filename))

        copyfile(src=os.path.join(MODEL_DIR, filename), dst=os.path.join(MODEL_DIR, "sac_{0}_latest.pth".format(self.env_name)))

    def validate(self) -> tuple[np.ndarray, float]:
        episode_reward_lst = np.zeros(shape=(self.validation_num_episodes,), dtype=float)

        for i in range(self.validation_num_episodes):
            episode_reward = 0

            observation, _ = self.test_env.reset()

            done = False

            while not done:
                action = self.policy.get_action(observation, exploration=False)

                next_observation, reward, terminated, truncated, _ = self.test_env.step(action)

                episode_reward += reward
                observation = next_observation
                done = terminated or truncated

            episode_reward_lst[i] = episode_reward

        return episode_reward_lst, np.average(episode_reward_lst)


def main() -> None:
    print("TORCH VERSION:", torch.__version__)
    # ENV_NAME = "Ant-v5"
    ENV_NAME = "HalfCheetah-v5"
    # ENV_NAME = "Pendulum-v1"

    # env
    env = gym.make(ENV_NAME)
    test_env = gym.make(ENV_NAME)

    config = {
        "env_name": ENV_NAME,                               # 환경의 이름
        "max_num_episodes": 200_000,                        # 훈련을 위한 최대 에피소드 횟수
        "batch_size": 256,                                  # 훈련시 배치에서 한번에 가져오는 랜덤 배치 사이즈
        "steps_between_train": 16,                          # 훈련 사이의 환경 스텝 수
        "replay_buffer_size": 1_000_000,                    # 리플레이 버퍼 사이즈
        "learning_rate": 0.0001,                            # 학습율
        "gamma": 0.99,                                      # 감가율
        "soft_update_tau": 0.995,                           # Soft Update Tau
        "print_episode_interval": 20,                       # Episode 통계 출력에 관한 에피소드 간격
        "train_num_episodes_before_next_validation": 100,   # 검증 사이 마다 각 훈련 episode 간격
        "validation_num_episodes": 3,                       # 검증에 수행하는 에피소드 횟수
        # "episode_reward_avg_solved": -150,                  # 훈련 종료를 위한 테스트 에피소드 리워드의 Average
        # "episode_reward_avg_solved": 5000,                  # 훈련 종료를 위한 테스트 에피소드 리워드의 Average
        "episode_reward_avg_solved": 9000,  # 훈련 종료를 위한 테스트 에피소드 리워드의 Average
        "learning_starts": 5000,                            # 충분한 경험 데이터 수집
        "automatic_entropy_tuning": True                    # Alpha Auto Tuning
    }

    use_wandb = False
    sac = SAC(env=env, test_env=test_env, config=config, use_wandb=use_wandb)
    sac.train_loop()


if __name__ == "__main__":
    main()
