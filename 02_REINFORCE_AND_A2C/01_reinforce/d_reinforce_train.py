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
from torch.distributions import Normal

from c_policy_and_value import DEVICE, MODEL_DIR, Buffer, Policy, StateValueNet, Transition


class REINFORCE:
    def __init__(self, env: gym.Env, test_env: gym.Env, config: dict, use_baseline: bool, use_wandb: bool):
        self.env = env
        self.test_env = test_env
        self.use_baseline = use_baseline
        self.use_wandb = use_wandb

        self.env_name = config["env_name"]

        self.current_time = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")

        if self.use_wandb:
            self.wandb = wandb.init(project="REINFORCE_{0}".format(self.env_name), name=self.current_time, config=config)

        self.max_num_episodes = config["max_num_episodes"]
        self.learning_rate = config["learning_rate"]
        self.gamma = config["gamma"]
        self.print_episode_interval = config["print_episode_interval"]
        self.train_num_episodes_before_next_validation = config["train_num_episodes_before_next_validation"]
        self.validation_num_episodes = config["validation_num_episodes"]
        self.episode_reward_avg_solved = config["episode_reward_avg_solved"]

        self.policy = Policy(n_features=3, n_actions=1)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)

        if self.use_baseline:
            self.state_value_net = StateValueNet(n_features=3)
            self.value_optimizer = optim.Adam(self.state_value_net.parameters(), lr=self.learning_rate)

        self.buffer = Buffer()

        self.time_steps = 0
        self.training_time_steps = 0

    def train_loop(self) -> None:
        total_train_start_time = time.time()

        validation_episode_reward_avg = -1500

        is_terminated = False

        for n_episode in range(1, self.max_num_episodes + 1):
            episode_reward = 0

            observation, _ = self.env.reset()

            done = False

            while not done:
                self.time_steps += 1

                action = self.policy.get_action(observation)

                next_observation, reward, terminated, truncated, _ = self.env.step(action * 2)

                episode_reward += reward

                transition = Transition(observation, action, next_observation, reward, terminated)

                self.buffer.append(transition)

                observation = next_observation
                done = terminated or truncated

            # TRAIN AFTER EPISODE DONE
            policy_loss, avg_mu_v, avg_std_v, avg_action = self.train()
            self.buffer.clear()

            if n_episode % self.print_episode_interval == 0:
                print(
                    "[Episode {:3,}, Steps {:6,}]".format(n_episode, self.time_steps),
                    "Episode Reward: {:>9.3f},".format(episode_reward),
                    "Policy Loss: {:>7.3f},".format(policy_loss),
                    "Training Steps: {:5,}".format(self.training_time_steps),
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
                    print("Solved in {0:,} steps ({1:,} training steps)!".format(self.time_steps, self.training_time_steps))
                    self.model_save(validation_episode_reward_avg)
                    is_terminated = True

            if self.use_wandb and n_episode > self.train_num_episodes_before_next_validation:
                self.log_wandb(
                    validation_episode_reward_avg, episode_reward, policy_loss, avg_mu_v, avg_std_v, avg_action, n_episode
                )

            if is_terminated:
                if self.run_wandb:
                    for _ in range(5):
                        self.log_wandb(
                            validation_episode_reward_avg, episode_reward, policy_loss, avg_mu_v, avg_std_v, avg_action, n_episode
                        )
                break

        total_training_time = time.time() - total_train_start_time
        total_training_time = time.strftime("%H:%M:%S", time.gmtime(total_training_time))
        print("Total Training End : {}".format(total_training_time))
        self.wandb.finish()

    def log_wandb(
        self,
        validation_episode_reward_avg: float,
        episode_reward: float,
        policy_loss: float,
        avg_mu_v: float,
        avg_std_v: float,
        avg_action: float,
        n_episode: int
    ) -> None:
        self.wandb.log(
            {
                "[VALIDATION] Mean Episode Reward ({0} Episodes)".format(
                    self.validation_num_episodes
                ): validation_episode_reward_avg,
                "[TRAIN] Episode Reward": episode_reward,
                "[TRAIN] Policy Loss": policy_loss,
                "[TRAIN] avg_mu_v": avg_mu_v,
                "[TRAIN] avg_std_v": avg_std_v,
                "[TRAIN] avg_action": avg_action,
                "Training Episode": n_episode,
                "Training Steps": self.training_time_steps,
            }
        )

    def train(self) -> tuple[float, float, float, float]:
        self.training_time_steps += 1

        observations, actions, next_observations, rewards, dones = self.buffer.get()

        reversed_rewards = rewards.squeeze(dim=-1).cpu().numpy()[::-1]

        G = 0.0
        return_lst = []
        for reward in reversed_rewards:
            G = reward + self.gamma * G
            return_lst.append(G)

        returns = torch.tensor(return_lst[::-1], dtype=torch.float32, device=DEVICE)

        mu_v, std_v = self.policy.forward(observations)
        dist = Normal(loc=mu_v, scale=std_v)
        action_log_probs = dist.log_prob(value=actions).squeeze(dim=-1)  # natural log

        # print(action_log_probs.mean(), torch.exp(action_log_probs).mean(), torch.normal(mu_v, std_v).mean())

        if self.use_baseline:
            values = self.state_value_net(observations).squeeze(dim=-1)
            v_loss = F.mse_loss(values, returns)
            self.value_optimizer.zero_grad()
            v_loss.backward()
            self.value_optimizer.step()

            returns_baseline = returns - values.detach()
            # normalization
            returns_baseline = (returns_baseline - torch.mean(returns_baseline)) / (torch.std(returns_baseline) + 1e-7)
            log_pi_returns = action_log_probs * returns_baseline
            log_pi_returns_sum = log_pi_returns.sum()
            # print(
            #     returns.shape, values.shape, returns_baseline.shape, action_log_probs.shape, log_pi_returns.shape,
            #     log_pi_returns_sum.shape, "!!!"
            # )
        else:
            # normalization
            returns = (returns - torch.mean(returns)) / (torch.std(returns) + 1e-7)
            log_pi_returns = action_log_probs * returns
            log_pi_returns_sum = log_pi_returns.sum()
            # print(
            # returns.shape, action_log_probs.shape, log_pi_returns.shape, log_pi_returns_sum.shape, "!!!"
            # )

        policy_loss = -1.0 * log_pi_returns_sum

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        return (
            policy_loss.item(),
            mu_v.mean().item(),
            std_v.mean().item(),
            actions.mean().item(),
        )

    def model_save(self, validation_episode_reward_avg: float) -> None:
        filename = "reinforce_{0}_{1:4.1f}_{2}.pth".format(self.env_name, validation_episode_reward_avg, self.current_time)
        torch.save(self.policy.state_dict(), os.path.join(MODEL_DIR, filename))

        copyfile(
            src=os.path.join(MODEL_DIR, filename), dst=os.path.join(MODEL_DIR, "reinforce_{0}_latest.pth".format(self.env_name))
        )

    def validate(self) -> tuple[np.ndarray, float]:
        episode_reward_lst = np.zeros(shape=(self.validation_num_episodes,), dtype=float)

        for i in range(self.validation_num_episodes):
            episode_reward = 0

            observation, _ = self.test_env.reset()

            done = False

            while not done:
                action = self.policy.get_action(observation, exploration=False)

                next_observation, reward, terminated, truncated, _ = self.test_env.step(action * 2)

                episode_reward += reward
                observation = next_observation
                done = terminated or truncated

            episode_reward_lst[i] = episode_reward

        return episode_reward_lst, np.average(episode_reward_lst)


def main() -> None:
    print("TORCH VERSION:", torch.__version__)
    ENV_NAME = "Pendulum-v1"

    # env
    env = gym.make(ENV_NAME)
    test_env = gym.make(ENV_NAME)

    config = {
        "env_name": ENV_NAME,  # 환경의 이름
        "max_num_episodes": 200_000,  # 훈련을 위한 최대 에피소드 횟수
        "learning_rate": 0.0003,  # 학습율
        "gamma": 0.99,  # 감가율
        "print_episode_interval": 20,  # Episode 통계 출력에 관한 에피소드 간격
        "train_num_episodes_before_next_validation": 100,  # 검증 사이 마다 각 훈련 episode 간격
        "validation_num_episodes": 3,  # 검증에 수행하는 에피소드 횟수
        "episode_reward_avg_solved": -150,  # 훈련 종료를 위한 테스트 에피소드 리워드의 Average
    }

    use_wandb = True
    reinforce = REINFORCE(env=env, test_env=test_env, config=config, use_baseline=True, use_wandb=use_wandb)
    reinforce.train_loop()


if __name__ == "__main__":
    main()
