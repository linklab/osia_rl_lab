# https://gymnasium.farama.org/environments/classic_control/pendulum/
import time
import os
import multiprocessing
import copy

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Normal
import wandb
from datetime import datetime
from shutil import copyfile

from b_actor_and_critic import MODEL_DIR, Actor, Critic, Transition, Buffer
from a_shared_adam import SharedAdam


class SharedInfo:
    def __init__(self):
        self.global_episodes = multiprocessing.Value('I', 0)                # I: unsigned int
        self.global_time_steps = multiprocessing.Value('I', 0)              # I: unsigned int
        self.global_training_time_steps = multiprocessing.Value('I', 0)     # I: unsigned int

        self.shared_info_train_result_lock = multiprocessing.Lock()
        self.last_policy_loss = multiprocessing.Value('d', 0.0)             # d: double
        self.last_critic_loss = multiprocessing.Value('d', 0.0)             # d: double
        self.last_avg_mu_v = multiprocessing.Value('d', 0.0)                # d: double
        self.last_avg_std_v = multiprocessing.Value('d', 0.0)               # d: double
        self.last_avg_action = multiprocessing.Value('d', 0.0)              # d: double
        self.last_avg_action_prob = multiprocessing.Value('d', 0.0)         # d: double

        self.is_terminated = multiprocessing.Value('I', 0)                  # I: unsigned int --> bool


class A3CAgent:
    def __init__(
        self, worker_id, global_actor, global_critic, global_actor_optimizer, global_critic_optimizer,
        shared_info, env, config
    ):
        self.worker_id = worker_id
        self.env_name = config["env_name"]
        self.env = env
        self.is_terminated = False

        self.global_actor = global_actor
        self.global_critic = global_critic

        self.local_actor = copy.deepcopy(global_actor)
        self.local_actor.load_state_dict(global_actor.state_dict())

        self.local_critic = copy.deepcopy(global_critic)
        self.local_critic.load_state_dict(global_critic.state_dict())

        self.global_actor_optimizer = global_actor_optimizer
        self.global_critic_optimizer = global_critic_optimizer

        self.max_num_episodes = config["max_num_episodes"]
        self.batch_size = config["batch_size"]
        self.learning_rate = config["learning_rate"]
        self.gamma = config["gamma"]
        self.entropy_beta = config["entropy_beta"]
        self.print_episode_interval = config["print_episode_interval"]
        self.train_num_episodes_before_next_validation = config["train_num_episodes_before_next_validation"]
        self.validation_num_episodes = config["validation_num_episodes"]
        self.episode_reward_avg_solved = config["episode_reward_avg_solved"]

        self.buffer = Buffer()

        self.shared_info = shared_info
        self.time_steps = 0
        self.training_time_steps = 0

        self.current_time = datetime.now().astimezone().strftime('%Y-%m-%d_%H-%M-%S')

    def train_loop(self):
        total_train_start_time = time.time()

        policy_loss = critic_loss = avg_mu_v = avg_std_v = avg_action = avg_action_prob = 0.0

        is_terminated = False

        for n_episode in range(1, self.max_num_episodes + 1):
            self.local_actor.load_state_dict(self.global_actor.state_dict())
            self.local_critic.load_state_dict(self.global_critic.state_dict())

            episode_reward = 0

            observation, _ = self.env.reset()
            done = False

            while not done:
                self.time_steps += 1
                with self.shared_info.global_time_steps.get_lock():
                    self.shared_info.global_time_steps.value += 1

                action = self.local_actor.get_action(observation)
                next_observation, reward, terminated, truncated, _ = self.env.step(action * 2)

                episode_reward += reward

                transition = Transition(observation, action, next_observation, reward, terminated)

                self.buffer.append(transition)

                observation = next_observation
                done = terminated or truncated

                if self.time_steps % self.batch_size == 0:
                    policy_loss, critic_loss, avg_mu_v, avg_std_v, avg_action, avg_action_prob = self.train()
                    with self.shared_info.shared_info_train_result_lock:
                        self.shared_info.policy_loss = policy_loss
                        self.shared_info.critic_loss = critic_loss
                        self.shared_info.avg_mu_v = avg_mu_v
                        self.shared_info.avg_std_v = avg_std_v
                        self.shared_info.avg_action = avg_action
                        self.shared_info.avg_action_prob = avg_action_prob

                    self.buffer.clear()

            with self.shared_info.global_episodes.get_lock():
                self.shared_info.global_episodes.value += 1

            total_training_time = time.time() - total_train_start_time
            total_training_time = time.strftime('%H:%M:%S', time.gmtime(total_training_time))

            if n_episode % self.print_episode_interval == 0:
                print(
                    "[Workder: {:2}, Episode {:3,}, Steps {:6,}]".format(
                        self.worker_id, n_episode, self.time_steps
                    ),
                    "Episode Reward: {:>9.3f},".format(episode_reward),
                    "Police Loss: {:>7.3f},".format(policy_loss),
                    "Critic Loss: {:>7.3f},".format(critic_loss),
                    "Training Steps: {:5,},".format(self.training_time_steps),
                    "Elapsed Time: {}".format(total_training_time)
                )

    def train(self):
        self.training_time_steps += 1
        with self.shared_info.global_training_time_steps.get_lock():
            self.shared_info.global_training_time_steps.value += 1

        # Getting values from buffer
        observations, actions, next_observations, rewards, dones = self.buffer.get()

        # Calculating target values
        values = self.local_critic(observations).squeeze(dim=-1)
        next_values = self.local_critic(next_observations).squeeze(dim=-1)
        next_values[dones] = 0.0

        q_values = rewards.squeeze(dim=-1) + self.gamma * next_values

        # CRITIC UPDATE
        critic_loss = F.mse_loss(q_values.detach(), values)
        self.global_critic_optimizer.zero_grad()
        for i in self.local_critic.parameters():
            i.grad = None
        critic_loss.backward()
        for local_param, global_param in zip(self.local_critic.parameters(), self.global_critic.parameters()):
            global_param.grad = local_param.grad
        self.global_critic_optimizer.step()
        self.local_critic.load_state_dict(self.global_critic.state_dict())

        # Advantage calculating
        advantages = q_values - values
        advantages = (advantages - torch.mean(advantages)) / (torch.std(advantages) + 1e-7)

        # Actor Loss computing
        mu, std = self.local_actor.forward(observations)
        dist = Normal(mu, std)
        action_log_probs = dist.log_prob(value=actions).squeeze(dim=-1)  # natural log

        log_pi_advantages = action_log_probs * advantages.detach()
        log_pi_advantages_sum = log_pi_advantages.sum()

        entropy = dist.entropy().squeeze(dim=-1)
        entropy_sum = entropy.sum()

        actor_loss = -1.0 * log_pi_advantages_sum - 1.0 * entropy_sum * self.entropy_beta

        # Actor Update
        self.global_actor_optimizer.zero_grad()
        for i in self.local_actor.parameters():
            i.grad = None
        actor_loss.backward()
        for local_param, global_param in zip(self.local_actor.parameters(), self.global_actor.parameters()):
            global_param.grad = local_param.grad
        self.global_actor_optimizer.step()
        self.local_actor.load_state_dict(self.global_actor.state_dict())

        return (
            actor_loss.item(),
            critic_loss.item(),
            mu.mean().item(),
            std.mean().item(),
            actions.mean().item(),
            action_log_probs.exp().mean().item()
        )

    def validate(self):
        episode_rewards = np.zeros(self.validation_num_episodes)

        for i in range(self.validation_num_episodes):
            observation, _ = self.test_env.reset()
            episode_reward = 0

            done = False

            while not done:
                action = self.local_actor.get_action(observation, exploration=False)
                next_observation, reward, terminated, truncated, _ = self.test_env.step(action * 2)
                episode_reward += reward
                observation = next_observation
                done = terminated or truncated

            episode_rewards[i] = episode_reward

        return episode_rewards, np.average(episode_rewards)

    def model_save(self, validation_episode_reward_avg):
        filename = "a3c_{0}_{1:4.1f}_{2}.pth".format(
            self.env_name, validation_episode_reward_avg, self.current_time
        )
        torch.save(self.local_actor.state_dict(), os.path.join(MODEL_DIR, filename))

        copyfile(
            src=os.path.join(MODEL_DIR, filename),
            dst=os.path.join(MODEL_DIR, "a3c_{0}_latest.pth".format(self.env_name))
        )

def master_loop(global_actor, global_critic, shared_info, run_wandb, test_env):
    if 0 == n_episode % self.train_num_episodes_before_next_validation:
        validation_episode_reward_lst, validation_episode_reward_avg = self.validate()

        print("[Validation Episode Reward: {0}] Average: {1:.3f}".format(
            validation_episode_reward_lst, validation_episode_reward_avg
        ))

        if validation_episode_reward_avg > self.episode_reward_avg_solved:
            print("Solved in {0:,} steps ({1:,} training steps)!".format(
                self.shared_info.time_steps.value, self.shared_info.training_time_steps.value
            ))
            self.model_save(validation_episode_reward_avg)
            is_terminated = True
            # break

    if not self.run_wandb:
        pass
    else:
        self.run_wandb.log({
            "[VALIDATION] Mean Episode Reward ({0} Episodes)".format(
                self.validation_num_episodes): episode_reward,
            "[TRAIN] Episode Reward": episode_reward,
            "[TRAIN] Policy Loss": policy_loss,
            "[TRAIN] Critic Loss": critic_loss,
            "[TRAIN] avg_mu_v": avg_mu_v,
            "[TRAIN] avg_std_v": avg_std_v,
            "[TRAIN] avg_action": avg_action,
            "[TRAIN] avg_action_prob": avg_action_prob,
            "Training Episode": n_episode,
            # "Training Steps": self.training_time_steps,
            "Training Steps": self.shared_info.training_time_steps.value,
        })

    if is_terminated:
        break

def worker_loop(
        process_id, global_actor, global_critic, global_actor_optimizer, global_critic_optimizer, shared_info, run_wandb,
        test_env, config
):
    env_name = config["env_name"]
    env = gym.make(env_name)

    agent = A3CAgent(
        worker_id=process_id,
        global_actor=global_actor,
        global_critic=global_critic,
        global_actor_optimizer=global_actor_optimizer,
        global_critic_optimizer=global_critic_optimizer,
        shared_info=shared_info,
        run_wandb=run_wandb,
        env=env,
        test_env=test_env,
        config=config,
    )

    agent.train_loop()


def main():
    ENV_NAME = "Pendulum-v1"

    config = {
        "env_name": ENV_NAME,  # 환경의 이름
        "num_workers": 3,
        "max_num_episodes": 50_000,  # 훈련을 위한 최대 에피소드 횟수
        "batch_size": 64,  # 훈련시 배치에서 한번에 가져오는 랜덤 배치 사이즈
        "learning_rate": 0.0005,  # 학습율
        "gamma": 0.99,  # 감가율
        "entropy_beta": 0.01,  # 엔트로피 가중치
        "print_episode_interval": 20,  # Episode 통계 출력에 관한 에피소드 간격
        "train_num_episodes_before_next_validation": 100,  # 검증 사이 마다 각 훈련 episode 간격
        "validation_num_episodes": 3,  # 검증에 수행하는 에피소드 횟수
        "episode_reward_avg_solved": -150  # 훈련 종료를 위한 테스트 에피소드 리워드의 Average
    }

    use_wandb = True
    num_workers = min(config["num_workers"], multiprocessing.cpu_count() - 1)

    test_env = gym.make(ENV_NAME)

    # Initialize global models and optimizers
    global_actor = Actor(n_features=3, n_actions=1).share_memory()
    global_critic = Critic(n_features=3).share_memory()

    global_actor_optimizer = SharedAdam(global_actor.parameters(), lr=config["learning_rate"], betas=(0.92, 0.999))
    global_critic_optimizer = SharedAdam(global_critic.parameters(), lr=config["learning_rate"], betas=(0.92, 0.999))

    manager = multiprocessing.Manager()
    shared_info = SharedInfo()

    if use_wandb:
        current_time = datetime.now().astimezone().strftime('%Y-%m-%d_%H-%M-%S')
        run_wandb = wandb.init(
            project="A3C_{0}".format(config["env_name"]),
            name=current_time,
            config=config
        )
    else:
        run_wandb = None

    worker_processes = []

    for i in range(num_workers):
        env = gym.make(ENV_NAME)
        worker_process = multiprocessing.Process(
            target=worker_loop,
            args=(
                i, global_actor, global_critic, global_actor_optimizer, global_critic_optimizer,
                shared_info, env, config
            )
        )
        worker_process.start()
        worker_processes.append(worker_process)

    test_env = gym.make(ENV_NAME)
    master_process = multiprocessing.Process(
        target=master_loop,
        args=(global_actor, global_critic, shared_info, run_wandb, test_env)
    )
    master_process.start()

    while True:
        r = res_queue.get()
        if r is not None:
            res.append(r)
        else:
            break

    for p in processes:
        p.join()

    if use_wandb and run_wandb:
        run_wandb.finish()


if __name__ == '__main__':
    main()
