import copy
import os
import time
from datetime import datetime
from shutil import copyfile

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F

from b_actor_and_critic_attn import MODEL_DIR, ActorAttn, Buffer, CriticAttn, Transition
from torch.distributions import Categorical
from torch.optim import Adam

import wandb

from _04_COP_POINTING_AND_ATTENTION._01_COMMON.a_common import env_config, ENV_NAME, STATIC_NUM_RESOURCES, NUM_ITEMS, \
    EarlyStopModelSaver
from _04_COP_POINTING_AND_ATTENTION._01_COMMON.b_mkp_env import MkpEnv


def master_loop(global_actor, shared_stat, wandb, global_lock, config):
    test_env = MkpEnv(env_config=env_config)

    class PPOMaster:
        def __init__(self, global_actor, shared_stat, wandb, test_env, global_lock, config):
            self.is_terminated = False

            self.global_actor = global_actor

            self.shared_stat = shared_stat
            self.wandb = wandb
            self.test_env = test_env
            self.global_lock = global_lock

            self.train_num_episodes_before_next_validation = config["train_num_episodes_before_next_validation"]
            self.validation_num_episodes = config["validation_num_episodes"]

            self.current_time = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")

            self.early_stop_model_saver = EarlyStopModelSaver(
                model_name="ppo_attn",
                model_dir=MODEL_DIR,
                patience=config["early_stop_patience"]
            )

        def validate_loop(self):
            total_train_start_time = time.time()

            validation_episode_reward_avg = 0.0
            validation_total_value_avg = 0.0

            while True:
                validation_conditions = [
                    self.shared_stat.global_episodes.value != 0,
                    self.shared_stat.global_episodes.value % self.train_num_episodes_before_next_validation == 0,
                ]
                if all(validation_conditions):
                    self.global_lock.acquire()
                    validation_episode_reward_lst, validation_episode_reward_avg, validation_total_value_lst, validation_total_value_avg = \
                        self.validate()

                    total_training_time = time.time() - total_train_start_time
                    total_training_time_str = time.strftime("%H:%M:%S", time.gmtime(total_training_time))

                    print("[Validation Episode Reward: {0}] Average: {1:.3f}".format(
                        validation_episode_reward_lst, validation_episode_reward_avg
                    ))
                    print("[ValidationTotal Value: {0}] Average: {1:.3f}, Elapsed Time: {2}".format(
                        validation_total_value_lst, validation_total_value_avg, total_training_time_str
                    ))

                    is_terminated = self.early_stop_model_saver.check(
                        validation_episode_reward_avg=validation_episode_reward_avg,
                        num_items=env_config["num_items"], env_name=ENV_NAME, current_time=self.current_time,
                        n_episode=self.shared_stat.global_episodes.value,
                        time_steps=self.shared_stat.global_time_steps.value,
                        training_time_steps=self.shared_stat.global_training_time_steps.value,
                        model=self.global_actor
                    )

                    if is_terminated:
                        self.shared_stat.is_terminated.value = 1  # break

                    self.global_lock.release()

                if self.wandb:
                    self.log_wandb(validation_episode_reward_avg, validation_total_value_avg)

                if bool(self.shared_stat.is_terminated.value):
                    if self.wandb:
                        for _ in range(5):
                            self.log_wandb(validation_episode_reward_avg, validation_total_value_avg)
                    break

        def validate(self):
            episode_reward_lst = np.zeros(shape=(self.validation_num_episodes,), dtype=float)
            total_value_lst = np.zeros(shape=(self.validation_num_episodes,), dtype=float)

            for i in range(self.validation_num_episodes):
                episode_reward = 0

                observation, info = self.test_env.reset()

                done = False

                while not done:
                    action = self.global_actor.get_action(observation, action_mask=info["ACTION_MASK"], exploration=False)

                    next_observation, reward, terminated, truncated, info = self.test_env.step(action)

                    episode_reward += reward
                    observation = next_observation
                    done = terminated or truncated

                episode_reward_lst[i] = episode_reward
                total_value_lst[i] = info["VALUE_ALLOCATED"]

            return episode_reward_lst, np.average(episode_reward_lst), total_value_lst, np.average(total_value_lst)

        def log_wandb(self, validation_episode_reward_avg, validation_total_value_avg):
            self.wandb.log(
                {
                    "[VALIDATION] Mean Episode Reward ({0} Episodes)".format(self.validation_num_episodes): validation_episode_reward_avg,
                    "[VALIDATION] Mean Total Value ({0} Episodes)".format(self.validation_num_episodes): validation_total_value_avg,
                    "[TRAIN] Episode Reward": self.shared_stat.last_episode_reward.value,
                    "[TRAIN] Policy Loss": self.shared_stat.last_policy_loss.value,
                    "[TRAIN] Critic Loss": self.shared_stat.last_critic_loss.value,
                    "Training Episode": self.shared_stat.global_episodes.value,
                    "Training Steps": self.shared_stat.global_training_time_steps.value,
                }
            )

        def model_save(self, validation_episode_reward_avg):
            filename = "ppo_{0}_{1:4.1f}_{2}.pth".format(ENV_NAME, validation_episode_reward_avg, self.current_time)
            torch.save(self.global_actor.state_dict(), os.path.join(MODEL_DIR, filename))

            copyfile(
                src=os.path.join(MODEL_DIR, filename), dst=os.path.join(MODEL_DIR, "ppo_{0}_latest.pth".format(ENV_NAME))
            )

    master = PPOMaster(
        global_actor=global_actor,
        shared_stat=shared_stat,
        wandb=wandb,
        test_env=test_env,
        global_lock=global_lock,
        config=config,
    )

    master.validate_loop()


def worker_loop(process_id, global_actor, global_critic, shared_stat, global_lock, config):
    env = MkpEnv(env_config=env_config)

    class PPOAgent:
        def __init__(self, worker_id, global_actor, global_critic, shared_stat, env, global_lock, config):
            self.worker_id = worker_id
            self.env = env
            self.is_terminated = False

            self.global_actor = global_actor
            self.global_critic = global_critic

            self.local_actor = copy.deepcopy(global_actor)
            self.local_actor.load_state_dict(global_actor.state_dict())

            self.local_critic = copy.deepcopy(global_critic)
            self.local_critic.load_state_dict(global_critic.state_dict())

            self.local_actor_optimizer = Adam(self.local_actor.parameters(), lr=config["learning_rate"])
            self.local_critic_optimizer = Adam(self.local_critic.parameters(), lr=config["learning_rate"])

            self.global_lock = global_lock

            self.max_num_episodes = config["max_num_episodes"]
            self.ppo_epochs = config["ppo_epochs"]
            self.ppo_clip_coefficient = config["ppo_clip_coefficient"]

            self.batch_size = config["batch_size"]
            self.learning_rate = config["learning_rate"]
            self.gamma = config["gamma"]
            self.entropy_beta = config["entropy_beta"]
            self.print_episode_interval = config["print_episode_interval"]

            self.buffer = Buffer()

            self.shared_stat = shared_stat
            self.time_steps = 0
            self.training_time_steps = 0

            self.current_time = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")

        def train_loop(self):
            policy_loss = critic_loss = 0.0

            for n_episode in range(1, self.max_num_episodes + 1):
                episode_reward = 0

                observation, info = self.env.reset()
                done = False

                while not done:
                    self.time_steps += 1
                    self.shared_stat.global_time_steps.value += 1

                    action = self.local_actor.get_action(observation, action_mask=info["ACTION_MASK"])
                    next_observation, reward, terminated, truncated, info = self.env.step(action)

                    episode_reward += reward

                    transition = Transition(
                        observation, action, next_observation, reward, terminated, info["ACTION_MASK"]
                    )

                    self.buffer.append(transition)

                    observation = next_observation
                    done = terminated or truncated

                    if self.time_steps % self.batch_size == 0:
                        policy_loss, critic_loss = self.train()
                        self.shared_stat.last_policy_loss.value = policy_loss
                        self.shared_stat.last_critic_loss.value = critic_loss

                        self.buffer.clear()

                self.shared_stat.global_episodes.value += 1
                self.shared_stat.last_episode_reward.value = episode_reward

                if n_episode % self.print_episode_interval == 0:
                    print(
                        "[Worker: {:2}, Episode {:3,}, Time Steps {:6,}]".format(self.worker_id, n_episode, self.time_steps),
                        "Episode Reward: {:>6.4f},".format(episode_reward),
                        "Police Loss: {:>7.3f},".format(policy_loss),
                        "Critic Loss: {:>7.3f},".format(critic_loss),
                        "Training Steps: {:5,}".format(self.training_time_steps),
                    )

                if bool(self.shared_stat.is_terminated.value):
                    break

        def train(self):
            self.training_time_steps += 1
            self.shared_stat.global_training_time_steps.value += 1

            # Getting values from buffer
            observations, actions, next_observations, rewards, dones, action_masks = self.buffer.get()
            # observations.shape: [256, 6]
            # actions.shape: [256, 1]
            # next_observations.shape: [256, 6]
            # rewards.shape: [256, 1]
            # dones.shape: [256]

            self.global_lock.acquire()
            self.local_critic.load_state_dict(self.global_critic.state_dict())
            self.local_actor.load_state_dict(self.global_actor.state_dict())
            self.global_lock.release()

            # Calculating target values
            values = self.local_critic(observations).squeeze(dim=-1)
            next_values = self.local_critic(next_observations).squeeze(dim=-1)
            next_values[dones] = 0.0
            target_values = rewards.squeeze(dim=-1) + self.gamma * next_values
            # Normalized advantage calculation
            advantages = target_values - values
            advantages = (advantages - torch.mean(advantages)) / (torch.std(advantages) + 1e-7)

            old_mu = self.local_actor.forward(observations)
            old_dist = Categorical(probs=old_mu)
            old_action_log_probs = old_dist.log_prob(value=actions.squeeze(dim=-1)).squeeze(dim=-1)
            # actions.shape: [256, 1]
            # actions.squeeze(dim=-1).shape: [256]
            # old_action_log_probs.shape: [256]

            for epoch in range(self.ppo_epochs):
                values = self.local_critic(observations).squeeze(dim=-1)

                # CRITIC UPDATE
                critic_loss = F.mse_loss(target_values.detach(), values)
                self.local_critic_optimizer.zero_grad()
                critic_loss.backward()
                self.local_critic_optimizer.step()

                # Actor Loss computing
                mu = self.local_actor.forward(observations)
                dist = Categorical(probs=mu)
                action_log_probs = dist.log_prob(value=actions.squeeze(dim=-1)).squeeze(dim=-1)

                ratio = torch.exp(action_log_probs - old_action_log_probs.detach())

                ratio_advantages = ratio * advantages.detach()
                clipped_ratio_advantages = (
                    torch.clamp(ratio, 1 - self.ppo_clip_coefficient, 1 + self.ppo_clip_coefficient) * advantages.detach()
                )
                ratio_advantages_sum = torch.min(ratio_advantages, clipped_ratio_advantages).sum()

                entropy = dist.entropy().squeeze(dim=-1)
                entropy_sum = entropy.sum()

                actor_loss = -1.0 * ratio_advantages_sum - 1.0 * entropy_sum * self.entropy_beta

                # Actor Update
                self.local_actor_optimizer.zero_grad()
                actor_loss.backward()
                self.local_actor_optimizer.step()

            # GLOBAL MODEL UPDATE
            self.global_lock.acquire()
            self.global_actor.load_state_dict(self.local_actor.state_dict())
            self.global_critic.load_state_dict(self.local_critic.state_dict())
            self.global_lock.release()

            return (
                actor_loss.item(),
                critic_loss.item(),
            )

    agent = PPOAgent(
        worker_id=process_id,
        global_actor=global_actor,
        global_critic=global_critic,
        shared_stat=shared_stat,
        env=env,
        global_lock=global_lock,
        config=config,
    )

    agent.train_loop()


class SharedStat:
    def __init__(self):
        self.global_episodes = mp.Value("I", 0)  # I: unsigned int
        self.global_time_steps = mp.Value("I", 0)  # I: unsigned int
        self.global_training_time_steps = mp.Value("I", 0)  # I: unsigned int

        self.last_episode_reward = mp.Value("d", 0.0)  # d: double
        self.last_policy_loss = mp.Value("d", 0.0)  # d: double
        self.last_critic_loss = mp.Value("d", 0.0)  # d: double

        self.is_terminated = mp.Value("I", 0)  # I: unsigned int --> bool


class PPO:
    def __init__(self, config, use_wandb):
        self.config = config
        self.use_wandb = use_wandb
        self.num_workers = min(config["num_workers"], mp.cpu_count() - 1)

        # Initialize global models and optimizers
        self.global_actor = ActorAttn(
            n_features=env_config["num_resources"] + 1
        ).share_memory()
        self.global_critic = CriticAttn(
            n_features=env_config["num_resources"] + 1, n_items=NUM_ITEMS
        ).share_memory()

        self.global_lock = mp.Lock()
        self.shared_stat = SharedStat()

        if self.use_wandb:
            current_time = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
            self.wandb = wandb.init(project="PPO_{0}".format(ENV_NAME), name=current_time, config=config)
        else:
            self.wandb = None

        self.worker_processes = []
        self.master_process = None

    def train_loop(self):
        for i in range(self.num_workers):
            worker_process = mp.Process(
                target=worker_loop,
                args=(i, self.global_actor, self.global_critic, self.shared_stat, self.global_lock, self.config)
            )
            worker_process.start()
            print(">>> Worker Process: {0} Started!".format(worker_process.pid))
            self.worker_processes.append(worker_process)

        master_process = mp.Process(
            target=master_loop, args=(self.global_actor, self.shared_stat, self.wandb, self.global_lock, self.config)
        )
        master_process.start()
        print(">>> Master Process: {0} Started!".format(master_process.pid))

        ###########################

        for worker_process in self.worker_processes:
            worker_process.join()
            print(">>> Worker Process: {0} Joined!".format(worker_process.pid))

        master_process.join()
        print(">>> Master Process: {0} Joined!".format(master_process.pid))

        if self.wandb:
            self.wandb.finish()


def main():
    print("TORCH VERSION:", torch.__version__)
    if env_config["use_static_item_resource_demand"]:
        env_config["num_resources"] = STATIC_NUM_RESOURCES

    ppo_config = {
        "num_workers": 4,                                   # 동시 수행 Worker Process 수
        "max_num_episodes": 15_000 * NUM_ITEMS,               # 훈련을 위한 최대 에피소드 횟수
        "ppo_epochs": 10,                                   # PPO 내부 업데이트 횟수
        "ppo_clip_coefficient": 0.2,                        # PPO Ratio Clip Coefficient
        "batch_size": 256,                                  # 훈련시 배치에서 한번에 가져오는 랜덤 배치 사이즈
        "learning_rate": 0.0001,                            # 학습율
        "gamma": 0.99,                                      # 감가율
        "entropy_beta": 0.03,                               # 엔트로피 가중치
        "print_episode_interval": 10,                       # Episode 통계 출력에 관한 에피소드 간격
        "train_num_episodes_before_next_validation": 1000,  # 검증 사이 마다 각 훈련 episode 간격
        "validation_num_episodes": 100,                     # 검증에 수행하는 에피소드 횟수
        "early_stop_patience": NUM_ITEMS * 10,              # episode_reward가 개선될 때까지 기다리는 기간
    }

    use_wandb = True
    ppo = PPO(use_wandb=use_wandb, config=ppo_config)
    ppo.train_loop()


if __name__ == "__main__":
    main()
