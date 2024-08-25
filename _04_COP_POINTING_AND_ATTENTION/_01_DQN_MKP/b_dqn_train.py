import time
from copy import deepcopy
import numpy as np

np.set_printoptions(edgeitems=3, linewidth=100000, formatter=dict(float=lambda x: "%5.3f" % x))

import torch
from torchinfo import summary

print("TORCH VERSION:", torch.__version__)

import torch.nn.functional as F
import torch.optim as optim
import wandb
from datetime import datetime


from _04_COP_POINTING_AND_ATTENTION._01_COMMON.a_common import env_config, ENV_NAME, STATIC_NUM_RESOURCES, NUM_ITEMS, \
    EarlyStopModelSaver
from _04_COP_POINTING_AND_ATTENTION._01_COMMON.b_mkp_env import MkpEnv
from _04_COP_POINTING_AND_ATTENTION._01_DQN_MKP.a_qnet import QNet, ReplayBuffer, Transition, MODEL_DIR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN:
    def __init__(self, model_name, model_dir, q, target_q, env, validation_env, config, env_config, use_wandb):
        self.model_name = model_name
        self.q = q
        self.target_q = target_q
        self.env = env
        self.validation_env = validation_env
        self.use_wandb = use_wandb

        self.env_name = ENV_NAME

        self.current_time = datetime.now().astimezone().strftime('%Y-%m-%d_%H-%M-%S')

        if self.use_wandb:
            project_name = "DQN_{0}_{1}_WITH_{2}".format(
                env_config["num_items"], self.env_name, config["max_num_episodes"]
            )
            if env_config["use_static_item_resource_demand"] is False and \
                    env_config["use_same_item_resource_demand"] is False:
                project_name += "_RANDOM_INSTANCES"
            elif env_config["use_static_item_resource_demand"] is True:
                project_name += "_STATIC_INSTANCES"
            elif env_config["use_same_item_resource_demand"] is True:
                project_name += "_SAME_INSTANCES"
            else:
                raise ValueError()

            self.wandb = wandb.init(
                project=project_name,
                name="{0}_{1}".format("AM", self.current_time),
                config=config | env_config
            )

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
        self.double_dqn = config["double_dqn"]

        self.epsilon_scheduled_last_episode = self.max_num_episodes * self.epsilon_final_scheduled_percent

        self.optimizer = optim.Adam(self.q.parameters(), lr=self.learning_rate)

        # agent
        self.replay_buffer = ReplayBuffer(self.replay_buffer_size)

        self.time_steps = 0
        self.total_time_steps = 0
        self.training_time_steps = 0

        self.early_stop_model_saver = EarlyStopModelSaver(
            model_name=self.model_name,
            model_dir=model_dir,
            patience=config["early_stop_patience"]
        )

    def epsilon_scheduled(self, current_episode):
        fraction = min(current_episode / self.epsilon_scheduled_last_episode, 1.0)

        epsilon = min(
            self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start),
            self.epsilon_start
        )
        return epsilon

    def train_loop(self):
        loss = 0.0

        total_train_start_time = time.time()
        validation_episode_reward_avg = 0.0
        validation_total_value_avg = 0.0

        is_terminated = False

        for n_episode in range(1, self.max_num_episodes + 1):
            epsilon = self.epsilon_scheduled(n_episode)

            episode_reward = 0
            observation, info = self.env.reset()

            done = False

            while not done:
                self.time_steps += 1
                self.total_time_steps += 1

                action = self.q.get_action(observation, epsilon, action_mask=info["ACTION_MASK"])

                next_observation, reward, terminated, truncated, info = self.env.step(action)

                transition = Transition(observation, action, next_observation, reward, terminated, info["ACTION_MASK"])

                self.replay_buffer.append(transition)

                episode_reward += reward
                observation = next_observation
                done = terminated or truncated

                if self.total_time_steps % self.steps_between_train == 0 and self.time_steps > self.batch_size:
                    loss = self.train()

            total_training_time = time.time() - total_train_start_time
            total_training_time_str = time.strftime('%H:%M:%S', time.gmtime(total_training_time))

            if n_episode % self.print_episode_interval == 0:
                print(
                    "[Episode {0:4,}/{1:5,}, Time Steps {2:6,}]".format(
                        n_episode, self.max_num_episodes, self.time_steps
                    ),
                    "Episode Reward: {:>4.2f},".format(episode_reward),
                    "Replay buffer: {:>6,},".format(self.replay_buffer.size()),
                    "Loss: {:.6f},".format(loss),
                    "Epsilon: {:4.2f},".format(epsilon),
                    "Training Steps: {:>5,}".format(self.training_time_steps)
                )

            # print(epsilon, self.epsilon_end, n_episode, self.train_num_episodes_before_next_validation, "!!!")
            if n_episode % self.train_num_episodes_before_next_validation == 0:
                validation_episode_reward_lst, validation_episode_reward_avg, validation_total_value_lst, validation_total_value_avg = \
                    self.validate()

                print("[Validation Episode Reward: {0}] Average: {1:.3f}".format(
                    validation_episode_reward_lst, validation_episode_reward_avg
                ))
                print("[ValidationTotal Value: {0}] Average: {1:.3f}, Elapsed Time: {2}".format(
                    validation_total_value_lst, validation_total_value_avg, total_training_time_str
                ))

                is_terminated = self.early_stop_model_saver.check(
                    validation_episode_reward_avg=validation_episode_reward_avg,
                    num_items=env_config["num_items"], env_name=ENV_NAME, current_time=self.current_time,
                    n_episode=n_episode, time_steps=self.time_steps, training_time_steps=self.training_time_steps,
                    model=self.q
                )

            if self.use_wandb:
                self.wandb.log({
                    "[VALIDATION] Mean Episode Reward ({0} Episodes)".format(self.validation_num_episodes): validation_episode_reward_avg,
                    "[VALIDATION] Mean Total Value ({0} Episodes)".format(self.validation_num_episodes): validation_total_value_avg,
                    "[TRAIN] Episode Reward (Total Value)": episode_reward,
                    "[TRAIN] Loss": loss if loss != 0.0 else 0.0,
                    "[TRAIN] Epsilon": epsilon,
                    "[TRAIN] Replay buffer": self.replay_buffer.size(),
                    "[TRAIN] Episodes per second": n_episode / total_training_time,
                    "Training Episode": n_episode,
                    "Training Steps": self.training_time_steps
                })

            if is_terminated:
                break

        total_training_time = time.time() - total_train_start_time
        total_training_time_str = time.strftime('%H:%M:%S', time.gmtime(total_training_time))
        print("Total Training End : {}".format(total_training_time_str))
        if self.use_wandb:
            self.wandb.finish()

    def train(self):
        self.training_time_steps += 1

        batch = self.replay_buffer.sample(self.batch_size)

        observations, actions, next_observations, rewards, dones, action_masks = batch

        q_out = self.q(observations)
        q_values = q_out.gather(dim=-1, index=actions)

        with torch.no_grad():
            if self.double_dqn:
                q_prime_out = self.q(next_observations)

                q_prime_out = q_prime_out.masked_fill(action_masks, -float('inf'))
                # target_argmax_action.shape: [256, 1]
                target_argmax_action = torch.argmax(q_prime_out, dim=-1, keepdim=True)

                q_prime_out = self.target_q(next_observations)
                next_q_values = q_prime_out.gather(dim=-1, index=target_argmax_action)
                next_q_values[dones] = 0.0

                targets = rewards + self.gamma * next_q_values
            else:
                q_prime_out = self.target_q(next_observations)

                q_prime_out = q_prime_out.masked_fill(action_masks, -float('inf'))

                max_q_prime = q_prime_out.max(dim=-1, keepdim=True).values
                max_q_prime[dones] = 0.0

                targets = rewards + self.gamma * max_q_prime

        loss = F.mse_loss(targets.detach(), q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # sync
        if self.time_steps % self.target_sync_step_interval == 0:
            self.target_q.load_state_dict(self.q.state_dict())

        return loss.item()

    def validate(self):
        episode_reward_lst = np.zeros(shape=(self.validation_num_episodes,), dtype=float)
        total_value_lst = np.zeros(shape=(self.validation_num_episodes,), dtype=float)

        for i in range(self.validation_num_episodes):
            episode_reward = 0

            observation, info = self.validation_env.reset()

            done = False

            while not done:
                action = self.q.get_action(observation, epsilon=0.0, action_mask=info["ACTION_MASK"])

                next_observation, reward, terminated, truncated, info = self.validation_env.step(action)

                episode_reward += reward
                observation = next_observation
                done = terminated or truncated

            episode_reward_lst[i] = episode_reward
            total_value_lst[i] = info["VALUE_ALLOCATED"]

        return episode_reward_lst, np.average(episode_reward_lst), total_value_lst, np.average(total_value_lst)


def main():
    if env_config["use_static_item_resource_demand"]:
        env_config["num_resources"] = STATIC_NUM_RESOURCES

    dqn_config = {
        "max_num_episodes": 10_000 * NUM_ITEMS,             # 훈련을 위한 최대 에피소드 횟수
        "batch_size": 256,                                  # 훈련시 배치에서 한번에 가져오는 랜덤 배치 사이즈
        "learning_rate": 0.0001,                            # 학습율
        "gamma": 1.0,                                       # 감가율
        "steps_between_train": 4,                           # 훈련 사이의 환경 스텝 수
        "target_sync_step_interval": 300 * NUM_ITEMS,       # 기존 Q 모델을 타깃 Q 모델로 동기화시키는 step 간격
        "replay_buffer_size": 10000 * NUM_ITEMS,            # 리플레이 버퍼 사이즈
        "epsilon_start": 0.95,                              # Epsilon 초기 값
        "epsilon_end": 0.01,                                # Epsilon 최종 값
        "epsilon_final_scheduled_percent": 0.25,            # Epsilon 최종 값으로 스케줄되는 마지막 에피소드 비율
        "print_episode_interval": 10,                       # Episode 통계 출력에 관한 에피소드 간격
        "train_num_episodes_before_next_validation": 1000,  # 검증 사이 마다 각 훈련 episode 간격
        "validation_num_episodes": 100,                     # 검증에 수행하는 에피소드 횟수
        "early_stop_patience": NUM_ITEMS * 5,               # episode_reward가 개선될 때까지 기다리는 기간
        "double_dqn": True
    }

    q = QNet(
        n_features=env_config["num_items"] * (env_config["num_resources"] + 1),
        n_actions=env_config["num_items"],
    )
    target_q = QNet(
        n_features=env_config["num_items"] * (env_config["num_resources"] + 1),
        n_actions=env_config["num_items"],
    )
    target_q.load_state_dict(q.state_dict())

    summary(q, input_size=(1, env_config["num_items"] * (env_config["num_resources"] + 1)))

    env = MkpEnv(env_config=env_config)
    validation_env = deepcopy(env)

    print("*" * 100)

    use_wandb = True
    dqn = DQN(
        model_name="dqn", model_dir=MODEL_DIR, q=q, target_q=target_q,
        env=env, validation_env=validation_env, config=dqn_config, env_config=env_config, use_wandb=use_wandb
    )
    dqn.train_loop()


if __name__ == '__main__':
    main()
