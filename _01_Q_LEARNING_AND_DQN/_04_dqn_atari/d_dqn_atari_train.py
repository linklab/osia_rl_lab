# https://ale.farama.org/environments/pong/
import gymnasium as gym
import torch
import ale_py
from c_qnet import QNetCNN
import os

from _01_Q_LEARNING_AND_DQN._02_dqn_cartpole.d_dqn_train_test import DqnTrainer

gym.register_envs(ale_py)

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

def main():
    print("TORCH VERSION:", torch.__version__)
    ENV_NAME = "PongNoFrameskip-v4"

    env = gym.make(ENV_NAME)
    env = gym.wrappers.AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=(84, 84),
        grayscale_obs=True,
        grayscale_newaxis=False,
        scale_obs=True
    )
    env = gym.wrappers.FrameStackObservation(env, stack_size=4)

    valid_env = gym.make(ENV_NAME)
    valid_env = gym.wrappers.AtariPreprocessing(
        valid_env,
        frame_skip=4,
        screen_size=(84, 84),
        grayscale_obs=True,
        grayscale_newaxis=False,
        scale_obs=True
    )
    valid_env = gym.wrappers.FrameStackObservation(valid_env, stack_size=4)

    config = {
        "env_name": ENV_NAME,                             # 환경의 이름
        "max_num_episodes": 10_000,                        # 훈련을 위한 최대 에피소드 횟수
        "batch_size": 32,                                 # 훈련시 배치에서 한번에 가져오는 랜덤 배치 사이즈
        "learning_rate": 0.00025,                         # 학습율
        "gamma": 0.99,                                    # 감가율
        "steps_between_train": 2,                         # 훈련 사이의 환경 스텝 수
        "replay_buffer_size": 500_000,                    # 리플레이 버퍼 사이즈
        "epsilon_start": 0.99,                            # Epsilon 초기 값
        "epsilon_end": 0.01,                              # Epsilon 최종 값
        "epsilon_final_scheduled_percent": 0.75,          # Epsilon 최종 값으로 스케줄되는 마지막 에피소드 비율
        "print_episode_interval": 5,                      # Episode 통계 출력에 관한 에피소드 간격
        "target_sync_time_steps_interval": 1_000,           # 기존 Q 모델을 타깃 Q 모델로 동기화시키는 step 간격
        "validation_time_steps_interval": 20_000,          # 검증 사이 마다 각 훈련 episode 간격
        "validation_num_episodes": 3,                     # 검증에 수행하는 에피소드 횟수
        "episode_reward_avg_solved": 20,                  # 훈련 종료를 위한 검증 에피소드 리워드의 Average
    }

    qnet = QNetCNN(n_actions=4)
    target_qnet = QNetCNN(n_actions=4)

    use_wandb = True
    dqn = DqnTrainer(
        env=env, valid_env=valid_env, qnet=qnet, target_qnet=target_qnet, config=config, use_wandb=use_wandb,
        current_dir=CURRENT_DIR
    )
    dqn.train_loop()

if __name__ == "__main__":
    main()
