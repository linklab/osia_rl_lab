from copy import deepcopy
import numpy as np
np.set_printoptions(edgeitems=3, linewidth=100000, formatter=dict(float=lambda x: "%5.3f" % x))

import torch
from torchinfo import summary

from _04_COP_POINTING_AND_ATTENTION._01_COMMON.a_common import env_config, STATIC_NUM_RESOURCES, NUM_ITEMS
from _04_COP_POINTING_AND_ATTENTION._01_COMMON.b_mkp_env import MkpEnv
from _04_COP_POINTING_AND_ATTENTION._01_DQN_MKP.b_dqn_train import DQN
from _04_COP_POINTING_AND_ATTENTION._02_DQN_ATTN_MKP.a_qnet_attn import QNetAttn, MODEL_DIR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    if env_config["use_static_item_resource_demand"]:
        env_config["num_resources"] = STATIC_NUM_RESOURCES

    dqn_config = {
        "max_num_episodes": 15_000 * NUM_ITEMS,              # 훈련을 위한 최대 에피소드 횟수
        "batch_size": 256,                                  # 훈련시 배치에서 한번에 가져오는 랜덤 배치 사이즈
        "learning_rate": 0.0001,                            # 학습율
        "gamma": 1.0,                                       # 감가율
        "steps_between_train": 4,                           # 훈련 사이의 환경 스텝 수
        "target_sync_time_steps_interval": 300 * NUM_ITEMS,       # 기존 Q 모델을 타깃 Q 모델로 동기화시키는 step 간격
        "replay_buffer_size": 10000 * NUM_ITEMS,            # 리플레이 버퍼 사이즈
        "epsilon_start": 0.95,                              # Epsilon 초기 값
        "epsilon_end": 0.01,                                # Epsilon 최종 값
        "epsilon_final_scheduled_percent": 0.25,            # Epsilon 최종 값으로 스케줄되는 마지막 에피소드 비율
        "print_episode_interval": 10,                       # Episode 통계 출력에 관한 에피소드 간격
        "validation_time_steps_interval": 1000,  # 검증 사이 마다 각 훈련 episode 간격
        "validation_num_episodes": 100,                     # 검증에 수행하는 에피소드 횟수
        "early_stop_patience": NUM_ITEMS * 15,               # episode_reward가 개선될 때까지 기다리는 기간
        "double_dqn": True
    }

    q_attn = QNetAttn(
        n_features=env_config["num_resources"] + 1, device=DEVICE
    )
    target_q_attn = QNetAttn(
        n_features=env_config["num_resources"] + 1, device=DEVICE
    )
    target_q_attn.load_state_dict(q_attn.state_dict())

    summary(q_attn, input_size=(1, env_config["num_resources"] + 1))

    env = MkpEnv(env_config=env_config)
    validation_env = deepcopy(env)

    print("*" * 100)

    use_wandb = True
    dqn = DQN(
        model_name="dqn_attn", model_dir=MODEL_DIR, q=q_attn, target_q=target_q_attn,
        env=env, validation_env=validation_env, config=dqn_config, env_config=env_config, use_wandb=use_wandb
    )
    dqn.train_loop()


if __name__ == '__main__':
    main()
