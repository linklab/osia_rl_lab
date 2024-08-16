ENV_NAME = "MKP"

### STATIC: START ###
STATIC_RESOURCE_DEMAND_SAMPLE = [
    [21, 21],
    [79, 89],
    [44, 25],
    [48, 32],
    [84, 98],
    [44, 47],
    [44, 49],
    [43, 53],
    [73, 69],
    [65, 64],
]
STATIC_VALUE_SAMPLE = [
    36,
    19,
    86,
    22,
    76,
    11,
    9,
    96,
    23,
    56,
]
STATIC_NUM_RESOURCES = 2
### STATIC: END ###

NUM_ITEMS = 10
NUM_RESOURCES = 2

env_config = {
    "num_items": NUM_ITEMS,                                             # 대기하는 아이템 개수
    "num_resources": NUM_RESOURCES,                                     # 자원 개수
    "use_static_item_resource_demand": False,                           # 항상 미리 정해 놓은 아이템 자원 요구량 사용 유무
    "use_same_item_resource_demand": False,                             # 각 에피소드 초기에 동일한 아이템 자원 요구량 사용 유무
    "lowest_item_resource_demand": [50, 50],                            # 아이템의 각 자원 최소 요구량
    "highest_item_resource_demand": [100, 100],                         # 아이템의 각 자원 최대 요구량
    "lowest_item_value": 1,                                             # 아이템의 최소 값어치
    "highest_item_value": 100,                                          # 아이템의 최대 값어치
    "initial_resources_capacity": [NUM_ITEMS * 30, NUM_ITEMS * 30],     # 초기 자원 용량
    "state_normalization": True,                                        # 상태 정보 정규화 유무
}

if env_config["use_same_item_resource_demand"]:
    assert env_config["use_static_item_resource_demand"] is False

if env_config["use_static_item_resource_demand"]:
    assert env_config["use_same_item_resource_demand"] is False

if env_config["use_static_item_resource_demand"]:
    assert env_config["num_items"] == 10

dqn_config = {
    "max_num_episodes": 5000 * NUM_ITEMS,               # 훈련을 위한 최대 에피소드 횟수
    "batch_size": 256,                                  # 훈련시 배치에서 한번에 가져오는 랜덤 배치 사이즈
    "learning_rate": 0.0001,                            # 학습율
    "gamma": 1.0,                                       # 감가율
    "steps_between_train": 4,                           # 훈련 사이의 환경 스텝 수
    "target_sync_step_interval": 100 * NUM_ITEMS,       # 기존 Q 모델을 타깃 Q 모델로 동기화시키는 step 간격
    "replay_buffer_size": 1000 * NUM_ITEMS,             # 리플레이 버퍼 사이즈
    "epsilon_start": 0.95,                              # Epsilon 초기 값
    "epsilon_end": 0.01,                                # Epsilon 최종 값
    "epsilon_final_scheduled_percent": 0.25,            # Epsilon 최종 값으로 스케줄되는 마지막 에피소드 비율
    "print_episode_interval": 10,                       # Episode 통계 출력에 관한 에피소드 간격
    "train_num_episodes_before_next_validation": 500,   # 검증 사이 마다 각 훈련 episode 간격
    "validation_num_episodes": 100,                     # 검증에 수행하는 에피소드 횟수
    "early_stop_patience": NUM_ITEMS * 3,               # episode_reward가 개선될 때까지 기다리는 기간
    "double_dqn": True
}

