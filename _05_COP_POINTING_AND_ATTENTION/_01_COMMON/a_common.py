import os
import numpy as np
import torch
from shutil import copyfile

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


class EarlyStopModelSaver:
    """주어진 patience 이후로 episode_reward가 개선되지 않으면 학습을 조기 중지"""

    def __init__(self, model_name, model_dir, patience):
        """
        Args:
            patience (int): episode_reward가 개선될 때까지 기다리는 기간
        """
        self.model_name = model_name
        self.model_dir = model_dir
        self.patience = patience
        self.counter = 0
        self.max_validation_episode_reward = -np.inf
        self.model_filename_saved = None

    def check(
            self, validation_episode_reward_avg, num_items, env_name, current_time,
            n_episode, time_steps, training_time_steps, model
    ):
        early_stop = False

        if validation_episode_reward_avg >= self.max_validation_episode_reward:
            print("[EARLY STOP] validation_episode_reward {0:.5f} is increased to {1:.5f}".format(
                self.max_validation_episode_reward, validation_episode_reward_avg
            ))
            self.model_save(validation_episode_reward_avg, num_items, env_name, n_episode, current_time, model)
            self.max_validation_episode_reward = validation_episode_reward_avg
            self.counter = 0
        else:
            self.counter += 1
            if self.counter < self.patience:
                print("[EARLY STOP] COUNTER: {0} (validation_episode_reward/max_validation_episode_reward={1:.5f}/{2:.5f})".format(
                    self.counter, validation_episode_reward_avg, self.max_validation_episode_reward
                ))
            else:
                early_stop = True
                print("[EARLY STOP] COUNTER: {0} - Solved in {1:,} episode, {2:,} steps ({3:,} training steps)!".format(
                    self.counter, n_episode, time_steps, training_time_steps
                ))
        return early_stop

    def model_save(self, validation_episode_reward_avg, num_items, env_name, n_episode, current_time, model):
        if self.model_filename_saved is not None:
            os.remove(self.model_filename_saved)

        filename = "{0}_{1}_{2}_{3}_{4:5.3f}_{5}.pth".format(
            self.model_name,
            num_items, env_name,
            current_time, validation_episode_reward_avg, n_episode
        )
        filename = os.path.join(self.model_dir, filename)
        torch.save(model.state_dict(), filename)
        self.model_filename_saved = filename
        print("*** MODEL SAVED TO {0}".format(filename))

        latest_file_name = "{0}_{1}_{2}_latest.pth".format(self.model_name, num_items, env_name)
        copyfile(
            src=os.path.join(self.model_dir, filename),
            dst=os.path.join(self.model_dir, latest_file_name)
        )
        print("*** MODEL UPDATED TO {0}".format(os.path.join(self.model_dir, latest_file_name)))

