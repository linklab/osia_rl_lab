import random

import numpy as np

from _04_COP_POINTING_AND_ATTENTION._01_DQN_MKP.c_mkp_env import MkpEnv
from _04_COP_POINTING_AND_ATTENTION._01_DQN_MKP.a_config import env_config, STATIC_NUM_RESOURCES


class Dummy_Agent:
    def __init__(self, num_items):
        self.num_items = num_items

    def get_action(self, observation, action_mask):
        # observation is not used
        available_actions = np.where(action_mask == 0)[0]
        action_id = random.choice(available_actions)
        return action_id


def main():
    print("START RUN!!!")
    # env_config = {
    #     "num_items": 10,  # 아이템 개수
    #     "use_static_item_resource_demand": False,    # 항상 미리 정해 놓은 아이템 자원 요구량 사용 유무
    #     "use_same_item_resource_demand": False,      # 각 에피소드 초기에 동일한 아이템 자원 요구량 사용 유무
    #     "lowest_item_resource_demand": [1, 1],       # 아이템 자원 초기화 시에 각 아이템 자원 최소 요구량
    #     "highest_item_resource_demand": [100, 100],  # 아이템 자원 초기화 시에 각 아이템 자원 최대 요구량
    #     "initial_resources_capacity": [250, 250],  # 초기 자원 용량
    # }

    if env_config["use_static_item_resource_demand"]:
        env_config["num_resources"] = STATIC_NUM_RESOURCES

    env = MkpEnv(env_config=env_config)

    agent = Dummy_Agent(env_config["num_items"])
    observation, info = env.reset()

    episode_step = 0
    done = False
    print("[Step: RESET] Info: {0}".format(info))
    #print(info['INTERNAL_STATE'].flatten())

    while not done:
        action = agent.get_action(observation, info["ACTION_MASK"])
        next_observation, reward, terminated, truncated, info = env.step(action)

        episode_step += 1
        print("[Step: {0:3}] Obs.: {1}, Action: {2:>2}, Next Obs.: {3}, "
              "Reward: {4:>6.3f}, Terminated: {5}, Truncated: {6}, Info: {7}".format(
            episode_step, observation.shape, action, next_observation.shape,
            reward, terminated, truncated, info
        ))
        #print(info['INTERNAL_STATE'].flatten())
        observation = next_observation
        done = terminated or truncated


if __name__ == "__main__":
    main()
