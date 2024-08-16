import os

import numpy as np

np.set_printoptions(edgeitems=3, linewidth=100000, formatter=dict(float=lambda x: "%5.3f" % x))

import torch
from _04_COP_POINTING_AND_ATTENTION._01_DQN_MKP.a_common import env_config, ENV_NAME, NUM_ITEMS, STATIC_NUM_RESOURCES
from _04_COP_POINTING_AND_ATTENTION._01_DQN_MKP.c_mkp_env import MkpEnv
from _04_COP_POINTING_AND_ATTENTION._02_DQN_ATTN_MKP.f_dqn_attn_train import QNetAttn
from _04_COP_POINTING_AND_ATTENTION._01_DQN_MKP.g_dqn_and_or_tool_test import test
from _04_COP_POINTING_AND_ATTENTION._02_DQN_ATTN_MKP.e_qnet_attn import MODEL_DIR

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(num_episodes, env_name):
    if env_config["use_static_item_resource_demand"]:
        env_config["num_resources"] = STATIC_NUM_RESOURCES

    env = MkpEnv(env_config=env_config)

    print("*" * 100)

    q = QNetAttn(
        n_features=env_config["num_resources"] + 1, device=DEVICE
    )

    model_params = torch.load(
        os.path.join(MODEL_DIR, "{0}_{1}_{2}_latest.pth".format("dqn_attn", NUM_ITEMS, env_name))
    )
    q.load_state_dict(model_params)

    results = test(env, q, num_episodes=num_episodes)

    print("[DQN_ATTN]   Episode Rewards: {0}, Average: {1:.3f}, Duration: {2}".format(
        results["rl_episode_reward_lst"], results["rl_episode_reward_avg"], results["rl_duration_avg"]
    ))
    print("[ OR TOOL] OR Tool Solutions: {0}, Average: {1:.3f}, Duration: {2}".format(
        results["or_tool_solution_lst"], results["or_tool_solutions_avg"], results["or_tool_duration_avg"]
    ))

    env.close()


if __name__ == "__main__":
    NUM_EPISODES = 10

    main(num_episodes=NUM_EPISODES, env_name=ENV_NAME)
