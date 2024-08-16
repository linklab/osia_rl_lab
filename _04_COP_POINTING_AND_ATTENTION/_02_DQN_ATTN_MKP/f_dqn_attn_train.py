import os, sys

from copy import deepcopy
import numpy as np
np.set_printoptions(edgeitems=3, linewidth=100000, formatter=dict(float=lambda x: "%5.3f" % x))

import torch
from torchinfo import summary

from _04_COP_POINTING_AND_ATTENTION._01_DQN_MKP.a_config import env_config, dqn_config, STATIC_NUM_RESOURCES
from _04_COP_POINTING_AND_ATTENTION._01_DQN_MKP.c_mkp_env import MkpEnv
from _04_COP_POINTING_AND_ATTENTION._01_DQN_MKP.f_dqn_train import DQN
from _04_COP_POINTING_AND_ATTENTION._02_DQN_ATTN_MKP.e_qnet_attn import QNetAttn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    current_path = os.path.dirname(os.path.realpath(__file__))
    project_home = os.path.abspath(os.path.join(current_path, os.pardir))
    if project_home not in sys.path:
        sys.path.append(project_home)

    model_dir = os.path.join(project_home, "_02_DQN_ATTN_MKP", "models")
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    if env_config["use_static_item_resource_demand"]:
        env_config["num_resources"] = STATIC_NUM_RESOURCES

    q = QNetAttn(
        n_features=env_config["num_resources"] + 1, n_actions=env_config["num_items"], device=DEVICE
    )
    target_q = QNetAttn(
        n_features=env_config["num_resources"] + 1, n_actions=env_config["num_items"], device=DEVICE
    )
    target_q.load_state_dict(q.state_dict())

    summary(q, input_size=(1, env_config["num_resources"] + 1))

    env = MkpEnv(env_config=env_config)
    validation_env = deepcopy(env)

    print("*" * 100)

    use_wandb = False
    dqn = DQN(
        q=q, target_q=target_q, model_dir=model_dir,
        env=env, validation_env=validation_env, config=dqn_config, env_config=env_config, use_wandb=use_wandb
    )
    dqn.train_loop()


if __name__ == '__main__':
    main()
