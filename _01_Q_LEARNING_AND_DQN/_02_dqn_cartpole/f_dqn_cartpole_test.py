# https://gymnasium.farama.org/environments/classic_control/cart_pole/
import os

import gymnasium as gym
import torch
from c_qnet import MODEL_DIR, QNet

from _01_Q_LEARNING_AND_DQN._02_dqn_cartpole.d_dqn_train_test import DqnTester

def main():
    NUM_EPISODES = 3
    ENV_NAME = "CartPole-v1"

    env = gym.make(ENV_NAME, render_mode="human")

    qnet = QNet(n_features=4, n_actions=2)
    model_params = torch.load(os.path.join(MODEL_DIR, "dqn_{0}_latest.pth".format(ENV_NAME)), weights_only=True)
    qnet.load_state_dict(model_params)
    qnet.eval()

    dqn_tester = DqnTester(env=env, qnet = qnet, num_episodes=NUM_EPISODES)
    dqn_tester.test()
    env.close()

if __name__ == "__main__":
    main()