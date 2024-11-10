# https://ale.farama.org/environments/pong/
import os

import gymnasium as gym
import torch
from c_qnet import MODEL_DIR, QNetCNN

from _01_Q_LEARNING_AND_DQN._02_dqn_cartpole.d_dqn_train_test import DqnTester

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

def main():
    NUM_EPISODES = 3
    ENV_NAME = "PongNoFrameskip-v4"

    env = gym.make(ENV_NAME, render_mode="human")

    qnet = QNetCNN(n_actions=4)

    dqn_tester = DqnTester(env=env, qnet = qnet, current_dir=CURRENT_DIR)
    dqn_tester.test()

    env.close()

if __name__ == "__main__":
    main()