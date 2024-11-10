# https://ale.farama.org/environments/pong/
import os

import gymnasium as gym
from c_qnet import QNetCNN

from _01_Q_LEARNING_AND_DQN._02_dqn_cartpole.d_dqn_train_test import DqnTester

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

def main():
    ENV_NAME = "PongNoFrameskip-v4"

    test_env = gym.make(ENV_NAME, render_mode="human")
    test_env = gym.wrappers.AtariPreprocessing(
        test_env,
        noop_max=30,
        frame_skip=4,
        screen_size=(84, 84),
        grayscale_obs=True,
        grayscale_newaxis=False,
        scale_obs=True
    )
    test_env = gym.wrappers.FrameStackObservation(test_env, stack_size=4)

    qnet = QNetCNN(n_actions=4)

    dqn_tester = DqnTester(env=test_env, qnet = qnet, env_name=ENV_NAME, current_dir=CURRENT_DIR)
    dqn_tester.test()

    test_env.close()

if __name__ == "__main__":
    main()