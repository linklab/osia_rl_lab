#https://github.com/markub3327/flappy-bird-gymnasium
import os
import gymnasium as gym
import flappy_bird_gymnasium
from c_qnet import QNet

from _01_Q_LEARNING_AND_DQN._02_dqn_cartpole.d_dqn_train_test import DqnTester

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

def main():
    ENV_NAME = "FlappyBird-v0"

    test_env = gym.make(ENV_NAME, render_mode="rgb_array")

    qnet = QNet(n_actions=2)

    dqn_tester = DqnTester(env=test_env, qnet = qnet, env_name=ENV_NAME, current_dir=CURRENT_DIR)
    dqn_tester.test()

    test_env.close()

if __name__ == "__main__":
    main()