import os
from datetime import datetime, timedelta
import numpy as np

np.set_printoptions(edgeitems=3, linewidth=100000, formatter=dict(float=lambda x: "%5.3f" % x))

import torch
from _04_COP_POINTING_AND_ATTENTION._01_COMMON.a_common import env_config, ENV_NAME, NUM_ITEMS, STATIC_NUM_RESOURCES
from _04_COP_POINTING_AND_ATTENTION._01_COMMON.b_mkp_env import MkpEnv
from _04_COP_POINTING_AND_ATTENTION._03_PPO_ATTN_MKP.c_ppo_attn_train import ActorAttn
from _04_COP_POINTING_AND_ATTENTION._03_PPO_ATTN_MKP.b_actor_and_critic_attn import MODEL_DIR
from _04_COP_POINTING_AND_ATTENTION._01_COMMON.d_mkp_with_google_or_tools import solve

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test(env, actor, num_episodes):
    rl_episode_reward_lst = np.zeros(shape=(num_episodes,), dtype=float)
    rl_duration_lst = []
    or_tool_solution_lst = np.zeros(shape=(num_episodes,), dtype=float)
    or_tool_duration_lst = []

    for i in range(num_episodes):
        observation, info = env.reset()
        episode_reward = 0  # cumulative_reward
        episode_steps = 0
        done = False
        print("[EPISODE: {0}]\nRESET, Info: {1}".format(i, info))

        rl_selected_tasks = []
        rl_start_time = datetime.now()
        while not done:
            episode_steps += 1
            action = actor.get_action(observation, action_mask=info["ACTION_MASK"], exploration=False)

            rl_selected_tasks.append(action.item() if type(action) is np.ndarray else action)
            next_observation, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            observation = next_observation
            done = terminated or truncated

        rl_duration = datetime.now() - rl_start_time
        rl_episode_reward_lst[i] = info["VALUE_ALLOCATED"]
        rl_duration_lst.append(rl_duration)
        print("*** RL RESULT ***")
        print("EPISODE_STEPS: {0:3d}, EPISODE REWARD: {1:5.3f}, SELECTED TASKS: {2}, VALUE_ALLOCATED: {3}, INFO:{4}".format(
            episode_steps, episode_reward, rl_selected_tasks, info['VALUE_ALLOCATED'], info
        ))

        print("*** GOOGLE OR TOOL RESULT ***")
        or_tool_start_time = datetime.now()

        or_tool_solution, or_tool_actions_list = solve(
            n_items=NUM_ITEMS, n_resources=2,
            item_resource_demands=env.item_resource_demand,
            item_values=env.item_values,
            resource_capacities=env.initial_resources_capacity
        )
        or_tool_duration = datetime.now() - or_tool_start_time
        or_tool_solution_lst[i] = or_tool_solution
        or_tool_duration_lst.append(or_tool_duration)
        print("OR TOOL: SELECTED TASKS,", or_tool_actions_list)
        print("*** RL VS. OR_TOOL COMPARISON ***")
        print("RL_SOLUTION | OR_TOOL_SOLUTION ---> {0:>6.3f} | {1:>6.3f}".format(
            info['VALUE_ALLOCATED'], or_tool_solution
        ))

        print("RL_DURATION | OR_TOOL_DURATION ---> {0} | {1}".format(
            rl_duration, or_tool_duration,
        ))

        print()

    return {
        "rl_episode_reward_lst": rl_episode_reward_lst,
        "rl_episode_reward_avg": np.average(rl_episode_reward_lst),
        "rl_duration_avg": sum(rl_duration_lst[1:], timedelta(0)) / (num_episodes -1),
        "or_tool_solution_lst": or_tool_solution_lst,
        "or_tool_solutions_avg": np.average(or_tool_solution_lst),
        "or_tool_duration_avg": sum(or_tool_duration_lst[1:], timedelta(0)) / (num_episodes - 1)
    }

def main(num_episodes, env_name):
    if env_config["use_static_item_resource_demand"]:
        env_config["num_resources"] = STATIC_NUM_RESOURCES

    env = MkpEnv(env_config=env_config)

    print("*" * 100)

    actor = ActorAttn(
        n_features=env_config["num_resources"] + 1
    )

    model_params = torch.load(
        os.path.join(MODEL_DIR, "{0}_{1}_{2}_latest.pth".format("ppo_attn", NUM_ITEMS, env_name))
    )
    actor.load_state_dict(model_params)

    results = test(env, actor, num_episodes=num_episodes)

    print("[PPO_ATTN]   Episode Rewards: {0}, Average: {1:.3f}, Duration: {2}".format(
        results["rl_episode_reward_lst"], results["rl_episode_reward_avg"], results["rl_duration_avg"]
    ))
    print("[ OR TOOL] OR Tool Solutions: {0}, Average: {1:.3f}, Duration: {2}".format(
        results["or_tool_solution_lst"], results["or_tool_solutions_avg"], results["or_tool_duration_avg"]
    ))

    env.close()


if __name__ == "__main__":
    NUM_EPISODES = 10

    main(num_episodes=NUM_EPISODES, env_name=ENV_NAME)
