from datetime import datetime, timedelta
import numpy as np
np.set_printoptions(edgeitems=3, linewidth=100000, formatter=dict(float=lambda x: "%5.3f" % x))

from _04_COP_POINTING_AND_ATTENTION._01_COMMON.a_common import env_config, STATIC_NUM_RESOURCES
from _04_COP_POINTING_AND_ATTENTION._01_COMMON.b_mkp_env import MkpEnv
from _04_COP_POINTING_AND_ATTENTION._01_COMMON.d_mkp_with_google_or_tools import solve


def main(num_episodes):
    if env_config["use_static_item_resource_demand"]:
        env_config["num_resources"] = STATIC_NUM_RESOURCES

    env = MkpEnv(env_config=env_config)

    or_tool_solution_lst = np.zeros(shape=(num_episodes,), dtype=float)
    or_tool_duration_lst = []

    for i in range(num_episodes):
        env.reset()

        print("*** GOOGLE OR TOOL RESULT ***")
        or_tool_start_time = datetime.now()

        or_tool_solution, selected_tasks = solve(
            n_items=env.num_items, n_resources=2,
            item_resource_demands=env.item_resource_demand,
            item_values=env.item_values,
            resource_capacities=env.initial_resources_capacity
        )

        or_tool_duration = datetime.now() - or_tool_start_time
        or_tool_solution_lst[i] = or_tool_solution
        or_tool_duration_lst.append(or_tool_duration)
        print("Selected Tasks:", selected_tasks)
        print()

    results = {
        "or_tool_solution_lst": or_tool_solution_lst,
        "or_tool_solutions_avg": np.average(or_tool_solution_lst),
        "or_tool_duration_avg": sum(or_tool_duration_lst[1:], timedelta(0)) / (num_episodes - 1)
    }

    print("[OR TOOL] OR Tool Solutions: {0}, Average: {1:.3f}, Duration: {2}".format(
        results["or_tool_solution_lst"], results["or_tool_solutions_avg"], results["or_tool_duration_avg"]
    ))

    env.close()


if __name__ == "__main__":
    NUM_EPISODES = 10

    main(num_episodes=NUM_EPISODES)
