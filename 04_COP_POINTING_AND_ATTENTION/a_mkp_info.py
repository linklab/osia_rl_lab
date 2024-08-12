import gymnasium as gym

from a_mkp_env import MultiDimKnapsack

gym.register_envs(MultiDimKnapsack)


def main() -> None:
    env: MultiDimKnapsack = gym.make("MultiDimKnapsack-v0", n_items=5, n_resources=3)
    obs, info = env.reset()

    print("Initial state:")
    print(env.get_wrapper_attr("_inintial_state").get_observation())
    print()
    print("Current obs:")
    print(obs)
    print()

    available_actions = info["available_actions"]
    action = available_actions[0]
    obs, reward, truncated, terminated, info = env.step(action)

    print("After taking action:")
    print(obs)
    print(f"Reward: {reward}")
    print(f"Truncated: {truncated}")
    print(f"Terminated: {terminated}")
    print(f"Info: {info}")


if __name__ == "__main__":
    main()
