import enum
import random
from datetime import datetime
from typing import Optional, List, Final, Tuple, Union

from gymnasium import spaces
import gymnasium as gym
import numpy as np
from gymnasium.core import RenderFrame

from _04_COP_POINTING_AND_ATTENTION._01_DQN_MKP.a_common import STATIC_RESOURCE_DEMAND_SAMPLE, STATIC_VALUE_SAMPLE


class MkpEnv(gym.Env):
    def __init__(self, env_config, verbose=False):
        super(MkpEnv, self).__init__()

        # Instance's Unique Values (static and for OR-Tools)
        self._n_items: Final[int] = env_config["num_items"]
        self._n_resources: Final[int] = env_config["num_resources"]
        self._item_values: Optional[np.ndarray] = None  # (n_items,)
        self._item_resource_demand: Optional[np.ndarray] = None  # (n_items, n_resources)
        self._total_resources_capacity: np.ndarray = np.array(
            env_config["initial_resources_capacity"], dtype=np.float64
        )
        self._total_value: Optional[int] = None

        # State
        self._action_mask: Optional[np.ndarray] = None  # (n_items,)
        self._curr_item_values: Optional[np.ndarray] = None  # (n_items,)
        self._curr_item_resource_demand: Optional[np.ndarray] = None  # (n_items, n_resources)
        self._remaining_resources_capacity: np.ndarray = self._total_resources_capacity.copy()  # (n_resources) reset

        # Info for monitoring, validation, etc.
        self._selected_actions = []
        self._value_in_knapsack = 0
        self._resources_in_knapsack = np.zeros(self._n_resources)

        # Distribution to sample from
        self._lowest_item_value: Final[int] = env_config["lowest_item_value"]
        self._highest_item_value: Final[int] = env_config["highest_item_value"]
        self._lowest_item_resource_demand: Final[List[int]] = env_config["lowest_item_resource_demand"]
        self._highest_item_resource_demand: Final[List[int]] = env_config["highest_item_resource_demand"]

        # Flags
        # self._use_static_item_resource_demand: Final[bool] = env_config["use_static_item_resource_demand"]
        # self._use_same_item_resource_demand: Final[bool] = env_config["use_same_item_resource_demand"]
        self._state_normalization: Final[bool] = env_config["state_normalization"]

        # Spaces
        self._action_space = spaces.Discrete(n=self._n_items)
        self._observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self._n_items, (1+self._n_resources))
        )

        if verbose:
            self._print_env_config(env_config)

    @property
    def num_items(self):  # TODO -> n_items
        return self._n_items

    @property
    def n_resources(self):
        return self._n_resources

    @property
    def item_values(self):
        return self._item_values

    @property
    def item_resource_demand(self):
        return self._item_resource_demand

    @property
    def initial_resources_capacity(self):
        return self._total_resources_capacity

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def full_state(self) -> np.ndarray:
        full_state = np.zeros(shape=(self._n_items + 1, 1 + self._n_resources), dtype=np.float32)
        full_state[:-1, 0] = self._curr_item_values
        full_state[:-1, 1:] = self._curr_item_resource_demand
        full_state[-1, 0] = self._value_in_knapsack
        full_state[-1, 1:] = self._remaining_resources_capacity

        return full_state

    def _initialize_state(self) -> Tuple[np.ndarray, np.ndarray]:
        # Sample from distribution for each item
        item_values = np.zeros(shape=(self._n_items,))
        item_resource_demand = np.zeros(shape=(self._n_items, self._n_resources))
        for item_idx in range(self._n_items):
            item_values[item_idx] = np.random.randint(
                low=self._lowest_item_value,  # scalar
                high=self._highest_item_value,  # scalar
                size=(1,)
            )
            item_resource_demand[item_idx] = np.random.randint(
                low=self._lowest_item_resource_demand,  # (n_resources,)
                high=self._highest_item_resource_demand,  # (n_resources,)
                size=(self._n_resources,),
            ).astype(np.float64)

        self._total_value = item_values.sum()

        return item_values, item_resource_demand

    def reset(self, **kwargs) -> Tuple[np.ndarray, dict]:
        # Initialize values and resource demand
        self._item_values, self._item_resource_demand = self._initialize_state()
        self._curr_item_values = self._item_values.copy()
        self._curr_item_resource_demand = self._item_resource_demand.copy()
        self._remaining_resources_capacity = self._total_resources_capacity.copy()

        # Reset etc.
        self._selected_actions = []
        self._value_in_knapsack = 0
        self._resources_in_knapsack = np.zeros(self._n_resources)
        self._action_mask = np.zeros(shape=(self._n_items,), dtype=bool)  # 0: available, 1: unavailable

        # Compute action mask: 0 if available, 1 if unavailable
        unavailable_items_indices = self._get_unavailable_items_indices()
        self._action_mask[unavailable_items_indices] = True

        # Make masked state (n_items, 1+n_resources)
        state = np.hstack((self._curr_item_values.reshape(-1, 1), self._curr_item_resource_demand))
        state[unavailable_items_indices, :] = 0

        # Make obs
        obs = state
        if self._state_normalization:
            obs = self._normalize_internal_state(obs)

        # Make info
        info = {}
        self.fill_info(info)

        # Check if all items are unavailable
        if len(unavailable_items_indices) == self._n_items:
            observation, info = self.reset(**kwargs)

        assert info.get("ACTION_MASK") is not None, "ACTION_MASK not in info"
        return obs, info

    def step(self, action_idx: int):
        # Select an item
        selected_item_value = self._curr_item_values[action_idx]
        selected_item_resources = self._curr_item_resource_demand[action_idx].copy()

        # Validate action
        assert (action_idx not in self._selected_actions), f"The Same Item Selected: {action_idx}"
        assert all([
            self._resources_in_knapsack[i] + selected_item_resources[i] <= self._total_resources_capacity[i]
            for i in range(self._n_resources)
        ]), f"{self._resources_in_knapsack} + {selected_item_resources} <= {self._total_resources_capacity}"

        # Apply action
        self._curr_item_values[action_idx] = 0
        self._curr_item_resource_demand[action_idx] = 0

        self._remaining_resources_capacity -= selected_item_resources
        self._selected_actions.append(action_idx)

        self._value_in_knapsack += selected_item_value
        self._resources_in_knapsack += selected_item_resources

        # Compute action mask: 0 if available, 1 if unavailable
        unavailable_items_indices = self._get_unavailable_items_indices()
        self._action_mask[unavailable_items_indices] = True
        self._curr_item_values[unavailable_items_indices] = 0
        self._curr_item_resource_demand[unavailable_items_indices] = 0

        # Make masked state (n_items, 1+n_resources)
        state = np.hstack((self._curr_item_values.reshape(-1, 1), self._curr_item_resource_demand))
        state[unavailable_items_indices, :] = 0

        # Make next_obs
        next_obs = state
        if self._state_normalization:
            next_obs = self._normalize_internal_state(next_obs)

        # Make reward
        reward = self.compute_reward(selected_item_value)

        # Make terminated
        terminated = len(unavailable_items_indices) == self._n_items

        # Make truncated
        truncated = False

        # Make info
        info = {}
        self.fill_info(info)

        return next_obs, reward, terminated, truncated, info

    def render(self) -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        return None

    def compute_reward(self, value_step: int) -> float:
        reward = value_step / self._total_value
        assert reward < 1.0
        return reward

    def fill_info(self, info: dict):
        info["VALUE_ALLOCATED"] = self._value_in_knapsack  # TODO rename to VALUE_IN_KNAPSACK
        info["ACTION_MASK"] = self._action_mask

    def _get_unavailable_items_indices(self) -> np.ndarray:
        # Already selected
        # selected_items = self._internal_state[:-1, 0] == 1.0
        selected_items_arr = np.zeros(shape=(self._n_items,), dtype=bool)
        selected_items_arr[self._selected_actions] = True

        # Lacking resources
        lacking_resources_items = np.zeros(shape=(self._n_items,), dtype=bool)
        for j in range(self._n_resources):
            resource_demand_arr = self._item_resource_demand[:, j]  # (n_items,)
            available_resources = self._remaining_resources_capacity[j]  # scalar
            lacking_resources_items = lacking_resources_items | (resource_demand_arr > available_resources)

        unavailable_items_indices = np.where(
            selected_items_arr | lacking_resources_items
        )[0]

        return unavailable_items_indices  # (n_items,)

    def _normalize_internal_state(self, state):
        assert state.shape == (self._n_items, 1 + self._n_resources)

        # Value scaled in 0~1 (0~max_value_at_item)
        if self._curr_item_values.max() != 0:
            state[:, 0] = state[:, 0] / self._curr_item_values.max()

        # Resource scaled in 0~1 (0~remaining_resources_capacity)
        for j in range(self._n_resources):
            remaining_capacity = self._remaining_resources_capacity[j]
            if remaining_capacity == 0:
                assert all(state[:, 1+j] == 0)
            else:
                state[:, 1+j] = state[:, 1+j] / remaining_capacity

        return state

    @staticmethod
    def _print_env_config(env_config: dict):
        for k, v in env_config.items():
            print("{0:>50}: {1}".format(k, v))
