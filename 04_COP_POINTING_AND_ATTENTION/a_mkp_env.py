from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

__all__ = ["MultiDimKnapsack"]


class MKPState:
    _demands: np.ndarray  # (n_items, n_resources)
    _values: np.ndarray  # (n_items,)
    _capacities: np.ndarray  # (n_resources,)
    _value_in_knapsack: float
    _available_items: np.ndarray  # (numbor of available items,)
    _total_value: float
    _selected_items: list[int]

    def __init__(self) -> None:
        pass

    @property
    def demands(self) -> np.ndarray:
        return self._demands

    @property
    def values(self) -> np.ndarray:
        return self._values

    @property
    def capacities(self) -> np.ndarray:
        return self._capacities

    @property
    def value_in_knapsack(self) -> float:
        return self._value_in_knapsack

    @property
    def available_items(self) -> np.ndarray:
        return self._available_items

    @property
    def total_value(self) -> float:
        return self._total_value

    @property
    def selected_items(self) -> list[int]:
        return self._selected_items

    def set_demands(self, demands: np.ndarray) -> None:
        assert demands.ndim == 2
        self._demands = demands
        self._selected_items = []

    def set_values(self, values: np.ndarray) -> None:
        assert values.ndim == 1
        self._values = values
        self._total_value = np.sum(values)

    def set_capacities(self, capacities: np.ndarray) -> None:
        assert capacities.ndim == 1
        self._capacities = capacities

    def set_value_in_knapsack(self, value_in_knapsack: float) -> None:
        self._value_in_knapsack = value_in_knapsack

    def set_state(self, state: np.ndarray) -> None:
        demands = state[:-1, :-1]
        values = state[:-1, -1].squeeze()
        capacities = state[-1, :-1].squeeze()

        self.set_demands(demands)
        self.set_values(values)
        self.set_capacities(capacities)
        self.set_value_in_knapsack(0)

    def apply_action(self, action: int) -> None:
        # Apply action
        self._capacities -= self.demands[action]
        self._value_in_knapsack += self.values[action]
        self._selected_items.append(action)

        # Remove item
        self.values[action] = 0
        self.demands[action] = 0

    def get_full_state(self) -> np.ndarray:
        n_items = self.values.shape[0]
        n_demands = self.demands.shape[1]
        obs = np.zeros((n_items + 1, n_demands + 1))
        obs[:n_items, :n_demands] = self.demands
        obs[:n_items, -1] = self.values
        obs[-1, :n_demands] = self.capacities
        obs[-1, -1] = self.value_in_knapsack
        return obs

    def get_observation(self) -> np.ndarray:
        n_items = self.values.shape[0]
        n_demands = self.demands.shape[1]
        obs = np.zeros((n_items, n_demands + 1))
        obs[:, :n_demands] = self.demands / self.capacities
        obs[:, -1] = self.values / self.total_value
        return obs

    def validate_available_items(self) -> None:
        n_items = self.values.shape[0]
        for i in range(n_items):
            # if np.all(self.demands[i] == 0):
            #     continue
            if np.any(self.demands[i] > self.capacities):
                self.values[i] = 0
                self.demands[i] = 0

    def get_available_items(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: An array containing the indices of available items.
        """
        self._available_items = np.where(self._values > 0)[0]
        return self._available_items


class MultiDimKnapsack(gym.Env):
    _inintial_state: MKPState
    _curr_state: MKPState

    def __init__(
        self,
        n_items: int,
        n_resources: int,
        capacity_rate: float = 0.7,
    ):
        super().__init__()

        # Environment parameters
        self._n_items = n_items
        self._n_resources = n_resources
        self._capacity_rate = capacity_rate

        # Action space
        self.action_space = spaces.Discrete(n_items)

        # Observation space
        # low = np.zeros((n_items + 1, n_resources + 1))
        # high = np.ones((n_items + 1, n_resources + 1))
        # high[-1] = n_items * capacity_rate
        # self.observation_space = spaces.Box(
        #     low=low,
        #     high=high,
        #     shape=(n_items + 1, n_resources + 1),
        #     dtype=float,
        # )
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(n_items, n_resources + 1),
            dtype=float,
        )

    @property
    def n_items(self) -> int:
        return self._n_items

    @property
    def n_resources(self) -> int:
        return self._n_resources

    def reset(
        self,
        state: np.ndarray | None = None,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed, options=options)

        reset_complete = False
        while not reset_complete:
            if state is not None:
                # Fix initial state
                assert state.shape == (self._n_items + 1, self._n_resources + 1)
                initial_demands = state[:-1, :-1]
                initial_values = state[:-1, -1].squeeze()
                initial_capacities = state[-1, :-1].squeeze()
            else:
                # Sample initial state
                rng = np.random.default_rng()
                initial_demands = rng.random(size=(self._n_items, self._n_resources))
                initial_values = rng.random(size=(self._n_items,))
                initial_capacities = rng.random(size=(self._n_resources,)) * self._n_items * self._capacity_rate

            # Check for zeros
            initial_demands[initial_demands == 0] = 0.01
            initial_values[initial_values == 0] = 0.01
            initial_capacities[initial_capacities == 0] = 0.01

            # Initialize state
            self._inintial_state = MKPState()
            self._inintial_state.set_demands(initial_demands.copy())
            self._inintial_state.set_values(initial_values.copy())
            self._inintial_state.set_capacities(initial_capacities.copy())
            self._inintial_state.set_value_in_knapsack(0)

            # Current state
            self._curr_state = MKPState()
            self._curr_state.set_demands(initial_demands.copy())
            self._curr_state.set_values(initial_values.copy())
            self._curr_state.set_capacities(initial_capacities.copy())
            self._curr_state.set_value_in_knapsack(0)

            # Validate available items
            self._curr_state.validate_available_items()
            available_actions = self._curr_state.get_available_items()

            reset_complete = len(available_actions) > 0
            if not reset_complete and state is not None:
                raise ValueError(f"No available items in the initial state: {state}")

        return self._curr_state.get_observation(), {"available_actions": available_actions}

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        # Check if action is valid
        if action not in self._curr_state.available_items:
            raise ValueError(f"Action {action} is not available. Available actions: {self._curr_state.available_items}")
        # Apply action
        self._curr_state.apply_action(action)
        self._curr_state.validate_available_items()

        # Get observation
        obs = self._curr_state.get_observation()

        # Get reward
        reward = self._inintial_state.values[action] / self._inintial_state.total_value

        # Available actions
        available_actions = self._curr_state.get_available_items()

        # Done
        terminated = np.all(self._curr_state.values == 0)
        truncated = False

        return obs, reward, terminated, truncated, {"available_actions": available_actions}

    def get_log(self) -> dict[str, Any]:
        return {
            "n_selected_items": len(self._curr_state.selected_items),  # int
            "value_allocated": self._curr_state.value_in_knapsack,  # value in knapsack
            "value_ratio": self._curr_state.value_in_knapsack / self._curr_state.total_value,  # value in knapsack / total value
            "selected_items": self._curr_state.selected_items,  # one-hot vector of selected items
        }

    def get_initial_state(self) -> MKPState:
        return self._inintial_state


# Register environment
if "MultiDimKnapsack-v0" not in gym.envs.registry:
    gym.register(
        id="MultiDimKnapsack-v0",
        entry_point="a_mkp_env:MultiDimKnapsack",
    )
