import numpy as np
from ortools.linear_solver import pywraplp

__all__ = ["solve"]

# Create the solver
solver = pywraplp.Solver("simple_item_allocation", pywraplp.Solver.SAT_INTEGER_PROGRAMMING)


def solve(
    n_items: int,
    n_resources: int,
    demands: np.ndarray,
    values: np.ndarray,
    capacities: np.ndarray,
) -> tuple[int, float, float, np.ndarray]:
    # Define the variables
    xs = []
    for n in range(n_items):
        xs.append(solver.IntVar(0, 1, "x_" + str(n)))

    # Define the constraints
    for m in range(n_resources):
        each_demand = [xs[n] * demands[n][m] for n in range(n_items)]
        resource_constraints = solver.Sum(each_demand) <= capacities[m]
        solver.Add(resource_constraints)

    # Define the objective function
    each_value = [xs[n] * values[n] for n in range(n_items)]
    sum_value = solver.Sum(each_value)
    solver.Maximize(sum_value)

    # Solve the problem
    status = solver.Solve()
    assert status == pywraplp.Solver.OPTIMAL, f"Solver failed to find optimal solution. Solver status: {status}"

    # Print the solution
    num_selected_items = 0
    value_allocated = solver.Objective().Value()

    selected_item_value = np.zeros(shape=(n_items,), dtype=int)
    selected_item_demand = np.zeros(shape=(n_resources,), dtype=int)
    for i in range(n_items):

        if xs[i].solution_value() == 1.0:
            num_selected_items += 1
            selected_item_value[i] = values[i]
            for j in range(n_resources):
                selected_item_demand[j] += demands[i][j]

    selected_items = [i for i in range(n_items) if xs[i].solution_value() == 1]

    return {
        "n_selected_items": num_selected_items,  # int
        "value_allocated": value_allocated,  # value in knapsack
        "value_ratio": value_allocated / sum(values),  # value in knapsack / total value
        "selected_items": selected_items,  # list of selected items
    }
