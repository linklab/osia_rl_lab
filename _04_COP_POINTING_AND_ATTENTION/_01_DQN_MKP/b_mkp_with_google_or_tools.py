import numpy as np
from ortools.linear_solver import pywraplp

# Create the solver
solver = pywraplp.Solver('simple_item_allocation', pywraplp.Solver.BOP_INTEGER_PROGRAMMING)


def solve(n_items, n_resources, item_resource_demands, item_values, resource_capacities):
    # Define the variables
    xs = []
    for n in range(n_items):
        xs.append(solver.IntVar(0, 1, 'x_' + str(n)))

    # Define the constraints
    for m in range(n_resources):
        solver.Add(solver.Sum([xs[n] * item_resource_demands[n][m] for n in range(n_items)]) <= resource_capacities[m])

    # Define the objective function
    total_value = [xs[n] * item_values[n] for n in range(n_items)]
    solver.Maximize(solver.Sum(total_value))

    # Solve the problem
    status = solver.Solve()

    # Print the solution
    if status == pywraplp.Solver.OPTIMAL:
        total_value = solver.Objective().Value()
        selected_item_value = np.zeros(shape=(n_items,), dtype=int)
        selected_item_demand = np.zeros(shape=(n_resources,), dtype=int)
        for i in range(n_items):

            if xs[i].solution_value() == 1.0:
                selected_item_value[i] = item_values[i]
                for j in range(n_resources):
                    selected_item_demand[j] += item_resource_demands[i][j]

            print("Task {0} [{1:>3},{2:>3}] : [{3:>3}] is {4:<12} ([{5:>3},{6:>3}] : [{7:>3}])".format(
                i, item_resource_demands[i][0], item_resource_demands[i][1], item_values[i],
                "selected" if xs[i].solution_value() == 1.0 else "not selected",
                selected_item_demand[0], selected_item_demand[1],
                sum(selected_item_value)
            ))
    else:
        total_value = None
        print("Solver status: ", status)

    return total_value


if __name__ == "__main__":
    n_items = 10
    n_resources = 2

    use_random_item_demand = True

    if use_random_item_demand:
        item_resource_demands = np.zeros(shape=(n_items, n_resources), dtype=int)
        item_values = np.zeros(shape=(n_items,), dtype=int)
        for item_idx in range(n_items):
            item_resource_demands[item_idx] = np.random.randint(
                low=[50] * n_resources, high=[100] * n_resources, size=(n_resources,)
            )
            item_values[item_idx] = np.random.randint(
                low=[1] * n_resources, high=[100] * n_resources, size=()
            )
    else:
        item_resource_demands = [
            [54, 53],
            [65, 96],
            [56, 78],
            [65, 92],
            [71, 51],
            [68, 65],
            [52, 86],
            [83, 86],
            [77, 87],
            [58, 98],
        ]
        item_values = [
            12, 22, 39, 98, 55, 19, 23, 76, 44, 81
        ]

    resource_capacities = [300, 300]
    print("resource_capacities: ", resource_capacities)

    total_value = solve(
        n_items=n_items, n_resources=2, item_resource_demands=item_resource_demands,
        item_values=item_values,
        resource_capacities=resource_capacities
    )

    print("Total Value: {0}".format(total_value))
