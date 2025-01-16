from pyomo.environ import ConcreteModel, Var, Objective, Constraint, NonNegativeReals
import numpy as np

def create_large_lp(num_vars=100, var_bound=(0, 10)):
    """
    Creates a large-scale LP with a known optimal solution.

    Parameters:
    - num_vars (int): Number of variables in the LP.
    - var_bound (tuple): Lower and upper bounds for the variables.

    Returns:
    - model (ConcreteModel): The Pyomo model instance.
    - expected_solution (dict): The known optimal solution.
    """
    model = ConcreteModel()

    # Define variables with bounds [0, 10]
    model.x = Var(range(num_vars), domain=NonNegativeReals, bounds=var_bound)

    # Objective: Minimize sum of all variables
    model.obj = Objective(expr=sum(model.x[i] for i in range(num_vars)), sense=1)  # sense=1 for minimization

    # Constraints:
    # 1. Each variable must be exactly 1: x_i >=1 and x_i <=1
    # 2. Additional redundant constraints to increase complexity
    constraint_list = []
    for i in range(num_vars):
        # Enforce x_i >=1
        constraint_name_min = f"con_min_{i}"
        model.add_component(constraint_name_min, Constraint(expr=model.x[i] >= 10))

        # Enforce x_i <=1
        constraint_name_max = f"con_max_{i}"
        model.add_component(constraint_name_max, Constraint(expr=model.x[i] <= 1))

    # Additional redundant constraints: sum of subsets equals sum of their individual parts
    # For example, for every 10 variables, add a constraint that their sum equals 10
    for j in range(10):
        subset_indices = range(j * 10, (j + 1) * 10)
        constraint_name = f"redundant_con_{j}"
        model.add_component(constraint_name, Constraint(expr=sum(model.x[i] for i in subset_indices) == 10))

    # Expected Solution: All variables equal to 1
    expected_solution = {f'x_{i}': 1.0 for i in range(num_vars)}

    return model, expected_solution

if __name__ == '__main__':
    model, expected_solution = create_large_lp()
    print(expected_solution)