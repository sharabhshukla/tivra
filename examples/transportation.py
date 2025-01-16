# Import necessary libraries
from pyomo.environ import ConcreteModel, Set, Param, Var, NonNegativeReals, Objective, Constraint, SolverFactory, value
from tivra import TivraSolver

# Create a Concrete Model
model = ConcreteModel()

# Define Sets
model.origins = Set(initialize=['Factory_A', 'Factory_B', 'Factory_C'])
model.destinations = Set(initialize=['Warehouse_X', 'Warehouse_Y', 'Warehouse_Z'])

# Define Parameters

# Supply at each origin
supply_data = {
    'Factory_A': 100,
    'Factory_B': 150,
    'Factory_C': 200
}
model.supply = Param(model.origins, initialize=supply_data)

# Demand at each destination
demand_data = {
    'Warehouse_X': 80,
    'Warehouse_Y': 120,
    'Warehouse_Z': 250
}
model.demand = Param(model.destinations, initialize=demand_data)

# Transportation costs per unit
cost_data = {
    ('Factory_A', 'Warehouse_X'): 2,
    ('Factory_A', 'Warehouse_Y'): 3,
    ('Factory_A', 'Warehouse_Z'): 1,
    ('Factory_B', 'Warehouse_X'): 5,
    ('Factory_B', 'Warehouse_Y'): 4,
    ('Factory_B', 'Warehouse_Z'): 8,
    ('Factory_C', 'Warehouse_X'): 5,
    ('Factory_C', 'Warehouse_Y'): 6,
    ('Factory_C', 'Warehouse_Z'): 8
}
model.cost = Param(model.origins, model.destinations, initialize=cost_data)

# Define Variables
# Amount transported from each origin to each destination
model.transport = Var(model.origins, model.destinations, domain=NonNegativeReals)

# Define Objective
# Minimize total transportation cost
def total_cost_rule(model):
    return sum(model.cost[o, d] * model.transport[o, d] for o in model.origins for d in model.destinations)
model.total_cost = Objective(rule=total_cost_rule, sense=1)  # sense=1 for minimization

# Define Constraints

# Supply constraints: Total shipped from each origin <= supply
def supply_rule(model, o):
    return sum(model.transport[o, d] for d in model.destinations) <= model.supply[o]
model.supply_constraint = Constraint(model.origins, rule=supply_rule)

# Demand constraints: Total received at each destination == demand
def demand_rule(model, d):
    return sum(model.transport[o, d] for o in model.origins) == model.demand[d]
model.demand_constraint = Constraint(model.destinations, rule=demand_rule)

# Solve the model using an appropriate solver
# Ensure that the solver (e.g., GLPK) is installed and accessible
solver = SolverFactory('appsi_highs')  # You can use 'cbc', 'gurobi', etc., if installed
result = solver.solve(model, tee=True)

# Display Results
print("\nOptimal Transportation Plan:")
for o in model.origins:
    for d in model.destinations:
        print(f"Transport {value(model.transport[o, d])} units from {o} to {d}")

print(f"\nTotal Transportation Cost: {value(model.total_cost)}")
print("Now, running PDHG")

tivra_solver = TivraSolver(tol=1e-8, max_iter=8000, theta=0.5, verbose=True)
x = tivra_solver.solve(model)
print(x)