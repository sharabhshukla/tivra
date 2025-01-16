from pyomo.environ import *
from tivra import TivraSolver, TivraAccelerator

# Define simple Pyomo model
model = ConcreteModel()
model.x = Var(domain=NonNegativeReals)
model.y = Var(domain=NonNegativeReals)
model.obj = Objective(expr=10 * model.x + model.y, sense=minimize)
model.constraint = Constraint(expr=model.x + 2 * model.y >= 1)

# Use Tivra to solve the model
solver = TivraSolver(verbose=True, tol=1e-10)
results = solver.solve(model)

print(results)

solver = SolverFactory('appsi_highs')
solver.solve(model, tee=True)