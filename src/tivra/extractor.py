from typing import Tuple
import numpy as np
from pyomo.environ import ConcreteModel, value, Var, Constraint, Objective
from pyomo.repn import generate_standard_repn
from tivra.base import Extractor

LARGE_PINF = 1E20
LARGE_NINF = -1E20


class PyomoExtractor(Extractor):
    def __init__(self, model: ConcreteModel):
        """ Initialize the Pyomo extractor"""
        if not isinstance(model, ConcreteModel):
            raise TypeError(
                'Pyomo extractor only works with pyomo concrete models!!, model is not a ConcreteModel from '
                'pyomo')
        self.model = model
        self.constraints = list(model.component_data_objects(ctype=Constraint, active=True))
        self.variables = list(model.component_data_objects(ctype=Var, active=True))
        self.objective = next(model.component_data_objects(ctype=Objective, active=True))

        # Create a lookup from var -> index to help in building A
        self.var_index_map = {var.name: i for i, var in enumerate(self.variables)}

    def _extract_constraint_matrix(self):
        """
        Extract the constraint matrix A.

        Returns:
            A (np.ndarray): 2D numpy array representing the constraint coefficients.
        """
        num_constraints = len(self.constraints)
        num_variables = len(self.variables)
        A = np.zeros((num_constraints, num_variables))

        for i, constr in enumerate(self.constraints):
            # Generate the standard representation of the expression
            repn = generate_standard_repn(constr.body)
            # repn.linear_vars is a list of variables with non-zero coefficients
            # repn.linear_coefs is the corresponding list of coefficients
            if repn.is_linear():
                for var, coef in zip(repn.linear_vars, repn.linear_coefs):
                    col_idx = self.var_index_map[var.name]
                    A[i, col_idx] = coef
            else:
                raise ValueError(f"Constraint {constr.name} is not linear.")
        return A

    def _extract_objective_vector(self):
        """
        Extract the objective cost vector c.
        """
        c = np.zeros(len(self.variables))
        # TODO: make a small change to extract the constant term if any present in the objective
        from pyomo.repn import generate_standard_repn
        repn = generate_standard_repn(self.objective.expr)
        if repn.is_linear():
            for var, coef in zip(repn.linear_vars, repn.linear_coefs):
                col_idx = self.var_index_map[var.name]
                c[col_idx] = coef
        else:
            raise ValueError("Objective is not linear.")

        return c

    def _extract_constr_bounds(self):
        """
        Extract lower and upper bounds for each constraint.
        Some constraints might have None (meaning -inf or +inf).
        We'll convert None to float('inf') or -float('inf').
        """
        n = len(self.constraints)
        b_lower = np.empty(n, dtype=float)
        b_upper = np.empty(n, dtype=float)

        for i, constr in enumerate(self.constraints):
            # Convert None to +/- INF
            lower_val = constr.lower if constr.lower is not None else float(LARGE_NINF)
            upper_val = constr.upper if constr.upper is not None else float(LARGE_PINF)

            # Convert Pyomo 'value' into a numeric scalar
            b_lower[i] = value(lower_val)
            b_upper[i] = value(upper_val)

        return b_lower, b_upper

    def _extract_variable_bounds(self):

        """
        Extract lower and upper bounds for all variables.
        Returns two arrays (lb, ub). If a bound is None, treat as ±inf.
        """
        n = len(self.variables)
        lb = np.zeros(n, dtype=float)
        ub = np.zeros(n, dtype=float)

        for i, var in enumerate(self.variables):
            # Convert None to -inf or +inf
            lb[i] = value(var.lb) if var.lb is not None else LARGE_NINF
            ub[i] = value(var.ub) if var.ub is not None else LARGE_PINF

        return lb, ub

    def _get_constraint_senses(self):
        """
        Extract the senses of each constraint as a list.
        Returns:
            List[str]: A list of constraint senses ('<=', '>=', '=').
            = -> 0
            <= -> -1
            >= -> +1
        """
        senses = []
        for constr in self.constraints:
            if constr.lower is not None and constr.upper is not None and constr.lower == constr.upper:
                senses.append(0)
            elif constr.upper is not None:
                senses.append(-1)
            elif constr.lower is not None:
                senses.append(1)
            else:
                raise ValueError(f"Constraint {constr.name} has no bounds and cannot have a sense determined.")
        return senses

    @property
    def no_vars(self):
        return len(self.variables)

    @property
    def no_constraints(self):
        return len(self.constraints)

    def extract_all(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        A = self._extract_constraint_matrix()
        c = self._extract_objective_vector()
        b_lower, b_upper = self._extract_constr_bounds()
        var_lb, var_ub = self._extract_variable_bounds()
        senses = self._get_constraint_senses()
        return A, c, b_lower, b_upper, senses, var_lb, var_ub


if __name__ == "__main__":
    # Example usage
    model = ConcreteModel()
    model.x = Var(range(3), initialize=0)

    # Some linear constraints (<=)
    model.con1 = Constraint(expr=2 * model.x[0] + model.x[1] <= 10)
    model.con2 = Constraint(expr=3 * model.x[1] - model.x[2] <= 5)
    model.con3 = Constraint(expr=model.x[0] + model.x[2] <= 7)
    model.con4 = Constraint(expr=-model.x[0] + 4 * model.x[1] == 3)
    model.con5 = Constraint(expr=5 * model.x[2] >= 15)

    # Objective
    model.obj = Objective(expr=3 * model.x[0] + 2 * model.x[1])

    extractor = PyomoExtractor(model)
    A, c, b_lower, b_upper, senses, lb, ub = extractor.extract_all()
    print("A =\n", A)
    print("c =", c)
    print("b_ =", b_lower)
    print("b_ =", b_upper)
    print("lb =", lb)
    print("ub =", ub)
    print("senses =", senses)
