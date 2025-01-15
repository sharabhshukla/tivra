import pytest
import numpy as np
from pyomo.environ import ConcreteModel, Var, Constraint, Objective
from tivra.extractor import PyomoExtractor  # Replace 'your_module' with the actual module name


def test_invalid_model_type():
    """
    Test that initializing PyomoExtractor with an invalid model type raises a TypeError.
    """
    with pytest.raises(TypeError, match="Pyomo extractor only works with pyomo concrete models!!.*"):
        PyomoExtractor("not_a_model")


def test_variable_extraction():
    """
    Test that variables are properly extracted by PyomoExtractor.
    """
    model = ConcreteModel()
    model.x = Var(range(3), initialize=0)
    model.obj = Objective(expr=3 * model.x[0] + 2 * model.x[1])
    extractor = PyomoExtractor(model)

    # Expecting 3 variables
    assert extractor.no_vars == 3
    assert sorted([var.name for var in extractor.variables]) == ["x[0]", "x[1]", "x[2]"]


def test_constraint_extraction():
    """
    Test that constraints are properly extracted by PyomoExtractor.
    """
    model = ConcreteModel()
    model.x = Var(range(2), initialize=0)
    model.con1 = Constraint(expr=model.x[0] + model.x[1] <= 10)
    model.con2 = Constraint(expr=2 * model.x[0] - 3 * model.x[1] == 5)
    model.obj = Objective(expr=3 * model.x[0] + 2 * model.x[1])

    extractor = PyomoExtractor(model)

    # Expecting 2 constraints
    assert extractor.no_constraints == 2
    assert sorted([constr.name for constr in extractor.constraints]) == ["con1", "con2"]


def test_constraint_matrix_extraction():
    """
    Test that the constraint matrix 'A' is correctly extracted.
    """
    model = ConcreteModel()
    model.x = Var(range(2), initialize=0)
    model.con1 = Constraint(expr=model.x[0] + model.x[1] <= 10)
    model.con2 = Constraint(expr=2 * model.x[0] - 3 * model.x[1] == 5)
    model.obj = Objective(expr=3 * model.x[0] + 2 * model.x[1])
    extractor = PyomoExtractor(model)
    A = extractor._extract_constraint_matrix()

    # Constraint coefficients: [[1, 1], [2, -3]]
    expected_A = np.array([[1, 1], [2, -3]])
    np.testing.assert_array_almost_equal(A, expected_A)


def test_objective_vector_extraction():
    """
    Test that the objective vector 'c' is correctly extracted.
    """
    model = ConcreteModel()
    model.x = Var(range(3), initialize=0)
    model.obj = Objective(expr=3 * model.x[0] + 2 * model.x[1] - model.x[2])

    extractor = PyomoExtractor(model)
    c = extractor._extract_objective_vector()

    # Objective coefficients: [3, 2, -1]
    expected_c = np.array([3, 2, -1])
    np.testing.assert_array_almost_equal(c, expected_c)


def test_constraint_bounds_extraction():
    """
    Test that constraint bounds (b_lower, b_upper) are properly extracted.
    """
    model = ConcreteModel()
    model.x = Var(range(2), initialize=0)
    model.con1 = Constraint(expr=model.x[0] + model.x[1] <= 10)
    model.con2 = Constraint(expr=model.x[0] - 2 * model.x[1] >= 5)
    model.con3 = Constraint(expr=2 * model.x[0] + 3 * model.x[1] == 7)
    model.obj = Objective(expr=3 * model.x[0] + 2 * model.x[1])
    extractor = PyomoExtractor(model)
    b_lower, b_upper = extractor._extract_constr_bounds()

    assert b_lower.tolist() == [-1E20, 5, 7]  # -∞, 5, 7
    assert b_upper.tolist() == [10, 1E20, 7]  # 10, ∞, 7


def test_variable_bounds_extraction():
    """
    Test that variable bounds (lb, ub) are properly extracted.
    """
    model = ConcreteModel()
    model.x = Var(range(3))
    model.obj = Objective(expr=0.00)
    model.x[0].setlb(0)
    model.x[1].setub(10)
    model.x[2].setlb(-5)
    model.x[2].setub(5)

    extractor = PyomoExtractor(model)
    lb, ub = extractor._extract_variable_bounds()

    assert lb.tolist() == [0, -1E20, -5]  # 0, -∞, -5
    assert ub.tolist() == [1E20, 10, 5]  # ∞, 10, 5


def test_constraint_senses_extraction():
    """
    Test that constraint senses are properly extracted.
    """
    model = ConcreteModel()
    model.x = Var(range(2), initialize=0)
    model.con1 = Constraint(expr=model.x[0] + model.x[1] <= 10)
    model.con2 = Constraint(expr=model.x[0] >= 5)
    model.con3 = Constraint(expr=model.x[0] + model.x[1] == 5)
    model.obj = Objective(expr=3 * model.x[0] + 2 * model.x[1])
    extractor = PyomoExtractor(model)
    senses = extractor._get_constraint_senses()

    # Expecting senses: [-1 (<=), 1 (>=), 0 (=)]
    assert senses == [-1, 1, 0]


def test_extract_all():
    """
    Test the extract_all method for correctness.
    """
    model = ConcreteModel()
    model.x = Var(range(2), initialize=0)
    model.con1 = Constraint(expr=model.x[0] + model.x[1] <= 10)
    model.con2 = Constraint(expr=2 * model.x[0] - 3 * model.x[1] == 5)
    model.obj = Objective(expr=3 * model.x[0] + 2 * model.x[1])

    extractor = PyomoExtractor(model)
    A, c, b_lower, b_upper, senses, lb, ub = extractor.extract_all()

    # Expected results
    expected_A = np.array([[1, 1], [2, -3]])
    expected_c = np.array([3, 2])
    expected_b_lower = np.array([-1E20, 5])
    expected_b_upper = np.array([10, 5])
    expected_senses = [-1, 0]
    expected_lb = np.array([-1E20, -1E20])
    expected_ub = np.array([1E20, 1E20])

    np.testing.assert_array_almost_equal(A, expected_A)
    np.testing.assert_array_almost_equal(c, expected_c)
    np.testing.assert_array_almost_equal(b_lower, expected_b_lower)
    np.testing.assert_array_almost_equal(b_upper, expected_b_upper)
    assert senses == expected_senses
    np.testing.assert_array_almost_equal(lb, expected_lb)
    np.testing.assert_array_almost_equal(ub, expected_ub)
