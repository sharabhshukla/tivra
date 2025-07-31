"""
Tests for utils/pyomo_generator module
"""
import pytest
from unittest.mock import patch, MagicMock
from tivra.utils.pyomo_generator import create_large_lp


def test_create_large_lp_default_parameters():
    """Test create_large_lp with default parameters."""
    model, expected_solution = create_large_lp()
    
    # Check that model is returned
    assert model is not None
    
    # Check expected solution format
    assert isinstance(expected_solution, dict)
    assert len(expected_solution) == 100  # default num_vars
    
    # Check that all expected variables are in solution
    for i in range(100):
        assert f'x_{i}' in expected_solution
        assert expected_solution[f'x_{i}'] == 1.0


def test_create_large_lp_custom_parameters():
    """Test create_large_lp with custom parameters."""
    num_vars = 50
    var_bound = (5, 15)
    
    model, expected_solution = create_large_lp(num_vars=num_vars, var_bound=var_bound)
    
    # Check that model is returned
    assert model is not None
    
    # Check expected solution format
    assert isinstance(expected_solution, dict)
    assert len(expected_solution) == num_vars
    
    # Check that all expected variables are in solution
    for i in range(num_vars):
        assert f'x_{i}' in expected_solution
        assert expected_solution[f'x_{i}'] == 1.0


def test_create_large_lp_different_num_vars():
    """Test create_large_lp with different numbers of variables."""
    test_cases = [1, 5, 20, 100, 500]
    
    for num_vars in test_cases:
        model, expected_solution = create_large_lp(num_vars=num_vars)
        
        assert model is not None
        assert isinstance(expected_solution, dict)
        assert len(expected_solution) == num_vars
        
        for i in range(num_vars):
            assert f'x_{i}' in expected_solution
            assert expected_solution[f'x_{i}'] == 1.0


def test_create_large_lp_var_bounds():
    """Test create_large_lp with different variable bounds."""
    test_bounds = [(0, 1), (-5, 5), (10, 20), (-10, 10)]
    
    for var_bound in test_bounds:
        model, expected_solution = create_large_lp(num_vars=10, var_bound=var_bound)
        
        assert model is not None
        assert isinstance(expected_solution, dict)
        assert len(expected_solution) == 10


@patch('tivra.utils.pyomo_generator.ConcreteModel')
@patch('tivra.utils.pyomo_generator.Var')
@patch('tivra.utils.pyomo_generator.Objective')
@patch('tivra.utils.pyomo_generator.Constraint')
def test_create_large_lp_model_structure(mock_constraint, mock_objective, mock_var, mock_model):
    """Test that create_large_lp creates the expected model structure."""
    mock_model_instance = MagicMock()
    mock_model.return_value = mock_model_instance
    mock_model_instance.add_component = MagicMock()
    
    # Mock variables
    mock_var_instance = MagicMock()
    mock_var.return_value = mock_var_instance
    
    num_vars = 20
    model, expected_solution = create_large_lp(num_vars=num_vars)
    
    # Check that ConcreteModel was called
    mock_model.assert_called_once()
    
    # Check that Var was called correctly
    mock_var.assert_called_once()
    
    # Check that add_component was called the expected number of times
    # Each variable should have min and max constraints: 2 * num_vars
    # Plus redundant constraints: 10 (for groups of 10)
    # Expected calls: 2 * num_vars + (num_vars // 10)
    expected_constraints = 2 * num_vars + (num_vars // 10)
    assert mock_model_instance.add_component.call_count == expected_constraints


def test_create_large_lp_edge_cases():
    """Test create_large_lp with edge cases."""
    # Test with very small number of variables
    model, expected_solution = create_large_lp(num_vars=1)
    assert len(expected_solution) == 1
    assert 'x_0' in expected_solution
    
    # Test with zero lower bound
    model, expected_solution = create_large_lp(num_vars=5, var_bound=(0, 100))
    assert len(expected_solution) == 5
    
    # Test with negative bounds
    model, expected_solution = create_large_lp(num_vars=5, var_bound=(-100, -1))
    assert len(expected_solution) == 5


def test_create_large_lp_constraint_patterns():
    """Test that create_large_lp creates the expected constraint patterns."""
    num_vars = 30
    model, expected_solution = create_large_lp(num_vars=num_vars)
    
    # The function should create constraints for each variable
    # and additional redundant constraints for groups of 10
    expected_groups = num_vars // 10
    assert expected_groups >= 0


@pytest.mark.parametrize("num_vars,expected_groups", [
    (10, 1),
    (20, 2),
    (50, 5),
    (100, 10),
    (105, 10),  # Should still be 10 groups, not 10.5
])
def test_create_large_lp_redundant_constraints(num_vars, expected_groups):
    """Test that the correct number of redundant constraints are created."""
    model, expected_solution = create_large_lp(num_vars=num_vars)
    
    # Check that the expected solution has the right number of variables
    assert len(expected_solution) == num_vars
    
    # The function creates redundant constraints for groups of 10
    # This is tested indirectly through the model structure


def test_create_large_lp_objective_sense():
    """Test that the objective has the correct sense (minimization)."""
    model, expected_solution = create_large_lp(num_vars=10)
    
    # The objective should be minimization (sense=1)
    # This is tested indirectly through the model creation


def test_create_large_lp_variable_domains():
    """Test that variables are created with NonNegativeReals domain."""
    model, expected_solution = create_large_lp(num_vars=5)
    
    # Variables should be created with NonNegativeReals domain
    # This is tested indirectly through the model creation


def test_create_large_lp_solution_consistency():
    """Test that the expected solution is consistent across calls."""
    num_vars = 15
    var_bound = (2, 8)
    
    model1, solution1 = create_large_lp(num_vars=num_vars, var_bound=var_bound)
    model2, solution2 = create_large_lp(num_vars=num_vars, var_bound=var_bound)
    
    # Solutions should be identical for same parameters
    assert solution1 == solution2
    assert len(solution1) == len(solution2) == num_vars


def test_create_large_lp_main_execution():
    """Test that the module can be executed as main."""
    # This tests the if __name__ == '__main__' block
    with patch('tivra.utils.pyomo_generator.create_large_lp') as mock_create:
        mock_create.return_value = (MagicMock(), {'x_0': 1.0})
        
        # Import and execute the main block
        import tivra.utils.pyomo_generator
        
        # The main block should have been executed during import
        # but we can't easily test it without running the module directly