"""
Simple working tests for extractor functionality
"""
import pytest
from unittest.mock import patch, Mock
from tivra.extractor import PyomoExtractor, LARGE_PINF, LARGE_NINF


def test_extractor_constants():
    """Test that extractor constants have expected values."""
    assert LARGE_PINF == 1E20
    assert LARGE_NINF == -1E20


def test_extractor_type_checking():
    """Test extractor type checking."""
    with pytest.raises(TypeError, match="Pyomo extractor only works with pyomo concrete models"):
        PyomoExtractor("not_a_model")


def test_extractor_with_none():
    """Test extractor with None input."""
    with pytest.raises(TypeError):
        PyomoExtractor(None)


def test_extractor_inheritance():
    """Test that PyomoExtractor inherits from Extractor."""
    from tivra.base import Extractor
    assert issubclass(PyomoExtractor, Extractor)


@patch('tivra.extractor.ConcreteModel')
def test_extractor_mock_model(mock_concrete_model):
    """Test extractor with properly mocked model."""
    # Create a mock model that passes isinstance check
    mock_model = Mock()
    mock_model.__class__.__name__ = 'ConcreteModel'
    
    # Mock the component_data_objects method
    mock_constraints = [Mock(name="con1")]
    mock_variables = [Mock(name="var1")]
    mock_objective = Mock(name="obj1")
    
    def mock_component_data_objects(ctype=None, active=True):
        if hasattr(ctype, '__name__'):
            if ctype.__name__ == 'Constraint':
                return mock_constraints
            elif ctype.__name__ == 'Var':
                return mock_variables
            elif ctype.__name__ == 'Objective':
                return [mock_objective]
        return []
    
    mock_model.component_data_objects = mock_component_data_objects
    
    # Mock the isinstance check
    with patch('builtins.isinstance', return_value=True):
        with patch('tivra.extractor.Constraint') as mock_constraint_type:
            with patch('tivra.extractor.Var') as mock_var_type:
                with patch('tivra.extractor.Objective') as mock_objective_type:
                    mock_constraint_type.__name__ = 'Constraint'
                    mock_var_type.__name__ = 'Var'
                    mock_objective_type.__name__ = 'Objective'
                    
                    extractor = PyomoExtractor(mock_model)
                    assert extractor.model == mock_model
                    assert len(extractor.constraints) == 1
                    assert len(extractor.variables) == 1
                    assert extractor.objective == mock_objective


def test_extractor_properties():
    """Test extractor properties with mocked model."""
    mock_model = Mock()
    mock_model.__class__.__name__ = 'ConcreteModel'
    
    # Create specific mocks for constraints and variables
    mock_constraints = [Mock() for _ in range(3)]
    mock_variables = [Mock() for _ in range(5)]
    mock_objective = Mock()
    
    def mock_component_data_objects(ctype=None, active=True):
        if hasattr(ctype, '__name__'):
            if ctype.__name__ == 'Constraint':
                return mock_constraints
            elif ctype.__name__ == 'Var':
                return mock_variables
            elif ctype.__name__ == 'Objective':
                return [mock_objective]
        return []
    
    mock_model.component_data_objects = mock_component_data_objects
    
    with patch('builtins.isinstance', return_value=True):
        with patch('tivra.extractor.Constraint') as mock_constraint_type:
            with patch('tivra.extractor.Var') as mock_var_type:
                with patch('tivra.extractor.Objective') as mock_objective_type:
                    mock_constraint_type.__name__ = 'Constraint'
                    mock_var_type.__name__ = 'Var'
                    mock_objective_type.__name__ = 'Objective'
                    
                    extractor = PyomoExtractor(mock_model)
                    
                    assert extractor.no_vars == 5
                    assert extractor.no_constraints == 3


def test_extractor_var_index_map():
    """Test that var_index_map is created correctly."""
    mock_model = Mock()
    mock_model.__class__.__name__ = 'ConcreteModel'
    
    # Create variables with specific names
    mock_variables = []
    for i in range(3):
        var = Mock()
        var.name = f"x[{i}]"
        mock_variables.append(var)
    
    mock_objective = Mock()
    
    def mock_component_data_objects(ctype=None, active=True):
        if hasattr(ctype, '__name__'):
            if ctype.__name__ == 'Constraint':
                return []
            elif ctype.__name__ == 'Var':
                return mock_variables
            elif ctype.__name__ == 'Objective':
                return [mock_objective]
        return []
    
    mock_model.component_data_objects = mock_component_data_objects
    
    with patch('builtins.isinstance', return_value=True):
        with patch('tivra.extractor.Constraint') as mock_constraint_type:
            with patch('tivra.extractor.Var') as mock_var_type:
                with patch('tivra.extractor.Objective') as mock_objective_type:
                    mock_constraint_type.__name__ = 'Constraint'
                    mock_var_type.__name__ = 'Var'
                    mock_objective_type.__name__ = 'Objective'
                    
                    extractor = PyomoExtractor(mock_model)
                    
                    expected_map = {"x[0]": 0, "x[1]": 1, "x[2]": 2}
                    assert extractor.var_index_map == expected_map


@pytest.mark.parametrize("num_vars,num_constraints", [
    (1, 1),
    (5, 3),
    (10, 8),
    (0, 0),
])
def test_extractor_different_sizes(num_vars, num_constraints):
    """Test extractor with different model sizes."""
    mock_model = Mock()
    mock_model.__class__.__name__ = 'ConcreteModel'
    
    mock_constraints = [Mock() for _ in range(num_constraints)]
    mock_variables = [Mock() for _ in range(num_vars)]
    mock_objective = Mock() if num_vars > 0 else None
    
    def mock_component_data_objects(ctype=None, active=True):
        if hasattr(ctype, '__name__'):
            if ctype.__name__ == 'Constraint':
                return mock_constraints
            elif ctype.__name__ == 'Var':
                return mock_variables
            elif ctype.__name__ == 'Objective':
                return [mock_objective] if mock_objective else []
        return []
    
    mock_model.component_data_objects = mock_component_data_objects
    
    with patch('builtins.isinstance', return_value=True):
        with patch('tivra.extractor.Constraint') as mock_constraint_type:
            with patch('tivra.extractor.Var') as mock_var_type:
                with patch('tivra.extractor.Objective') as mock_objective_type:
                    mock_constraint_type.__name__ = 'Constraint'
                    mock_var_type.__name__ = 'Var'
                    mock_objective_type.__name__ = 'Objective'
                    
                    if num_vars == 0 and mock_objective is None:
                        # Empty model case - might raise StopIteration
                        with pytest.raises(StopIteration):
                            PyomoExtractor(mock_model)
                    else:
                        extractor = PyomoExtractor(mock_model)
                        assert extractor.no_vars == num_vars
                        assert extractor.no_constraints == num_constraints


def test_extractor_module_imports():
    """Test that extractor module imports work correctly."""
    from tivra.extractor import PyomoExtractor
    from tivra.base import Extractor
    
    assert PyomoExtractor is not None
    assert Extractor is not None
    assert issubclass(PyomoExtractor, Extractor)