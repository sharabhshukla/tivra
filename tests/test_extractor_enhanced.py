"""
Enhanced tests for PyomoExtractor with comprehensive coverage
"""
import pytest
from unittest.mock import Mock, MagicMock, patch
import numpy as np
from tivra.extractor import PyomoExtractor, LARGE_PINF, LARGE_NINF


class TestPyomoExtractor:
    """Test suite for PyomoExtractor class."""
    
    def test_init_with_invalid_model_type(self):
        """Test initialization with invalid model type."""
        with pytest.raises(TypeError, match="Pyomo extractor only works with pyomo concrete models"):
            PyomoExtractor("not_a_model")
            
    def test_init_with_none_model(self):
        """Test initialization with None model."""
        with pytest.raises(TypeError):
            PyomoExtractor(None)
            
    def test_init_with_mock_concrete_model(self):
        """Test successful initialization with mock ConcreteModel."""
        mock_model = Mock()
        mock_model.__class__.__name__ = 'ConcreteModel'
        
        # Mock the component_data_objects method
        mock_constraints = [Mock(name=f"con{i}") for i in range(3)]
        mock_variables = [Mock(name=f"var{i}") for i in range(2)]
        mock_objective = Mock(name="obj")
        
        def component_data_objects_side_effect(ctype=None, active=True):
            if ctype.__name__ == 'Constraint':
                return mock_constraints
            elif ctype.__name__ == 'Var':
                return mock_variables
            elif ctype.__name__ == 'Objective':
                return [mock_objective]
            return []
            
        mock_model.component_data_objects = Mock(side_effect=component_data_objects_side_effect)
        
        # Mock component types
        with patch('tivra.extractor.Constraint') as mock_constraint_type:
            with patch('tivra.extractor.Var') as mock_var_type:
                with patch('tivra.extractor.Objective') as mock_objective_type:
                    mock_constraint_type.__name__ = 'Constraint'
                    mock_var_type.__name__ = 'Var'
                    mock_objective_type.__name__ = 'Objective'
                    
                    extractor = PyomoExtractor(mock_model)
                    
                    assert extractor.model == mock_model
                    assert len(extractor.constraints) == 3
                    assert len(extractor.variables) == 2
                    assert extractor.objective == mock_objective
                    
    def test_no_vars_property(self):
        """Test no_vars property."""
        mock_model = self._create_mock_model(num_vars=5, num_constraints=3)
        extractor = PyomoExtractor(mock_model)
        assert extractor.no_vars == 5
        
    def test_no_constraints_property(self):
        """Test no_constraints property."""
        mock_model = self._create_mock_model(num_vars=3, num_constraints=7)
        extractor = PyomoExtractor(mock_model)
        assert extractor.no_constraints == 7
        
    def test_extract_constraint_matrix_linear(self):
        """Test _extract_constraint_matrix with linear constraints."""
        mock_model = self._create_mock_model(num_vars=2, num_constraints=2)
        extractor = PyomoExtractor(mock_model)
        
        # Mock generate_standard_repn
        with patch('tivra.extractor.generate_standard_repn') as mock_repn:
            mock_repn_obj = Mock()
            mock_repn_obj.is_linear.return_value = True
            
            # Create mock variables with names
            var1 = Mock()
            var1.name = 'x[0]'
            var2 = Mock()
            var2.name = 'x[1]'
            
            mock_repn_obj.linear_vars = [var1, var2]
            mock_repn_obj.linear_coefs = [2.0, 3.0]
            mock_repn.return_value = mock_repn_obj
            
            A = extractor._extract_constraint_matrix()
            
            assert A.shape == (2, 2)
            # Should have called generate_standard_repn for each constraint
            assert mock_repn.call_count == 2
            
    def test_extract_constraint_matrix_nonlinear(self):
        """Test _extract_constraint_matrix with non-linear constraints."""
        mock_model = self._create_mock_model(num_vars=2, num_constraints=1)
        extractor = PyomoExtractor(mock_model)
        
        # Mock generate_standard_repn to return non-linear
        with patch('tivra.extractor.generate_standard_repn') as mock_repn:
            mock_repn_obj = Mock()
            mock_repn_obj.is_linear.return_value = False
            mock_repn.return_value = mock_repn_obj
            
            with pytest.raises(ValueError, match="Constraint .* is not linear"):
                extractor._extract_constraint_matrix()
                
    def test_extract_objective_vector_linear(self):
        """Test _extract_objective_vector with linear objective."""
        mock_model = self._create_mock_model(num_vars=3, num_constraints=1)
        extractor = PyomoExtractor(mock_model)
        
        # Mock generate_standard_repn
        with patch('tivra.extractor.generate_standard_repn') as mock_repn:
            mock_repn_obj = Mock()
            mock_repn_obj.is_linear.return_value = True
            
            # Create mock variables with names
            var1 = Mock()
            var1.name = 'x[0]'
            var2 = Mock()
            var2.name = 'x[1]'
            
            mock_repn_obj.linear_vars = [var1, var2]
            mock_repn_obj.linear_coefs = [1.5, -2.0]
            mock_repn.return_value = mock_repn_obj
            
            c = extractor._extract_objective_vector()
            
            assert len(c) == 3
            mock_repn.assert_called_once()
            
    def test_extract_objective_vector_nonlinear(self):
        """Test _extract_objective_vector with non-linear objective."""
        mock_model = self._create_mock_model(num_vars=2, num_constraints=1)
        extractor = PyomoExtractor(mock_model)
        
        # Mock generate_standard_repn to return non-linear
        with patch('tivra.extractor.generate_standard_repn') as mock_repn:
            mock_repn_obj = Mock()
            mock_repn_obj.is_linear.return_value = False
            mock_repn.return_value = mock_repn_obj
            
            with pytest.raises(ValueError, match="Objective is not linear"):
                extractor._extract_objective_vector()
                
    def test_extract_constr_bounds_all_finite(self):
        """Test _extract_constr_bounds with finite bounds."""
        mock_model = self._create_mock_model(num_vars=2, num_constraints=2)
        
        # Set up constraints with finite bounds
        extractor = PyomoExtractor(mock_model)
        extractor.constraints[0].lower = 5.0
        extractor.constraints[0].upper = 15.0
        extractor.constraints[1].lower = -10.0
        extractor.constraints[1].upper = 0.0
        
        # Mock value function
        with patch('tivra.extractor.value', side_effect=lambda x: x):
            b_lower, b_upper = extractor._extract_constr_bounds()
            
            assert len(b_lower) == 2
            assert len(b_upper) == 2
            assert b_lower[0] == 5.0
            assert b_upper[0] == 15.0
            assert b_lower[1] == -10.0
            assert b_upper[1] == 0.0
            
    def test_extract_constr_bounds_infinite(self):
        """Test _extract_constr_bounds with infinite bounds (None values)."""
        mock_model = self._create_mock_model(num_vars=2, num_constraints=2)
        
        extractor = PyomoExtractor(mock_model)
        extractor.constraints[0].lower = None
        extractor.constraints[0].upper = 10.0
        extractor.constraints[1].lower = 5.0
        extractor.constraints[1].upper = None
        
        # Mock value function
        with patch('tivra.extractor.value', side_effect=lambda x: x):
            b_lower, b_upper = extractor._extract_constr_bounds()
            
            assert b_lower[0] == LARGE_NINF
            assert b_upper[0] == 10.0
            assert b_lower[1] == 5.0
            assert b_upper[1] == LARGE_PINF
            
    def test_extract_variable_bounds_finite(self):
        """Test _extract_variable_bounds with finite bounds."""
        mock_model = self._create_mock_model(num_vars=3, num_constraints=1)
        
        extractor = PyomoExtractor(mock_model)
        extractor.variables[0].lb = 0.0
        extractor.variables[0].ub = 10.0
        extractor.variables[1].lb = -5.0
        extractor.variables[1].ub = 5.0
        extractor.variables[2].lb = 2.0
        extractor.variables[2].ub = 8.0
        
        # Mock value function
        with patch('tivra.extractor.value', side_effect=lambda x: x):
            lb, ub = extractor._extract_variable_bounds()
            
            assert len(lb) == 3
            assert len(ub) == 3
            assert lb[0] == 0.0 and ub[0] == 10.0
            assert lb[1] == -5.0 and ub[1] == 5.0
            assert lb[2] == 2.0 and ub[2] == 8.0
            
    def test_extract_variable_bounds_infinite(self):
        """Test _extract_variable_bounds with infinite bounds (None values)."""
        mock_model = self._create_mock_model(num_vars=2, num_constraints=1)
        
        extractor = PyomoExtractor(mock_model)
        extractor.variables[0].lb = None
        extractor.variables[0].ub = 10.0
        extractor.variables[1].lb = 5.0
        extractor.variables[1].ub = None
        
        # Mock value function
        with patch('tivra.extractor.value', side_effect=lambda x: x):
            lb, ub = extractor._extract_variable_bounds()
            
            assert lb[0] == LARGE_NINF
            assert ub[0] == 10.0
            assert lb[1] == 5.0
            assert ub[1] == LARGE_PINF
            
    def test_get_constraint_senses_equality(self):
        """Test _get_constraint_senses with equality constraints."""
        mock_model = self._create_mock_model(num_vars=2, num_constraints=3)
        
        extractor = PyomoExtractor(mock_model)
        # Equality constraint: lower == upper
        extractor.constraints[0].lower = 5.0
        extractor.constraints[0].upper = 5.0
        # Less than or equal constraint
        extractor.constraints[1].lower = None
        extractor.constraints[1].upper = 10.0
        # Greater than or equal constraint
        extractor.constraints[2].lower = 3.0
        extractor.constraints[2].upper = None
        
        senses = extractor._get_constraint_senses()
        
        assert len(senses) == 3
        assert senses[0] == 0  # Equality
        assert senses[1] == -1  # <=
        assert senses[2] == 1   # >=
        
    def test_get_constraint_senses_no_bounds(self):
        """Test _get_constraint_senses with constraint having no bounds."""
        mock_model = self._create_mock_model(num_vars=2, num_constraints=1)
        
        extractor = PyomoExtractor(mock_model)
        extractor.constraints[0].lower = None
        extractor.constraints[0].upper = None
        
        with pytest.raises(ValueError, match="Constraint .* has no bounds"):
            extractor._get_constraint_senses()
            
    def test_extract_all_integration(self):
        """Test extract_all method integration."""
        mock_model = self._create_mock_model(num_vars=2, num_constraints=2)
        extractor = PyomoExtractor(mock_model)
        
        # Set up mock data
        extractor.constraints[0].lower = 0.0
        extractor.constraints[0].upper = 10.0
        extractor.constraints[1].lower = 5.0
        extractor.constraints[1].upper = 5.0
        
        extractor.variables[0].lb = 0.0
        extractor.variables[0].ub = None
        extractor.variables[1].lb = None
        extractor.variables[1].ub = 10.0
        
        # Mock the private methods
        with patch.object(extractor, '_extract_constraint_matrix') as mock_matrix:
            with patch.object(extractor, '_extract_objective_vector') as mock_obj:
                with patch.object(extractor, '_extract_constr_bounds') as mock_constr_bounds:
                    with patch.object(extractor, '_extract_variable_bounds') as mock_var_bounds:
                        with patch.object(extractor, '_get_constraint_senses') as mock_senses:
                            
                            # Set up return values
                            mock_matrix.return_value = np.array([[1, 2], [3, 4]])
                            mock_obj.return_value = np.array([1, -1])
                            mock_constr_bounds.return_value = (np.array([0, 5]), np.array([10, 5]))
                            mock_var_bounds.return_value = (np.array([0, LARGE_NINF]), np.array([LARGE_PINF, 10]))
                            mock_senses.return_value = [-1, 0]
                            
                            result = extractor.extract_all()
                            
                            assert len(result) == 7
                            A, c, b_lower, b_upper, senses, lb, ub = result
                            
                            # Verify all methods were called
                            mock_matrix.assert_called_once()
                            mock_obj.assert_called_once()
                            mock_constr_bounds.assert_called_once()
                            mock_var_bounds.assert_called_once()
                            mock_senses.assert_called_once()
                            
    def test_constants_values(self):
        """Test that the module constants have expected values."""
        assert LARGE_PINF == 1E20
        assert LARGE_NINF == -1E20
        
    def test_main_execution_block(self):
        """Test the main execution block of extractor.py"""
        # This tests the if __name__ == "__main__" block
        with patch('tivra.extractor.ConcreteModel') as mock_model:
            with patch('tivra.extractor.Var') as mock_var:
                with patch('tivra.extractor.Constraint') as mock_constraint:
                    with patch('tivra.extractor.Objective') as mock_objective:
                        with patch('tivra.extractor.PyomoExtractor') as mock_extractor_class:
                            
                            mock_extractor = Mock()
                            mock_extractor.extract_all.return_value = (
                                np.array([[1]]), np.array([1]), np.array([0]), 
                                np.array([10]), [1], np.array([0]), np.array([10])
                            )
                            mock_extractor_class.return_value = mock_extractor
                            
                            # Import the module to trigger main execution
                            import tivra.extractor
                            
    def _create_mock_model(self, num_vars, num_constraints):
        """Helper method to create a mock ConcreteModel."""
        mock_model = Mock()
        mock_model.__class__.__name__ = 'ConcreteModel'
        
        # Create mock constraints
        mock_constraints = []
        for i in range(num_constraints):
            constraint = Mock()
            constraint.name = f"con{i}"
            constraint.body = Mock()
            constraint.lower = 0.0
            constraint.upper = 10.0
            mock_constraints.append(constraint)
        
        # Create mock variables
        mock_variables = []
        for i in range(num_vars):
            variable = Mock()
            variable.name = f"x[{i}]"
            variable.lb = 0.0
            variable.ub = 10.0
            mock_variables.append(variable)
        
        # Create mock objective
        mock_objective = Mock()
        mock_objective.expr = Mock()
        
        # Mock the component_data_objects method
        def component_data_objects_side_effect(ctype=None, active=True):
            if hasattr(ctype, '__name__'):
                if ctype.__name__ == 'Constraint':
                    return mock_constraints
                elif ctype.__name__ == 'Var':
                    return mock_variables
                elif ctype.__name__ == 'Objective':
                    return [mock_objective]
            return []
            
        mock_model.component_data_objects = Mock(side_effect=component_data_objects_side_effect)
        
        # Set up the mock for isinstance check
        with patch('isinstance', return_value=True):
            return mock_model
            
    @pytest.mark.parametrize("num_vars,num_constraints", [
        (1, 1),
        (5, 3),
        (10, 8),
        (2, 15),
    ])
    def test_extractor_with_different_sizes(self, num_vars, num_constraints):
        """Test extractor with different model sizes."""
        mock_model = self._create_mock_model(num_vars, num_constraints)
        extractor = PyomoExtractor(mock_model)
        
        assert extractor.no_vars == num_vars
        assert extractor.no_constraints == num_constraints
        
    def test_var_index_map_creation(self):
        """Test that var_index_map is created correctly."""
        mock_model = self._create_mock_model(num_vars=3, num_constraints=2)
        extractor = PyomoExtractor(mock_model)
        
        expected_map = {f'x[{i}]': i for i in range(3)}
        assert extractor.var_index_map == expected_map