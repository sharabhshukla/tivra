"""
Edge cases and error handling tests for tivra package
"""
import pytest
from unittest.mock import patch, Mock, MagicMock
import numpy as np
from tivra.core import TivraSolver, TivraAccelerator
from tivra.device import get_torch_device
from tivra.extractor import PyomoExtractor


class TestEdgeCases:
    """Test suite for edge cases and error conditions."""
    
    def test_solver_zero_max_iter(self):
        """Test solver with zero max iterations."""
        solver = TivraSolver(max_iter=0)
        
        with patch('tivra.core.PyomoExtractor') as mock_extractor_class:
            mock_extractor = Mock()
            mock_extractor_class.return_value = mock_extractor
            mock_extractor.extract_all.return_value = (
                np.array([[1]]), np.array([1]), np.array([0]), 
                np.array([10]), [1], np.array([0]), np.array([10])
            )
            
            with patch('torch.tensor') as mock_tensor:
                with patch('torch.linalg.norm', return_value=1.0):
                    mock_tensor_obj = Mock()
                    mock_tensor_obj.shape = (1, 1)
                    mock_tensor_obj.device = Mock()
                    mock_tensor_obj.t.return_value = mock_tensor_obj
                    mock_tensor_obj.detach.return_value.cpu.return_value.numpy.return_value = np.array([1])
                    mock_tensor.return_value = mock_tensor_obj
                    
                    with patch('torch.rand', return_value=mock_tensor_obj):
                        with patch('torch.zeros', return_value=mock_tensor_obj):
                            
                            mock_model = Mock()
                            result = solver.solve(mock_model)
                            
                            # Should still return a result (initial guess)
                            assert result is not None
                            
    def test_solver_negative_tolerance(self):
        """Test solver with negative tolerance."""
        solver = TivraSolver(tol=-1e-6)
        assert solver.tol == -1e-6  # Should accept negative tolerance
        
    def test_solver_zero_tolerance(self):
        """Test solver with zero tolerance."""
        solver = TivraSolver(tol=0.0)
        assert solver.tol == 0.0
        
    def test_solver_large_tolerance(self):
        """Test solver with very large tolerance."""
        solver = TivraSolver(tol=1e10)
        assert solver.tol == 1e10
        
    def test_solver_theta_boundary_values(self):
        """Test solver with boundary theta values."""
        # Test theta = 0
        solver1 = TivraSolver(theta=0.0)
        assert solver1.theta == 0.0
        
        # Test theta = 1
        solver2 = TivraSolver(theta=1.0)
        assert solver2.theta == 1.0
        
        # Test theta > 1
        solver3 = TivraSolver(theta=2.0)
        assert solver3.theta == 2.0
        
        # Test negative theta
        solver4 = TivraSolver(theta=-0.5)
        assert solver4.theta == -0.5
        
    def test_solver_extreme_logging_intervals(self):
        """Test solver with extreme logging intervals."""
        # Zero logging interval
        solver1 = TivraSolver(logging_interval=0)
        assert solver1.logging_interval == 0
        
        # Very large logging interval
        solver2 = TivraSolver(logging_interval=1000000)
        assert solver2.logging_interval == 1000000
        
        # Negative logging interval
        solver3 = TivraSolver(logging_interval=-10)
        assert solver3.logging_interval == -10
        
    def test_device_unknown_accelerator_types(self):
        """Test get_torch_device with various invalid accelerator types."""
        invalid_types = [
            "invalid",
            123,
            [],
            {},
            lambda x: x,
            object(),
        ]
        
        for invalid_type in invalid_types:
            with pytest.raises(ValueError, match="Unknown accelerator type"):
                get_torch_device(invalid_type)
                
    def test_device_missing_cuda(self):
        """Test CUDA device when torch.cuda module is missing."""
        with patch('torch.cuda.is_available', side_effect=AttributeError("No cuda module")):
            with pytest.raises(AttributeError):
                get_torch_device(TivraAccelerator.CUDA)
                
    def test_device_missing_xpu(self):
        """Test XPU device when torch.backends.xpu is missing."""
        with patch('torch.backends.xpu.is_available', side_effect=AttributeError("No XPU module")):
            with pytest.raises(AttributeError):
                get_torch_device(TivraAccelerator.XPU)
                
    def test_device_hpu_partial_availability(self):
        """Test HPU device with partial availability."""
        # Test when hasattr returns True but hpu.is_available() raises exception
        with patch('hasattr', return_value=True):
            with patch('torch.hpu.is_available', side_effect=RuntimeError("HPU error"), create=True):
                with pytest.raises(RuntimeError):
                    get_torch_device(TivraAccelerator.HPU)
                    
    def test_device_tpu_import_errors(self):
        """Test TPU device with various import errors."""
        # Test when torch_xla import fails
        with patch('builtins.__import__', side_effect=ImportError("No torch_xla")):
            with pytest.raises(ValueError, match="TPU backend is not available"):
                get_torch_device(TivraAccelerator.TPU)
                
    def test_extractor_empty_model(self):
        """Test extractor with model having no components."""
        mock_model = Mock()
        mock_model.__class__.__name__ = 'ConcreteModel'
        
        # Mock component_data_objects to return empty lists
        def component_data_objects_side_effect(ctype=None, active=True):
            if hasattr(ctype, '__name__'):
                if ctype.__name__ == 'Constraint':
                    return []
                elif ctype.__name__ == 'Var':
                    return []
                elif ctype.__name__ == 'Objective':
                    return []
            return []
            
        mock_model.component_data_objects = Mock(side_effect=component_data_objects_side_effect)
        
        with patch('isinstance', return_value=True):
            with patch('tivra.extractor.Constraint') as mock_constraint_type:
                with patch('tivra.extractor.Var') as mock_var_type:
                    with patch('tivra.extractor.Objective') as mock_objective_type:
                        mock_constraint_type.__name__ = 'Constraint'
                        mock_var_type.__name__ = 'Var'
                        mock_objective_type.__name__ = 'Objective'
                        
                        # Should raise StopIteration when trying to get objective from empty list
                        with pytest.raises(StopIteration):
                            PyomoExtractor(mock_model)
                            
    def test_extractor_constraint_matrix_with_zero_constraints(self):
        """Test constraint matrix extraction with zero constraints."""
        mock_model = self._create_mock_model_with_components(num_vars=2, num_constraints=0)
        extractor = PyomoExtractor(mock_model)
        
        A = extractor._extract_constraint_matrix()
        assert A.shape == (0, 2)
        
    def test_extractor_constraint_matrix_with_zero_variables(self):
        """Test constraint matrix extraction with zero variables."""
        mock_model = self._create_mock_model_with_components(num_vars=0, num_constraints=2)
        extractor = PyomoExtractor(mock_model)
        
        A = extractor._extract_constraint_matrix()
        assert A.shape == (2, 0)
        
    def test_extractor_objective_vector_with_zero_variables(self):
        """Test objective vector extraction with zero variables."""
        mock_model = self._create_mock_model_with_components(num_vars=0, num_constraints=1)
        extractor = PyomoExtractor(mock_model)
        
        with patch('tivra.extractor.generate_standard_repn') as mock_repn:
            mock_repn_obj = Mock()
            mock_repn_obj.is_linear.return_value = True
            mock_repn_obj.linear_vars = []
            mock_repn_obj.linear_coefs = []
            mock_repn.return_value = mock_repn_obj
            
            c = extractor._extract_objective_vector()
            assert len(c) == 0
            
    def test_extractor_bounds_with_nan_values(self):
        """Test bounds extraction with NaN values."""
        mock_model = self._create_mock_model_with_components(num_vars=2, num_constraints=2)
        extractor = PyomoExtractor(mock_model)
        
        # Set constraints with NaN bounds
        extractor.constraints[0].lower = float('nan')
        extractor.constraints[0].upper = float('nan')
        extractor.constraints[1].lower = None
        extractor.constraints[1].upper = None
        
        with patch('tivra.extractor.value', side_effect=lambda x: x if x is not None else None):
            b_lower, b_upper = extractor._extract_constr_bounds()
            
            assert np.isnan(b_lower[0])
            assert np.isnan(b_upper[0])
            assert b_lower[1] == -1E20
            assert b_upper[1] == 1E20
            
    def test_extractor_bounds_with_inf_values(self):
        """Test bounds extraction with infinity values."""
        mock_model = self._create_mock_model_with_components(num_vars=2, num_constraints=2)
        extractor = PyomoExtractor(mock_model)
        
        # Set constraints with infinite bounds
        extractor.constraints[0].lower = float('-inf')
        extractor.constraints[0].upper = float('inf')
        extractor.constraints[1].lower = float('inf')
        extractor.constraints[1].upper = float('-inf')
        
        with patch('tivra.extractor.value', side_effect=lambda x: x):
            b_lower, b_upper = extractor._extract_constr_bounds()
            
            assert b_lower[0] == float('-inf')
            assert b_upper[0] == float('inf')
            assert b_lower[1] == float('inf')
            assert b_upper[1] == float('-inf')
            
    def test_extractor_variable_bounds_edge_cases(self):
        """Test variable bounds extraction with edge cases."""
        mock_model = self._create_mock_model_with_components(num_vars=3, num_constraints=1)
        extractor = PyomoExtractor(mock_model)
        
        # Set variables with edge case bounds
        extractor.variables[0].lb = 0.0
        extractor.variables[0].ub = 0.0  # Same lower and upper bound
        extractor.variables[1].lb = float('inf')
        extractor.variables[1].ub = float('-inf')  # Invalid: lb > ub
        extractor.variables[2].lb = None
        extractor.variables[2].ub = None
        
        with patch('tivra.extractor.value', side_effect=lambda x: x if x is not None else None):
            lb, ub = extractor._extract_variable_bounds()
            
            assert lb[0] == 0.0 and ub[0] == 0.0
            assert lb[1] == float('inf') and ub[1] == float('-inf')
            assert lb[2] == -1E20 and ub[2] == 1E20
            
    def test_solver_with_singular_matrix(self):
        """Test solver behavior with singular constraint matrix."""
        solver = TivraSolver(max_iter=2)
        
        with patch('tivra.core.PyomoExtractor') as mock_extractor_class:
            mock_extractor = Mock()
            mock_extractor_class.return_value = mock_extractor
            
            # Singular matrix (rank deficient)
            A = np.array([[1, 2], [2, 4]])  # Second row is 2x first row
            c = np.array([1, 1])
            mock_extractor.extract_all.return_value = (
                A, c, np.array([0, 0]), np.array([10, 10]), 
                [1, 1], np.array([0, 0]), np.array([10, 10])
            )
            
            with patch('torch.tensor') as mock_tensor:
                with patch('torch.linalg.norm', return_value=0.0):  # Zero norm for singular matrix
                    mock_tensor_obj = Mock()
                    mock_tensor_obj.shape = (2, 2)
                    mock_tensor_obj.device = Mock()
                    mock_tensor_obj.t.return_value = mock_tensor_obj
                    mock_tensor_obj.detach.return_value.cpu.return_value.numpy.return_value = np.array([1, 1])
                    mock_tensor.return_value = mock_tensor_obj
                    
                    with patch('torch.rand', return_value=mock_tensor_obj):
                        with patch('torch.zeros', return_value=mock_tensor_obj):
                            with patch('torch.zeros_like', return_value=mock_tensor_obj):
                                mock_model = Mock()
                                
                                # Should handle division by zero gracefully
                                result = solver.solve(mock_model)
                                assert result is not None
                                
    def test_solver_with_very_large_values(self):
        """Test solver with very large coefficient values."""
        solver = TivraSolver(max_iter=2)
        
        with patch('tivra.core.PyomoExtractor') as mock_extractor_class:
            mock_extractor = Mock()
            mock_extractor_class.return_value = mock_extractor
            
            # Very large values
            large_val = 1e20
            A = np.array([[large_val, 1], [1, large_val]])
            c = np.array([large_val, 1])
            mock_extractor.extract_all.return_value = (
                A, c, np.array([0, 0]), np.array([large_val, large_val]), 
                [1, 1], np.array([0, 0]), np.array([large_val, large_val])
            )
            
            with patch('torch.tensor') as mock_tensor:
                with patch('torch.linalg.norm', return_value=large_val):
                    mock_tensor_obj = Mock()
                    mock_tensor_obj.shape = (2, 2)
                    mock_tensor_obj.device = Mock()
                    mock_tensor_obj.t.return_value = mock_tensor_obj
                    mock_tensor_obj.detach.return_value.cpu.return_value.numpy.return_value = np.array([1, 1])
                    mock_tensor.return_value = mock_tensor_obj
                    
                    with patch('torch.rand', return_value=mock_tensor_obj):
                        with patch('torch.zeros', return_value=mock_tensor_obj):
                            with patch('torch.zeros_like', return_value=mock_tensor_obj):
                                with patch('torch.norm', return_value=1e-10):
                                    mock_model = Mock()
                                    result = solver.solve(mock_model)
                                    assert result is not None
                                    
    def test_solver_with_empty_problem(self):
        """Test solver with empty problem (no variables or constraints)."""
        solver = TivraSolver(max_iter=2)
        
        with patch('tivra.core.PyomoExtractor') as mock_extractor_class:
            mock_extractor = Mock()
            mock_extractor_class.return_value = mock_extractor
            
            # Empty problem
            A = np.array([]).reshape(0, 0)
            c = np.array([])
            mock_extractor.extract_all.return_value = (
                A, c, np.array([]), np.array([]), 
                [], np.array([]), np.array([])
            )
            
            with patch('torch.tensor') as mock_tensor:
                mock_tensor_obj = Mock()
                mock_tensor_obj.shape = (0, 0)
                mock_tensor_obj.device = Mock()
                mock_tensor_obj.detach.return_value.cpu.return_value.numpy.return_value = np.array([])
                mock_tensor.return_value = mock_tensor_obj
                
                with patch('torch.linalg.norm', return_value=0.0):
                    with patch('torch.rand', return_value=mock_tensor_obj):
                        mock_model = Mock()
                        result = solver.solve(mock_model)
                        assert result is not None
                        assert len(result) == 0
                        
    def test_pyomo_generator_edge_cases(self):
        """Test pyomo_generator with edge case parameters."""
        from tivra.utils.pyomo_generator import create_large_lp
        
        # Test with very small number of variables
        model, solution = create_large_lp(num_vars=1, var_bound=(0, 1))
        assert len(solution) == 1
        
        # Test with large variable bounds range
        model, solution = create_large_lp(num_vars=5, var_bound=(-1000, 1000))
        assert len(solution) == 5
        
        # Test with zero-width bounds
        model, solution = create_large_lp(num_vars=3, var_bound=(5, 5))
        assert len(solution) == 3
        
        # Test with inverted bounds (lower > upper)
        model, solution = create_large_lp(num_vars=2, var_bound=(10, 5))
        assert len(solution) == 2
        
    def _create_mock_model_with_components(self, num_vars, num_constraints):
        """Helper to create mock model with specified number of components."""
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
                    return [mock_objective] if num_vars > 0 else []
            return []
            
        mock_model.component_data_objects = Mock(side_effect=component_data_objects_side_effect)
        
        with patch('isinstance', return_value=True):
            return mock_model