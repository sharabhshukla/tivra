"""
Integration tests for the tivra package
"""
import pytest
from unittest.mock import patch, Mock, MagicMock
import numpy as np
from tivra import TivraSolver, TivraAccelerator


class TestTivraIntegration:
    """Integration tests for the tivra package."""
    
    def test_package_imports(self):
        """Test that all main components can be imported."""
        from tivra import TivraSolver, TivraAccelerator
        from tivra.core import TivraSolver as CoreSolver, logger
        from tivra.device import get_torch_device
        from tivra.extractor import PyomoExtractor
        from tivra.base import Extractor
        from tivra.utils.pyomo_generator import create_large_lp
        
        assert TivraSolver is not None
        assert TivraAccelerator is not None
        assert CoreSolver is not None
        assert get_torch_device is not None
        assert PyomoExtractor is not None
        assert Extractor is not None
        assert create_large_lp is not None
        assert logger is not None
        
    def test_solver_extractor_integration(self):
        """Test integration between TivraSolver and PyomoExtractor."""
        solver = TivraSolver(max_iter=2)
        
        # Mock the entire solve workflow
        with patch('tivra.core.PyomoExtractor') as mock_extractor_class:
            mock_extractor = Mock()
            mock_extractor_class.return_value = mock_extractor
            
            # Mock extracted data
            mock_extractor.extract_all.return_value = (
                np.array([[1, 2], [3, 4]]),  # A
                np.array([1, 2]),            # c
                np.array([0, 0]),            # b_lower
                np.array([10, 10]),          # b_upper
                [1, 1],                      # senses
                np.array([0, 0]),            # lb
                np.array([10, 10])           # ub
            )
            
            # Mock torch operations
            with patch('torch.tensor') as mock_tensor:
                with patch('torch.linalg.norm', return_value=2.0):
                    with patch('torch.rand') as mock_rand:
                        with patch('torch.zeros'):
                            with patch('torch.zeros_like'):
                                with patch('torch.norm', return_value=1e-10):  # Quick convergence
                                    mock_tensor_obj = Mock()
                                    mock_tensor_obj.shape = (2, 2)
                                    mock_tensor_obj.device = Mock()
                                    mock_tensor_obj.t.return_value = mock_tensor_obj
                                    mock_tensor_obj.detach.return_value.cpu.return_value.numpy.return_value = np.array([1, 2])
                                    
                                    mock_tensor.return_value = mock_tensor_obj
                                    mock_rand.return_value = mock_tensor_obj
                                    
                                    # Create a mock model
                                    mock_model = Mock()
                                    
                                    result = solver.solve(mock_model)
                                    
                                    assert result is not None
                                    assert isinstance(result, np.ndarray)
                                    mock_extractor_class.assert_called_once_with(mock_model)
                                    mock_extractor.extract_all.assert_called_once()
                                    
    def test_solver_device_integration(self):
        """Test integration between TivraSolver and device management."""
        with patch('tivra.core.get_torch_device') as mock_get_device:
            mock_device = Mock()
            mock_dtype = Mock()
            mock_get_device.return_value = (mock_device, mock_dtype)
            
            solver = TivraSolver(accelerator=TivraAccelerator.CUDA)
            
            mock_get_device.assert_called_once()
            assert solver.device == mock_device
            assert solver.data_type == mock_dtype
            
    def test_end_to_end_mock_workflow(self):
        """Test complete end-to-end workflow with mocks."""
        # Create solver
        solver = TivraSolver(max_iter=5, tol=1e-8, verbose=True)
        
        # Mock all dependencies
        with patch('tivra.core.PyomoExtractor') as mock_extractor_class:
            with patch('tivra.core.logger') as mock_logger:
                # Set up extractor mock
                mock_extractor = Mock()
                mock_extractor_class.return_value = mock_extractor
                
                # Simple 2x2 problem
                A = np.array([[1, 1], [1, -1]])
                c = np.array([1, 1])
                b_lower = np.array([1, 0])
                b_upper = np.array([1, 0])
                senses = [0, 0]  # Both equality
                lb = np.array([0, 0])
                ub = np.array([10, 10])
                
                mock_extractor.extract_all.return_value = (A, c, b_lower, b_upper, senses, lb, ub)
                
                # Mock torch operations for realistic behavior
                with patch('torch.tensor') as mock_tensor:
                    with patch('torch.linalg.norm', return_value=1.414):  # Norm of 2x2 matrix
                        with patch('torch.rand') as mock_rand:
                            with patch('torch.zeros') as mock_zeros:
                                with patch('torch.zeros_like') as mock_zeros_like:
                                    with patch('torch.matmul') as mock_matmul:
                                        with patch('torch.norm', side_effect=[1.0, 0.1, 1e-9]):  # Converge after 3 iterations
                                            
                                            # Set up tensor mocks
                                            mock_tensor_obj = Mock()
                                            mock_tensor_obj.shape = (2, 2)
                                            mock_tensor_obj.device = Mock()
                                            mock_tensor_obj.t.return_value = mock_tensor_obj
                                            mock_tensor_obj.copy_ = Mock()
                                            mock_tensor_obj.detach.return_value.cpu.return_value.numpy.return_value = np.array([0.5, 0.5])
                                            
                                            mock_tensor.return_value = mock_tensor_obj
                                            mock_rand.return_value = mock_tensor_obj
                                            mock_zeros.return_value = mock_tensor_obj
                                            mock_zeros_like.return_value = mock_tensor_obj
                                            mock_matmul.return_value = mock_tensor_obj
                                            
                                            # Create mock model
                                            mock_model = Mock()
                                            
                                            # Solve the problem
                                            result = solver.solve(mock_model)
                                            
                                            # Verify result
                                            assert result is not None
                                            assert isinstance(result, np.ndarray)
                                            assert len(result) == 2
                                            
                                            # Verify workflow
                                            mock_extractor_class.assert_called_once_with(mock_model)
                                            mock_extractor.extract_all.assert_called_once()
                                            
                                            # Check that convergence was logged
                                            mock_logger.info.assert_called()
                                            
    def test_different_accelerator_configurations(self):
        """Test solver with different accelerator configurations."""
        accelerators = [
            TivraAccelerator.CPU,
            TivraAccelerator.CUDA,
            TivraAccelerator.XPU,
            TivraAccelerator.HPU,
            TivraAccelerator.TPU
        ]
        
        for accelerator in accelerators:
            with patch('tivra.core.get_torch_device') as mock_get_device:
                mock_get_device.return_value = (Mock(), Mock())
                
                solver = TivraSolver(accelerator=accelerator)
                assert solver is not None
                mock_get_device.assert_called_once()
                
    def test_solver_parameter_variations(self):
        """Test solver with various parameter combinations."""
        parameter_sets = [
            {'max_iter': 100, 'tol': 1e-4, 'theta': 0.5, 'verbose': False},
            {'max_iter': 5000, 'tol': 1e-8, 'theta': 1.0, 'verbose': True},
            {'max_iter': 1000, 'tol': 1e-6, 'theta': 0.8, 'logging_interval': 25},
            {'accelerator': TivraAccelerator.CPU, 'verbose': True}
        ]
        
        for params in parameter_sets:
            with patch('tivra.core.get_torch_device') as mock_get_device:
                mock_get_device.return_value = (Mock(), Mock())
                solver = TivraSolver(**params)
                
                for key, value in params.items():
                    if key != 'accelerator':  # accelerator is processed differently
                        assert getattr(solver, key) == value
                        
    def test_error_propagation(self):
        """Test that errors propagate correctly through the integration."""
        solver = TivraSolver()
        
        # Test with invalid model type
        with patch('tivra.core.PyomoExtractor') as mock_extractor_class:
            mock_extractor_class.side_effect = TypeError("Invalid model type")
            
            mock_model = Mock()
            
            with pytest.raises(TypeError):
                solver.solve(mock_model)
                
    def test_large_problem_simulation(self):
        """Test with simulated large problem."""
        solver = TivraSolver(max_iter=10)
        
        with patch('tivra.core.PyomoExtractor') as mock_extractor_class:
            mock_extractor = Mock()
            mock_extractor_class.return_value = mock_extractor
            
            # Simulate large problem (100 variables, 50 constraints)
            n_vars = 100
            n_constraints = 50
            
            A = np.random.rand(n_constraints, n_vars)
            c = np.random.rand(n_vars)
            b_lower = np.zeros(n_constraints)
            b_upper = np.ones(n_constraints) * 10
            senses = [1] * n_constraints  # All >=
            lb = np.zeros(n_vars)
            ub = np.ones(n_vars) * 10
            
            mock_extractor.extract_all.return_value = (A, c, b_lower, b_upper, senses, lb, ub)
            
            # Mock torch operations
            with patch('torch.tensor') as mock_tensor:
                with patch('torch.linalg.norm', return_value=10.0):
                    with patch('torch.rand') as mock_rand:
                        with patch('torch.zeros'):
                            with patch('torch.zeros_like'):
                                with patch('torch.norm', return_value=1e-10):  # Quick convergence
                                    mock_tensor_obj = Mock()
                                    mock_tensor_obj.shape = (n_constraints, n_vars)
                                    mock_tensor_obj.device = Mock()
                                    mock_tensor_obj.t.return_value = mock_tensor_obj
                                    mock_tensor_obj.detach.return_value.cpu.return_value.numpy.return_value = np.random.rand(n_vars)
                                    
                                    mock_tensor.return_value = mock_tensor_obj
                                    mock_rand.return_value = mock_tensor_obj
                                    
                                    mock_model = Mock()
                                    result = solver.solve(mock_model)
                                    
                                    assert result is not None
                                    assert len(result) == n_vars
                                    
    def test_pyomo_generator_integration(self):
        """Test integration with pyomo_generator utility."""
        with patch('tivra.utils.pyomo_generator.ConcreteModel') as mock_model:
            with patch('tivra.utils.pyomo_generator.Var') as mock_var:
                with patch('tivra.utils.pyomo_generator.Objective') as mock_obj:
                    with patch('tivra.utils.pyomo_generator.Constraint') as mock_con:
                        
                        mock_model_instance = Mock()
                        mock_model.return_value = mock_model_instance
                        mock_model_instance.add_component = Mock()
                        
                        from tivra.utils.pyomo_generator import create_large_lp
                        
                        model, solution = create_large_lp(num_vars=10)
                        
                        assert model is not None
                        assert isinstance(solution, dict)
                        assert len(solution) == 10
                        
    def test_module_level_constants(self):
        """Test that module-level constants are accessible."""
        from tivra.extractor import LARGE_PINF, LARGE_NINF
        from tivra.device import TivraAccelerator
        
        assert LARGE_PINF == 1E20
        assert LARGE_NINF == -1E20
        assert TivraAccelerator.CPU == "cpu"
        
    def test_logging_integration(self):
        """Test that logging works correctly across modules."""
        solver = TivraSolver(verbose=True, max_iter=2)
        
        with patch('tivra.core.logger') as mock_logger:
            with patch('tivra.core.PyomoExtractor') as mock_extractor_class:
                mock_extractor = Mock()
                mock_extractor_class.return_value = mock_extractor
                mock_extractor.extract_all.return_value = (
                    np.array([[1]]), np.array([1]), np.array([0]), 
                    np.array([10]), [1], np.array([0]), np.array([10])
                )
                
                # Mock torch for quick convergence
                with patch('torch.tensor') as mock_tensor:
                    with patch('torch.linalg.norm', return_value=1.0):
                        with patch('torch.rand'):
                            with patch('torch.zeros'):
                                with patch('torch.zeros_like'):
                                    with patch('torch.norm', return_value=1e-10):
                                        mock_tensor_obj = Mock()
                                        mock_tensor_obj.shape = (1, 1)
                                        mock_tensor_obj.device = Mock()
                                        mock_tensor_obj.t.return_value = mock_tensor_obj
                                        mock_tensor_obj.detach.return_value.cpu.return_value.numpy.return_value = np.array([1])
                                        mock_tensor.return_value = mock_tensor_obj
                                        
                                        mock_model = Mock()
                                        solver.solve(mock_model)
                                        
                                        # Verify logging was called
                                        mock_logger.info.assert_called()
                                        
    def test_package_metadata(self):
        """Test package metadata and structure."""
        import tivra
        
        # Test that __all__ exports are available
        expected_exports = ["TivraSolver", "TivraAccelerator"]
        
        for export in expected_exports:
            assert hasattr(tivra, export)
            assert getattr(tivra, export) is not None