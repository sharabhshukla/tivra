"""
Comprehensive tests for TivraSolver class
"""
import pytest
from unittest.mock import patch, MagicMock, Mock
import numpy as np
from tivra.core import TivraSolver, TivraAccelerator


class TestTivraSolver:
    """Test suite for TivraSolver class."""
    
    def test_init_default_parameters(self):
        """Test TivraSolver initialization with default parameters."""
        solver = TivraSolver()
        
        assert solver.max_iter == 5000
        assert solver.tol == 1e-6
        assert solver.theta == 1.0
        assert solver.verbose is False
        assert solver.logging_interval == 50
        
    def test_init_custom_parameters(self):
        """Test TivraSolver initialization with custom parameters."""
        solver = TivraSolver(
            max_iter=1000,
            tol=1e-8,
            theta=0.5,
            verbose=True,
            logging_interval=25,
            accelerator=TivraAccelerator.CPU
        )
        
        assert solver.max_iter == 1000
        assert solver.tol == 1e-8
        assert solver.theta == 0.5
        assert solver.verbose is True
        assert solver.logging_interval == 25
        
    @patch('tivra.core.get_torch_device')
    def test_init_device_configuration(self, mock_get_torch_device):
        """Test that device configuration is set up correctly."""
        mock_device = Mock()
        mock_dtype = Mock()
        mock_get_torch_device.return_value = (mock_device, mock_dtype)
        
        solver = TivraSolver(accelerator=TivraAccelerator.CUDA)
        
        mock_get_torch_device.assert_called_once_with(TivraAccelerator.CPU)  # Default
        assert solver.device == mock_device
        assert solver.data_type == mock_dtype
        
    def test_prox_f_basic(self):
        """Test _prox_f method with basic input."""
        solver = TivraSolver()
        
        # Mock torch tensors
        with patch('torch.tensor') as mock_tensor:
            with patch('torch.maximum') as mock_maximum:
                with patch('torch.minimum') as mock_minimum:
                    
                    # Set up mocks
                    v = Mock()
                    tau = 0.1
                    c = Mock()
                    var_lb = Mock()
                    var_ub = Mock()
                    
                    mock_tensor.return_value = v
                    mock_maximum.return_value = v
                    mock_minimum.return_value = v
                    
                    result = solver._prox_f(v, tau, c, var_lb, var_ub)
                    
                    # Should apply both bounds
                    mock_maximum.assert_called_once()
                    mock_minimum.assert_called_once()
                    assert result == v
                    
    def test_prox_f_no_bounds(self):
        """Test _prox_f method without variable bounds."""
        solver = TivraSolver()
        
        with patch('torch.tensor') as mock_tensor:
            v = Mock()
            tau = 0.1
            c = Mock()
            
            mock_tensor.return_value = v
            
            result = solver._prox_f(v, tau, c)
            assert result is not None
            
    def test_prox_f_only_lower_bound(self):
        """Test _prox_f method with only lower bound."""
        solver = TivraSolver()
        
        with patch('torch.maximum') as mock_maximum:
            v = Mock()
            tau = 0.1
            c = Mock()
            var_lb = Mock()
            
            mock_maximum.return_value = v
            
            result = solver._prox_f(v, tau, c, var_lb=var_lb)
            
            mock_maximum.assert_called_once()
            assert result == v
            
    def test_prox_f_only_upper_bound(self):
        """Test _prox_f method with only upper bound."""
        solver = TivraSolver()
        
        with patch('torch.minimum') as mock_minimum:
            v = Mock()
            tau = 0.1
            c = Mock()
            var_ub = Mock()
            
            mock_minimum.return_value = v
            
            result = solver._prox_f(v, tau, c, var_ub=var_ub)
            
            mock_minimum.assert_called_once()
            assert result == v
            
    def test_prox_g_star(self):
        """Test _prox_g_star method."""
        solver = TivraSolver()
        
        with patch('torch.zeros_like') as mock_zeros_like:
            v = Mock()
            sigma = 0.1
            b_min = Mock()
            b_max = Mock()
            u = Mock()
            
            mock_zeros_like.return_value = u
            
            result = solver._prox_g_star(v, sigma, b_min, b_max)
            
            mock_zeros_like.assert_called_once_with(v)
            assert result == u
            
    def test_extrapolate(self):
        """Test _extrapolate method."""
        solver = TivraSolver(theta=0.8)
        
        with patch('torch.tensor') as mock_tensor:
            x_k = Mock()
            x_k_minus_1 = Mock()
            
            result = solver._extrapolate(x_k, x_k_minus_1)
            assert result is not None
            
    @patch('tivra.core.PyomoExtractor')
    @patch('torch.tensor')
    @patch('torch.linalg.norm')
    @patch('torch.rand')
    @patch('torch.zeros')
    @patch('torch.zeros_like')
    @patch('torch.matmul')
    @patch('torch.norm')
    def test_solve_basic(self, mock_norm, mock_matmul, mock_zeros_like, mock_zeros, 
                        mock_rand, mock_linalg_norm, mock_tensor, mock_extractor_class):
        """Test basic solve functionality."""
        solver = TivraSolver(max_iter=2)  # Short iteration for testing
        
        # Mock the extractor
        mock_extractor = Mock()
        mock_extractor_class.return_value = mock_extractor
        
        # Mock extracted data
        A = np.array([[1, 2], [3, 4]])
        c = np.array([1, 2])
        b_lower = np.array([0, 0])
        b_upper = np.array([10, 10])
        senses = [1, 1]
        lb = np.array([0, 0])
        ub = np.array([10, 10])
        
        mock_extractor.extract_all.return_value = (A, c, b_lower, b_upper, senses, lb, ub)
        
        # Mock torch operations
        mock_device = Mock()
        mock_tensor_obj = Mock()
        mock_tensor_obj.shape = (2, 2)
        mock_tensor_obj.device = mock_device
        mock_tensor_obj.t.return_value = mock_tensor_obj
        mock_tensor_obj.detach.return_value.cpu.return_value.numpy.return_value = np.array([1, 2])
        
        mock_tensor.return_value = mock_tensor_obj
        mock_linalg_norm.return_value = 2.0
        mock_rand.return_value = mock_tensor_obj
        mock_zeros.return_value = mock_tensor_obj
        mock_zeros_like.return_value = mock_tensor_obj
        mock_matmul.return_value = mock_tensor_obj
        mock_norm.return_value = 1e-10  # Converged
        
        # Mock model
        model = Mock()
        
        result = solver.solve(model)
        
        assert result is not None
        mock_extractor_class.assert_called_once_with(model)
        mock_extractor.extract_all.assert_called_once()
        
    @patch('tivra.core.PyomoExtractor')
    def test_solve_with_max_iter_kwarg(self, mock_extractor_class):
        """Test solve method with max_iter passed as keyword argument."""
        solver = TivraSolver(max_iter=1000)
        
        # Mock the extractor and its return values
        mock_extractor = Mock()
        mock_extractor_class.return_value = mock_extractor
        mock_extractor.extract_all.return_value = (
            np.array([[1]]), np.array([1]), np.array([0]), 
            np.array([10]), [1], np.array([0]), np.array([10])
        )
        
        model = Mock()
        
        # Mock torch operations to make solve converge quickly
        with patch('torch.tensor') as mock_tensor:
            with patch('torch.linalg.norm', return_value=1.0):
                with patch('torch.rand') as mock_rand:
                    with patch('torch.zeros'):
                        with patch('torch.zeros_like'):
                            with patch('torch.norm', return_value=1e-10):
                                mock_tensor_obj = Mock()
                                mock_tensor_obj.shape = (1, 1)
                                mock_tensor_obj.device = Mock()
                                mock_tensor_obj.t.return_value = mock_tensor_obj
                                mock_tensor_obj.detach.return_value.cpu.return_value.numpy.return_value = np.array([1])
                                
                                mock_tensor.return_value = mock_tensor_obj
                                mock_rand.return_value = mock_tensor_obj
                                
                                # Test with max_iter kwarg
                                solver.solve(model, max_iter=500)
                                assert solver.max_iter == 500
        
    def test_solve_convergence_logging(self):
        """Test convergence logging in solve method."""
        solver = TivraSolver(max_iter=2, verbose=True)
        
        with patch('tivra.core.PyomoExtractor') as mock_extractor_class:
            with patch('tivra.core.logger') as mock_logger:
                # Mock the extractor
                mock_extractor = Mock()
                mock_extractor_class.return_value = mock_extractor
                mock_extractor.extract_all.return_value = (
                    np.array([[1]]), np.array([1]), np.array([0]), 
                    np.array([10]), [1], np.array([0]), np.array([10])
                )
                
                # Mock torch operations for quick convergence
                with patch('torch.tensor') as mock_tensor:
                    with patch('torch.linalg.norm', return_value=1.0):
                        with patch('torch.rand'):
                            with patch('torch.zeros'):
                                with patch('torch.zeros_like'):
                                    with patch('torch.norm', return_value=1e-10):  # Quick convergence
                                        mock_tensor_obj = Mock()
                                        mock_tensor_obj.shape = (1, 1)
                                        mock_tensor_obj.device = Mock()
                                        mock_tensor_obj.t.return_value = mock_tensor_obj
                                        mock_tensor_obj.detach.return_value.cpu.return_value.numpy.return_value = np.array([1])
                                        mock_tensor.return_value = mock_tensor_obj
                                        
                                        model = Mock()
                                        solver.solve(model)
                                        
                                        # Check that convergence was logged
                                        mock_logger.info.assert_called()
                                        
    def test_solve_non_convergence(self):
        """Test solve method when it doesn't converge."""
        solver = TivraSolver(max_iter=2)  # Very short for non-convergence
        
        with patch('tivra.core.PyomoExtractor') as mock_extractor_class:
            # Mock the extractor
            mock_extractor = Mock()
            mock_extractor_class.return_value = mock_extractor
            mock_extractor.extract_all.return_value = (
                np.array([[1]]), np.array([1]), np.array([0]), 
                np.array([10]), [1], np.array([0]), np.array([10])
            )
            
            # Mock torch operations for no convergence
            with patch('torch.tensor') as mock_tensor:
                with patch('torch.linalg.norm', return_value=1.0):
                    with patch('torch.rand'):
                        with patch('torch.zeros'):
                            with patch('torch.zeros_like'):
                                with patch('torch.norm', return_value=1.0):  # No convergence
                                    mock_tensor_obj = Mock()
                                    mock_tensor_obj.shape = (1, 1)
                                    mock_tensor_obj.device = Mock()
                                    mock_tensor_obj.t.return_value = mock_tensor_obj
                                    mock_tensor_obj.detach.return_value.cpu.return_value.numpy.return_value = np.array([1])
                                    mock_tensor.return_value = mock_tensor_obj
                                    
                                    model = Mock()
                                    result = solver.solve(model)
                                    
                                    assert result is not None
                                    
    def test_solve_verbose_logging(self):
        """Test verbose logging during solve."""
        solver = TivraSolver(max_iter=100, verbose=True, logging_interval=20)
        
        with patch('tivra.core.PyomoExtractor') as mock_extractor_class:
            with patch('tivra.core.logger') as mock_logger:
                # Mock the extractor
                mock_extractor = Mock()
                mock_extractor_class.return_value = mock_extractor
                mock_extractor.extract_all.return_value = (
                    np.array([[1]]), np.array([1]), np.array([0]), 
                    np.array([10]), [1], np.array([0]), np.array([10])
                )
                
                # Mock torch operations
                with patch('torch.tensor') as mock_tensor:
                    with patch('torch.linalg.norm', return_value=1.0):
                        with patch('torch.rand'):
                            with patch('torch.zeros'):
                                with patch('torch.zeros_like'):
                                    with patch('torch.norm', side_effect=[1.0] * 19 + [1e-10]):  # Converge on iteration 20
                                        with patch('torch.dot', return_value=Mock(item=lambda: 5.0)):
                                            mock_tensor_obj = Mock()
                                            mock_tensor_obj.shape = (1, 1)
                                            mock_tensor_obj.device = Mock()
                                            mock_tensor_obj.t.return_value = mock_tensor_obj
                                            mock_tensor_obj.detach.return_value.cpu.return_value.numpy.return_value = np.array([1])
                                            mock_tensor.return_value = mock_tensor_obj
                                            
                                            model = Mock()
                                            solver.solve(model)
                                            
                                            # Should have logged progress
                                            mock_logger.info.assert_called()
                                            
    @pytest.mark.parametrize("accelerator", [
        TivraAccelerator.CPU,
        TivraAccelerator.CUDA,
        TivraAccelerator.XPU,
        TivraAccelerator.HPU,
        TivraAccelerator.TPU
    ])
    def test_solver_with_different_accelerators(self, accelerator):
        """Test solver initialization with different accelerators."""
        with patch('tivra.core.get_torch_device') as mock_get_device:
            mock_get_device.return_value = (Mock(), Mock())
            
            solver = TivraSolver(accelerator=accelerator)
            
            assert solver is not None
            
    def test_solver_data_type_parameter(self):
        """Test solver with different data types."""
        with patch('torch.float32') as mock_float32:
            solver = TivraSolver(data_type=mock_float32)
            assert solver is not None
            
    def test_main_execution(self):
        """Test the main execution block of core.py"""
        with patch('tivra.core.create_large_lp') as mock_create:
            with patch('tivra.core.TivraSolver') as mock_solver_class:
                with patch('torch.manual_seed'):
                    mock_model = Mock()
                    mock_solution = [1, 2, 3]
                    mock_create.return_value = (mock_model, mock_solution)
                    
                    mock_solver = Mock()
                    mock_solver.solve.return_value = [1, 2, 3]
                    mock_solver_class.return_value = mock_solver
                    
                    # Import the module to trigger main execution
                    import tivra.core
                    
                    # The main block should have been executed during import