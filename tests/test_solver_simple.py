"""
Simple working tests for TivraSolver core functionality
"""
import pytest
from unittest.mock import patch, Mock
from tivra.core import TivraSolver, TivraAccelerator


def test_tivra_solver_init_default():
    """Test TivraSolver initialization with default parameters."""
    solver = TivraSolver()
    assert solver.max_iter == 5000
    assert solver.tol == 1e-6
    assert solver.theta == 1.0
    assert solver.verbose is False
    assert solver.logging_interval == 50


def test_tivra_solver_init_custom():
    """Test TivraSolver initialization with custom parameters."""
    solver = TivraSolver(
        max_iter=1000,
        tol=1e-8,
        theta=0.5,
        verbose=True,
        logging_interval=25
    )
    assert solver.max_iter == 1000
    assert solver.tol == 1e-8
    assert solver.theta == 0.5
    assert solver.verbose is True
    assert solver.logging_interval == 25


@patch('tivra.core.get_torch_device')
def test_tivra_solver_device_setup(mock_get_device):
    """Test that device is set up correctly."""
    mock_device = Mock()
    mock_dtype = Mock()
    mock_get_device.return_value = (mock_device, mock_dtype)
    
    solver = TivraSolver(accelerator=TivraAccelerator.CPU)
    
    mock_get_device.assert_called_once()
    assert solver.device == mock_device
    assert solver.data_type == mock_dtype


def test_tivra_solver_different_accelerators():
    """Test TivraSolver initialization with different accelerators."""
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


def test_tivra_solver_theta_boundary_values():
    """Test TivraSolver with boundary theta values."""
    test_thetas = [0.0, 0.5, 1.0, 1.5, 2.0]
    for theta in test_thetas:
        solver = TivraSolver(theta=theta)
        assert solver.theta == theta


def test_tivra_solver_tolerance_values():
    """Test TivraSolver with different tolerance values."""
    test_tolerances = [1e-10, 1e-6, 1e-3, 0.1]
    for tol in test_tolerances:
        solver = TivraSolver(tol=tol)
        assert solver.tol == tol


def test_tivra_solver_max_iter_values():
    """Test TivraSolver with different max_iter values."""
    test_max_iters = [1, 10, 100, 1000, 10000]
    for max_iter in test_max_iters:
        solver = TivraSolver(max_iter=max_iter)
        assert solver.max_iter == max_iter


def test_tivra_solver_logging_interval_values():
    """Test TivraSolver with different logging intervals."""
    test_intervals = [1, 10, 50, 100]
    for interval in test_intervals:
        solver = TivraSolver(logging_interval=interval)
        assert solver.logging_interval == interval


def test_tivra_solver_verbose_flag():
    """Test TivraSolver verbose flag."""
    solver_quiet = TivraSolver(verbose=False)
    solver_verbose = TivraSolver(verbose=True)
    
    assert solver_quiet.verbose is False
    assert solver_verbose.verbose is True