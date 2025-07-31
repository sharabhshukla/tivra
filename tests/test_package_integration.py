"""
Simple comprehensive tests for package integration
"""
import pytest
from unittest.mock import patch, Mock


def test_package_imports():
    """Test that all main components can be imported."""
    from tivra import TivraSolver, TivraAccelerator
    
    assert TivraSolver is not None
    assert TivraAccelerator is not None


def test_package_exports():
    """Test that __all__ exports are correct."""
    import tivra
    
    expected_exports = ["TivraSolver", "TivraAccelerator"]
    
    for export in expected_exports:
        assert hasattr(tivra, export)


def test_module_level_imports():
    """Test that all modules can be imported."""
    from tivra import core, device, extractor, base
    from tivra.utils import pyomo_generator
    
    assert core is not None
    assert device is not None
    assert extractor is not None
    assert base is not None
    assert pyomo_generator is not None


def test_enum_accessibility():
    """Test that TivraAccelerator enum is accessible."""
    from tivra import TivraAccelerator
    
    # Test enum values
    assert TivraAccelerator.CPU == "cpu"
    assert TivraAccelerator.CUDA == "cuda"
    assert TivraAccelerator.XPU == "xpu"
    assert TivraAccelerator.HPU == "hpu"
    assert TivraAccelerator.TPU == "xla"


def test_solver_accessibility():
    """Test that TivraSolver is accessible."""
    from tivra import TivraSolver
    
    # Should be able to create solver instance
    with patch('tivra.core.get_torch_device') as mock_get_device:
        mock_get_device.return_value = (Mock(), Mock())
        solver = TivraSolver()
        assert solver is not None


def test_core_module_components():
    """Test core module components."""
    from tivra.core import TivraSolver, TivraAccelerator, logger
    
    assert TivraSolver is not None
    assert TivraAccelerator is not None
    assert logger is not None


def test_device_module_components():
    """Test device module components."""
    from tivra.device import TivraAccelerator, get_torch_device
    
    assert TivraAccelerator is not None
    assert get_torch_device is not None


def test_extractor_module_components():
    """Test extractor module components."""
    from tivra.extractor import PyomoExtractor, LARGE_PINF, LARGE_NINF
    
    assert PyomoExtractor is not None
    assert LARGE_PINF == 1E20
    assert LARGE_NINF == -1E20


def test_base_module_components():
    """Test base module components."""
    from tivra.base import Extractor
    
    assert Extractor is not None


def test_package_structure():
    """Test overall package structure."""
    import tivra
    import tivra.core
    import tivra.device
    import tivra.extractor
    import tivra.base
    import tivra.utils
    
    # All modules should be importable
    assert tivra is not None
    assert tivra.core is not None
    assert tivra.device is not None
    assert tivra.extractor is not None
    assert tivra.base is not None
    assert tivra.utils is not None


def test_solver_device_integration():
    """Test integration between solver and device management."""
    from tivra import TivraSolver, TivraAccelerator
    
    with patch('tivra.core.get_torch_device') as mock_get_device:
        mock_get_device.return_value = (Mock(), Mock())
        
        # Test with different accelerators
        for accelerator in [TivraAccelerator.CPU, TivraAccelerator.CUDA]:
            solver = TivraSolver(accelerator=accelerator)
            assert solver is not None
            mock_get_device.assert_called()


def test_constants_accessibility():
    """Test that module constants are accessible."""
    from tivra.extractor import LARGE_PINF, LARGE_NINF
    from tivra.device import TivraAccelerator
    
    # Check constants have expected values
    assert LARGE_PINF == 1E20
    assert LARGE_NINF == -1E20
    
    # Check enum values
    assert hasattr(TivraAccelerator, 'CPU')
    assert hasattr(TivraAccelerator, 'CUDA')
    assert hasattr(TivraAccelerator, 'XPU')
    assert hasattr(TivraAccelerator, 'HPU')
    assert hasattr(TivraAccelerator, 'TPU')


def test_class_inheritance():
    """Test class inheritance relationships."""
    from tivra.extractor import PyomoExtractor
    from tivra.base import Extractor
    
    assert issubclass(PyomoExtractor, Extractor)


def test_abstract_base_class():
    """Test abstract base class functionality."""
    from tivra.base import Extractor
    
    # Should not be able to instantiate abstract base class
    with pytest.raises(TypeError):
        Extractor()


def test_enum_inheritance():
    """Test enum inheritance."""
    from tivra.device import TivraAccelerator
    from enum import Enum
    
    assert issubclass(TivraAccelerator, str)
    assert issubclass(TivraAccelerator, Enum)


def test_module_docstrings():
    """Test that modules have docstrings or are importable."""
    import tivra.core
    import tivra.device
    import tivra.extractor
    import tivra.base
    
    # Modules should be importable (docstrings are optional)
    assert tivra.core is not None
    assert tivra.device is not None
    assert tivra.extractor is not None
    assert tivra.base is not None


def test_function_accessibility():
    """Test that key functions are accessible."""
    from tivra.device import get_torch_device
    from tivra.utils.pyomo_generator import create_large_lp
    
    assert callable(get_torch_device)
    assert callable(create_large_lp)


def test_cross_module_functionality():
    """Test functionality that crosses module boundaries."""
    from tivra import TivraSolver, TivraAccelerator
    
    # Test that solver can use different accelerator types
    with patch('tivra.core.get_torch_device') as mock_get_device:
        mock_get_device.return_value = (Mock(), Mock())
        
        for accelerator in TivraAccelerator:
            try:
                solver = TivraSolver(accelerator=accelerator)
                assert solver is not None
            except ValueError:
                # Some accelerators may not be available, which is expected
                pass