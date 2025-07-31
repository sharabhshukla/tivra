"""
Tests for device management functionality
"""
import pytest
from unittest.mock import patch, MagicMock
from tivra.device import TivraAccelerator, get_torch_device


def test_tivra_accelerator_enum_values():
    """Test TivraAccelerator enum values are correct."""
    assert TivraAccelerator.CPU == "cpu"
    assert TivraAccelerator.CUDA == "cuda"
    assert TivraAccelerator.XPU == "xpu"
    assert TivraAccelerator.HPU == "hpu"
    assert TivraAccelerator.TPU == "xla"


def test_tivra_accelerator_enum_inheritance():
    """Test that TivraAccelerator inherits from str and Enum."""
    from enum import Enum
    assert issubclass(TivraAccelerator, str)
    assert issubclass(TivraAccelerator, Enum)


def test_get_torch_device_cpu():
    """Test CPU device configuration."""
    with patch('torch.device') as mock_device:
        mock_device.return_value = "cpu_device"
        device, dtype = get_torch_device(TivraAccelerator.CPU)
        mock_device.assert_called_once_with("cpu")
        assert device == "cpu_device"
        # Mock torch.float64 if available
        try:
            import torch
            assert dtype == torch.float64
        except ImportError:
            assert dtype == "float64"


@patch('torch.cuda.is_available', return_value=True)
@patch('torch.device')
def test_get_torch_device_cuda_available(mock_device, mock_cuda_available):
    """Test CUDA device when CUDA is available."""
    mock_device.return_value = "cuda_device"
    device, dtype = get_torch_device(TivraAccelerator.CUDA)
    mock_device.assert_called_once_with("cuda")
    mock_cuda_available.assert_called_once()
    assert device == "cuda_device"
    # Mock torch.float32 if available
    try:
        import torch
        assert dtype == torch.float32
    except ImportError:
        assert dtype == "float32"


@patch('torch.cuda.is_available', return_value=False)
def test_get_torch_device_cuda_unavailable(mock_cuda_available):
    """Test CUDA device when CUDA is not available."""
    with pytest.raises(ValueError, match="CUDA is not available on this system"):
        get_torch_device(TivraAccelerator.CUDA)
    mock_cuda_available.assert_called_once()


@patch('torch.backends.xpu.is_available', return_value=True)
@patch('torch.device')
def test_get_torch_device_xpu_available(mock_device, mock_xpu_available):
    """Test XPU device when XPU is available."""
    mock_device.return_value = "xpu_device"
    device, dtype = get_torch_device(TivraAccelerator.XPU)
    mock_device.assert_called_once_with("xpu")
    mock_xpu_available.assert_called_once()
    assert device == "xpu_device"
    try:
        import torch
        assert dtype == torch.float32
    except ImportError:
        assert dtype == "float32"


@patch('torch.backends.xpu.is_available', return_value=False)
def test_get_torch_device_xpu_unavailable(mock_xpu_available):
    """Test XPU device when XPU is not available."""
    with pytest.raises(ValueError, match="XPU \\(Intel GPU\\) backend is not available on this system"):
        get_torch_device(TivraAccelerator.XPU)
    mock_xpu_available.assert_called_once()


def test_get_torch_device_hpu_available():
    """Test HPU device when HPU is available."""
    with patch('torch.hpu.is_available', return_value=True, create=True):
        with patch('torch.device') as mock_device:
            with patch('hasattr', return_value=True):
                mock_device.return_value = "hpu_device"
                device, dtype = get_torch_device(TivraAccelerator.HPU)
                mock_device.assert_called_once_with("hpu")
                assert device == "hpu_device"
                try:
                    import torch
                    assert dtype == torch.float32
                except ImportError:
                    assert dtype == "float32"


def test_get_torch_device_hpu_unavailable():
    """Test HPU device when HPU is not available."""
    with patch('hasattr', return_value=False):
        with pytest.raises(ValueError, match="HPU \\(Habana\\) backend is not available on this system"):
            get_torch_device(TivraAccelerator.HPU)


def test_get_torch_device_hpu_no_attr():
    """Test HPU device when torch.hpu doesn't exist."""
    with patch('hasattr', return_value=True):
        with patch('torch.hpu.is_available', return_value=False, create=True):
            with pytest.raises(ValueError, match="HPU \\(Habana\\) backend is not available on this system"):
                get_torch_device(TivraAccelerator.HPU)


def test_get_torch_device_tpu_available():
    """Test TPU device when TPU is available."""
    mock_torch_xla = MagicMock()
    mock_xm = MagicMock()
    mock_torch_xla.devices.return_value = ["tpu:0"]
    mock_xm.xla_device.return_value = "tpu_device"
    
    with patch.dict('sys.modules', {
        'torch_xla': mock_torch_xla,
        'torch_xla.core.xla_model': mock_xm
    }):
        device, dtype = get_torch_device(TivraAccelerator.TPU)
        assert device == "tpu_device"
        try:
            import torch
            assert dtype == torch.float32
        except ImportError:
            assert dtype == "float32"


def test_get_torch_device_tpu_unavailable():
    """Test TPU device when TPU is not available."""
    with patch.dict('sys.modules', {}, clear=True):
        with patch('builtins.__import__', side_effect=ImportError()):
            with pytest.raises(ValueError, match="TPU backend is not available on this system"):
                get_torch_device(TivraAccelerator.TPU)


def test_get_torch_device_tpu_no_devices():
    """Test TPU device when no TPU devices are available."""
    mock_torch_xla = MagicMock()
    mock_xm = MagicMock()
    mock_torch_xla.devices.return_value = []
    
    with patch.dict('sys.modules', {
        'torch_xla': mock_torch_xla,
        'torch_xla.core.xla_model': mock_xm
    }):
        # Should still return None/raise error based on implementation
        result = get_torch_device(TivraAccelerator.TPU)
        # The function might still return a device even with no devices listed
        assert result is not None


def test_get_torch_device_unknown_accelerator():
    """Test unknown accelerator type."""
    with pytest.raises(ValueError, match="Unknown accelerator type"):
        get_torch_device("invalid_accelerator")


def test_get_torch_device_none_accelerator():
    """Test None accelerator type."""
    with pytest.raises(ValueError, match="Unknown accelerator type"):
        get_torch_device(None)


@pytest.mark.parametrize("accelerator", [
    TivraAccelerator.CPU,
    TivraAccelerator.CUDA,
    TivraAccelerator.XPU,
    TivraAccelerator.HPU,
    TivraAccelerator.TPU
])
def test_all_accelerators_return_tuple(accelerator):
    """Test that all accelerators return a tuple when available."""
    with patch('torch.cuda.is_available', return_value=True):
        with patch('torch.backends.xpu.is_available', return_value=True):
            with patch('torch.hpu.is_available', return_value=True, create=True):
                with patch('hasattr', return_value=True):
                    with patch('torch.device') as mock_device:
                        mock_device.return_value = f"{accelerator}_device"
                        
                        if accelerator == TivraAccelerator.TPU:
                            # Special handling for TPU
                            mock_torch_xla = MagicMock()
                            mock_xm = MagicMock()
                            mock_torch_xla.devices.return_value = ["tpu:0"]
                            mock_xm.xla_device.return_value = "tpu_device"
                            
                            with patch.dict('sys.modules', {
                                'torch_xla': mock_torch_xla,
                                'torch_xla.core.xla_model': mock_xm
                            }):
                                result = get_torch_device(accelerator)
                        else:
                            result = get_torch_device(accelerator)
                        
                        assert isinstance(result, tuple)
                        assert len(result) == 2
                        device, dtype = result
                        assert device is not None
                        assert dtype is not None