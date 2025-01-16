import pytest
import torch
from tivra.core import TivraAccelerator
from tivra.device import get_torch_device
from unittest.mock import patch


# Test CPU accelerator case
def test_cpu_device():
    device, data_type = get_torch_device(TivraAccelerator.CPU)
    assert device == torch.device("cpu")
    assert data_type == torch.float64


# Test MPS (Metal Performance Shaders) available case
@patch("torch.backends.mps.is_available", return_value=True)
def test_mps_device(mock_mps_available):
    device, data_type = get_torch_device(TivraAccelerator.MPS)
    assert device == torch.device("mps")
    assert data_type == torch.float32
    mock_mps_available.assert_called_once()


# Test MPS unavailable case
@patch("torch.backends.mps.is_available", return_value=False)
def test_mps_device_unavailable(mock_mps_available):
    with pytest.raises(ValueError, match="MPS \\(Metal\\) backend is not available on this system."):
        get_torch_device(TivraAccelerator.MPS)
    mock_mps_available.assert_called_once()


# Test CUDA available case
@patch("torch.cuda.is_available", return_value=True)
def test_cuda_device(mock_cuda_available):
    device, data_type = get_torch_device(TivraAccelerator.CUDA)
    assert device == torch.device("cuda")
    assert data_type == torch.float32
    mock_cuda_available.assert_called_once()


# Test CUDA unavailable case
@patch("torch.cuda.is_available", return_value=False)
def test_cuda_device_unavailable(mock_cuda_available):
    with pytest.raises(ValueError, match="CUDA is not available on this system."):
        get_torch_device(TivraAccelerator.CUDA)
    mock_cuda_available.assert_called_once()



# Test unknown accelerator
def test_unknown_accelerator():
    with pytest.raises(ValueError, match="Unknown accelerator type: .*"):
        get_torch_device("unknown_accelerator")


# Test invalid accelerator type
def test_invalid_accelerator_type():
    with pytest.raises(ValueError, match="Unknown accelerator type: .*"):
        get_torch_device(None)