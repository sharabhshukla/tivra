from enum import Enum
from typing import Tuple
import torch


class TivraAccelerator(str, Enum):
    CPU = "cpu"
    CUDA = "cuda"
    XPU = "xpu"
    HPU = "hpu"
    TPU = "xla"

def get_torch_device(accelerator: TivraAccelerator) -> Tuple[torch.device, torch.dtype]:
    """Return the appropriate PyTorch device based on the accelerator type."""

    if accelerator == TivraAccelerator.CPU:
        return torch.device("cpu"), torch.float64


    elif accelerator == TivraAccelerator.CUDA:
        if torch.cuda.is_available():
            return torch.device("cuda"), torch.float32
        else:
            raise ValueError("CUDA is not available on this system.")

    elif accelerator == TivraAccelerator.XPU:
        if torch.backends.xpu.is_available():
            return torch.device("xpu"), torch.float32
        else:
            raise ValueError("XPU (Intel GPU) backend is not available on this system.")


    elif accelerator == TivraAccelerator.HPU:
        if hasattr(torch, 'hpu') and torch.hpu.is_available():
            return torch.device("hpu"), torch.float32
        else:
            raise ValueError(
                "HPU (Habana) backend is not available on this system. Ensure Habana drivers and PyTorch HPU extensions are installed.")

    elif accelerator == TivraAccelerator.TPU:
        try:
            import torch_xla
            import torch_xla.core.xla_model as xm
        except ImportError:
            raise ValueError("TPU backend is not available on this system.")
        if len(torch_xla.devices()) >= 1:
            return xm.xla_device(), torch.float32

    else:
        raise ValueError(f"Unknown accelerator type: {accelerator}")

if __name__ == '__main__':
    device, data_type = get_torch_device(TivraAccelerator.MPS)
    print(device)
    a = torch.tensor([1, 2, 3], device=device, dtype=data_type)
    print(a)
