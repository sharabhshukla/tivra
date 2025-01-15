import torch
from tivra.core import TivraAccelerator

def get_torch_device(accelerator: TivraAccelerator) -> torch.device:
    """Return the appropriate PyTorch device based on the accelerator type."""

    if accelerator == TivraAccelerator.CPU:
        return torch.device("cpu")

    elif accelerator == TivraAccelerator.MPS:
        if torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            raise ValueError("MPS (Metal) backend is not available on this system.")

    elif accelerator == TivraAccelerator.CUDA:
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            raise ValueError("CUDA is not available on this system.")

    elif accelerator == TivraAccelerator.XPU:
        if torch.backends.xpu.is_available():
            return torch.device("xpu")
        else:
            raise ValueError("XPU (Intel GPU) backend is not available on this system.")


    elif accelerator == TivraAccelerator.HPU:
        if hasattr(torch, 'hpu') and torch.hpu.is_available():
            return torch.device("hpu")
        else:
            raise ValueError(
                "HPU (Habana) backend is not available on this system. Ensure Habana drivers and PyTorch HPU extensions are installed.")


    else:
        raise ValueError(f"Unknown accelerator type: {accelerator}")

if __name__ == '__main__':
    device = get_torch_device(TivraAccelerator.MPS)
    print(device)
    a = torch.tensor([1, 2, 3], device=device)
    print(a)
