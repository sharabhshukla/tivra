import pytest
from tivra.core import TivraAccelerator
import torch

def test_enum_values():
    """
    Test that TivraAccelerator contains the correct values.
    """
    assert TivraAccelerator.CPU.value == "cpu"
    assert TivraAccelerator.MPS.value == "mps"
    assert TivraAccelerator.CUDA.value == "cuda"
    assert TivraAccelerator.XPU.value == "xpu"
    assert TivraAccelerator.HPU.value == "hpu"


def test_enum_membership():
    """
    Test that all expected members are present in TivraAccelerator.
    """
    members = {member.name for member in TivraAccelerator}
    expected_members = {"CPU", "MPS", "CUDA", "XPU", "HPU"}
    assert members == expected_members


def test_enum_iteration():
    """
    Test that we can iterate over all members of the enum.
    """
    expected_values = ["cpu", "mps", "cuda", "xpu", "hpu"]
    actual_values = [member.value for member in TivraAccelerator]
    assert actual_values == expected_values


def test_enum_comparison():
    """
    Test that enum members can be compared with strings.
    """
    assert TivraAccelerator.CPU == "cpu"
    assert TivraAccelerator.MPS == "mps"
    assert TivraAccelerator.CUDA == "cuda"
    assert TivraAccelerator.XPU == "xpu"
    assert TivraAccelerator.HPU == "hpu"


def test_invalid_enum_value():
    """
    Ensure that invalid values are not part of the enum.
    """
    with pytest.raises(ValueError):
        TivraAccelerator("invalid_value")
