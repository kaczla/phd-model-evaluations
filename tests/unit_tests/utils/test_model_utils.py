import pytest
import torch

from phd_model_evaluations.utils.model_utils import get_torch_device


def test_get_torch_device_cpu() -> None:
    device = get_torch_device(-1)
    assert device == torch.device("cpu"), "Invalid torch device, expected CPU device"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_get_torch_device_cuda() -> None:
    device = get_torch_device(0)
    assert device == torch.device(0), "Invalid torch device, expected CUDA device"
