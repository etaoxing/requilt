import math

import warp as wp
from warp.types import Array, DType


def zeros_(array: Array[DType]):
    array.zero_()


def ones_(array: Array[DType]):
    array.fill_(1.0)


def uniform_(array: Array[DType], a: float = 0.0, b: float = 1.0):
    import torch.nn.init as init

    tensor = wp.to_torch(array)
    init.uniform_(tensor, a, b)


def _calculate_fan_in_and_fan_out(array: Array[DType]):
    dimensions = array.ndim
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for array with fewer than 2 dimensions")

    num_input_fmaps = array.shape[1]
    num_output_fmaps = array.shape[0]
    receptive_field_size = 1
    if array.ndim > 2:
        receptive_field_size = math.prod(array.shape[2:])
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out
