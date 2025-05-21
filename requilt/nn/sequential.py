from typing import Sequence

import warp as wp
from warp.types import Array, DType

from ._module import Module


class Sequential(Module):
    modules = tuple[Module]

    def __init__(self, modules: Sequence[Module], device=None, dtype=None):
        device = wp.get_preferred_device() if device is None else device
        dtype = wp.float32 if dtype is None else dtype

        self.modules = tuple(modules)

    def parameters(self):
        params = []
        for module in self.modules:
            params.extend(module.parameters())
        return params

    def __call__(self, x: Array[DType], params: dict[str, Array[DType]] = None, y: Array[DType] = None):
        for module in self.modules:
            # TODO: pass params, y
            x = module(x)
        return x
