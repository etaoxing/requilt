from typing import Any, Callable

import warp as wp
from warp.types import Array, DType

from ..warp_util import kernel as nested_kernel
from ._module import Module

# TODO: support generic dtypes
# TODO: tiled versions


@wp.func
def elu(x: wp.float32, alpha: float = 1.0):
    if x > 0.0:
        return x
    else:
        return alpha * (wp.exp(x) - 1.0)


@wp.func
def leaky_relu(x: wp.float32, alpha: float = 0.01):
    if x >= 0.0:
        return x
    else:
        return alpha * x


@wp.func
def relu(x: wp.float32):
    return wp.max(x, float(0.0))


@wp.func
def sigmoid(x: wp.float32):
    return 1.0 / (1.0 + wp.exp(-x))


@wp.func
def silu(x: wp.float32):
    return x * sigmoid(x)


tanh = wp.tanh


ACTIVATION = {
    "elu": elu,
    "leaky_relu": leaky_relu,
    "relu": relu,
    "sigmoid": sigmoid,
    "silu": silu,  # aka swish
    "tanh": tanh,
}


class Lambda(Module):
    fn: Callable[..., Any]

    def __init__(self, fn, device=None, dtype=None):
        device = wp.get_preferred_device() if device is None else device
        dtype = wp.float32 if dtype is None else dtype

        self.fn = fn

    def __call__(self, x: Array[DType], params: dict[str, Array[DType]] = None, y: Array[DType] = None):
        B = x.shape[0]
        if params is None:
            params = {}
        if y is None:
            y = wp.empty((B, x.shape[1]), dtype=x.dtype, device=x.device, requires_grad=x.requires_grad)
        wp.map(
            self.fn,
            x,
            out=y,
            device=x.device,
            block_dim=wp.num_threads,
        )
        # TODO: replace this with launch_tiled and tile_map? how to handle arbitrary array shapes? wp.array.reshape?
        return y


class Activation(Lambda):
    r"""Element-wise activation."""

    def __init__(self, act: str, device=None, dtype=None):
        act_fn = ACTIVATION[act]
        # act_fn = getattr(wp, act)
        super().__init__(act_fn, device=device, dtype=dtype)


@wp.func
def _exp(x: float):
    return wp.exp(x)


def logsoftmax_kernel(DIM_BATCH, DIM_IN, AXIS, TILES, DTYPE):
    # if TILES is None:
    #     # TODO: default tile sizes heuristic
    #     raise RuntimeError

    TILE_M = wp.constant(1)
    TILE_K = wp.constant(DIM_IN)

    @nested_kernel
    def logsoftmax(x: wp.array2d(dtype=DTYPE), y: wp.array2d(dtype=DTYPE)):
        i = wp.tid()
        a = wp.tile_load(x, shape=(TILE_M, TILE_K), offset=(i * TILE_M, 0))
        shifted = a
        shifted_exp = wp.tile_map(_exp, shifted)  # BUG: Warp NVRTC compilation error, need to wrap wp.exp in wp.func
        shifted_sumexp = wp.tile_sum(shifted_exp)
        shifted_logsumexp = wp.tile_map(wp.log, shifted_sumexp)
        result = wp.tile_map(wp.sub, shifted, wp.tile_broadcast(shifted_logsumexp, shape=(TILE_M, TILE_K)))
        wp.tile_store(y, result, offset=(i * TILE_M, 0))

    kernel_dims = (DIM_BATCH,)
    return logsoftmax, kernel_dims


class LogSoftmax(Module):
    axis: int

    tiles: tuple[int] | None

    def __init__(
        self,
        axis: int = -1,
        tiles: tuple[int] | None = None,
        device=None,
        dtype=None,
    ):
        device = wp.get_preferred_device() if device is None else device
        dtype = wp.float32 if dtype is None else dtype

        self.axis = axis  # TODO

        self.tiles = tiles
        self.reset_kernels()

    def reset_kernels(self):
        self._kernel = None
        self._kernel_dims = None

    def __call__(self, x: Array[DType], params: dict[str, Array[DType]] = None, y: Array[DType] = None):
        B, K = x.shape[0], x.shape[1]
        if params is None:
            params = {}
        if y is None:
            y = wp.empty((B, K), dtype=x.dtype, device=x.device, requires_grad=x.requires_grad)
        if self._kernel is None:
            self._kernel, self._kernel_dims = logsoftmax_kernel(B, K, self.axis, self.tiles, x.dtype)
        inputs, outputs = [x], [y]
        wp.launch_tiled(
            self._kernel,
            dim=self._kernel_dims,
            inputs=inputs,
            outputs=outputs,
            device=x.device,
            block_dim=wp.num_threads,
        )
        return y
