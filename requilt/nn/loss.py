import warp as wp
from warp.types import Array, DType

from ..warp_util import kernel as nested_kernel
from ._module import Module


def nll_kernel(DIM_BATCH, DIM_IN, TILES, DTYPE):
    @nested_kernel
    def nll(x: wp.array2d(dtype=DTYPE), target: wp.array1d(dtype=wp.int64), y: wp.array1d(dtype=DTYPE)):
        i = wp.tid()
        l = -x[i, target[i]]
        wp.atomic_add(y, 0, l)

    kernel_dims = (DIM_BATCH,)
    return nll, kernel_dims


class NLLLoss(Module):
    reduction: str | None

    tiles: tuple[int] | None

    def __init__(
        self,
        reduction: str | None = "sum",
        tiles: tuple[int] | None = None,
        device=None,
        dtype=None,
    ):
        device = wp.get_preferred_device() if device is None else device
        dtype = wp.float32 if dtype is None else dtype

        self.reduction = reduction  # TODO

        self.tiles = tiles
        self.reset_kernels()

    def reset_kernels(self):
        self._kernel = None
        self._kernel_dims = None

    def __call__(
        self, x: Array[DType], target: Array[DType], params: dict[str, Array[DType]] = None, y: Array[DType] = None
    ):
        B = x.shape[0]
        K = x.shape[1]
        if params is None:
            params = {}
        if y is None:
            y = wp.zeros((1,), dtype=x.dtype, device=x.device, requires_grad=x.requires_grad)
        if self._kernel is None:
            self._kernel, self._kernel_dims = nll_kernel(B, K, self.tiles, x.dtype)
        inputs = [x, target]
        wp.launch(
            self._kernel,
            dim=self._kernel_dims,
            inputs=inputs,
            outputs=[y],
            device=x.device,
            block_dim=wp.num_threads,
        )
        # TODO: launch_tiled when tile indexed load/store is supported
        return y
