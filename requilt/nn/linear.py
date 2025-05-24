import math

import warp as wp
from warp.types import Array, DType

from ..warp_util import kernel as nested_kernel
from ._module import Module
from .utils import init


def linear_kernel(DIM_BATCH, DIM_IN, DIM_OUT, USE_BIAS, TILES, DTYPE):
    if TILES is None:
        # TODO: default tile sizes heuristic
        raise RuntimeError

    TILE_M = wp.constant(TILES[0])
    TILE_K = wp.constant(TILES[1])
    TILE_N = wp.constant(TILES[2])

    assert DIM_BATCH % TILE_M == 0
    assert DIM_IN % TILE_K == 0
    assert DIM_OUT % TILE_N == 0

    @nested_kernel
    def linear_gemm(
        x: wp.array2d(dtype=DTYPE),
        weight: wp.array2d(dtype=DTYPE),
        bias: wp.array2d(dtype=DTYPE),
        y: wp.array2d(dtype=DTYPE),
    ):
        # output tile index
        i, j = wp.tid()

        # M = x.shape[0]
        K = x.shape[1]
        # N = weight.shape[1]

        c = wp.tile_zeros(shape=(TILE_M, TILE_N), dtype=DTYPE)
        for k in range(0, int(K / TILE_K)):
            a = wp.tile_load(x, shape=(TILE_M, TILE_K), offset=(i * TILE_M, k * TILE_K))
            b = wp.tile_load(weight, shape=(TILE_K, TILE_N), offset=(k * TILE_K, j * TILE_N))

            # sum += a*b
            wp.tile_matmul(a, b, c)

        if wp.static(USE_BIAS):
            b = wp.tile_load(bias, shape=(1, TILE_N), offset=(0, j * TILE_N))
            c += wp.tile_broadcast(b, shape=(TILE_M, TILE_N))

        wp.tile_store(y, c, offset=(i * TILE_M, j * TILE_N))

    kernel_dims = (int(DIM_BATCH / TILE_M), int(DIM_OUT / TILE_N))
    return linear_gemm, kernel_dims


class Linear(Module):
    r"""Linear layer.

    y = x * weight + bias
    """

    in_features: int
    out_features: int
    use_bias: bool

    tiles: tuple[int, int, int] | None
    weight: Array[DType]
    bias: Array[DType] | None

    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        tiles: tuple[int, int, int] | None = None,
        device=None,
        dtype=None,
    ) -> None:
        device = wp.get_preferred_device() if device is None else device
        dtype = wp.float32 if dtype is None else dtype

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias

        # NOTE: transposed
        self.weight = wp.empty((in_features, out_features), dtype=dtype, device=device, requires_grad=True)
        self.bias = wp.empty((1, out_features), dtype=dtype, device=device, requires_grad=True) if use_bias else None
        self.reset_parameters()

        self.tiles = tiles
        self.reset_kernels()

    def reset_parameters(self):
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.weight, -bound, bound)
        if self.use_bias:
            init.uniform_(self.bias, -bound, bound)

    def reset_kernels(self):
        self._kernel = None
        self._kernel_dims = None

    def __call__(self, x: Array[DType], params: dict[str, Array[DType]] = None, y: Array[DType] = None):
        B = x.shape[0]
        if params is None:
            params = {"weight": self.weight, "bias": self.bias}
        if y is None:
            y = wp.empty((B, self.out_features), dtype=x.dtype, device=x.device, requires_grad=x.requires_grad)
        if self._kernel is None:
            self._kernel, self._kernel_dims = linear_kernel(
                B, self.in_features, self.out_features, self.use_bias, self.tiles, x.dtype
            )
        inputs = [x, params["weight"], params["bias"]]
        wp.launch_tiled(
            self._kernel,
            dim=self._kernel_dims,
            inputs=inputs,
            outputs=[y],
            device=x.device,
            block_dim=wp.num_threads,
        )
        return y
