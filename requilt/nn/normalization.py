import warp as wp
from warp.types import Array, DType

from ..warp_util import kernel as nested_kernel
from ._module import Module
from .utils import init


def layernorm_kernel(DIM_BATCH, DIM_IN, EPS, USE_WEIGHT, USE_BIAS, TILES, DTYPE):
    # if TILES is None:
    #     # TODO: default tile sizes heuristic
    #     raise RuntimeError

    EPS = wp.constant(EPS)
    TILE_M = wp.constant(1)  # TODO
    TILE_K = wp.constant(DIM_IN)

    @wp.func
    def abs_sq(x: float):
        return x * x

    @wp.func
    def rsqrt(x: float):
        return 1.0 / wp.sqrt(x)

    @nested_kernel
    def layernorm(
        x: wp.array2d(dtype=DTYPE),
        weight: wp.array2d(dtype=DTYPE),
        bias: wp.array2d(dtype=DTYPE),
        y: wp.array2d(dtype=DTYPE),
    ):
        i = wp.tid()

        K = x.shape[1]
        INV_K = 1.0 / float(K)

        a = wp.tile_load(x, shape=(TILE_M, TILE_K), offset=(i * TILE_M, 0))
        a_mean = wp.tile_sum(a) * INV_K
        a_abs_sq = wp.tile_map(abs_sq, a)
        a_mean2 = wp.tile_sum(a_abs_sq) * INV_K
        a_var = wp.tile_map(wp.max, a_mean2 - wp.tile_map(abs_sq, a_mean), wp.tile_zeros(shape=(TILE_M,), dtype=DTYPE))

        inv = wp.tile_map(rsqrt, a_var + EPS * wp.tile_ones(shape=(TILE_M,), dtype=DTYPE))
        centered = a - wp.tile_broadcast(a_mean, shape=(TILE_M, TILE_K))
        out = wp.tile_map(wp.mul, centered, wp.tile_broadcast(inv, shape=(TILE_M, TILE_K)))

        if wp.static(USE_WEIGHT):
            w = wp.tile_load(weight, shape=(1, TILE_K), offset=(0, 0))
            out = wp.tile_map(wp.mul, out, wp.tile_broadcast(w, shape=(TILE_M, TILE_K)))
        if wp.static(USE_BIAS):
            b = wp.tile_load(bias, shape=(1, TILE_K), offset=(0, 0))
            out = wp.tile_map(wp.add, out, wp.tile_broadcast(b, shape=(TILE_M, TILE_K)))
        wp.tile_store(y, out, offset=(i * TILE_M, 0))

    kernel_dims = (DIM_BATCH,)
    return layernorm, kernel_dims


class LayerNorm(Module):
    num_features: int
    eps: float
    use_weight: bool
    use_bias: bool

    weight: Array[DType] | None
    bias: Array[DType] | None

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        use_weight: bool = True,
        use_bias: bool = True,
        tiles: tuple[int] | None = None,
        device=None,
        dtype=None,
    ):
        device = wp.get_preferred_device() if device is None else device
        dtype = wp.float32 if dtype is None else dtype

        self.num_features = num_features
        self.eps = eps
        self.use_weight = use_weight
        self.use_bias = use_bias

        self.weight = (
            wp.empty(
                (
                    1,
                    num_features,
                ),
                dtype=dtype,
                device=device,
                requires_grad=True,
            )
            if use_weight
            else None
        )
        self.bias = (
            wp.empty(
                (
                    1,
                    num_features,
                ),
                dtype=dtype,
                device=device,
                requires_grad=True,
            )
            if use_bias
            else None
        )
        self.reset_parameters()

        self.tiles = tiles
        self.reset_kernels()

    def reset_parameters(self):
        if self.use_weight:
            init.ones_(self.weight)
        if self.use_bias:
            init.zeros_(self.bias)

    def reset_kernels(self):
        self._kernel = None
        self._kernel_dims = None

    def __call__(self, x: Array[DType], params: dict[str, Array[DType]] = None, y: Array[DType] = None):
        B = x.shape[0]
        if params is None:
            params = {"weight": self.weight, "bias": self.bias}
        if y is None:
            y = wp.empty((B, self.num_features), dtype=x.dtype, device=x.device, requires_grad=x.requires_grad)
        if self._kernel is None:
            self._kernel, self._kernel_dims = layernorm_kernel(
                B, self.num_features, self.eps, self.use_weight, self.use_bias, self.tiles, x.dtype
            )
        inputs, outputs = [x, params["weight"], params["bias"]], [y]
        wp.launch_tiled(
            self._kernel,
            dim=self._kernel_dims,
            inputs=inputs,
            outputs=outputs,
            device=x.device,
            block_dim=wp.num_threads,
        )
        return y
