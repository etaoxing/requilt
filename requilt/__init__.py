import warp as wp

wp.init()

wp.dtype = wp.float32
wp.device = wp.get_device()
wp.num_threads = 64

from . import nn, optim
