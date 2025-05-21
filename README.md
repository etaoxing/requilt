# Requilt

Requilt is an optimization and neural network library for Warp.

# Setup

```bash
conda create -n requilt python=3.12
# linux only
pip install -U --pre warp-lang --extra-index-url=https://pypi.nvidia.com/
```

```bash
# for mnist data
pip install torch torchvision

python requilt/examples/mlp_mnist.py
```

# Acknowledgements

- Warp
- PyTorch
- JAX
- Flax
- Equinox
