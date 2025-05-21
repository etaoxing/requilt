# Copyright 2025 The Newton Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from typing import Callable

import warp as wp
from warp.context import Module, get_module


# @kernel decorator to automatically set up modules based on nested
# function names
def kernel(
    f: Callable | None = None,
    *,
    enable_backward: bool | None = None,
    module: Module | None = None,
):
    """
    Decorator to register a Warp kernel from a Python function.
    The function must be defined with type annotations for all arguments.
    The function must not return anything.

    Example::

        @kernel
        def my_kernel(a: wp.array(dtype=float), b: wp.array(dtype=float)):
            tid = wp.tid()
            b[tid] = a[tid] + 1.0


        @kernel(enable_backward=False)
        def my_kernel_no_backward(a: wp.array(dtype=float, ndim=2), x: float):
            # the backward pass will not be generated
            i, j = wp.tid()
            a[i, j] = x


        @kernel(module="unique")
        def my_kernel_unique_module(a: wp.array(dtype=float), b: wp.array(dtype=float)):
            # the kernel will be registered in new unique module created just for this
            # kernel and its dependent functions and structs
            tid = wp.tid()
            b[tid] = a[tid] + 1.0

    Args:
        f: The function to be registered as a kernel.
        enable_backward: If False, the backward pass will not be generated.
        module: The :class:`warp.context.Module` to which the kernel belongs. Alternatively, if a string `"unique"` is provided, the kernel is assigned to a new module named after the kernel name and hash. If None, the module is inferred from the function's module.

    Returns:
        The registered kernel.
    """
    if module is None:
        # create a module name based on the name of the nested function
        # get the qualified name, e.g. "main.<locals>.nested_kernel"
        qualname = f.__qualname__
        parts = [part for part in qualname.split(".") if part != "<locals>"]
        outer_functions = parts[:-1]
        module = get_module(".".join([f.__module__] + outer_functions))

    return wp.kernel(f, enable_backward=enable_backward, module=module)
