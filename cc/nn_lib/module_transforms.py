import equinox as eqx

from ..core import make_module_from_eqx_module
from .module_utils import filter_scan_module


def close_loop_transform(module1, module2, merge_x_y, y0, name="closed-loop"):
    class ClosedLoop(eqx.Module):
        def __call__(self, state, x):
            (module1, module2, y) = state
            module1, u = module1(merge_x_y(x, y))
            module2, y = module2(u)
            return (module1, module2, y), (y, u)

    return make_module_from_eqx_module(ClosedLoop(), (module1, module2, y0), name=name)


def module_input_transform(module, input_transform, name="input-transform"):
    class InputTransform(eqx.Module):
        def __call__(self, state, x):
            module = state
            x = input_transform(x)
            module, y = module(x)
            return module, y

    return make_module_from_eqx_module(InputTransform(), module, name=name)


def module_output_transform(module, output_transform, name="output_transform"):
    class OutputTransform(eqx.Module):
        def __call__(self, state, x):
            module = state
            module, y = module(x)
            y = output_transform(y)
            return module, y

    return make_module_from_eqx_module(OutputTransform(), module, name=name)


def unroll_module_transform(module, length=None, reset=True, name="unrolled-module"):
    def scan_fn(carry, x):
        module = carry
        new_module, y = module(x)
        return new_module, y

    class UnrolledModule(eqx.Module):
        def __call__(self, state, xs):
            module = state

            # TODO
            # this is not nice. Otherwise we can not propagate any gradients
            # to the initial state
            if reset:
                module = module.reset()

            _, ys = filter_scan_module(scan_fn, init=module, xs=xs, length=length)
            return module, ys

    return make_module_from_eqx_module(UnrolledModule(), module, name=name)
