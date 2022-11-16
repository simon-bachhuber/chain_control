import equinox as eqx

from ..core import AbstractController, AbstractModel


class WrapController(eqx.Module):
    controller: AbstractController

    def __call__(self, *args):
        controller, out = self.controller.step(*args)
        return WrapController(controller), out

    def reset(self):
        return WrapController(self.controller.reset())


class WrapModel(eqx.Module):
    model: AbstractModel

    def __call__(self, *args):
        model, out = self.model.step(*args)
        return WrapModel(model), out

    def reset(self):
        return WrapModel(self.model.reset())

    def y0(self):
        return self.model.y0()
